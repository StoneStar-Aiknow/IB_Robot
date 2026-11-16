"""Process-owned control transport for runtimes without a serving ROS host."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import queue
import threading
from contextlib import contextmanager
from uuid import uuid4

from inference_manifest.runtime_target import DEFAULT_ADMISSION_SAFETY_MARGIN_MS, validate_runtime_endpoint
from inference_service.ros_executor import release_cached_wait_set
from inference_service.unified_runtime.lifecycle import lifecycle_budget, lifecycle_lock

from .runtime_protocol import DEFAULT_RUNTIME_ENDPOINT, ControlError, NotStartedControlError


class RuntimeEnvironment:
    """Share infrastructure across in-process models, without business callbacks."""

    def __init__(self, *, endpoint=DEFAULT_RUNTIME_ENDPOINT, scheduler_enabled=False):
        validate_runtime_endpoint(endpoint)
        validate_runtime_endpoint(endpoint + "/ready")
        self.endpoint = endpoint
        self.scheduler_enabled = scheduler_enabled
        self.initialization_timeout = 60.0
        self.configured_target_ids = ()
        self._lock = threading.RLock()
        # Startup holds _lock across ROS construction. Finishing a creation
        # must still be able to release its transaction count within its budget.
        self._idle = threading.Condition()
        self._close_lock = threading.Lock()
        self._initializing = 0
        self._configuration_frozen = False
        self._domain_id = None
        self._closing = False
        self._node = None
        self._startup_cleanup = None
        self._handles = []
        self._closed = False
        self._retired = queue.Queue()
        self._stopping = threading.Event()

    def configure_from_node(self, node):
        """Inherit host configuration before the first runtime creation starts."""
        with self._lock:
            context = getattr(node, "context", None)
            domain_id = context.get_domain_id() if context is not None else None
            enabled = self._scheduler_enabled_from_node(node)
            if type(enabled) is not bool:
                raise ValueError("runtime_scheduler_enabled must be boolean")
            if self._configuration_frozen and enabled != self.scheduler_enabled:
                raise ValueError("cannot change runtime scheduling after runtime creation starts")
            if not enabled:
                self.scheduler_enabled = enabled
                return
            if self._configuration_frozen and domain_id != self._domain_id:
                raise ValueError("cannot change ROS domain after runtime creation starts")
            from rcl_interfaces.msg import ParameterDescriptor

            name = "runtime_scheduler_endpoint"
            endpoint = (
                node.get_parameter(name) if node.has_parameter(name) else node.declare_parameter(name, self.endpoint)
            ).value
            timeout = self._initialization_timeout_from_node(node)
            name = "configured_target_ids"
            instances = (
                node.get_parameter(name)
                if node.has_parameter(name)
                else node.declare_parameter(name, [], ParameterDescriptor(dynamic_typing=True))
            ).value
            if not isinstance(instances, list | tuple) or any(not isinstance(item, str) for item in instances):
                raise ValueError("configured_target_ids must be a list of full catalogue target IDs")
            instances = tuple(instances)
            if any(not item for item in instances):
                raise ValueError("configured_target_ids cannot contain empty identities")
            if self._configuration_frozen and instances != self.configured_target_ids:
                raise ValueError("cannot change model identities after runtime creation starts")
            if self._configuration_frozen and endpoint != self.endpoint:
                raise ValueError("cannot change runtime authority after runtime creation starts")
            if self._configuration_frozen and timeout != self.initialization_timeout:
                raise ValueError("cannot change runtime initialization timeout after runtime creation starts")
            validate_runtime_endpoint(endpoint)
            validate_runtime_endpoint(endpoint + "/ready")
            self.scheduler_enabled = enabled
            self.configured_target_ids = instances
            self.endpoint = endpoint
            self.initialization_timeout = float(timeout)
            self._domain_id = domain_id

    def _scheduler_enabled_from_node(self, node):
        # Both explicit-timeout configuration and default-budget discovery can
        # be the first entrant. Their check/declaration shares this short lock.
        with self._idle:
            name = "runtime_scheduler_enabled"
            return (
                node.get_parameter(name)
                if node.has_parameter(name)
                else node.declare_parameter(name, self.scheduler_enabled)
            ).value

    def _initialization_timeout_from_node(self, node):
        # Read the default before waiting for the transport/membership lock.
        # Serialize only parameter declaration with other creation entrants.
        with self._idle:
            enabled = self._scheduler_enabled_from_node(node)
            if not enabled:
                return self.initialization_timeout
            name = "runtime_initialization_timeout"
            timeout = (
                node.get_parameter(name)
                if node.has_parameter(name)
                else node.declare_parameter(name, self.initialization_timeout)
            ).value
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("runtime initialization timeout must be finite and positive")
        return float(timeout)

    def _start(self):
        if self._closed:
            raise RuntimeError("runtime environment is closed")
        if self._startup_cleanup is not None:
            self._startup_cleanup.close()
            self._startup_cleanup = None
            self._node = None
        if self._node is not None:
            return
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node

        from inference_service.unified_runtime.loading import LoadRollback

        from .runtime_ros import _service_type

        rollback = LoadRollback()
        try:
            self._context = Context()
            rollback.defer(self._context.try_shutdown)
            self._context.init(domain_id=self._domain_id)
            self._domain_id = self._context.get_domain_id()
            name = f"model_runtime_{uuid4().hex}"
            self._node = Node(name, context=self._context, use_global_arguments=False)
            rollback.defer(self._node.destroy_node)
            # Only control callbacks run on this thread; retirement uses its
            # quiescent boundary between callbacks.
            self._executor = SingleThreadedExecutor(context=self._context)
            rollback.defer(self._executor.shutdown)
            self._executor.add_node(self._node)
            self._client = self._node.create_client(_service_type(), self.endpoint)
            self._spin = threading.Thread(target=self._spin_control, name="model-runtime-control", daemon=True)
            self._spin.start()
        except BaseException:
            self._startup_cleanup = rollback
            rollback.close()
            self._startup_cleanup = None
            self._node = None
            raise
        rollback.commit()

    def _spin_control(self):
        try:
            while not self._stopping.is_set():
                self._executor.spin_once(timeout_sec=0.1)
                if self._retired.empty():
                    continue
                release_cached_wait_set(self._executor)
                for _ in range(self._retired.qsize()):
                    adapter = self._retired.get_nowait()
                    try:
                        adapter.destroy_transport(_quiescent=True)
                    except Exception:
                        self._retired.put(adapter)
                        logging.getLogger(__name__).exception("Runtime transport retirement will retry")
                    finally:
                        del adapter
                        self._retired.task_done()
        finally:
            self._executor.shutdown()

    def retire(self, handle, bindings):
        """Forget handles after proven cleanup and Global ownership release."""
        # Transport retirement may wait for Global. Never keep the process
        # membership lock while another model is constructing or retiring.
        for binding in bindings:
            binding.adapter.close()
        with lifecycle_lock(self._lock):
            if handle not in self._handles:
                return
            for binding in bindings:
                # A clean construction rollback may retire a bound handle
                # before its public Close ever ran. Stop callbacks first.
                self._retired.put(binding.adapter)
            self._handles.remove(handle)
            if self._node is not None:
                self._executor.wake()

    def prepare(
        self, validated, options, timeout, request_timeout_ms, *, configured_target_id=None, adapter="model_runtime"
    ):
        """Resolve a profile template and allocate a distinct lifecycle identity."""
        from inference_manifest.runtime_identity import physical_device, runtime_composition_fingerprint
        from inference_service.unified_runtime.errors import DeadlineExceeded
        from inference_service.unified_runtime.lifecycle import check_startup, wait_for_startup

        from .runtime_ros import CONTROL_RESPONSE_WAIT_SECONDS, _reply, _request

        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("runtime initialization timeout must be finite and positive")
        device = physical_device(validated, options, torch_device_index=options.get("torch_device_index"))
        with lifecycle_budget(timeout), self.initialization(timeout=timeout):
            with lifecycle_lock(self._lock):
                self._start()
            try:
                available = wait_for_startup(
                    lambda seconds: self._client.wait_for_service(timeout_sec=seconds), timeout
                )
            except DeadlineExceeded as exc:
                raise NotStartedControlError("GLOBAL_UNAVAILABLE") from exc
            if not available:
                raise NotStartedControlError("GLOBAL_UNAVAILABLE")
            composition = runtime_composition_fingerprint(adapter, options)
            request = _request(
                {
                    "verb": "describe_resource",
                    "physical_device": device,
                    "deployment_fingerprint": validated.fingerprint,
                    "composition_fingerprint": composition,
                    "configured_target_id": configured_target_id,
                    "configured_target_ids": list(self.configured_target_ids),
                }
            )
            try:
                while True:
                    check_startup()
                    # Discovery is read-only. Retry lost responses within the
                    # inherited creation budget, without beginning SDK work.
                    future = self._client.call_async(request)
                    done = threading.Event()
                    future.add_done_callback(lambda _, signal=done: signal.set())
                    try:
                        if wait_for_startup(done.wait, CONTROL_RESPONSE_WAIT_SECONDS):
                            resource = _reply(future.result())
                            break
                    finally:
                        self._client.remove_pending_request(future)
            except DeadlineExceeded as exc:
                raise NotStartedControlError("RESOURCE_DISCOVERY_TIMEOUT") from exc
            if resource.get("mode", "serial") == "ascend_concurrent":
                contract = getattr(validated.deployment, "execution_contract", None)
                if validated.deployment.backend != "ascend" or resource.get("verified_concurrency") is not True:
                    raise NotStartedControlError("BACKEND_CONCURRENCY_UNAVAILABLE")
                if contract is not None and (contract.stateful or contract.state_links):
                    raise NotStartedControlError("STATEFUL_CONCURRENCY_UNSUPPORTED")
            configured = resource.get("runtime_target")
            if configured is not None:
                from .runtime_protocol import RuntimeExecutionTarget

                identity = uuid4().hex
                configured = {
                    **configured,
                    "template_only": False,
                    "target_id": f"runtime/{identity}",
                    "lifecycle_group": f"runtime/{identity}",
                    "endpoint": f"/inference/runtime/hosts/runtime/r_{identity}",
                }
                return self._node, RuntimeExecutionTarget(**configured).to_dict()
            if resource.get("predictive_admission"):
                error = NotStartedControlError("RUNTIME_PROFILE_REQUIRED")
                error.args = (
                    f"RUNTIME_PROFILE_REQUIRED: prediction on {device} requires a configured runtime target "
                    f"and offline profile for deployment={validated.fingerprint}, composition={composition}; "
                    "no matching configured target/profile is available; this stage supports policy/perception "
                    "targets only, and local-model profile configuration is not supported yet",
                )
                raise error
            identity = uuid4().hex
            target_id = f"runtime/{identity}"
            model = validated.manifest.model
            target = {
                "target_id": target_id,
                "endpoint": f"/inference/runtime/hosts/runtime/r_{identity}",
                "lifecycle_group": target_id,
                "hardware_resource_id": resource["hardware_resource_id"],
                "physical_device": device,
                "resource_mode": resource.get("mode", "serial"),
                "deployment_fingerprint": validated.fingerprint,
                "composition_fingerprint": composition,
                "configuration_generation": hashlib.sha256(json.dumps(resource, sort_keys=True).encode()).hexdigest(),
                "interface": model.interface,
                "model_type": model.model_type,
                "operation": model.operation or "policy",
                "hardware_profile_fingerprint": resource.get("hardware_profile_fingerprint", ""),
                "admission_safety_margin_ms": resource.get(
                    "admission_safety_margin_ms", DEFAULT_ADMISSION_SAFETY_MARGIN_MS
                ),
                "request_timeout_ms": request_timeout_ms,
            }
            from .runtime_protocol import RuntimeExecutionTarget

            return self._node, RuntimeExecutionTarget(**target).to_dict()

    def retain(self, handle):
        with lifecycle_lock(self._lock):
            if self._closed or (self._closing and not self._initializing):
                raise RuntimeError("runtime environment is closed")
            if handle not in self._handles:
                self._handles.append(handle)

    @contextmanager
    def initialization(self, node=None, *, timeout=None):
        """Configure and freeze atomically before admitting a model constructor."""
        if timeout is None:
            timeout = self.initialization_timeout if node is None else self._initialization_timeout_from_node(node)
        with lifecycle_budget(timeout):
            with lifecycle_lock(self._lock):
                if self._closed or self._closing:
                    raise RuntimeError("runtime environment is closed")
                if node is not None:
                    self.configure_from_node(node)
                self._configuration_frozen = True
                with self._idle:
                    self._initializing += 1
            try:
                yield
            finally:
                with self._idle:
                    self._initializing -= 1
                    self._idle.notify_all()

    @staticmethod
    def publish(bindings, timeout):
        from inference_service.scheduler.runtime_binding import closing_control_transport
        from inference_service.unified_runtime import Deadline, ExecutionContext
        from inference_service.unified_runtime.lifecycle import check_startup, lifecycle_deadline

        context = ExecutionContext("runtime-registration", lifecycle_deadline(Deadline.after(timeout)))
        for binding in bindings:
            with closing_control_transport(binding.adapter.node):
                check_startup()
                binding.adapter.wait_for_registration(context, startup=True)
                binding.adapter.refresh_registration(context, startup=True)
                check_startup()

    def close(self):
        from inference_service.unified_runtime.lifecycle import lifecycle_deadline, lifecycle_lock, lifecycle_remaining

        with lifecycle_budget(self.initialization_timeout), lifecycle_lock(self._close_lock):
            if self._closed:
                return
            with lifecycle_lock(self._lock):
                self._closing = True
            with lifecycle_lock(self._idle):
                while self._initializing:
                    lifecycle_deadline().check("runtime_environment_initialization")
                    self._idle.wait(timeout=lifecycle_remaining(0.05))
            if self._startup_cleanup is not None:
                self._startup_cleanup.close()
                self._startup_cleanup = None
                self._node = None
            # Any uncertain Close keeps the control transport and providers
            # alive for reconciliation, but does not prevent independent owners
            # from closing within the remaining shared shutdown budget.
            errors = []
            for handle in tuple(self._handles):
                try:
                    lifecycle_deadline().check("runtime_environment_close")
                    handle.close()
                except Exception as exc:
                    errors.append(exc)
            if self._node is not None:
                from .runtime_initialization import settle_initializations

                try:
                    settle_initializations(self._node)
                except Exception as exc:
                    errors.append(exc)
            if errors:
                raise errors[0]
            if self._node is not None:
                # Keep the control thread alive until every retired transport
                # has either completed or can be retried by this owner.
                while self._retired.unfinished_tasks:
                    lifecycle_deadline().check("runtime_transport_retirement")
                    self._executor.wake()
                    self._stopping.wait(lifecycle_remaining(0.05))
                self._stopping.set()
                self._executor.wake()
                self._spin.join(timeout=lifecycle_remaining(5))
                if self._spin.is_alive():
                    raise ControlError("CONTROL_SHUTDOWN_PENDING")
                self._node.destroy_node()
                self._context.shutdown()
            self._handles.clear()
            while not self._retired.empty():
                self._retired.get_nowait()
            self._closed = True


__all__ = ["RuntimeEnvironment"]
