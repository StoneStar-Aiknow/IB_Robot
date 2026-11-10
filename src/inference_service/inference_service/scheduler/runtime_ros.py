"""Thin ROS control transport; images, point clouds and results never cross Global."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import TimeoutError
from contextlib import suppress
from uuid import uuid4

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.qos import QoSProfile

from inference_manifest.runtime_target import (
    MAX_RUNTIME_GROUP_MEMBERS,
    validate_runtime_endpoint,
    validate_runtime_group_budget,
)
from inference_service.ros_executor import release_cached_wait_set
from inference_service.unified_runtime import Deadline, ExecutionContext, ModelRequest
from inference_service.unified_runtime.contracts import OutcomeEvidence, OutcomeState
from inference_service.unified_runtime.errors import CancellationRequested, DeadlineExceeded, ExecutionFailure
from inference_service.unified_runtime.lifecycle import lifecycle_budget, lifecycle_lock

from .runtime_controller import RuntimeExecutionController
from .runtime_profiles import DEFAULT_MIN_SAMPLES, PRODUCTION_MIN_SAMPLES
from .runtime_protocol import (
    DEFAULT_CONTROL_RPC_TIMEOUT_NS,
    DEFAULT_RUNTIME_ENDPOINT,
    DEFAULT_STATUS_STALE_TIMEOUT_NS,
    HOST_HEARTBEAT_INTERVAL_NS,
    PROTOCOL_VERSION,
    RUNTIME_HANDLE_OPERATION,
    TERMINAL,
    CompletedControlError,
    ControlError,
    NotStartedControlError,
    RuntimeExecutionTarget,
    Submission,
    decode_control,
    encode_control,
)
from .runtime_scheduler import RuntimeScheduler

LOGGER = logging.getLogger(__name__)
HOST_QUERY_INTERVAL_SECONDS = 0.25
CONTROL_RETRY_INTERVAL_SECONDS = 0.25
CONTROL_RESPONSE_WAIT_SECONDS = 0.25
BUSINESS_POLL_INTERVAL_SECONDS = 0.25
# A register RPC wait budget, independent of the Global status freshness policy.
REGISTRATION_RESPONSE_TIMEOUT_SECONDS = 2.0
TERMINAL_RECEIPT_TIMEOUT_SECONDS = 2.0
# Per Host callback work and outstanding RPCs, independent of retained history.
TERMINAL_REPORT_BATCH_SIZE = 32
MAX_TERMINAL_RECEIPTS = 64
_RETRYABLE_REGISTRATION_ERRORS = frozenset({"STALE_STATUS", "RESET_COMPLETION_EVIDENCE_PENDING"})


class _RemoteControlError(ControlError):
    """An explicit control-service rejection, distinct from reply decoding failure."""

    def __init__(self, code, *, not_started=False):
        super().__init__(code)
        self.operation_started = not not_started
        self.outcome_known = not_started


def _service_type():
    # Disabled deployments never import or create the new generated interface.
    from ibrobot_msgs.srv import RuntimeControlV1

    return RuntimeControlV1


def _request(payload):
    request = _service_type().Request()
    request.payload = encode_control(payload)
    return request


def _reply(response):
    payload = decode_control(response.payload)
    if "control_error" in payload:
        raise _RemoteControlError(payload["control_error"], not_started=payload.get("not_started") is True)
    return payload


def _serve(handler):
    def callback(request, response):
        verb = "decode"
        try:
            message = decode_control(request.payload)
            verb = str(message.get("verb", "missing"))[:64]
            response.payload = encode_control(handler(message))
        except Exception as exc:
            if not isinstance(exc, ControlError):
                LOGGER.exception("Runtime control handler failed (verb=%s)", verb)
            error = {"control_error": getattr(exc, "code", "INVALID_CONTROL")}
            if getattr(exc, "operation_started", None) is False or (
                isinstance(exc, ExecutionFailure)
                and exc.evidence.state is OutcomeState.NOT_STARTED
                and exc.evidence.outcome_known
            ):
                error["not_started"] = True
            response.payload = encode_control(error)
        return response

    return callback


class GlobalRuntimeEndpoint:
    """Generic control transport owned exclusively by GlobalRuntimeSchedulerNode."""

    def __init__(
        self,
        node,
        targets,
        *,
        endpoint=DEFAULT_RUNTIME_ENDPOINT,
        policy="fifo",
        resources=None,
        max_registered_targets=256,
        stale_ns=DEFAULT_STATUS_STALE_TIMEOUT_NS,
        control_rpc_timeout_ns=DEFAULT_CONTROL_RPC_TIMEOUT_NS,
        profile_min_samples=DEFAULT_MIN_SAMPLES,
        profile_max_age_days=30,
    ):
        validate_runtime_endpoint(endpoint)
        validate_runtime_endpoint(endpoint + "/ready")
        if type(control_rpc_timeout_ns) is not int or not 0 < control_rpc_timeout_ns <= 2**63 - 1:
            raise ValueError("control RPC timeout must be a positive int64 number of nanoseconds")
        if type(profile_min_samples) is not int or profile_min_samples < PRODUCTION_MIN_SAMPLES:
            raise ValueError(f"profile_min_samples must be >= {PRODUCTION_MIN_SAMPLES} for a ROS endpoint")
        self.node = node
        self.control_rpc_timeout_ns = control_rpc_timeout_ns
        self._dynamic_registration = resources is not None
        self.core = RuntimeScheduler(
            targets,
            policy=policy,
            resources=resources,
            max_registered_targets=max_registered_targets,
            stale_ns=stale_ns,
            profile_min_samples=profile_min_samples,
            profile_max_age_days=profile_max_age_days,
        )
        self.endpoint = endpoint
        if any(target.endpoint in {endpoint, endpoint + "/ready"} for target in targets):
            raise ValueError("runtime global endpoint conflicts with a host endpoint")
        self._pending = {}
        self._control_schedule = {}
        from .runtime_recovery import ResourceRecovery

        self.recovery = ResourceRecovery(self)
        self._client_endpoints = {}
        self._client_errors = {}
        self._clients = {}
        self._retired_clients = []
        self.service = self._dispatch_guard = self.timer = None
        try:
            self._sync_clients()
            self.service = node.create_service(
                _service_type(),
                endpoint,
                _serve(self._handle),
                callback_group=ReentrantCallbackGroup(),
                qos_profile=QoSProfile(depth=16),
            )
            group = MutuallyExclusiveCallbackGroup()
            self._dispatch_guard = node.create_guard_condition(self._tick, callback_group=group)
            self.timer = node.create_timer(0.01, self._tick, callback_group=group)
        except BaseException:
            for entity, destroy in (
                (self.timer, node.destroy_timer),
                (self._dispatch_guard, node.destroy_guard_condition),
                (self.service, node.destroy_service),
                *((client, node.destroy_client) for client in self._clients.values()),
            ):
                if entity is not None:
                    with suppress(Exception):
                        destroy(entity)
            raise

    def _sync_clients(self):
        """Change transport membership under the same lock as the resource ledger."""
        for target_id, target in self.core.targets.items():
            if target_id in self._clients and self._client_endpoints[target_id] != target.endpoint:
                # A retired identity may be reused only after its ledger has
                # been collected. Dispose late transport before rebinding it.
                old_client = self._clients[target_id]
                for key, (future, _, pending_target, submission) in tuple(self._pending.items()):
                    if pending_target.target_id != target_id:
                        continue
                    if self.core.has_operation(submission.operation_id):
                        raise ControlError("TARGET_TRANSPORT_BUSY")
                    old_client.remove_pending_request(future)
                    del self._pending[key]
                self._retire_client(self._clients.pop(target_id))
            if target_id not in self._clients:
                # Bound transport while the spin owner reaches a quiescent
                # collection boundary; retry membership on the next tick.
                if len(self._clients) + len(self._retired_clients) >= self.core.max_registered_targets * 2:
                    continue
                try:
                    self._clients[target_id] = self.node.create_client(
                        _service_type(),
                        target.endpoint,
                        callback_group=ReentrantCallbackGroup(),
                        qos_profile=QoSProfile(depth=8),
                    )
                except Exception as exc:
                    if self.service is None:
                        raise  # Initial endpoint construction still rolls back atomically.
                    error = f"{type(exc).__name__}: {str(exc)[:256]}"
                    if self._client_errors.get(target_id) != error:
                        LOGGER.warning("Runtime client creation will retry (target=%s): %s", target_id, error)
                    self._client_errors[target_id] = error
                    continue
                self._client_errors.pop(target_id, None)
                self._client_endpoints[target_id] = target.endpoint
        self._client_errors = {key: value for key, value in self._client_errors.items() if key in self.core.targets}
        retained = {target.target_id for _, _, target, _ in self._pending.values()}
        retained.update(
            target.target_id
            for record in self.recovery.records.values()
            if record["state"] == "running"
            for target in record["targets"]
        )
        for target_id in tuple(self._clients):
            if target_id not in self.core.targets and target_id not in retained:
                self._retire_client(self._clients.pop(target_id))
                self._client_endpoints.pop(target_id, None)

    def _retire_client(self, client):
        if getattr(self.node, "executor", None) is None:
            self.node.destroy_client(client)
        else:
            self._retired_clients.append(client)

    def collect_retired_clients(self, executor):
        """Call on the sole spin thread between spin_once calls (never a callback).

        Drop Humble's cached wait set before destroying entities. A client
        already handed to a worker remains retained until its response is taken.
        """
        with self.core.transaction():
            if not self._retired_clients:
                return
            release_cached_wait_set(executor)
            for client in tuple(self._retired_clients):
                if client._executor_event:
                    continue
                try:
                    self.node.destroy_client(client)
                except Exception:
                    LOGGER.exception("Retired Global client cleanup will retry")
                else:
                    self._retired_clients.remove(client)

    def _handle(self, message):
        if self.timer is None:
            raise NotStartedControlError("GLOBAL_NOT_READY")
        verb = message["verb"]
        if verb == "describe_resource":
            with self.core.transaction():
                for resource_id, resource in (self.core.resources or {}).items():
                    if resource["physical_device"] == message.get("physical_device"):
                        target = self.core.discover_target(
                            resource_id,
                            message.get("deployment_fingerprint"),
                            message.get("composition_fingerprint"),
                            message.get("configured_target_id"),
                            message.get("configured_target_ids", ()),
                        )
                        return {**resource, "hardware_resource_id": resource_id, "runtime_target": target}
                raise ControlError("UNKNOWN_RESOURCE")
        if verb == "inspect_targets":
            ids = message.get("target_ids")
            if (
                not isinstance(ids, list)
                or not 1 <= len(ids) <= MAX_RUNTIME_GROUP_MEMBERS
                or any(not isinstance(t, str) for t in ids)
            ):
                raise ControlError("INVALID_RUNTIME_GROUP")
            with self.core.transaction():
                return {
                    "epoch": self.core.epoch,
                    "targets": {tid: self.core.targets[tid].to_dict() for tid in ids if tid in self.core.targets},
                    "boots": {tid: self.core.host_boot(tid) for tid in ids},
                    "reason": self.core.targets_ready_reason(ids),
                    "active": self.core.has_active_target(ids),
                }
        if verb == "describe":
            offset = message.get("offset", 0)
            if type(offset) is not int or offset < 0:
                raise ControlError("INVALID_TARGET_OFFSET")
            with self.core.transaction():
                targets = list(self.core.targets.values())
                return {
                    "protocol_version": PROTOCOL_VERSION,
                    "epoch": self.core.epoch,
                    "clock_id": self.core.clock_id,
                    "target_count": len(targets),
                    "targets": [target.to_dict() for target in targets[offset : offset + 1]],
                    "measurement_count": len(self.core.measurements),
                }
        if verb in {"register", "register_group", "begin_initialization"}:
            statuses = message["statuses"] if verb != "register" else [message["status"]]
            if not isinstance(statuses, list) or not 1 <= len(statuses) <= MAX_RUNTIME_GROUP_MEMBERS:
                raise ControlError("INVALID_RUNTIME_GROUP")
            # Core registration validates descriptors. Do not reconstruct the
            # same immutable target again on every transport heartbeat.
            if any(
                status["target"].get("endpoint") in {self.endpoint, self.endpoint + "/ready"} for status in statuses
            ):
                raise ControlError("TARGET_ENDPOINT_CONFLICT")
            if message.get("pipeline") is not None:
                raise ControlError("PRODUCT_REGISTRATION_REQUIRES_POLICY_SERVICE")
            if verb == "begin_initialization":
                return self.core.begin_initialization(
                    statuses, message["token"], accept_until_ns=message["accept_until_ns"]
                )
            result = self.core.register_group(statuses) if verb == "register_group" else self.core.register(statuses[0])
            with self.core.transaction():
                if any(
                    self._client_endpoints.get(status["target"]["target_id"]) != status["target"]["endpoint"]
                    or status["target"]["target_id"] not in self._clients
                    for status in statuses
                ):
                    self._sync_clients()
            return {**result, "dynamic_registration": self._dynamic_registration, "stale_ns": self.core.stale_ns}
        if verb == "abort_initialization":
            if message.get("clock_id") != self.core.clock_id:
                raise ControlError("CLOCK_MISMATCH")
            result = self.core.abort_initialization(
                message["resource"], message["token"], accept_until_ns=message["accept_until_ns"]
            )
            self._wake()
            return result
        if message.get("epoch") != self.core.epoch:
            raise ControlError("EPOCH_MISMATCH")
        if verb == "check_initialization":
            return self.core.check_initialization(message["resource"], message["token"])
        if verb == "finish_initialization":
            result = self.core.finish_initialization(
                message["resource"],
                message["token"],
                clean=message["clean"],
                retire=message.get("retire", False),
                acknowledged=message.get("acknowledged", False),
            )
            self._wake()
            return result
        if verb == "recover":
            with self.core.transaction():
                return self.recovery.request(message)
        if verb == "unregister":
            with self.core.transaction():
                result = self.core.unregister(message["status"])
                self.recovery.settle_retired()
                return result
        if verb == "terminal_report":
            result = self.core.terminal_report(message["report"])
            self._wake()
            return result
        if verb == "measurements":
            return self.core.read_measurements(message.get("after_sequence", 0))
        if verb not in {"submit", "cancel", "query"}:
            raise ControlError("INVALID_VERB")
        submission = Submission(**message["submission"])
        if verb == "submit":
            result = self.core.submit(submission)
            self._wake()
            return result
        if verb == "cancel":
            return self.core.cancel(submission)
        if verb == "query":
            return self.core.query(submission)
        raise ControlError("INVALID_VERB")

    def _send(self, target, submission, verb):
        key = (submission.operation_id, verb)
        if key in self._pending:
            return
        if verb in {"query", "cancel", "ack"}:
            now = time.monotonic()
            interval = HOST_QUERY_INTERVAL_SECONDS if verb == "query" else CONTROL_RETRY_INTERVAL_SECONDS
            # Queries retain their initial reconciliation delay. Cleanup starts
            # immediately, but prompt replies must not drive a busy retry loop.
            next_send = self._control_schedule.setdefault(key, now + interval if verb == "query" else now)
            if now < next_send:
                return
            self._control_schedule[key] = now + interval
        client = self._clients.get(target.target_id)
        try:
            ready = client is not None and client.service_is_ready()
            request = _request({"verb": verb, "epoch": self.core.epoch, "submission": submission.to_dict()})
        except Exception:
            LOGGER.exception("Runtime control request preparation failed (target=%s, verb=%s)", target.target_id, verb)
            if verb == "execute":
                self.core.reject_unsent(submission, "CONTROL_SEND_FAILED")
            return
        if not ready:
            if verb == "execute":
                self.core.reject_unsent(submission, "HOST_UNAVAILABLE")
            return
        try:
            future = client.call_async(request)
        except Exception:
            # call_async may have handed the request to middleware already.
            LOGGER.exception("Runtime control send failed (target=%s, verb=%s)", target.target_id, verb)
            self.core.mark_transport_unknown(submission)
            return
        self._pending[key] = (future, time.monotonic_ns(), target, submission)
        future.add_done_callback(lambda _: self._wake())

    def _wake(self):
        guard = getattr(self, "_dispatch_guard", None)
        if guard is not None:
            with suppress(Exception):  # Node teardown can race a late reply.
                guard.trigger()

    def _tick(self):
        if self.timer is None:
            return
        with self.core.transaction():
            self._tick_locked()

    def _tick_locked(self):
        self.recovery.tick()
        for key, (future, started, target, submission) in tuple(self._pending.items()):
            if not future.done():
                if time.monotonic_ns() - started > self.control_rpc_timeout_ns:
                    self._clients[target.target_id].remove_pending_request(future)
                    del self._pending[key]
                    self.core.mark_transport_unknown(submission)
                continue
            del self._pending[key]
            try:
                try:
                    report = _reply(future.result())
                except _RemoteControlError as exc:
                    if key[1] != "execute" or exc.operation_started:
                        raise
                    # An explicit first-Execute rejection releases the same
                    # reservation as an unsent command. Cancel/ACK still retire
                    # the Host snapshot; contradictory evidence stays UNKNOWN.
                    self.core.reject_unsent(submission, exc.code)
                    continue
                if key[1] == "ack":
                    if report.get("state") != "acknowledged":
                        raise ControlError("INVALID_ACK_RESPONSE")
                    self.core.acknowledge_host(submission)
                    continue
                if Submission(**report["submission"]) != submission:
                    raise ControlError("SUBMISSION_MISMATCH")
                self.core.report(report)
            except Exception:
                LOGGER.exception("Runtime control reply failed (target=%s, verb=%s)", target.target_id, key[1])
                self.core.mark_transport_unknown(submission)
        for target, submission in self.core.dispatchable():
            self._send(target, submission, "execute")
        commands = self.core.pending_reports()
        for target, submission, verb in commands:
            self._send(target, submission, verb)
        self.core.collect()
        # A terminal operation still needs Cancel/ACK until the host confirms
        # cleanup. Retire schedules when their command is no longer needed.
        needed = {(submission.operation_id, verb) for _, submission, verb in commands}
        self._control_schedule = {key: value for key, value in self._control_schedule.items() if key in needed}
        self._sync_clients()


class SchedulerRuntimeAdapter:
    """Local business adapter submits immutable references and waits for evidence."""

    def __init__(
        self,
        node,
        target,
        handle,
        *,
        endpoint=DEFAULT_RUNTIME_ENDPOINT,
        boot_id=None,
        registration_enabled=True,
    ):
        validate_runtime_endpoint(endpoint)
        self.target = RuntimeExecutionTarget(**target) if isinstance(target, dict) else target
        from inference_manifest.runtime_identity import validate_runtime_device_visibility

        try:
            validate_runtime_device_visibility(self.target.physical_device, self.target.target_id)
        except ValueError as exc:
            raise ControlError("DEVICE_VISIBILITY_REMAP_UNSUPPORTED") from exc
        owner = getattr(handle, "owner", handle)
        assembly = getattr(owner, "assembly", None)
        load_context = getattr(assembly, "load_context", None)
        validated = getattr(load_context, "validated_manifest", None)
        if validated is not None:
            from inference_manifest.runtime_identity import physical_device

            if not handle.local_ready:
                raise ControlError("RUNTIME_LOAD_REQUIRED")
            if (
                validated.fingerprint != self.target.deployment_fingerprint
                or physical_device(
                    validated,
                    load_context.runtime_options,
                    torch_device_index=getattr(assembly.session, "runtime_device_index", None),
                )
                != self.target.physical_device
            ):
                raise ControlError("LOADED_RUNTIME_IDENTITY_MISMATCH")
        self._validate_backend(handle)
        if boot_id is not None:
            from .runtime_protocol import uuid_value

            uuid_value(boot_id)
        self.controller = None
        self.node = node
        self._registration_enabled = registration_enabled
        self._registration_peers = (self,)
        self._terminal_receipts = {}
        self._retirement_replay_cursor = 0
        self._recovery_lock = threading.Lock()
        self._recovery_requests = {}
        self._reset_lock = threading.Lock()
        self._reset_requests = {}
        self._registration = None
        self._registration_started = 0
        self._resource_ready = False
        self._last_registered = 0.0
        self._stale_seconds = DEFAULT_STATUS_STALE_TIMEOUT_NS / 1e9
        self._registration_error = ""
        self._closing = False
        self._close_requested = False
        self._close_lock = threading.Lock()
        self._client = self._service = self._timer = self._terminal_guard = None
        try:
            # These callbacks only exchange control evidence; native work has
            # its own workers. One group also supplies the retirement barrier.
            group = MutuallyExclusiveCallbackGroup()
            self._client = node.create_client(
                _service_type(), endpoint, callback_group=group, qos_profile=QoSProfile(depth=8)
            )
            self._service = node.create_service(
                _service_type(),
                self.target.endpoint,
                _serve(self._handle),
                callback_group=group,
                qos_profile=QoSProfile(depth=8),
            )
            self._terminal_guard = node.create_guard_condition(self._terminal_tick, callback_group=group)
            self._timer = node.create_timer(HOST_HEARTBEAT_INTERVAL_NS / 1e9, self._heartbeat, callback_group=group)
            # ROS constructors validate service names and may fail. Bind only
            # after all entities exist: Handle ownership cannot be rolled back.
            controller = RuntimeExecutionController(self.target, handle)
            if boot_id is not None:
                controller.boot_id = boot_id
            controller.policy.delegate = self
            self.controller = controller
            controller.on_terminal = self._notify_terminal
        except BaseException:
            for name, destroy in (
                ("_timer", node.destroy_timer),
                ("_terminal_guard", node.destroy_guard_condition),
                ("_service", node.destroy_service),
                ("_client", node.destroy_client),
            ):
                entity = getattr(self, name)
                if entity is not None:
                    with suppress(Exception):
                        destroy(entity)
            raise

    def _validate_backend(self, handle):
        owner = getattr(handle, "owner", handle)
        assembly = getattr(owner, "assembly", None)
        session = getattr(assembly, "session", None)
        capacity = getattr(assembly, "max_active_executions", 1)
        if session is not None:
            capacity = min(capacity, getattr(getattr(session, "capabilities", None), "max_in_flight_per_instance", 1))
        if self.target.max_in_flight > capacity:
            raise ControlError("INSTANCE_CAPACITY_MISMATCH")
        if (
            handle.local_ready
            and self.target.runtime_version
            and getattr(session, "runtime_version", "") != self.target.runtime_version
        ):
            raise ControlError("RUNTIME_VERSION_MISMATCH")
        if handle.local_ready and self.target.resource_mode == "ascend_concurrent":
            capabilities = getattr(session, "capabilities", None)
            mapping = getattr(capabilities, "priority_mapping", None)
            if mapping is None:
                raise ControlError("BACKEND_PRIORITY_UNAVAILABLE")
            evidence = getattr(capabilities, "admission_evidence", None)
            if not (
                getattr(capabilities, "supports_isolated_stage_execution", False)
                and getattr(capabilities, "supports_multiple_instances", False)
                and evidence is not None
                and all(
                    getattr(evidence, name, False)
                    for name in (
                        "sdk_initialization",
                        "multi_instance_execution",
                        "failure_isolation",
                        "independent_close",
                    )
                )
            ):
                raise ControlError("BACKEND_CONCURRENCY_UNAVAILABLE")

    def _heartbeat(self):
        if self.controller is None:
            return
        with self.controller.transaction():
            if getattr(self, "_closing", False):
                return
            self._heartbeat_locked()

    def _heartbeat_locked(self):
        self._flush_terminal_evidence()
        if not self._registration_enabled:
            self.controller.collect()
            return
        peers = self._registration_peers
        if any(peer._closing for peer in peers):
            return
        if self._registration is not None:
            if not self._registration.done():
                if time.monotonic() - self._registration_started > REGISTRATION_RESPONSE_TIMEOUT_SECONDS:
                    self._client.remove_pending_request(self._registration)
                    self._registration = None
                    for peer in peers:
                        peer._set_registration_error("REGISTRATION_TIMEOUT")
                return
            try:
                reply = _reply(self._registration.result())
                for peer in peers:
                    peer._apply_registration(reply, self._registration_started)
            except Exception as exc:
                for peer in peers:
                    peer._set_registration_error(getattr(exc, "code", "REGISTRATION_FAILED"))
            self._registration = None
        self.controller.collect()
        try:
            if self._client.service_is_ready():
                statuses = []
                for peer in peers:
                    status = peer.controller.status()
                    try:
                        peer._validate_backend(peer.controller.handle)
                    except ControlError:
                        peer.controller.fence()
                        status.update(ready=False, clean=False)
                    statuses.append(status)
                payload = (
                    {"verb": "register_group", "statuses": statuses}
                    if len(peers) > 1
                    else {"verb": "register", "status": statuses[0]}
                )
                self._registration_started = min(status["status_ns"] for status in statuses) / 1e9
                self._registration = self._client.call_async(_request(payload))
        except Exception as exc:
            LOGGER.exception("Runtime heartbeat send failed (target=%s)", self.target.target_id)
            self._set_registration_error(getattr(exc, "code", "REGISTRATION_SEND_FAILED"))

    def _apply_registration(self, reply, sent_at):
        with self.controller.transaction():
            self.controller.accept_epoch(reply["epoch"])
            self._stale_seconds = reply.get("stale_ns", DEFAULT_STATUS_STALE_TIMEOUT_NS) / 1e9
            self._resource_ready = reply["resource_ready"]
            self._last_registered = sent_at
            self._set_registration_error("")

    def _handle(self, message):
        if self.controller is None:
            raise NotStartedControlError("RUNTIME_NOT_READY")
        verb, epoch = message["verb"], message["epoch"]
        if verb == "recover":
            if (message["boot_id"], message["lifecycle_group"]) != (
                self.controller.boot_id,
                self.target.lifecycle_group,
            ):
                raise ControlError("RECOVERY_IDENTITY_MISMATCH")
            return self.controller.recover(
                epoch,
                message["operation_id"],
                message["command"],
                message["accept_until_ns"],
                expected_generation=message.get("expected_generation"),
                resume_until_ns=message.get("resume_until_ns"),
                stream_id=message.get("stream_id"),
                acknowledged=message.get("acknowledged", False),
            )
        if verb not in {"ack", "execute", "query", "cancel"}:
            raise ControlError("INVALID_VERB")
        submission = Submission(**message["submission"])
        if verb == "ack":
            self.controller.acknowledge(submission, epoch)
            return {"state": "acknowledged"}
        handler = {
            "execute": self.controller.execute,
            "query": self.controller.query,
            "cancel": self.controller.cancel,
        }.get(verb)
        if handler is None:
            raise ControlError("INVALID_VERB")
        if verb == "execute":

            def preflight():
                if getattr(self, "_closing", False):
                    raise NotStartedControlError("RUNTIME_CLOSING")
                self._validate_backend(self.controller.handle)

            return self.controller.execute(submission, epoch, preflight=preflight)
        return handler(submission, epoch)

    @property
    def registration_error(self) -> str:
        """Last registration failure, cleared only by a successful registration."""
        return self._registration_error

    def _set_registration_error(self, code: str) -> None:
        if code == self._registration_error:
            return
        self._registration_error = code
        if code:
            self._resource_ready = False
            self.node.get_logger().warning(f"Runtime registration failed for {self.target.target_id}: {code}")

    @property
    def ready(self):
        return (
            self.controller is not None
            and self._resource_ready
            and time.monotonic() - self._last_registered <= self._stale_seconds
            and self.controller.ready
        )

    @property
    def stream_input_available(self):
        """A healthy local reset may buffer PCM while model admission is paused."""
        with self.controller.transaction():
            return time.monotonic() - self._last_registered <= self._stale_seconds and self.controller.input_available

    @property
    def control_available(self):
        return self._client.service_is_ready()

    def _call(self, verb, submission, timeout, *, context=None):
        return self._call_control(
            {"verb": verb, "epoch": self.controller.epoch, "submission": submission.to_dict()}, timeout, context=context
        )

    def _call_control(self, payload, timeout, *, context=None):
        if not self._client.service_is_ready():
            raise NotStartedControlError("GLOBAL_UNAVAILABLE")
        try:
            request = _request(payload)
        except Exception as exc:
            raise NotStartedControlError(getattr(exc, "code", "INVALID_CONTROL")) from exc
        future = self._client.call_async(request)
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        cancellation_wake = done.set
        if context is not None:
            context.cancellation_token.add_callback(cancellation_wake)
        try:
            done.wait(timeout)
            if context is not None:
                context.check("global_control_wait")
            if not future.done():
                raise TimeoutError("global control response timed out")
            return _reply(future.result())
        finally:
            if context is not None:
                context.cancellation_token.remove_callback(cancellation_wake)
            if not future.done():
                self._client.remove_pending_request(future)

    def execute(
        self,
        request,
        context,
        *,
        priority=None,
        work="execute",
        snapshot_valid_until_ns=0,
        retry_not_ready=True,
        on_settled=None,
        _pending_reset=None,
    ):
        submission = None if _pending_reset is None else _pending_reset.get("submission")
        resuming = submission is not None
        try:
            if getattr(self, "_closing", False) or (getattr(self, "_close_requested", False) and work != "close"):
                raise NotStartedControlError("RUNTIME_CLOSING")
            if submission is None:
                submission = self.controller.prepare(
                    request,
                    context,
                    priority=priority,
                    work=work,
                    snapshot_valid_until_ns=snapshot_valid_until_ns,
                    retain_for_caller=True,
                )
                if _pending_reset is not None:
                    _pending_reset["submission"] = submission
        except Exception as exc:
            # Preparation never submits work to Global or invokes the model.
            # Tensor copying/encoding can raise RuntimeError as well as the
            # explicit validation errors; all are proven NOT_STARTED here.
            raise NotStartedControlError(getattr(exc, "code", "INVALID_REQUEST")) from exc
        # A retry renews only the business wait. It never prepares/submits work
        # again or extends the original operation's execution deadline.
        verb = "query" if resuming else "submit"
        wait_until_ns = submission.deadline_ns
        if resuming:
            remaining = context.deadline.remaining_seconds()
            timeout = self.target.request_timeout_ms / 1000 if remaining is None else max(0, remaining)
            wait_until_ns = time.monotonic_ns() + int(timeout * 1e9)
        completed = self.controller.completion(submission)
        receipt = self.controller.terminal_receipt(submission)
        if on_settled is not None:
            receipt.add_done_callback(lambda _: on_settled())
        wake = threading.Event()
        completed.add_done_callback(lambda _: wake.set())
        receipt.add_done_callback(lambda _: wake.set())
        cancellation_wake = wake.set
        context.cancellation_token.add_callback(cancellation_wake)
        # Once Submit may have been sent, only Global may acknowledge the
        # host's terminal evidence, independently of this business waiter.
        submit_sent = resuming
        submit_uncertain = False
        submit_rejected = False
        try:
            while True:
                context.check("scheduler_wait")
                if completed.done():
                    local = self.controller.query(submission, self.controller.epoch)
                    if local["state"] == "not_started":
                        raise NotStartedControlError(local["error"] or "NOT_STARTED")
                    if resuming and local["state"] in TERMINAL:
                        verb = "terminal_report"
                remaining = (wait_until_ns - time.monotonic_ns()) / 1e9
                if remaining <= 0:
                    raise TimeoutError("runtime request deadline expired")
                message_verb = verb
                try:
                    if verb == "submit":
                        submit_sent = True
                        submit_rejected = False
                    if verb == "terminal_report":
                        report = self.controller.query(submission, self.controller.epoch)
                    else:
                        report = self._call(
                            verb, submission, min(remaining, CONTROL_RESPONSE_WAIT_SECONDS), context=context
                        )
                    state = report["state"]
                except ControlError as exc:
                    # A first explicit admission rejection proves that no
                    # coordinator record was created. After any lost Submit
                    # reply, however, a later rejection cannot revoke earlier
                    # possible admission (notably after the retry window).
                    submit_rejected = (
                        verb == "submit"
                        and not submit_uncertain
                        and isinstance(exc, _RemoteControlError | NotStartedControlError)
                    )
                    if (
                        work == "execute"
                        and (retry_not_ready or exc.code == "RESOURCE_INITIALIZING")
                        and submit_rejected
                        and exc.code
                        in {
                            "RESOURCE_INITIALIZING",
                            "RESOURCE_NOT_READY",
                        }
                    ):
                        # Retry only proven rejection, retaining the same operation
                        # identity and deadline. Lost replies remain uncertain.
                        wake.wait(min(0.05, max(0, (wait_until_ns - time.monotonic_ns()) / 1e9)))
                        continue
                    raise
                except TimeoutError:
                    if verb == "submit":
                        submit_uncertain = True
                    continue  # Same immutable operation, never a new submission.
                verb = "query"
                local = self.controller.query(submission, self.controller.epoch)
                if state in TERMINAL and message_verb != "terminal_report":
                    if local["state"] not in TERMINAL:
                        local = self.controller.cancel(submission, self.controller.epoch)
                    if state == local["state"] == "not_started" and report.get("error"):
                        raise NotStartedControlError(report["error"])
                    return self.controller.result(submission)
                if local["state"] in TERMINAL:
                    # Both callers and the background guard use the same bounded
                    # receipt request. The Controller retains its confirmed outcome.
                    verb = "terminal_report"
                    wake.clear()
                    self._terminal_tick()
                    if receipt.done():
                        outcome = receipt.result()
                        if local["state"] == outcome.get("state") == "not_started" and outcome.get("error"):
                            raise NotStartedControlError(outcome["error"])
                        return self.controller.result(submission)
                    wake.wait(min(BUSINESS_POLL_INTERVAL_SECONDS, max(0, (wait_until_ns - time.monotonic_ns()) / 1e9)))
                    continue
                if state == "unknown" or local["state"] == "unknown":
                    raise ControlError("OUTCOME_UNKNOWN")
                # Completion wakes the waiter immediately. Queries only
                # reconcile queued expiry, lost transport or Global restart.
                wake.wait(min(BUSINESS_POLL_INTERVAL_SECONDS, max(0, (wait_until_ns - time.monotonic_ns()) / 1e9)))
                if completed.done():
                    verb = "terminal_report"
        except BaseException as exc:
            # Cancel shares the endpoint claim lock. If already running, its
            # input and global capacity survive this business waiter.
            local = self.controller.cancel(submission, self.controller.epoch)
            if submit_sent and not submit_rejected:
                with suppress(Exception):
                    # Notify Global without extending an expired/cancelled
                    # business wait. Global still owns reconciliation and ACK.
                    self._call("cancel", submission, 0)
            elif local["state"] == "not_started":
                # No Submit was attempted, or its first reply proves rejection.
                # Do not send Cancel here: an unknown Global Cancel can create
                # its own operation, which would need terminal evidence again.
                self.controller.acknowledge(submission, self.controller.epoch)
            if local["state"] == "not_started" and isinstance(exc, Exception):
                raise NotStartedControlError(getattr(exc, "code", local["error"] or "NOT_STARTED")) from exc
            if local["state"] not in TERMINAL and isinstance(exc, Exception):
                raise ControlError("OUTCOME_UNKNOWN") from exc
            if local["state"] == "completed" and isinstance(exc, Exception):
                if local["error"]:
                    return self.controller.result(submission)
                raise CompletedControlError(getattr(exc, "code", "COMPLETED_DELIVERY_FAILED")) from exc
            raise
        finally:
            context.cancellation_token.remove_callback(cancellation_wake)
            if _pending_reset is None:
                self.controller.release_result_waiter(submission)

    def reset(self, context, *, stream_id=None):
        """Retain ordinary Reset identity until result and registration are delivered."""
        remaining = context.deadline.remaining_seconds()
        timeout = self.target.request_timeout_ms / 1000 if remaining is None else max(0, remaining)
        entered = False
        try:
            with lifecycle_budget(timeout), lifecycle_lock(self._reset_lock, context):
                entered = True
                return self._reset_locked(context, stream_id=stream_id)
        except (DeadlineExceeded, CancellationRequested) as exc:
            if not entered:
                exc.evidence = OutcomeEvidence.not_started("control_lock")
            raise

    def _release_reset(self, key):
        pending = self._reset_requests.pop(key, {})
        if "submission" in pending:
            self.controller.release_result_waiter(pending["submission"])

    def release_reset_receipts(self, context, *, stream_id=None):
        """Release abandoned waiters only after their owner has closed."""
        with lifecycle_lock(self._reset_lock, context):
            keys = tuple(self._reset_requests) if stream_id is None else (("reset_stream", stream_id),)
            for key in keys:
                self._release_reset(key)

    def _reset_locked(self, context, *, stream_id):
        command = "reset" if stream_id is None else "reset_stream"
        key = command if stream_id is None else (command, stream_id)
        if self.has_recovery(key):
            return self.recover(context, command=command, stream_id=stream_id)
        pending = self._reset_requests.setdefault(key, {})
        request = (
            ModelRequest()
            if stream_id is None
            else ModelRequest(
                metadata={RUNTIME_HANDLE_OPERATION: command, "runtime_stream_id": stream_id},
                identity_metadata_keys={"runtime_stream_id"},
            )
        )
        try:
            if "result" not in pending:
                pending["result"] = self.execute(
                    request,
                    context,
                    work="reset" if stream_id is None else "execute",
                    retry_not_ready=False,
                    _pending_reset=pending,
                )
        except NotStartedControlError as exc:
            self._release_reset(key)
            if exc.code not in {
                "RUNTIME_NOT_REGISTERED",
                "RESOURCE_NOT_READY",
                "RUNTIME_FENCED",
                "REQUEST_CAPACITY",
                "INPUT_CAPACITY",
            }:
                raise
            if not self.controller.epoch:
                raise
            return self.recover(context, command=command, stream_id=stream_id)
        except Exception as exc:
            submission = pending.get("submission")
            if submission is not None:
                report = self.controller.query(submission, self.controller.epoch)
                if report["state"] == "completed" and not report["error"]:
                    # Only delivery failed: stream policy must not request a
                    # second reset to repair a state that is already reset.
                    error = CompletedControlError(getattr(exc, "code", "COMPLETED_DELIVERY_FAILED"))
                    error.evidence = OutcomeEvidence.completed("reset_delivery")
                    raise error from exc
                if report["error"]:
                    # Native failure has been delivered, rather than lost in
                    # transport. Preserve the existing explicit recovery path.
                    self._release_reset(key)
            raise
        try:
            self._refresh_after_reset(context)
        except Exception as exc:
            error = CompletedControlError(getattr(exc, "code", "RESET_REGISTRATION_FAILED"))
            error.evidence = OutcomeEvidence.completed("reset_delivery")
            raise error from exc
        result = pending["result"]
        self._release_reset(key)
        return result

    def cancel(self, request_id, *, deadline=None):
        for submission in self.controller.cancel_pending(request_id):
            remaining = None if deadline is None else deadline.remaining_seconds()
            with suppress(Exception):
                self._call("cancel", submission, 0.1 if remaining is None else max(0, min(remaining, 0.1)))

    def close(self, context=None):
        if getattr(self, "_shutdown_complete", False):
            if getattr(self.node, "context", None) is None or self.node.executor is None:
                self.destroy_transport()
            return
        context = context or ExecutionContext("runtime-close", Deadline.after(self.target.request_timeout_ms / 1000))
        with lifecycle_lock(self._close_lock, context):
            if self._closing:
                self.controller.shutdown(context)
                self.release_reset_receipts(context)
                self._shutdown_complete = True
                self.controller.on_terminal = None
                if getattr(self.node, "context", None) is None or self.node.executor is None:
                    self.destroy_transport()
                return
            owner = self.controller.handle
            if self.has_recovery("close"):
                self._close_requested = True
                self.recover(context, command="close")
            elif not getattr(owner, "closed", False):
                self._close_requested = True
                try:
                    self.execute(ModelRequest(), context, work="close", retry_not_ready=False)
                except NotStartedControlError:
                    self.wait_for_registration(context)
                    self.recover(context, command="close")
            if self.controller.epoch:
                # Managed public Close must observe retirement before destroying
                # the only endpoint that can replay its terminal evidence. Leave
                # transport alive on timeout so a later Close can resume safely.
                remaining = context.deadline.remaining_seconds()
                expires = time.monotonic() + (self.target.request_timeout_ms / 1000 if remaining is None else remaining)
                while True:
                    context.check("runtime_retirement")
                    remaining = expires - time.monotonic()
                    if remaining <= 0:
                        raise ControlError("RETIREMENT_TIMEOUT")
                    if self._unregister_clean_host(timeout=min(0.5, remaining)):
                        break
                    time.sleep(min(0.01, max(0, expires - time.monotonic())))
            with self.controller.transaction():
                if self._closing:
                    return
                self._closing = True
                self.controller.cancel_pending(closing=True)
            # A live ROS executor may still hold these entities in its wait set.
            # Destroying a client here can stop a shared spin thread with
            # InvalidHandle and strand other owners' Close replies. Stop callbacks
            # now; the spin owner releases ROS handles at a synchronized safe
            # boundary between spin_once calls, or after executor shutdown.
            node_owns_transport = getattr(self.node, "context", None) is not None and self.node.executor is not None
            self._timer.cancel()
            if self._registration is not None:
                self._client.remove_pending_request(self._registration)
                self._registration = None
            for future in tuple(self._terminal_receipts):
                self._client.remove_pending_request(future)
            self._terminal_receipts.clear()
            self.controller.shutdown(context)
            self.release_reset_receipts(context)
            self._shutdown_complete = True
            self.controller.on_terminal = None
            if not node_owns_transport:
                self.destroy_transport()

    def rollback_unpublished(self):
        """Stop transport after binding failed; construction still owns the model."""
        if self._registration_enabled or self.controller.epoch:
            raise ControlError("RUNTIME_ALREADY_PUBLISHED")
        self._closing = True
        self._timer.cancel()
        self.controller.shutdown()
        self._shutdown_complete = True
        self.controller.on_terminal = None
        if getattr(self.node, "context", None) is None or self.node.executor is None:
            self.destroy_transport()

    def destroy_transport(self, *, _quiescent=False):
        """Release transport after detachment or at the spin owner's quiescent boundary."""
        if getattr(self, "_transport_destroyed", False):
            return
        live_node = getattr(self.node, "context", None) is not None and self.node.executor is not None
        if live_node and not _quiescent:
            raise RuntimeError("runtime transport requires a quiescent node")
        if not self._closing:
            self.controller.require_transport_release()
            if self.controller.epoch:
                # Local ACKs cannot prove that Global received the final
                # receipt. Registered hosts must pass close()'s unregister barrier.
                raise ControlError("RUNTIME_RETIREMENT_REQUIRED")
            self._closing = True
        if not getattr(self, "_shutdown_complete", False):
            self.controller.shutdown()
            self._shutdown_complete = True
            self.controller.on_terminal = None
        if self._registration is not None:
            self._client.remove_pending_request(self._registration)
            self._registration = None
        for name, destroy in (
            ("_timer", self.node.destroy_timer),
            ("_terminal_guard", self.node.destroy_guard_condition),
            ("_service", self.node.destroy_service),
            ("_client", self.node.destroy_client),
        ):
            entity = getattr(self, name, None)
            if entity is not None:
                destroy(entity)
                setattr(self, name, None)
        self._transport_destroyed = True

    def _terminal_tick(self):
        if self.controller is None:
            return
        with self.controller.transaction():
            if not self._closing:
                self._flush_terminal_evidence()

    def _terminal_receipt_done(self, future):
        with self.controller.transaction(), suppress(Exception):
            pending = self._terminal_receipts.pop(future, None)
            if pending is not None:
                reply = _reply(future.result())
                self._acknowledge_terminal(pending[0], reply)
                self._notify_terminal()

    def _acknowledge_terminal(self, submission, reply):
        """Apply Global's outcome consistently across receipt and retirement paths."""
        if reply.get("state") == "acknowledged":
            self.controller.acknowledge(submission, self.controller.epoch, outcome=reply.get("outcome"))

    def _notify_terminal(self):
        if not self._closing:
            with suppress(Exception):  # A periodic heartbeat can reconcile a lost wake.
                self._terminal_guard.trigger()

    def _flush_terminal_evidence(self):
        """Bound scanning, sending and in-flight requests independently of history."""
        try:
            for future, (submission, started, receipt) in tuple(self._terminal_receipts.items()):
                if future.done():
                    self._terminal_receipts.pop(future, None)
                    with suppress(Exception):
                        self._acknowledge_terminal(submission, _reply(future.result()))
                elif receipt.done() or time.monotonic() - started > TERMINAL_RECEIPT_TIMEOUT_SECONDS:
                    self._client.remove_pending_request(future)
                    self._terminal_receipts.pop(future, None)
            available = MAX_TERMINAL_RECEIPTS - len(self._terminal_receipts)
            if available <= 0 or not self._client.service_is_ready():
                return
            in_flight = {pending[0].operation_id for pending in self._terminal_receipts.values()}
            reports = self.controller.pending_terminal_reports(
                limit=min(TERMINAL_REPORT_BATCH_SIZE, available), retry_interval_ns=HOST_HEARTBEAT_INTERVAL_NS
            )
            for report in reports:
                submission = Submission(**report["submission"])
                if submission.operation_id in in_flight:
                    continue
                future = self._client.call_async(
                    _request({"verb": "terminal_report", "epoch": self.controller.epoch, "report": report})
                )
                self._terminal_receipts[future] = (
                    submission,
                    time.monotonic(),
                    self.controller.terminal_receipt(submission),
                )
                if hasattr(self, "_terminal_guard"):
                    future.add_done_callback(self._terminal_receipt_done)
        except Exception:
            # Retained evidence remains retryable on the next guard/heartbeat;
            # one transport failure must not stop the shared control executor.
            LOGGER.exception("Runtime terminal receipt send failed (target=%s)", self.target.target_id)

    def has_recovery(self, command):
        return command in self._recovery_requests

    def wait_for_registration(self, context, *, startup=False):
        """Obtain lifecycle authority even when Close precedes the first heartbeat.

        Registration is independent of resource readiness: Close must still
        recover a resource whose peers are busy or unavailable.
        """
        remaining = context.deadline.remaining_seconds()
        timeout = self.target.request_timeout_ms / 1000 if remaining is None else remaining
        expires = time.monotonic() + max(0, timeout)
        while not self.controller.epoch:
            context.check("runtime_registration")
            if startup:
                from inference_service.unified_runtime.lifecycle import check_startup

                check_startup()
            if not self.control_available:
                raise ControlError("GLOBAL_UNAVAILABLE")
            if time.monotonic() >= expires:
                raise ControlError("REGISTRATION_TIMEOUT")
            self._heartbeat()
            if not self.controller.epoch:
                time.sleep(min(0.01, max(0, expires - time.monotonic())))

    def recover(self, context, *, command="reset", stream_id=None, completed_only=False):
        """Resume one immutable recovery receipt across caller timeouts.

        Receipts are retained until completion has been observed and settled.
        Reset also requires fresh registration before settling; successful
        lifecycle calls then release their ID for the next distinct operation.
        """
        if command not in {"reset", "close", "reset_stream", "close_stream"}:
            raise ControlError("INVALID_RECOVERY_COMMAND")
        if not self.controller.epoch:
            raise ControlError("RUNTIME_NOT_REGISTERED")
        remaining = context.deadline.remaining_seconds()
        timeout = self.target.request_timeout_ms / 1000 if remaining is None else max(0, remaining)
        entered = False
        try:
            with lifecycle_budget(timeout), lifecycle_lock(self._recovery_lock, context):
                entered = True
                key = command if stream_id is None else (command, stream_id)
                if completed_only and not self._recovery_requests.get(key, {}).get("acknowledged"):
                    # Check under the same lock as receipt removal. Another
                    # waiter may already have settled it; never start a mutation.
                    return None
                try:
                    return self._recover_locked(context, command, stream_id=stream_id)
                except Exception as exc:
                    if self._recovery_requests.get(key, {}).get("acknowledged"):
                        # The lifecycle result is known; only registration or
                        # receipt delivery failed. Keep that proof on retries.
                        raise CompletedControlError(getattr(exc, "code", "COMPLETED_DELIVERY_FAILED")) from exc
                    raise
        except (DeadlineExceeded, CancellationRequested) as exc:
            if not entered:
                # This waiter never acquired recovery authority. Its departure
                # cannot change the shared operation another caller owns.
                exc.evidence = OutcomeEvidence.not_started("control_lock")
            raise

    def _recover_locked(self, context, command, *, stream_id=None):
        # Product Open/Close use reset as their drain barrier. A fenced runtime
        # cannot enter ordinary admission; use the existing recovery transport
        # and change only this lifecycle owner after resource-wide drain.
        remaining = context.deadline.remaining_seconds()
        timeout = self.target.request_timeout_ms / 1000 if remaining is None else remaining
        caller_expiry = time.monotonic_ns() + int(max(0, timeout) * 1e9)
        key = command if stream_id is None else (command, stream_id)
        message = self._recovery_requests.get(key)
        if message is None:
            # The receipt outlives a short business waiter. A later caller can
            # observe completion without sending a second drain/close command.
            message = {
                "verb": "recover",
                "epoch": self.controller.epoch,
                "resource": self.target.hardware_resource_id,
                "target_id": self.target.target_id,
                "boot_id": self.controller.boot_id,
                "operation_id": str(uuid4()),
                "command": command,
                "accept_until_ns": time.monotonic_ns() + self.target.retry_window_ms * 1000000,
                **({"stream_id": stream_id} if stream_id is not None else {}),
            }
        elif time.monotonic_ns() >= message.get("resume_until_ns", message["accept_until_ns"]):
            # Renew only the wait/transport window; Global retains the phase,
            # operation and per-host receipts that authorize safe continuation.
            message["resume_until_ns"] = time.monotonic_ns() + self.target.retry_window_ms * 1000000
        while True:
            context.check("runtime_recovery")
            if message.get("acknowledged") and command != "close":
                self._refresh_after_reset(context)
            acceptance = message.get("resume_until_ns", message["accept_until_ns"])
            remaining = (min(caller_expiry, acceptance) - time.monotonic_ns()) / 1e9
            if remaining <= 0:
                raise ControlError("RECOVERY_EXPIRED" if time.monotonic_ns() >= acceptance else "RECOVERY_TIMEOUT")
            previously_attempted = key in self._recovery_requests
            try:
                self._recovery_requests[key] = message
                result = self._call_control(message, min(remaining, CONTROL_RESPONSE_WAIT_SECONDS))
            except NotStartedControlError:
                if not previously_attempted:
                    # Local serialization/discovery failed before sending. An
                    # earlier ambiguous attempt still owns its original ID.
                    self._recovery_requests.pop(key, None)
                raise
            except TimeoutError:
                continue  # Keep the same recovery identity after a lost reply.
            except _RemoteControlError as exc:
                if exc.code == "OPERATION_EXPIRED" and not previously_attempted:
                    # A first-send rejection proves Global never admitted this
                    # ID. After any lost reply, expiry alone proves nothing.
                    self._recovery_requests.pop(key, None)
                if exc.code == "RECOVERY_RECEIPT_UNAVAILABLE" and command == "close":
                    # Missing history is not NOT_STARTED. Close is an
                    # irreversible, repeatable cleanup barrier, so a new
                    # transaction may safely drain and finish this same owner.
                    # Reset has no such contract and never takes this path.
                    message = {
                        **message,
                        "operation_id": str(uuid4()),
                        "accept_until_ns": time.monotonic_ns() + self.target.retry_window_ms * 1000000,
                    }
                    message.pop("resume_until_ns", None)
                    continue
                if exc.code in {"RECOVERY_SUPERSEDED", "RECOVERY_RECEIPT_UNAVAILABLE"}:
                    # Global only collects settled/superseded receipts in this
                    # epoch. Report the old outcome as unknown, without replay;
                    # a later explicit call may request a new lifecycle action.
                    self._recovery_requests.pop(key, None)
                    raise
                if (
                    exc.code in {"INITIALIZATION_EVIDENCE_PENDING", "RECOVERY_HOST_UNREACHABLE"}
                    and "resume_until_ns" not in message
                ):
                    self._recovery_requests.pop(key, None)
                if exc.code != "RECOVERY_BUSY":
                    raise
                if "resume_until_ns" not in message:
                    # Global rejected this new operation before recording it.
                    # Existing IDs are looked up before its busy check, even
                    # if an earlier reply was lost. Resumes retain their ID.
                    # A later caller must not try to resume a nonexistent
                    # receipt after waiting for the current resource owner.
                    self._recovery_requests.pop(key, None)
                time.sleep(min(0.01, remaining))
                continue
            if result["state"] == "acknowledged" and message.get("acknowledged"):
                self._recovery_requests.pop(key, None)
                return None
            if result["state"] == "completed":
                # Retain completion before refreshing registration. The same
                # ACK intent is replayable even after Global collects history;
                # the next iteration refreshes before sending that ACK.
                message["acknowledged"] = True
                continue
            if result["state"] != "running":
                message["resume_until_ns"] = 0
                raise ControlError(result.get("error") or "RECOVERY_UNKNOWN")
            time.sleep(min(0.01, remaining))

    def _refresh_after_reset(self, context):
        # Publish this lifecycle's completion without waiting for other models
        # to recover business readiness. New work still checks resource_ready.
        return self.refresh_registration(context, require_ready=True)

    def refresh_registration(self, context, *, require_ready=False, startup=False):
        """Publish fresh lifecycle evidence after initialization or recovery."""
        # A heartbeat taken during reset advertises ready=False. Completion
        # advances the generation, but cannot prove that the handle is ready.
        # Register fresh evidence for every stage before releasing the caller.
        remaining = context.deadline.remaining_seconds()
        timeout = self.target.request_timeout_ms / 1000 if remaining is None else remaining
        expiry = time.monotonic() + timeout
        while True:
            context.check("runtime_registration")
            if startup:
                from inference_service.unified_runtime.lifecycle import check_startup

                check_startup()
            peers_ready = True
            adapters = getattr(self, "_registration_peers", (self,))
            if len(adapters) > 1:
                remaining = expiry - time.monotonic()
                if remaining <= 0:
                    raise ControlError("REGISTRATION_READINESS_TIMEOUT")
                with self.controller.transaction():
                    statuses = []
                    for adapter in adapters:
                        adapter._validate_backend(adapter.controller.handle)
                        statuses.append(adapter.controller.status())
                    payload = {"verb": "register_group", "statuses": statuses}
                try:
                    reply = self._call_control(payload, min(remaining, CONTROL_RESPONSE_WAIT_SECONDS), context=context)
                except TimeoutError:
                    pass
                except ControlError as exc:
                    if exc.code not in _RETRYABLE_REGISTRATION_ERRORS:
                        raise
                else:
                    for adapter, status in zip(adapters, statuses, strict=True):
                        adapter._apply_registration(reply, status["status_ns"] / 1e9)
                    if not require_ready or all(status["ready"] for status in statuses):
                        return
                time.sleep(min(0.01, max(0, expiry - time.monotonic())))
                continue
            for peer in self.controller.lifecycle_peers:
                remaining = expiry - time.monotonic()
                if remaining <= 0:
                    raise ControlError("REGISTRATION_READINESS_TIMEOUT")
                self._validate_backend(peer.handle)
                status = peer.status()
                peers_ready = peers_ready and status["ready"]
                try:
                    reply = self._call_control(
                        {"verb": "register", "status": status},
                        min(remaining, CONTROL_RESPONSE_WAIT_SECONDS),
                        context=context,
                    )
                except TimeoutError:
                    break
                except ControlError as exc:
                    if exc.code not in _RETRYABLE_REGISTRATION_ERRORS:
                        raise
                    break  # Refresh status after a heartbeat or Reset evidence race.
                peer.accept_epoch(reply["epoch"])
                if peer is self.controller:
                    self._apply_registration(reply, status["status_ns"] / 1e9)
            else:
                if not require_ready or peers_ready:
                    return
            time.sleep(min(0.01, max(0, expiry - time.monotonic())))

    def _unregister_clean_host(self, *, timeout=0.5):
        """Try the Global barrier first; replay evidence only when it is unmet."""
        # Cap each RPC and reserve one such slot for the final unregister.
        rpc_timeout = 0.1
        expires = time.monotonic() + timeout
        with self.controller.transaction():
            self.controller.cancel_pending(closing=True)
            # Registration must stop before the first retirement RPC. Keep
            # terminal evidence callbacks alive until unregister is observed.
            # The shared controller lock also serializes the group heartbeat.
            for peer in self._registration_peers:
                peer._registration_enabled = False
                if peer._registration is not None:
                    peer._client.remove_pending_request(peer._registration)
                    peer._registration = None

        def unregister(remaining):
            return (
                self._call_control(
                    {"verb": "unregister", "epoch": self.controller.epoch, "status": self.controller.status()},
                    min(remaining, rpc_timeout),
                ).get("state")
                == "unregistered"
            )

        def record_error(error):
            reason = f"{getattr(error, 'code', type(error).__name__)}: {str(error)[:256]}"
            if reason != getattr(self, "_retirement_error", ""):
                self._retirement_error = reason
                self.node.get_logger().warning(f"Runtime retirement pending for {self.target.target_id}: {reason}")

        try:
            return unregister(timeout)
        except ControlError as exc:
            record_error(exc)
            if exc.code != "HOST_NOT_DRAINED":
                return False
        except TimeoutError as exc:
            # A stopped executor can lose every reply. Preserve one bounded
            # compensating pass when the barrier's response is itself unknown.
            record_error(exc)
        except Exception as exc:
            record_error(exc)
            return False
        # A local ACK can precede Global's observation of its reply. Replay
        # retained terminal evidence round-robin so bounded attempts progress.
        reports = self.controller.terminal_reports()
        for _ in range(len(reports)):
            remaining = expires - time.monotonic() - rpc_timeout
            if remaining <= 0:
                break
            report = reports[self._retirement_replay_cursor % len(reports)]
            self._retirement_replay_cursor += 1
            try:
                reply = self._call_control(
                    {"verb": "terminal_report", "epoch": self.controller.epoch, "report": report},
                    min(remaining, rpc_timeout),
                )
                self._acknowledge_terminal(Submission(**report["submission"]), reply)
            except Exception:
                continue
        remaining = expires - time.monotonic()
        if remaining > 0:
            try:
                return unregister(remaining)
            except Exception as exc:
                record_error(exc)
        return False


def bind_registration_group(adapters):
    """Publish lifecycle peers together using one heartbeat and one incarnation.

    Configure before exposing the adapters to callers. Sharing the controller
    lock also makes lifecycle transitions and group status snapshots atomic.
    """
    if not 2 <= len(adapters) <= MAX_RUNTIME_GROUP_MEMBERS:
        raise ValueError(f"runtime group requires 2..{MAX_RUNTIME_GROUP_MEMBERS} members")
    validate_runtime_group_budget([adapter.target for adapter in adapters])
    leader = adapters[0]
    peers = [adapter.controller for adapter in adapters]
    for adapter in adapters:
        adapter.controller.join_lifecycle(leader.controller, peers)
        adapter._registration_peers = tuple(adapters)
        adapter._registration_enabled = adapter is leader
