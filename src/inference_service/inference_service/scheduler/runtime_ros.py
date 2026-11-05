"""Thin ROS control transport; images, point clouds and results never cross Global."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import TimeoutError
from contextlib import suppress

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.qos import QoSProfile

from inference_service.unified_runtime import Deadline, ExecutionContext, ModelRequest
from inference_service.unified_runtime.lifecycle import lifecycle_lock

from .runtime_controller import RuntimeExecutionController
from .runtime_profiles import DEFAULT_MIN_SAMPLES, PRODUCTION_MIN_SAMPLES
from .runtime_protocol import (
    DEFAULT_CONTROL_RPC_TIMEOUT_NS,
    DEFAULT_RUNTIME_ENDPOINT,
    DEFAULT_STATUS_STALE_TIMEOUT_NS,
    HOST_HEARTBEAT_INTERVAL_NS,
    PROTOCOL_VERSION,
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
            if getattr(exc, "operation_started", None) is False:
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
        stale_ns=DEFAULT_STATUS_STALE_TIMEOUT_NS,
        control_rpc_timeout_ns=DEFAULT_CONTROL_RPC_TIMEOUT_NS,
        profile_min_samples=DEFAULT_MIN_SAMPLES,
        profile_max_age_days=30,
    ):
        if type(control_rpc_timeout_ns) is not int or not 0 < control_rpc_timeout_ns <= 2**63 - 1:
            raise ValueError("control RPC timeout must be a positive int64 number of nanoseconds")
        if type(profile_min_samples) is not int or profile_min_samples < PRODUCTION_MIN_SAMPLES:
            raise ValueError(f"profile_min_samples must be >= {PRODUCTION_MIN_SAMPLES} for a ROS endpoint")
        self.node = node
        self.control_rpc_timeout_ns = control_rpc_timeout_ns
        self.core = RuntimeScheduler(
            targets,
            policy=policy,
            resources=resources,
            stale_ns=stale_ns,
            profile_min_samples=profile_min_samples,
            profile_max_age_days=profile_max_age_days,
        )
        self.endpoint = endpoint
        if any(target.endpoint == endpoint for target in targets):
            raise ValueError("runtime global endpoint conflicts with a host endpoint")
        self._pending = {}
        self._control_schedule = {}
        self._clients = {}
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
        for target_id, target in self.core.targets.items():
            if target_id not in self._clients:
                self._clients[target_id] = self.node.create_client(
                    _service_type(),
                    target.endpoint,
                    callback_group=ReentrantCallbackGroup(),
                    qos_profile=QoSProfile(depth=8),
                )

    def _handle(self, message):
        if self.timer is None:
            raise NotStartedControlError("GLOBAL_NOT_READY")
        verb = message["verb"]
        if verb == "describe":
            offset = message.get("offset", 0)
            if type(offset) is not int or offset < 0:
                raise ControlError("INVALID_TARGET_OFFSET")
            targets = list(self.core.targets.values())
            return {
                "protocol_version": PROTOCOL_VERSION,
                "epoch": self.core.epoch,
                "clock_id": self.core.clock_id,
                "target_count": len(targets),
                "targets": [target.to_dict() for target in targets[offset : offset + 1]],
                "measurement_count": len(self.core.measurements),
            }
        if verb == "register":
            result = self.core.register(message["status"])
            return {**result, "stale_ns": self.core.stale_ns}
        if verb not in {"measurements", "submit", "cancel", "query"}:
            raise ControlError("INVALID_VERB")
        if message.get("epoch") != self.core.epoch:
            raise ControlError("EPOCH_MISMATCH")
        if verb == "measurements":
            return self.core.read_measurements(message.get("after_sequence", 0))
        submission = Submission(**message["submission"])
        result = getattr(self.core, verb)(submission)
        self._wake()
        return result

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
        client = self._clients[target.target_id]
        try:
            ready = client.service_is_ready()
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
    ):
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
        self._registration = None
        self._registration_started = 0
        self._resource_ready = False
        self._last_registered = 0.0
        self._stale_seconds = DEFAULT_STATUS_STALE_TIMEOUT_NS / 1e9
        self._registration_error = ""
        self._closing = False
        self._close_requested = False
        self._close_lock = threading.Lock()
        self._client = self._service = self._timer = None
        try:
            self._client = node.create_client(
                _service_type(), endpoint, callback_group=ReentrantCallbackGroup(), qos_profile=QoSProfile(depth=8)
            )
            self._service = node.create_service(
                _service_type(),
                self.target.endpoint,
                _serve(self._handle),
                callback_group=ReentrantCallbackGroup(),
                qos_profile=QoSProfile(depth=8),
            )
            group = MutuallyExclusiveCallbackGroup()
            self._timer = node.create_timer(HOST_HEARTBEAT_INTERVAL_NS / 1e9, self._heartbeat, callback_group=group)
            # ROS constructors validate service names and may fail. Bind only
            # after all entities exist: Handle ownership cannot be rolled back.
            controller = RuntimeExecutionController(self.target, handle)
            if boot_id is not None:
                controller.boot_id = boot_id
            controller.policy.delegate = self
            self.controller = controller
        except BaseException:
            for name, destroy in (
                ("_timer", node.destroy_timer),
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
        if self._registration is not None:
            if not self._registration.done():
                if time.monotonic() - self._registration_started > REGISTRATION_RESPONSE_TIMEOUT_SECONDS:
                    self._client.remove_pending_request(self._registration)
                    self._registration = None
                    self._set_registration_error("REGISTRATION_TIMEOUT")
                return
            try:
                reply = _reply(self._registration.result())
                self._apply_registration(reply, self._registration_started)
            except Exception as exc:
                self._set_registration_error(getattr(exc, "code", "REGISTRATION_FAILED"))
            self._registration = None
        self.controller.collect()
        try:
            if self._client.service_is_ready():
                status = self.controller.status()
                try:
                    self._validate_backend(self.controller.handle)
                except ControlError:
                    self.controller.fence()
                    status.update(ready=False, clean=False)
                self._registration_started = status["status_ns"] / 1e9
                self._registration = self._client.call_async(_request({"verb": "register", "status": status}))
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
        self, request, context, *, priority=None, work="execute", snapshot_valid_until_ns=0, retry_not_ready=True
    ):
        try:
            if getattr(self, "_closing", False) or (getattr(self, "_close_requested", False) and work != "close"):
                raise NotStartedControlError("RUNTIME_CLOSING")
            submission = self.controller.prepare(
                request,
                context,
                priority=priority,
                work=work,
                snapshot_valid_until_ns=snapshot_valid_until_ns,
                retain_for_caller=True,
            )
        except Exception as exc:
            # Preparation never submits work to Global or invokes the model.
            # Tensor copying/encoding can raise RuntimeError as well as the
            # explicit validation errors; all are proven NOT_STARTED here.
            raise NotStartedControlError(getattr(exc, "code", "INVALID_REQUEST")) from exc
        verb = "submit"
        completed = self.controller.completion(submission)
        wake = threading.Event()
        completed.add_done_callback(lambda _: wake.set())
        cancellation_wake = wake.set
        context.cancellation_token.add_callback(cancellation_wake)
        # Once Submit may have been sent, only Global may acknowledge the
        # host's terminal evidence, independently of this business waiter.
        submit_sent = False
        submit_uncertain = False
        submit_rejected = False
        try:
            while True:
                context.check("scheduler_wait")
                if completed.done():
                    local = self.controller.query(submission, self.controller.epoch)
                    if local["state"] == "not_started":
                        raise NotStartedControlError(local["error"] or "NOT_STARTED")
                remaining = (submission.deadline_ns - time.monotonic_ns()) / 1e9
                if remaining <= 0:
                    raise TimeoutError("runtime request deadline expired")
                try:
                    if verb == "submit":
                        submit_sent = True
                        submit_rejected = False
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
                        and retry_not_ready
                        and submit_rejected
                        and exc.code
                        in {
                            "RESOURCE_NOT_READY",
                        }
                    ):
                        # Retry only proven rejection, retaining the same operation
                        # identity and deadline. Lost replies remain uncertain.
                        wake.wait(min(0.05, max(0, (submission.deadline_ns - time.monotonic_ns()) / 1e9)))
                        continue
                    raise
                except TimeoutError:
                    if verb == "submit":
                        submit_uncertain = True
                    continue  # Same immutable operation, never a new submission.
                verb = "query"
                local = self.controller.query(submission, self.controller.epoch)
                if state in TERMINAL:
                    if local["state"] not in TERMINAL:
                        local = self.controller.cancel(submission, self.controller.epoch)
                    if state == local["state"] == "not_started" and report.get("error"):
                        raise NotStartedControlError(report["error"])
                    return self.controller.result(submission)
                if state == "unknown" or local["state"] == "unknown":
                    raise ControlError("OUTCOME_UNKNOWN")
                # Completion wakes the waiter immediately. Queries only
                # reconcile queued expiry, lost transport or Global restart.
                wake.wait(
                    min(BUSINESS_POLL_INTERVAL_SECONDS, max(0, (submission.deadline_ns - time.monotonic_ns()) / 1e9))
                )
                if completed.done():
                    wake.clear()
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
            self.controller.release_result_waiter(submission)

    def reset(self, context):
        self.execute(ModelRequest(), context, work="reset", retry_not_ready=False)
        try:
            while True:
                context.check("reset_registration")
                remaining = context.deadline.remaining_seconds()
                try:
                    status = self.controller.status()
                    reply = self._call_control(
                        {"verb": "register", "status": status},
                        CONTROL_RESPONSE_WAIT_SECONDS
                        if remaining is None
                        else max(0, min(remaining, CONTROL_RESPONSE_WAIT_SECONDS)),
                        context=context,
                    )
                except ControlError as exc:
                    if exc.code != "STALE_STATUS":
                        raise
                    continue
                self._apply_registration(reply, status["status_ns"] / 1e9)
                return
        except Exception as exc:
            raise CompletedControlError(getattr(exc, "code", "RESET_REGISTRATION_FAILED")) from exc

    def cancel(self, request_id, *, deadline=None):
        for submission in self.controller.cancel_pending(request_id):
            remaining = None if deadline is None else deadline.remaining_seconds()
            with suppress(Exception):
                self._call("cancel", submission, 0.1 if remaining is None else max(0, min(remaining, 0.1)))

    def close(self, context=None):
        if self.controller.handle.closed:
            return
        context = context or ExecutionContext("runtime-close", Deadline.after(self.target.request_timeout_ms / 1000))
        # Ordinary Close uses the same device queue. Unknown or unavailable
        # ownership stays retained until cleanup can be proven complete.
        with lifecycle_lock(self._close_lock, context):
            if self.controller.handle.closed:
                return
            self._close_requested = True
            self.execute(ModelRequest(), context, work="close", retry_not_ready=False)
            self._closing = True
            self.controller.shutdown(context)

    def destroy_transport(self):
        """Destroy services only after the host owns no unacknowledged work."""
        if any(not record.acknowledged for record in self.controller._records.values()):
            raise ControlError("TERMINAL_EVIDENCE_PENDING")
        if not self.controller.handle.closed:
            raise ControlError("RUNTIME_CLOSE_REQUIRED")
        self._closing = True
        self.controller.shutdown()
        if self._registration is not None:
            self._client.remove_pending_request(self._registration)
        self.node.destroy_timer(self._timer)
        self.node.destroy_service(self._service)
        self.node.destroy_client(self._client)
