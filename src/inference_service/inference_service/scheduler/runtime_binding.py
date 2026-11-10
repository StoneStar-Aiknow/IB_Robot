"""Infrastructure binding for transparent ModelRuntimeHandle dispatch.

Business callers keep the ordinary handle API. Only this module translates
handle operations to scheduler messages and holds the local execution token.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager, nullcontext
from weakref import WeakKeyDictionary

from inference_service.host_lifecycle import HOST_SHUTDOWN_TIMEOUT_SEC
from inference_service.unified_runtime import Deadline, ExecutionContext, ModelRequest, ModelResult
from inference_service.unified_runtime.contracts import OutcomeEvidence

from .runtime_protocol import (
    DEFAULT_RUNTIME_ENDPOINT,
    RUNTIME_HANDLE_OPERATION,
    ControlError,
    NotStartedControlError,
    RuntimeExecutionTarget,
)
from .runtime_ros import SchedulerRuntimeAdapter

_STREAM = "runtime_stream_id"
_close_transport_locks = WeakKeyDictionary()
_close_transport_cleanup = WeakKeyDictionary()
_close_transport_registry_lock = threading.Lock()


@contextmanager
def closing_control_transport(node):
    """Keep control callbacks live for a host detached from its serving executor."""
    context = getattr(node, "context", None)
    if context is None or not context.ok():
        yield
        return
    from inference_service.unified_runtime.lifecycle import lifecycle_lock

    # Serialize detached shutdown per node. A second binding must not borrow
    # a temporary executor whose owning Close can destroy it underneath it.
    with _close_transport_registry_lock:
        lock = _close_transport_locks.setdefault(node, threading.RLock())
    with lifecycle_lock(lock):
        cleanup = _close_transport_cleanup.get(node)
        if cleanup is not None:
            cleanup()
            del _close_transport_cleanup[node]
        previous = getattr(node, "executor", None)
        if previous is not None and node in previous.get_nodes():
            yield
            return
        from rclpy.executors import SingleThreadedExecutor

        executor = SingleThreadedExecutor(context=context)
        stopping = threading.Event()
        spin = None

        def run():
            while context.ok() and not stopping.is_set():
                executor.spin_once(timeout_sec=0.05)

        def cleanup():
            from inference_service.unified_runtime.lifecycle import lifecycle_remaining

            # All callbacks run on this one thread. Join it before shutdown:
            # rclpy.shutdown(False) stops spinning without completing cleanup,
            # and a repeated shutdown cannot establish callback completion.
            stopping.set()
            executor.wake()
            if spin is not None and spin.ident is not None:
                spin.join(timeout=lifecycle_remaining(HOST_SHUTDOWN_TIMEOUT_SEC))
                if spin.is_alive():
                    raise TimeoutError("runtime close control callbacks are still running")
            if not executor.shutdown(timeout_sec=lifecycle_remaining(HOST_SHUTDOWN_TIMEOUT_SEC)):
                raise TimeoutError("runtime close control transport is still running")
            if node.executor is executor:
                node.executor = None

        try:
            executor.add_node(node)
            spin = threading.Thread(target=run, name="runtime-close-control", daemon=True)
            spin.start()
            yield
        finally:
            try:
                cleanup()
            except BaseException:
                _close_transport_cleanup[node] = cleanup
                raise


def runtime_load_options(targets):
    """Derive host loading policy from the scheduler's validated resource mode."""
    targets = [RuntimeExecutionTarget(**target) if isinstance(target, dict) else target for target in targets]
    modes = {target.resource_mode for target in targets}
    if len(modes) > 1:
        raise ValueError("one runtime cannot mix serial and ascend_concurrent resources")
    return {
        "managed_execution": bool(targets),
        "priority_scheduling": modes == {"ascend_concurrent"},
    }


class _LocalRuntime:
    def __init__(self, owner):
        self.owner = owner
        self.authority = None
        self.close_requested = False

    @property
    def local_ready(self):
        # Public Close has already stopped owner admission. Keep the control
        # path available until Close obtains its resource reservation; the
        # controller gate still quarantines recovery and unknown operations.
        return self.owner.local_ready or (self.close_requested and not self.owner.closed)

    @property
    def closed(self):
        return self.owner.closed

    def bind_execution_policy(self, policy):
        self.authority = object()
        return self.authority

    def _check_authority(self, authority):
        if authority is None or authority is not self.authority:
            raise ControlError("DISPATCH_TOKEN_REQUIRED")

    def execute(self, request, context, *, _dispatch_key=None):
        if _dispatch_key is None or _dispatch_key is not self.authority:
            raise ControlError("DISPATCH_TOKEN_REQUIRED")
        metadata = dict(request.metadata)
        operation = metadata.pop(RUNTIME_HANDLE_OPERATION, "execute")
        stream_id = metadata.pop(_STREAM, None)
        request = ModelRequest(request.inputs, metadata, request.identity_metadata_keys)
        if operation == "open_stream":
            stream = self.owner._open_stream_local(context, _dispatch_key=self.owner_authority)
            self.owner._set_stream_delivery_deadline(stream, context.deadline)
            return ModelResult({"stream_id": stream.stream_id}, 0.0, OutcomeEvidence.completed(operation))
        if operation == "step":
            return self.owner._step_local(stream_id, request, context, _dispatch_key=self.owner_authority)
        if operation in {"reset_stream", "close_stream"}:
            getattr(self.owner, f"_{operation}_local")(stream_id, context, _dispatch_key=self.owner_authority)
            return ModelResult({}, 0.0, OutcomeEvidence.completed(operation))
        if operation != "execute":
            raise ControlError("INVALID_RUNTIME_OPERATION")
        return self.owner._execute_local(request, context, _dispatch_key=self.owner_authority)

    def _reset_local(self, context, *, _dispatch_key):
        self._check_authority(_dispatch_key)
        self.owner._reset_local(context, _dispatch_key=self.owner_authority)

    def recover_stream(self, command, stream_id, context, *, _dispatch_key):
        self._check_authority(_dispatch_key)
        # A runtime-wide failure requires the owner's full reset contract.
        # Per-stream failures otherwise leave unrelated streams untouched.
        if self.owner.reset_required:
            self._reset_local(context, _dispatch_key=self.authority)
            if command == "reset_stream":
                return
        getattr(self.owner, f"_{command}_local")(stream_id, context, _dispatch_key=self.owner_authority)

    def _close_local(self, context, *, _dispatch_key):
        self._check_authority(_dispatch_key)
        self.owner._close_local(context, _dispatch_key=self.owner_authority)

    def settle_stream_close(self, stream_id):
        self.owner._settle_stream_close(stream_id)

    def settle_recovery(self, command, stream_id):
        if command == "close_stream":
            self.settle_stream_close(stream_id)

    def drain_managed(self, context, *, _dispatch_key):
        self._check_authority(_dispatch_key)
        if self.owner.closed:
            return  # Completed local Close already proved all native resources released.
        # Controller recovery already fences all dispatch until activation.
        # Device completion must not turn a stream fault into a runtime reset.
        self.owner.drain_managed(context, _dispatch_key=self.owner_authority, preserve_recovery_scope=True)


class RuntimeBinding:
    """Execution policy plus transport ownership, created by a runtime host."""

    def __init__(self, adapter, local, *, execution_scope=None, request_transform=None, lifecycle_timeout=60.0):
        self.adapter, self.local = adapter, local
        self._request_transform = request_transform
        self._execution_scope = execution_scope
        self._close_lock = threading.Lock()
        self._closed = False
        self._close_requested = False
        self._close_submission = None
        # The creation budget is only a fallback for an unbounded stream open.
        # Reset and Close keep their own caller/target deadlines.
        self._open_stream_timeout = lifecycle_timeout

    def failure_diagnostic(self, failure):
        return self.controller.policy.failure_diagnostic(failure)

    @property
    def controller(self):
        return self.adapter.controller

    @property
    def target(self):
        return self.adapter.target

    @property
    def ready(self):
        return not self._close_requested and not self._closed and self.adapter.ready

    @property
    def registration_error(self) -> str:
        return self.adapter.registration_error

    @property
    def input_ready(self):
        return not self._close_requested and not self._closed and self.adapter.stream_input_available

    def execute(self, request, context, *, priority=None):
        if self._request_transform is not None:
            request = self._request_transform(request)
        if RUNTIME_HANDLE_OPERATION in request.metadata or _STREAM in request.metadata:
            raise ValueError("reserved runtime control metadata")
        if priority is None:
            priority = request.metadata.get("priority")
        scope = self._execution_scope(request, context) if self._execution_scope is not None else nullcontext()
        with scope:
            return self.adapter.execute(request, context, priority=priority)

    def _stream_operation(self, operation, stream, request, context):
        stream_id = stream if isinstance(stream, str) else stream.stream_id
        metadata = {**request.metadata, RUNTIME_HANDLE_OPERATION: operation, _STREAM: stream_id}
        return self.adapter.execute(
            ModelRequest(request.inputs, metadata, request.identity_metadata_keys | {_STREAM}),
            context,
            priority=request.metadata.get("priority"),
        )

    def open_stream(self, context):
        if context.deadline.is_unbounded:
            context = context.with_deadline(Deadline.after(self._open_stream_timeout))
        result = self.adapter.execute(ModelRequest(metadata={RUNTIME_HANDLE_OPERATION: "open_stream"}), context)
        return self.local.owner._set_stream_delivery_deadline(result.outputs["stream_id"], None)

    def step(self, stream, request, context):
        if RUNTIME_HANDLE_OPERATION in request.metadata or _STREAM in request.metadata:
            raise ValueError("reserved runtime control metadata")
        return self._stream_operation("step", stream, request, context)

    def reset_stream(self, stream, context):
        stream_id = stream if isinstance(stream, str) else stream.stream_id
        self.adapter.reset(context, stream_id=stream_id)

    def close_stream(self, stream, context):
        self._control_stream("close_stream", stream, context)
        stream_id = stream if isinstance(stream, str) else stream.stream_id
        self.adapter.release_reset_receipts(context, stream_id=stream_id)

    def _control_stream(self, operation, stream, context):
        stream_id = stream if isinstance(stream, str) else stream.stream_id
        if operation == "close_stream" and self.adapter.has_recovery(("reset_stream", stream_id)):
            # Settle a known result before abandoning this stream. This replays
            # registration/ACK only; it cannot execute another stream reset.
            try:
                self.adapter.recover(context, command="reset_stream", stream_id=stream_id, completed_only=True)
            except ControlError as exc:
                # Reset completion is not evidence that this Close started.
                raise NotStartedControlError(exc.code) from exc
        if not self.adapter.has_recovery((operation, stream_id)):
            if operation == "close_stream" and self.local.owner._stream_is_closed(stream):
                return
            try:
                return self.adapter.execute(
                    ModelRequest(
                        metadata={RUNTIME_HANDLE_OPERATION: operation, _STREAM: stream_id},
                        identity_metadata_keys={_STREAM},
                    ),
                    context,
                    retry_not_ready=False,
                    on_settled=(lambda: self.local.settle_stream_close(stream_id))
                    if operation == "close_stream"
                    else None,
                )
            except NotStartedControlError as exc:
                if exc.code not in {
                    "RUNTIME_NOT_REGISTERED",
                    "RESOURCE_NOT_READY",
                    "RUNTIME_FENCED",
                    "REQUEST_CAPACITY",
                    "INPUT_CAPACITY",
                } and not (
                    operation == "close_stream" and exc.code in {"PROFILE_EXPIRED", "PROFILE_INPUT_OUT_OF_RANGE"}
                ):
                    raise
        return self.adapter.recover(context, command=operation, stream_id=stream_id)

    def reset(self, context):
        return self.adapter.reset(context)

    def cancel(self, request_id, *, deadline=None):
        self.adapter.cancel(request_id, deadline=deadline)

    def close(self, context=None):
        if context is None:
            # Host teardown uses the same public barrier with a fresh deadline.
            # Keep its executor and transport alive if that barrier is unknown.
            return self.local.owner.close(
                ExecutionContext("runtime-close", Deadline.after(self.target.request_timeout_ms / 1000))
            )
        from inference_service.unified_runtime.lifecycle import lifecycle_lock

        with lifecycle_lock(self._close_lock, context), closing_control_transport(self.adapter.node):
            if self._closed:
                return
            self._close_requested = True
            self.local.close_requested = True
            controller = self.controller
            unsettled, needs_recovery, before = controller.close_snapshot(self._close_submission)
            cleanup_complete = self.local.owner.closed and not unsettled
            recover = self.adapter.has_recovery("close") or needs_recovery
            if cleanup_complete and not self.adapter.has_recovery("close"):
                self.adapter.close(context)
                self._closed = True
                return
            # Missing registration evidence does not prove that SDK loading
            # never happened, or that another host is not using this device.
            self.adapter.wait_for_registration(context)
            if recover:
                # Recovery fences and drains the entire resource, including
                # every stage peer, before closing this lifecycle owner.
                self.adapter.recover(context, command="close")
            else:
                try:
                    self.adapter.execute(ModelRequest(), context, work="close")
                except NotStartedControlError:
                    # Only this Close is proven not started. Other operations
                    # may still own device work, so local cleanup is unsafe.
                    self.adapter.recover(context, command="close")
                except BaseException:
                    self._close_submission = controller.find_close_submission(before, self._close_submission)
                    # Even when local Close finished, keep terminal evidence
                    # transport alive until the next caller observes it.
                    raise
            self.adapter.close(context)
            self._closed = True


def bind_runtime(
    node,
    handle,
    target,
    *,
    endpoint=DEFAULT_RUNTIME_ENDPOINT,
    context_provider=None,
    execution_scope=None,
    request_transform=None,
    boot_id=None,
    lifecycle_timeout=60.0,
) -> RuntimeBinding | SchedulerRuntimeAdapter:
    """Bind an executable owner, or publish an unavailable owner's cleanup evidence.

    Owners without bind_execution_policy (including UnavailableRuntime) return
    the transport adapter directly. Only executable owners return RuntimeBinding
    with an .adapter and consume context_provider, execution_scope,
    request_transform and lifecycle_timeout.
    """
    from .runtime_controller import UnavailableRuntime

    if isinstance(handle, UnavailableRuntime) or not callable(getattr(handle, "bind_execution_policy", None)):
        return SchedulerRuntimeAdapter(node, target, handle, endpoint=endpoint, boot_id=boot_id)
    local = _LocalRuntime(handle)
    adapter = SchedulerRuntimeAdapter(
        node, target, local, endpoint=endpoint, boot_id=boot_id, registration_enabled=False
    )
    binding = RuntimeBinding(
        adapter,
        local,
        execution_scope=execution_scope,
        request_transform=request_transform,
        lifecycle_timeout=lifecycle_timeout,
    )
    try:
        local.owner_authority = handle.bind_execution_policy(binding, context_provider=context_provider)
    except BaseException as error:
        # The model remains owned by its construction transaction. No host
        # registration or device work was published before policy installation.
        from inference_service.unified_runtime.construction import current_runtime_construction

        construction = current_runtime_construction()
        if construction is not None:
            construction.bindings.append(binding)
        adapter.rollback_unpublished()
        error.cleanup_owner = adapter
        raise
    adapter._registration_enabled = True
    return binding
