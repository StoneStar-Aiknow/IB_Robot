"""The public handle API remains stable when infrastructure owns dispatch."""

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from inference_service.unified_runtime import (
    Deadline,
    ExecutionContext,
    ExecutionFailure,
    LifecycleState,
    ModelRequest,
    ModelRuntimeHandle,
    OutcomeEvidence,
    OutcomeState,
    StreamState,
)


class Executor:
    def __init__(self):
        self.events = []

    def execute(self, request, context):
        self.events.append(("execute", context))
        return {"value": request.inputs["value"]}

    def reset(self, context):
        self.events.append(("reset", context))

    def close(self):
        self.events.append(("close", None))


class LoopbackPolicy:
    """A separate worker owns local calls, as an external dispatcher would."""

    def __init__(self, owner):
        self.owner = owner
        self.authority = None
        self.calls = []
        self.available = True
        self.input_ready = True
        self.worker = ThreadPoolExecutor(max_workers=1)

    @property
    def ready(self):
        # Reading public health must not hold the handle lock while a dispatch
        # controller consults local health on another thread.
        return self.available and self.worker.submit(lambda: self.owner.local_ready).result(timeout=1)

    def local(self, operation, *args):
        self.calls.append(operation)
        method = getattr(self.owner, f"_{operation}_local")
        return self.worker.submit(method, *args, _dispatch_key=self.authority).result(timeout=1)

    def execute(self, request, context):
        return self.local("execute", request, context)

    def open_stream(self, context):
        return self.local("open_stream", context)

    def step(self, stream, request, context):
        return self.local("step", stream, request, context)

    def reset(self, context):
        return self.local("reset", context)

    def close(self, context):
        return self.local("close", context)

    def reset_stream(self, stream, context):
        return self.local("reset_stream", stream, context)

    def close_stream(self, stream, context):
        return self.local("close_stream", stream, context)

    def cancel(self, request_id, *, deadline=None):
        self.calls.append(("cancel", request_id, deadline))


def managed_handle(*, streaming_runtime=None, context_provider=None):
    executor = Executor()
    options = {"state_scope": "stream", "stateful": True} if streaming_runtime is not None else {}
    owner = ModelRuntimeHandle(executor=executor, streaming_runtime=streaming_runtime, resettable=True, **options)
    owner.load()
    policy = LoopbackPolicy(owner)
    policy.authority = owner.bind_execution_policy(policy, context_provider=context_provider)
    return owner, executor, policy


def dispose(owner, policy):
    try:
        owner.close()
    finally:
        policy.worker.shutdown()


@pytest.mark.parametrize("operation", ["reset", "close", "drain_managed"])
@pytest.mark.parametrize("failed", [False, True])
def test_policy_frame_delivery_rejects_control_reentry_on_its_own_worker(operation, failed):
    owner, executor, policy = managed_handle()
    deliver = threading.Event()
    callback_done = threading.Event()
    errors = []
    payload = {"value": 7}

    def submit(request, context):
        def dispatch():
            assert deliver.wait(2)
            local = owner._submit_frame_local(request, context, _dispatch_key=policy.authority)
            return local.result(timeout=1)

        return policy.worker.submit(dispatch)

    def native_submit(*_, **__):
        future = Future()
        if failed:
            error = RuntimeError("completed frame rejected")
            error.evidence = OutcomeEvidence.completed()
            future.set_exception(error)
        else:
            future.set_result(payload)
        return future

    def reenter(_):
        try:
            context = ExecutionContext("callback-control", Deadline.after(0.1))
            if operation == "drain_managed":
                owner.drain_managed(context, _dispatch_key=policy.authority)
            else:
                getattr(owner, operation)(context)
        except Exception as exc:
            errors.append(exc)
        finally:
            callback_done.set()

    executor.submit_frame = native_submit
    policy.submit_frame = submit
    try:
        result = owner.submit_frame(ModelRequest(), ExecutionContext("frame"))
        result.add_done_callback(reenter)
        deliver.set()
        assert callback_done.wait(3)
        if failed:
            with pytest.raises(ExecutionFailure, match="completed frame rejected"):
                result.result(timeout=1)
        else:
            assert result.result(timeout=1) is payload
        assert len(errors) == 1 and isinstance(errors[0], ExecutionFailure)
        assert errors[0].code == "runtime_reentrant" and errors[0].evidence.state is OutcomeState.NOT_STARTED
        assert policy.calls == [] and executor.events == [] and owner.local_ready
        owner.reset(ExecutionContext("outside-callback"))
        assert policy.calls == ["reset"] and owner.ready
    finally:
        deliver.set()
        dispose(owner, policy)


@pytest.mark.parametrize("entry", ["execute", "step", "open_stream", "frame_callback"])
@pytest.mark.parametrize("bounded", [False, True])
def test_managed_drain_rejects_active_reentry_without_revoking_admission(entry, bounded):
    native = Streaming() if entry in {"step", "open_stream"} else None
    owner, executor, policy = managed_handle(streaming_runtime=native)
    failures = []

    def reenter(*_):
        context = ExecutionContext("drain", Deadline.after(0.02) if bounded else Deadline.unbounded())
        with pytest.raises(ExecutionFailure) as caught:
            owner.drain_managed(context, _dispatch_key=policy.authority)
        failures.append(caught.value)
        return {"count": 0, "value": 1}

    try:
        if entry == "execute":
            executor.execute = reenter
            owner.execute(ModelRequest(), ExecutionContext("execute"))
        elif entry == "step":
            stream = owner.open_stream(ExecutionContext("open"))
            native.step = reenter
            owner.step(stream, ModelRequest(), ExecutionContext("step"))
        elif entry == "open_stream":
            native.open_stream = reenter
            owner.open_stream(ExecutionContext("open"))
        else:
            pending = Future()
            executor.submit_frame = lambda *_, **__: pending
            frame = owner._submit_frame_local(ModelRequest(), ExecutionContext("frame"), _dispatch_key=policy.authority)
            frame.add_done_callback(reenter)
            pending.set_result({"value": 1})
            frame.result(timeout=1)
        assert len(failures) == 1 and failures[0].code == "runtime_reentrant"
        assert failures[0].evidence.state is OutcomeState.NOT_STARTED
        assert owner.local_ready and owner.ready
    finally:
        dispose(owner, policy)


@pytest.mark.parametrize("operation", ["execute", "step", "submit_frame", "open_stream"])
@pytest.mark.parametrize("expired", [False, True])
def test_invalid_request_is_rejected_before_policy_queue(operation, expired):
    owner, executor, policy = managed_handle(
        streaming_runtime=Streaming() if operation in {"step", "open_stream"} else None
    )
    stream = owner.open_stream(ExecutionContext("open")) if operation == "step" else None
    policy.calls.clear()
    policy.submit_frame = lambda *_: pytest.fail("expired frame reached policy")
    context = ExecutionContext("invalid", Deadline.after(0) if expired else Deadline.unbounded())
    if not expired:
        context.token.cancel("already canceled")
    try:
        with pytest.raises(ExecutionFailure) as failure:
            if operation == "step":
                owner.step(stream, ModelRequest(), context)
            elif operation == "open_stream":
                owner.open_stream(context)
            elif operation == "submit_frame":
                owner.submit_frame(ModelRequest(), context).result(timeout=1)
            else:
                owner.execute(ModelRequest(), context)
        assert failure.value.evidence.state is OutcomeState.NOT_STARTED
        assert policy.calls == [] and executor.events == []
        assert owner.ready
    finally:
        dispose(owner, policy)


def test_policy_preserves_pipeline_evidence_on_original_exception():
    owner, executor, policy = managed_handle()
    evidence = OutcomeEvidence.completed("model.encoder", stage="encoder")
    error = RuntimeError("canceled after encoder")
    error.evidence = evidence
    error.details = {"diagnostic": "original"}

    def fail(*_):
        raise error

    policy.execute = fail
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest(), ExecutionContext("pipeline"))
        assert caught.value.evidence is evidence
        assert caught.value.details["diagnostic"] == "original"
        assert owner.ready and executor.events == []
    finally:
        dispose(owner, policy)


def test_public_execute_reset_close_delegate_without_recursive_dispatch_or_lifecycle_deadlock():
    owner, executor, policy = managed_handle()
    context = ExecutionContext("business", Deadline.after(3))
    try:
        assert owner.execute(ModelRequest({"value": 7}), context).outputs == {"value": 7}
        owner.reset(context)
        owner.close(context)
        owner.close(context)
        assert policy.calls == ["execute", "reset", "close"]
        assert [event for event, _ in executor.events] == ["execute", "reset", "close"]
        assert owner.state is LifecycleState.CLOSED
    finally:
        dispose(owner, policy)


@pytest.mark.parametrize("operation", ["execute", "reset", "close"])
def test_bound_handle_rejects_local_entry_without_capability(operation):
    owner, executor, policy = managed_handle()
    context = ExecutionContext("unauthorized")
    args = (ModelRequest({"value": 1}), context) if operation == "execute" else (context,)
    try:
        with pytest.raises(ExecutionFailure, match="authorized local dispatch") as caught:
            getattr(owner, f"_{operation}_local")(*args, _dispatch_key=object())
        assert caught.value.code == "dispatch_token_required"
        assert caught.value.evidence.state is OutcomeState.NOT_STARTED
        assert not executor.events and not policy.calls
        assert owner.local_ready
    finally:
        dispose(owner, policy)


def test_readiness_and_input_readiness_do_not_call_policy_under_handle_lock():
    owner, _executor, policy = managed_handle()
    try:
        assert owner.ready and owner.health.ready and owner.diagnostics().health.ready
        policy.available = False
        assert owner.local_ready and not owner.ready
        # RuntimeHealth describes backend lifecycle; dispatch availability is a
        # separate public property and must not rewrite a healthy model state.
        assert owner.health.ready and owner.diagnostics().health.ready
        assert owner.input_ready
        policy.input_ready = False
        assert not owner.input_ready
    finally:
        dispose(owner, policy)


def test_context_provider_runs_once_and_preserves_native_cancellation():
    provided = []

    def provider(context):
        provided.append(context)
        return context.with_deadline(Deadline.after(1))

    owner, executor, policy = managed_handle(context_provider=provider)
    context = ExecutionContext("bounded")
    try:
        owner.execute(ModelRequest({"value": 1}), context)
        assert provided == [context]
        dispatched = executor.events[0][1]
        assert dispatched.cancellation_token is context.cancellation_token
        assert dispatched.request_id == context.request_id
        assert dispatched.deadline.remaining_seconds() <= 1
        owner.cancel(context.request_id, deadline=context.deadline)
        assert policy.calls[-1] == ("cancel", context.request_id, context.deadline)
    finally:
        dispose(owner, policy)


def test_close_failure_is_irreversible_and_retry_keeps_using_policy():
    owner, executor, policy = managed_handle()
    close = policy.close

    class UnknownClose(RuntimeError):
        operation_started = True
        outcome_known = False
        state_mutated = False

    def fail_close(context):
        assert owner.state is LifecycleState.CLOSING
        raise UnknownClose("drain uncertain")

    policy.close = fail_close
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.close()
        assert not caught.value.evidence.outcome_known
        assert owner.state is LifecycleState.CLOSING
        assert not owner.ready and not owner.input_ready
        with pytest.raises(ExecutionFailure, match="closed"):
            owner.execute(ModelRequest({"value": 1}), ExecutionContext("after-close"))
        with pytest.raises(ExecutionFailure, match="closed"):
            owner.reset()
        assert not executor.events
        policy.close = close
        owner.close()
        assert owner.closed
    finally:
        policy.close = close
        dispose(owner, policy)


def test_policy_not_started_does_not_degrade_backend_but_unknown_quarantines_it():
    owner, _executor, policy = managed_handle()

    class Rejected(RuntimeError):
        admission_rejected = True
        operation_started = False
        outcome_known = True
        state_mutated = False

    class Uncertain(RuntimeError):
        operation_started = True
        outcome_known = False
        state_mutated = False

    def reject(*_):
        raise Rejected("capacity")

    def uncertain(*_):
        raise Uncertain("lost execution evidence")

    try:
        policy.execute = reject
        with pytest.raises(ExecutionFailure):
            owner.execute(ModelRequest(), ExecutionContext("rejected"))
        assert owner.local_ready and owner.health.failure_count == 0
        policy.execute = uncertain
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest(), ExecutionContext("uncertain"))
        assert not caught.value.evidence.outcome_known
        assert owner.state is LifecycleState.RESET_REQUIRED
        assert not owner.ready
        owner.reset()
        assert owner.ready
    finally:
        dispose(owner, policy)


def test_normalized_uncertain_policy_failure_still_requires_recovery():
    owner, _executor, policy = managed_handle()
    failure = owner.failure_factory.create(
        "dispatch_uncertain", "missing completion receipt", evidence=OutcomeEvidence.started(outcome_known=False)
    )

    def fail(*_):
        raise failure

    policy.execute = fail
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest(), ExecutionContext("uncertain"))
        assert caught.value is failure
        assert owner.state is LifecycleState.RESET_REQUIRED
        assert not owner.ready
    finally:
        dispose(owner, policy)


@pytest.mark.parametrize("stop", ["cancel", "deadline", "plain"])
@pytest.mark.parametrize("future", [False, True])
@pytest.mark.parametrize("proven_unstarted", [False, True])
def test_policy_control_error_after_dispatch_preserves_only_explicit_nonstart(stop, future, proven_unstarted):
    from inference_service.unified_runtime.errors import DeadlineExceeded

    owner, _executor, policy = managed_handle()
    context = ExecutionContext("dispatch")

    def fail(*_):
        try:
            if stop == "deadline":
                raise DeadlineExceeded(phase="dispatch_wait")
            if stop == "plain":
                raise TimeoutError("completion receipt lost")
            context.token.cancel()
            context.check("dispatch_wait")
        except Exception as exc:
            if proven_unstarted:
                exc.evidence = OutcomeEvidence.not_started("dispatch")
            raise

    policy.execute = fail
    if future:

        def submit(*_):
            result = Future()
            try:
                fail()
            except Exception as exc:
                result.set_exception(exc)
            return result

        policy.submit_frame = submit
    try:
        with pytest.raises(ExecutionFailure) as caught:
            if future:
                owner.submit_frame(ModelRequest(), context).result(timeout=1)
            else:
                owner.execute(ModelRequest(), context)
        assert caught.value.evidence.state is (OutcomeState.NOT_STARTED if proven_unstarted else OutcomeState.STARTED)
        assert caught.value.evidence.outcome_known is proven_unstarted
        assert owner.ready is proven_unstarted
    finally:
        dispose(owner, policy)


@pytest.mark.parametrize("via_policy", [False, True])
@pytest.mark.parametrize("known", [False, True])
@pytest.mark.parametrize("action", ["reset_runtime", "reload"])
def test_request_preserves_explicit_runtime_recovery(via_policy, known, action):
    from inference_service.unified_runtime import RecoveryAction, RecoveryRequirement, RecoveryScope

    owner, executor, policy = managed_handle()
    failure = ExecutionFailure(
        "runtime_fault",
        "native runtime recovery required",
        evidence=OutcomeEvidence.started(outcome_known=known),
        recovery=RecoveryRequirement(RecoveryScope.RUNTIME, RecoveryAction(action)),
        recoverable=True,
    )

    def fail(*_):
        raise failure

    if via_policy:
        policy.execute = fail
    else:
        executor.execute = fail
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest(), ExecutionContext("failed"))
        assert caught.value.recovery is failure.recovery
        assert not owner.ready
        assert owner.health.recovery_requirement == failure.recovery
        expected = LifecycleState.FAILED if action == "reload" else LifecycleState.RESET_REQUIRED
        assert owner.state is expected
        if action == "reload":
            with pytest.raises(ExecutionFailure):
                owner.reset()
        else:
            owner.reset()
            assert owner.ready
    finally:
        dispose(owner, policy)


def test_close_racing_reset_does_not_reopen_local_admission():
    owner, executor, policy = managed_handle()
    entered = threading.Event()
    release = threading.Event()
    closing = threading.Event()
    reset_context = ExecutionContext("reset", Deadline.after(3))
    close = policy.close

    def reset(_context):
        entered.set()
        assert release.wait(2)

    def close_policy(context):
        closing.set()
        return close(context)

    executor.reset = reset
    policy.close = close_policy
    try:
        with ThreadPoolExecutor(max_workers=2) as callers:
            resetting = callers.submit(owner.reset, reset_context)
            assert entered.wait(1)
            close_future = callers.submit(owner.close, ExecutionContext("close", Deadline.after(3)))
            assert closing.wait(1)
            assert owner.state is LifecycleState.CLOSING
            release.set()
            resetting.result(timeout=2)
            close_future.result(timeout=2)
        assert owner.closed and not owner.ready
        with pytest.raises(ExecutionFailure, match="closed"):
            owner.execute(ModelRequest({"value": 1}), ExecutionContext("after-close"))
        assert [event for event, _ in executor.events] == ["close"]
    finally:
        release.set()
        dispose(owner, policy)


def test_composite_frame_policy_delegates_only_outer_orchestration_and_has_no_fallback_bypass():
    owner, executor, policy = managed_handle()
    frames = []

    def submit(request, *, context):
        frames.append(context.request_id)
        result = Future()
        result.set_result(request.inputs)
        return result

    executor.submit_frame = submit
    try:
        with pytest.raises(ExecutionFailure, match="does not support frame submission"):
            owner.submit_frame(ModelRequest(), ExecutionContext("unsupported"))
        assert frames == []
        policy.submit_frame = lambda request, context: policy.local("submit_frame", request, context)
        result = owner.submit_frame(ModelRequest({"value": 5}), ExecutionContext("frame"))
        assert result.result(timeout=1) == {"value": 5}
        assert policy.calls == ["submit_frame"] and frames == ["frame"]
        assert owner.active_executions == 0
    finally:
        dispose(owner, policy)


class Streaming:
    def open_stream(self, _context):
        return {"count": 0}

    def step(self, stream, _request, _context):
        stream["count"] += 1
        return dict(stream)

    def reset_stream(self, stream, _context):
        stream["count"] = 0

    def close_stream(self, stream, _context):
        stream.clear()


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("invalid_request", [None, {}])
def test_invalid_frame_request_does_not_dispatch_or_change_health(managed, invalid_request):
    owner, executor, policy = managed_handle()
    if not managed:
        dispose(owner, policy)
        owner = ModelRuntimeHandle(executor=executor, resettable=True)
        owner.load()
    executor.events.clear()
    policy.calls.clear()
    executor.submit_frame = lambda *args, **kwargs: pytest.fail("invalid request reached executor")
    policy.submit_frame = lambda request, context: policy.local("submit_frame", request, context)
    before = owner.health
    try:
        with pytest.raises(TypeError, match="ModelRequest"):
            owner.submit_frame(invalid_request, ExecutionContext("invalid"))
        assert owner.health == before
        assert not executor.events and not policy.calls
    finally:
        dispose(owner, policy) if managed else owner.close()


@pytest.mark.parametrize("managed", [False, True])
def test_invalid_close_stream_context_does_not_revoke_stream(managed):
    owner, _, policy = managed_handle(streaming_runtime=Streaming())
    if not managed:
        dispose(owner, policy)
        owner = ModelRuntimeHandle(
            Executor(), streaming_runtime=Streaming(), state_scope="stream", stateful=True, resettable=True
        )
        owner.load()
    context = ExecutionContext("stream")
    try:
        stream = owner.open_stream(context)
        policy.calls.clear()
        with pytest.raises(TypeError, match="ExecutionContext"):
            owner.close_stream(stream, object())
        assert not policy.calls
        assert owner.step(stream, ModelRequest(), context).outputs == {"count": 1}
        owner.reset_stream(stream, context)
        assert owner.step(stream, ModelRequest(), context).outputs == {"count": 1}
    finally:
        dispose(owner, policy) if managed else owner.close()


@pytest.mark.parametrize("operation", ["step", "reset_stream", "close_stream"])
@pytest.mark.parametrize("managed", [False, True])
def test_foreign_stream_cannot_address_same_id_in_another_runtime(operation, managed):
    owners = [managed_handle(streaming_runtime=Streaming()) for _ in range(2)]
    first, _, first_policy = owners[0]
    second, _, second_policy = owners[1]
    if managed:
        # A transport serializes stream identity, so the public boundary must
        # reject a foreign owner before a policy can discard that identity.
        local = second_policy.local
        second_policy.local = lambda op, *args: local(op, *(getattr(arg, "stream_id", arg) for arg in args))
    else:
        # Exercise the same contract on an ordinary unbound runtime.
        dispose(second, second_policy)
        second = ModelRuntimeHandle(
            executor=Executor(), streaming_runtime=Streaming(), state_scope="stream", stateful=True, resettable=True
        )
        second.load()
    try:
        context = ExecutionContext("stream-ownership", Deadline.after(3))
        foreign = first.open_stream(context)
        own = second.open_stream(context)
        assert foreign.stream_id == own.stream_id
        second.step(own, ModelRequest(), context)
        args = (foreign, ModelRequest(), context) if operation == "step" else (foreign, context)
        with pytest.raises(ExecutionFailure) as caught:
            getattr(second, operation)(*args)
        assert caught.value.code == "stream_not_found"
        assert caught.value.evidence.state is OutcomeState.NOT_STARTED
        assert second.step(own, ModelRequest(), context).outputs == {"count": 2}
        second.close_stream(own)
        second.close_stream(own)
        # Closing a foreign handle must still fail when the matching ID is closed.
        with pytest.raises(ExecutionFailure, match="stream"):
            second.close_stream(foreign)
    finally:
        dispose(first, first_policy)
        if managed:
            dispose(second, second_policy)
        else:
            second.close()


def test_uncertain_stream_step_does_not_escalate_to_runtime_reset():
    class UncertainStreaming(Streaming):
        def step(self, _stream, _request, _context):
            class Unknown(RuntimeError):
                operation_started = True
                outcome_known = False
                state_mutated = True

            raise Unknown("stream result lost")

    owner, _executor, policy = managed_handle(streaming_runtime=UncertainStreaming())
    context = ExecutionContext("stream-unknown")
    try:
        stream = owner.open_stream(context)
        with pytest.raises(ExecutionFailure):
            owner.step(stream, ModelRequest(), context)
        assert stream.state is StreamState.RESET_REQUIRED
        assert owner.state is LifecycleState.READY
        assert owner.health.failure_count == 1
    finally:
        dispose(owner, policy)


def test_lost_policy_close_receipt_cannot_reopen_confirmed_closed_stream():
    owner, _, policy = managed_handle(streaming_runtime=Streaming())
    stream = owner.open_stream(ExecutionContext("open"))
    close = policy.close_stream

    def lost_receipt(stream, context):
        close(stream, context)
        raise ExecutionFailure("receipt_lost", "close receipt lost", evidence=OutcomeEvidence.started())

    policy.close_stream = lost_receipt
    try:
        with pytest.raises(ExecutionFailure):
            owner.close_stream(stream)
        assert stream.closed and owner.ready
        assert owner.open_stream_count == 0 and owner.health.failure_count == 1
        owner.close_stream(stream)
    finally:
        dispose(owner, policy)


@pytest.mark.parametrize("operation", ["step", "reset_stream", "close_stream"])
@pytest.mark.parametrize("bank", ["per_stream", "runtime_exclusive"])
@pytest.mark.parametrize("normalized", [False, True])
def test_stream_policy_uncertainty_respects_state_bank_scope(operation, bank, normalized):
    owner = ModelRuntimeHandle(
        executor=Executor(),
        streaming_runtime=Streaming(),
        state_scope="stream",
        stateful=True,
        resettable=True,
        state_bank_mode=bank,
        max_open_streams=1,
    )
    owner.load()
    policy = LoopbackPolicy(owner)
    policy.authority = owner.bind_execution_policy(policy)

    class RawUnknown(RuntimeError):
        operation_started = True
        outcome_known = False
        state_mutated = True

    def unknown(*_):
        if normalized:
            raise ExecutionFailure("receipt_lost", "OUTCOME_UNKNOWN", evidence=OutcomeEvidence.started())
        raise RawUnknown("OUTCOME_UNKNOWN")

    setattr(policy, operation, unknown)
    context = ExecutionContext("raw-stream-unknown")
    try:
        stream = owner.open_stream(context)
        args = (stream, ModelRequest(), context) if operation == "step" else (stream, context)
        with pytest.raises(ExecutionFailure) as caught:
            getattr(owner, operation)(*args)
        assert not caught.value.evidence.outcome_known
        stream_close = operation == "close_stream" and bank == "per_stream"
        assert stream.state is (StreamState.FAILED if stream_close else StreamState.RESET_REQUIRED)
        if stream_close:
            assert caught.value.close_pending
            assert caught.value.recovery.action.value == "close_stream"
            with pytest.raises(ExecutionFailure):
                owner.reset_stream(stream, ExecutionContext("cannot-reopen"))
        expected = LifecycleState.READY if bank == "per_stream" else LifecycleState.RESET_REQUIRED
        assert owner.state is expected
        assert owner.health.failure_count == 1
    finally:
        dispose(owner, policy)


@pytest.mark.parametrize("with_policy", [False, True])
def test_domain_admission_follows_current_dispatch_authority(with_policy):
    from inference_service.backends import BackendCapabilities, ResourceDomainAdmissions
    from inference_service.unified_runtime import RuntimeAssembly, RuntimeProviders

    provider = ResourceDomainAdmissions()
    domain = provider.register("test-device", 1)
    executor = Executor()
    session = SimpleNamespace(name="test-model", capabilities=BackendCapabilities(resource_domain="test-device"))
    owner = ModelRuntimeHandle(
        RuntimeAssembly(runtime_executor=executor, session=session, providers=RuntimeProviders(object(), provider))
    )
    owner.load()
    policy = LoopbackPolicy(owner) if with_policy else None
    if policy:
        policy.authority = owner.bind_execution_policy(policy)
    domain.acquire()
    try:
        context = ExecutionContext("domain", Deadline.after(0.1))
        if policy:
            assert owner.execute(ModelRequest({"value": 3}), context).outputs == {"value": 3}
        else:
            with pytest.raises(ExecutionFailure, match="deadline"):
                owner.execute(ModelRequest({"value": 3}), context)
            assert not executor.events
    finally:
        domain.release()
        domain.close()
        dispose(owner, policy) if policy else owner.close()


@pytest.mark.parametrize("streaming,missing", [(False, "ready"), (False, "cancel"), (True, "open_stream")])
def test_policy_contract_is_validated_before_binding(streaming, missing):
    options = {"streaming_runtime": Streaming(), "state_scope": "stream"} if streaming else {}
    owner = ModelRuntimeHandle(Executor(), **options)
    members = {
        name: lambda *_: None
        for name in ("execute", "reset", "close", "cancel", "open_stream", "step", "reset_stream", "close_stream")
    }
    members["ready"] = True
    members.pop(missing)
    with pytest.raises(TypeError, match=missing):
        owner.bind_execution_policy(SimpleNamespace(**members))
    assert not owner.execution_policy_bound
    owner.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("rejected", [False, True])
def test_policy_failure_without_completion_evidence_quarantines_stateless_runtime(asynchronous, rejected):
    from inference_service.backends.errors import BackendAdmissionError

    owner, _, policy = managed_handle()

    def fail(*_):
        error = BackendAdmissionError("queue full") if rejected else TimeoutError("receipt lost")
        if asynchronous:
            future = Future()
            future.set_exception(error)
            return future
        raise error

    setattr(policy, "submit_frame" if asynchronous else "execute", fail)
    try:
        with pytest.raises(ExecutionFailure) as failure:
            if asynchronous:
                owner.submit_frame(ModelRequest(), ExecutionContext("frame")).result(timeout=1)
            else:
                owner.execute(ModelRequest(), ExecutionContext("request"))
        assert failure.value.evidence.outcome_known is rejected
        assert (failure.value.evidence.state is OutcomeState.NOT_STARTED) is rejected
        assert owner.ready is rejected
    finally:
        dispose(owner, policy)


def test_remote_cancel_failure_still_delivers_local_token():
    owner, executor, policy = managed_handle()
    entered, release = threading.Event(), threading.Event()
    context = ExecutionContext("cancel")

    def execute(*_):
        entered.set()
        assert release.wait(2)
        return {}

    def cancel(*_, **__):
        raise TimeoutError("cancel receipt lost")

    executor.execute, policy.cancel = execute, cancel
    with ThreadPoolExecutor(1) as worker:
        running = worker.submit(owner.execute, ModelRequest(), context)
        try:
            assert entered.wait(1)
            with pytest.raises(TimeoutError, match="cancel receipt"):
                owner.cancel(context.request_id)
            assert context.cancellation_token.is_cancelled
        finally:
            release.set()
        with pytest.raises(ExecutionFailure):
            running.result(timeout=1)
    dispose(owner, policy)


@pytest.mark.parametrize("local_failure", [False, True])
@pytest.mark.parametrize("remote_failure", [False, True])
def test_cancel_notifies_both_owners_once_even_when_a_hook_fails(local_failure, remote_failure):
    owner, executor, policy = managed_handle()
    entered, release = threading.Event(), threading.Event()
    context = ExecutionContext("cancel-both")
    delivered = []
    original = ValueError("policy transport root cause")
    deadline = Deadline.after(1)

    def execute(*_):
        entered.set()
        assert release.wait(2)
        return {}

    def remote(request_id, *, deadline=None):
        assert context.token.is_cancelled
        delivered.append(("policy", request_id, deadline))
        if remote_failure:
            raise TimeoutError("remote cancel failed") from original

    def local(request_id, *, deadline=None):
        assert delivered[0][0] == "policy"
        delivered.append(("local", request_id, deadline))
        if local_failure:
            # The signature is compatible; this is an internal TypeError after
            # a side effect and must never cause a second cancel invocation.
            raise TypeError("internal cancel failed")

    executor.execute, executor.cancel, policy.cancel = execute, local, remote
    try:
        with ThreadPoolExecutor(1) as worker:
            active = worker.submit(owner.execute, ModelRequest(), context)
            try:
                assert entered.wait(1)
                if remote_failure or local_failure:
                    with pytest.raises(TimeoutError if remote_failure else TypeError) as caught:
                        owner.cancel(context.request_id, deadline=deadline)
                    if remote_failure:
                        assert caught.value.__cause__ is original
                        if local_failure:
                            assert "internal cancel failed" in caught.value.__notes__[0]
                else:
                    owner.cancel(context.request_id, deadline=deadline)
                assert delivered == [
                    ("policy", context.request_id, deadline),
                    ("local", context.request_id, deadline.deadline_at),
                ]
            finally:
                release.set()
            with pytest.raises(ExecutionFailure):
                active.result(timeout=1)
    finally:
        release.set()
        dispose(owner, policy)


def test_policy_cancel_arrives_before_a_blocking_local_hook():
    owner, executor, policy = managed_handle()
    executing, hook_entered, finish, hook_finish = (threading.Event() for _ in range(4))
    context = ExecutionContext("blocked-hook")

    def execute(*_):
        executing.set()
        assert finish.wait(2)
        return {}

    def cancel(_request_id):
        hook_entered.set()
        assert hook_finish.wait(2)

    executor.execute, executor.cancel = execute, cancel
    try:
        with ThreadPoolExecutor(2) as pool:
            active = pool.submit(owner.execute, ModelRequest(), context)
            assert executing.wait(1)
            cancellation = pool.submit(owner.cancel, context.request_id)
            try:
                assert hook_entered.wait(1)
                assert ("cancel", context.request_id, None) in policy.calls and context.token.is_cancelled
            finally:
                hook_finish.set()
                finish.set()
            cancellation.result(timeout=1)
            with pytest.raises(ExecutionFailure):
                active.result(timeout=1)
    finally:
        hook_finish.set()
        finish.set()
        dispose(owner, policy)


def test_stream_calls_use_policy_and_runtime_reset_does_not_reschedule_inner_stream_reset():
    owner, _executor, policy = managed_handle(streaming_runtime=Streaming())
    context = ExecutionContext("stream")
    try:
        stream = owner.open_stream(context)
        assert owner.step(stream, ModelRequest(), context).outputs == {"count": 1}
        owner.reset_stream(stream, context)
        owner.reset(context)
        assert policy.calls == ["open_stream", "step", "reset_stream", "reset"]
        assert owner.step(stream, ModelRequest(), context).outputs == {"count": 1}
        for operation, args in (
            ("open_stream", (context,)),
            ("step", (stream, ModelRequest(), context)),
            ("reset_stream", (stream, context)),
            ("close_stream", (stream, context)),
        ):
            with pytest.raises(ExecutionFailure, match="authorized local dispatch"):
                getattr(owner, f"_{operation}_local")(*args)
        owner.close_stream(stream, context)
        assert stream.state is StreamState.CLOSED
        assert policy.calls[-1] == "close_stream"
    finally:
        dispose(owner, policy)


def test_stream_open_failure_after_device_entry_keeps_uncertain_runtime_quarantined():
    class BrokenOpen(Streaming):
        def open_stream(self, context):
            raise RuntimeError("device reset failed after entry")

    owner, _, policy = managed_handle(streaming_runtime=BrokenOpen())
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.open_stream(ExecutionContext("open"))
        assert caught.value.evidence.state is OutcomeState.STARTED
        assert not caught.value.evidence.outcome_known
        assert owner.state is LifecycleState.RESET_REQUIRED
        assert not owner.ready
        assert policy.calls == ["open_stream"]
    finally:
        dispose(owner, policy)


def test_policy_token_remains_supported_by_public_execute_dispatch_key():
    owner = ModelRuntimeHandle(executor=SimpleNamespace(execute=lambda *_: {"value": 3}))
    owner.load()
    policy = LoopbackPolicy(owner)
    authority = policy.authority = owner.bind_execution_policy(policy)
    try:
        assert owner.execute(ModelRequest(), ExecutionContext("business")).outputs == {"value": 3}
        assert owner.execute(ModelRequest(), ExecutionContext("controlled"), _dispatch_key=authority).outputs == {
            "value": 3
        }
        assert policy.calls == ["execute"]
        with pytest.raises(RuntimeError, match="immutable"):
            owner.bind_execution_policy(policy)
    finally:
        dispose(owner, policy)


@pytest.mark.parametrize("path", ["execute", "normalized_execute", "frame", "frame_submit"])
@pytest.mark.parametrize("rejection", [True, False])
def test_not_started_admission_does_not_degrade_model_health(path, rejection):
    from concurrent.futures import Future
    from types import SimpleNamespace

    from inference_service.backends.errors import BackendInferenceError

    error = BackendInferenceError(
        "request rejected" if rejection else "outcome unknown",
        operation_started=not rejection,
        outcome_known=rejection,
    )
    error.admission_rejected = rejection
    error.state_mutated = False

    def submit_frame(*args, **kwargs):
        if path == "frame_submit":
            raise error
        future = Future()
        future.set_exception(error)
        return future

    def execute(*args):
        if path == "normalized_execute":
            raise handle.failure_factory.from_exception(error)
        raise error

    handle = ModelRuntimeHandle(SimpleNamespace(execute=execute, submit_frame=submit_frame))
    handle.load()
    try:
        with pytest.raises(ExecutionFailure) as caught:
            if path in {"execute", "normalized_execute"}:
                handle.execute(ModelRequest(), ExecutionContext("request"))
            else:
                handle.submit_frame(ModelRequest(), ExecutionContext("request")).result(timeout=1)
        assert handle.diagnostics().health.failure_count == (0 if rejection else 1)
        assert (handle.diagnostics().last_failure is None) == rejection
        assert (caught.value.evidence.state is OutcomeState.NOT_STARTED) == rejection
        assert handle.ready is rejection
        if not rejection:
            with pytest.raises(ExecutionFailure, match="not_ready|recovery_required"):
                handle.execute(ModelRequest(), ExecutionContext("after-unknown"))
    finally:
        handle.close()


@pytest.mark.parametrize("operation", ["execute", "submit_frame", "step", "open_stream"])
@pytest.mark.parametrize("stop", ["deadline", "cancel"])
def test_policy_delivery_rechecks_context_after_local_completion(operation, stop):
    from datetime import datetime, timedelta, timezone

    now = [datetime(2026, 10, 30, tzinfo=timezone.utc)]
    context = ExecutionContext("delivery", Deadline.after(1, now=now[0], clock=lambda: now[0]))
    owner, _, policy = managed_handle(streaming_runtime=Streaming() if operation in {"step", "open_stream"} else None)
    stream = owner.open_stream(ExecutionContext("open")) if operation == "step" else None
    local = getattr(policy, operation) if operation in {"step", "open_stream"} else policy.execute

    def delayed(*args):
        value = local(*args)
        if stop == "deadline":
            now[0] += timedelta(seconds=2)
        else:
            context.cancellation_token.cancel("canceled before delivery")
        if operation == "submit_frame":
            future = Future()
            future.set_result(value)
            return future
        return value

    setattr(policy, operation, delayed)
    try:
        with pytest.raises(ExecutionFailure) as failure:
            if operation == "step":
                owner.step(stream, ModelRequest(), context)
            elif operation == "open_stream":
                owner.open_stream(context)
            elif operation == "submit_frame":
                owner.submit_frame(ModelRequest({"value": 3}), context).result(timeout=1)
            else:
                owner.execute(ModelRequest({"value": 3}), context)
        assert failure.value.code == ("deadline_exceeded" if stop == "deadline" else "request_canceled")
        assert failure.value.evidence.state is OutcomeState.COMPLETED
        assert failure.value.evidence.outcome_known
        assert owner.local_ready
        if operation == "open_stream":
            assert failure.value.close_pending and failure.value.details["stream_id"]
        else:
            assert owner.health.failure_count == 1
        if stream is not None:
            assert stream.state is StreamState.OPEN
    finally:
        dispose(owner, policy)


def test_policy_cancel_reaches_request_before_local_admission():
    owner, executor, policy = managed_handle()
    queued, release = threading.Event(), threading.Event()
    context = ExecutionContext("queued")
    local_execute = policy.execute

    def execute(request, context):
        queued.set()
        assert release.wait(2)
        return local_execute(request, context)

    policy.execute = execute
    policy.cancel = lambda request_id, **_: (
        context.cancellation_token.cancel() if request_id == context.request_id else None
    )
    try:
        with ThreadPoolExecutor(1) as callers:
            work = callers.submit(owner.execute, ModelRequest({"value": 1}), context)
            try:
                assert queued.wait(1)
                owner.cancel("queued")
                assert context.cancellation_token.is_cancelled
            finally:
                release.set()
            with pytest.raises(ExecutionFailure) as failure:
                work.result(1)
        assert failure.value.evidence.state is OutcomeState.NOT_STARTED
        assert executor.events == []
    finally:
        release.set()
        dispose(owner, policy)


@pytest.mark.parametrize("normalized", [False, True])
def test_stream_policy_admission_rejection_does_not_degrade_health(normalized):
    from inference_service.backends.errors import BackendAdmissionError

    owner, _, policy = managed_handle(streaming_runtime=Streaming())
    stream = owner.open_stream(ExecutionContext("open"))

    def rejected(*_):
        error = BackendAdmissionError("no capacity")
        raise owner.failure_factory.from_exception(error) if normalized else error

    policy.step = rejected
    try:
        with pytest.raises(ExecutionFailure) as failure:
            owner.step(stream, ModelRequest(), ExecutionContext("rejected"))
        assert failure.value.evidence.state is OutcomeState.NOT_STARTED
        assert owner.health.failure_count == 0 and owner.last_failure is None
        assert stream.state is StreamState.OPEN
        assert owner.stream_diagnostics(stream).last_failure_code is None
    finally:
        dispose(owner, policy)


@pytest.mark.parametrize("bank", ["per_stream", "runtime_exclusive"])
@pytest.mark.parametrize("managed", [False, True])
def test_known_stream_cleanup_failure_blocks_reentry_in_its_own_scope(bank, managed):
    calls = []

    class PendingStreaming(Streaming):
        def step(self, stream, request, context):
            calls.append(context.request_id)
            error = RuntimeError("device output complete; buffer release pending")
            error.operation_started = error.outcome_known = error.close_pending = True
            error.state_mutated = True
            raise error

    owner = ModelRuntimeHandle(
        Executor(),
        streaming_runtime=PendingStreaming(),
        state_scope="stream",
        stateful=True,
        resettable=True,
        state_bank_mode=bank,
        max_open_streams=1,
    )
    owner.load()
    policy = LoopbackPolicy(owner) if managed else None
    if policy is not None:
        policy.authority = owner.bind_execution_policy(policy)
    try:
        stream = owner.open_stream(ExecutionContext("open"))
        with pytest.raises(ExecutionFailure) as failure:
            owner.step(stream, ModelRequest(), ExecutionContext("first"))
        assert failure.value.close_pending and failure.value.evidence.outcome_known
        assert failure.value.recovery.scope.value == ("stream" if bank == "per_stream" else "runtime")
        assert stream.state is StreamState.RESET_REQUIRED
        assert owner.local_ready is (bank == "per_stream")
        with pytest.raises(ExecutionFailure):
            owner.step(stream, ModelRequest(), ExecutionContext("second"))
        assert calls == ["first"]
    finally:
        dispose(owner, policy) if policy else owner.close()


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("local_closed", [False, True])
def test_policy_close_failure_keeps_parent_manager_ownership(normalized, local_closed):
    from inference_service.pipeline.errors import PipelineManagerError
    from inference_service.pipeline.manager import InferencePipelineManager

    owner, executor, policy = managed_handle()
    local_close = policy.close
    attempts = []

    def close(context):
        attempts.append(True)
        if len(attempts) == 1:
            if local_closed:
                local_close(context)
            if normalized:
                raise ExecutionFailure("close_failed", "close receipt missing", evidence=OutcomeEvidence.not_started())
            raise RuntimeError("close receipt missing")
        return local_close(context)

    policy.close = close
    manager = InferencePipelineManager((SimpleNamespace(pipeline_id="policy", load=lambda: None, close=owner.close),))
    manager.start()
    try:
        with pytest.raises(PipelineManagerError) as failure:
            manager.close()
        assert failure.value.close_pending and owner.closed is local_closed
        manager.close()
        manager.close()
        assert owner.closed and len(attempts) == 2
        assert [event for event, _ in executor.events] == ["close"]
    finally:
        dispose(owner, policy)


@pytest.mark.parametrize("stop", ["deadline", "cancel"])
def test_frame_admission_raises_normalized_not_started_failure(stop):
    calls = []
    owner = ModelRuntimeHandle(SimpleNamespace(submit_frame=lambda *_args, **_kwargs: calls.append("submitted")))
    owner.load()
    context = ExecutionContext("rejected", Deadline.after(0) if stop == "deadline" else Deadline.unbounded())
    if stop == "cancel":
        context.cancellation_token.cancel()
    try:
        with pytest.raises(ExecutionFailure) as failure:
            owner.submit_frame(ModelRequest(), context)
        assert failure.value.evidence.state is OutcomeState.NOT_STARTED
        assert failure.value.evidence.outcome_known
        assert calls == [] and owner.active_executions == 0 and owner.ready
    finally:
        owner.close()


@pytest.mark.parametrize("operation", ["execute", "submit_frame"])
def test_rejected_policy_delivery_does_not_keep_payloads_in_diagnostics(operation):
    import gc
    import weakref

    class Payload:
        pass

    references = []
    stop = [True]

    class Policy:
        ready = True

        def execute(self, request, context):
            result = Payload()
            references.append(weakref.ref(result))
            if stop[0]:
                context.token.cancel()
            return result

        def submit_frame(self, request, context):
            future = Future()
            future.set_result(self.execute(request, context))
            return future

        def cancel(self, *_):
            pass

        def reset(self, *_):
            pass

        def close(self, context):
            owner._close_local(context, _dispatch_key=authority)

    owner = ModelRuntimeHandle(SimpleNamespace())
    owner.load()
    authority = owner.bind_execution_policy(Policy())

    def rejected_call():
        payload = Payload()
        references.append(weakref.ref(payload))
        with pytest.raises(ExecutionFailure) as caught:
            result = getattr(owner, operation)(ModelRequest({"payload": payload}), ExecutionContext("rejected"))
            if operation == "submit_frame":
                result.result()
        assert caught.value.evidence.state is OutcomeState.COMPLETED

    try:
        rejected_call()
        stop[0] = False
        getattr(owner, operation)(ModelRequest(), ExecutionContext("success"))
        gc.collect()
        assert references[0]() is None and references[1]() is None
        assert owner.ready and owner.last_failure.code == "request_canceled"
    finally:
        owner.close()


@pytest.mark.parametrize("identity", ["request_id", "token"])
def test_context_provider_cannot_replace_request_identity(identity):
    def provider(context):
        return ExecutionContext(
            "replacement" if identity == "request_id" else context.request_id,
            context.deadline,
            cancellation_token=context.token if identity == "request_id" else None,
        )

    owner, executor, policy = managed_handle(context_provider=provider)
    try:
        with pytest.raises(ValueError, match="preserve request ID and cancellation token"):
            owner.execute(ModelRequest(), ExecutionContext("caller"))
        assert executor.events == [] and policy.calls == []
    finally:
        owner._policy_context_provider = None
        dispose(owner, policy)


@pytest.mark.parametrize("provided_timeout", [None, 10, 0])
def test_context_provider_can_only_shorten_caller_budget(provided_timeout):
    original = ExecutionContext("budget", Deadline.after(1))
    proposed = Deadline.unbounded() if provided_timeout is None else Deadline.after(provided_timeout)
    owner, executor, policy = managed_handle(context_provider=lambda context: context.with_deadline(proposed))
    try:
        effective = owner._policy_context(original)
        assert effective.token is original.token and effective.request_id == original.request_id
        assert effective.deadline is (proposed if provided_timeout == 0 else original.deadline)
        if provided_timeout == 0:
            with pytest.raises(ExecutionFailure):
                owner.execute(ModelRequest(), original)
        else:
            seen = []
            policy.execute = lambda request, context: seen.append(context) or {}
            owner.execute(ModelRequest(), original)
            assert seen[0].deadline is original.deadline
    finally:
        owner._policy_context_provider = None
        dispose(owner, policy)


def test_direct_frame_rejected_delivery_does_not_retain_payloads():
    import gc
    import weakref

    references = []
    stop = [True]

    class Payload:
        pass

    def submit(request, *, context):
        result = Payload()
        references.append(weakref.ref(result))
        future = Future()
        if stop[0]:
            context.token.cancel()
        future.set_result(result)
        return future

    owner = ModelRuntimeHandle(SimpleNamespace(submit_frame=submit))
    owner.load()

    def reject():
        request = Payload()
        references.append(weakref.ref(request))
        with pytest.raises(ExecutionFailure):
            owner.submit_frame(ModelRequest({"payload": request}), ExecutionContext("rejected")).result()

    try:
        reject()
        stop[0] = False
        owner.submit_frame(ModelRequest(), ExecutionContext("success")).result()
        gc.collect()
        assert references[0]() is None and references[1]() is None
        assert owner.ready and owner.last_failure.code == "request_canceled"
        assert owner.last_failure.__traceback__ is None and owner.last_failure.cause is None
    finally:
        owner.close()


@pytest.mark.parametrize("operation", ["execute", "step"])
@pytest.mark.parametrize("source", ["direct", "policy", "local"])
def test_stateless_raw_failure_requires_completion_proof(operation, source):
    from inference_service.backends import BackendCapabilities, ResourceDomainAdmissions
    from inference_service.unified_runtime import RuntimeAssembly, RuntimeProviders

    provider = ResourceDomainAdmissions()
    capabilities = BackendCapabilities(resource_domain="raw-failure")
    executor, streaming = Executor(), Streaming()
    owner = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=executor,
            streaming_runtime=streaming if operation == "step" else None,
            state_scope="stream" if operation == "step" else "request",
            stateful=False,
            session=SimpleNamespace(name="raw-failure", capabilities=capabilities),
            providers=RuntimeProviders(object(), provider),
        )
    )
    owner.load()
    policy = LoopbackPolicy(owner) if source != "direct" else None
    if policy:
        policy.authority = owner.bind_execution_policy(policy)
    stream = owner.open_stream(ExecutionContext("open")) if operation == "step" else None

    def fail(*_):
        raise RuntimeError("native completion was not reported")

    setattr(policy if source == "policy" else streaming if operation == "step" else executor, operation, fail)
    try:
        args = (stream, ModelRequest()) if operation == "step" else (ModelRequest(),)
        with pytest.raises(ExecutionFailure) as caught:
            getattr(owner, operation)(*args, ExecutionContext("raw"))
        assert caught.value.evidence.state is OutcomeState.STARTED
        if source == "direct" and operation == "execute":
            assert caught.value.evidence.outcome_known and owner.local_ready
            assert not owner._admission.has_retained
            return
        assert not caught.value.evidence.outcome_known
        if operation == "step":
            assert stream.state is not StreamState.OPEN
        else:
            assert not owner.local_ready
        if source != "policy":
            # Managed dispatch owns domain admission externally, but local
            # native execution must still retain its instance token.
            assert not owner._admission._instance_gate.acquire(blocking=False)
        if source == "direct":
            peer = provider.register("raw-failure", 1)
            try:
                with pytest.raises(Exception, match="deadline"):
                    peer.acquire(context=ExecutionContext("peer", Deadline.after(0.02)))
            finally:
                peer.close()
    finally:
        dispose(owner, policy) if policy else owner.close()


@pytest.mark.parametrize("operation", ["reset_stream", "close_stream"])
@pytest.mark.parametrize("dispatch", ["direct", "inline", "worker"])
@pytest.mark.parametrize("bounded", [False, True])
def test_stream_control_reentry_is_rejected_before_dispatch_without_poisoning_stream(operation, dispatch, bounded):
    native = Streaming()
    owner, _, policy = managed_handle(streaming_runtime=native)
    if dispatch == "direct":
        dispose(owner, policy)
        owner = ModelRuntimeHandle(Executor(), streaming_runtime=native, state_scope="stream", stateful=True)
        owner.load()
        policy = None
    elif dispatch == "inline":

        def local(name, *args):
            policy.calls.append(name)
            return getattr(owner, f"_{name}_local")(*args, _dispatch_key=policy.authority)

        policy.local = local
    stream = owner.open_stream(ExecutionContext("open"))
    original_step = native.step
    failures = []

    def step(handle, request, context):
        deadline = Deadline.after(0.05) if bounded else Deadline.unbounded()
        with pytest.raises(ExecutionFailure) as caught:
            getattr(owner, operation)(stream, ExecutionContext("reentry", deadline))
        failures.append(caught.value)
        return original_step(handle, request, context)

    native.step = step
    if policy:
        policy.calls.clear()
    try:
        assert owner.step(stream, ModelRequest(), ExecutionContext("first")).outputs == {"count": 1}
        assert failures[0].code == "stream_reentrant"
        assert failures[0].evidence.state is OutcomeState.NOT_STARTED
        assert owner.local_ready
        assert not stream.closed
        if policy:
            assert policy.calls == ["step"]
        native.step = original_step
        assert owner.step(stream, ModelRequest(), ExecutionContext("next")).outputs == {"count": 2}
    finally:
        dispose(owner, policy) if policy else owner.close()


@pytest.mark.parametrize("operation", ["reset", "close"])
@pytest.mark.parametrize("managed", [False, True])
def test_open_stream_reentry_rejects_runtime_control_before_waiting_for_itself(operation, managed):
    native = Streaming()
    owner = ModelRuntimeHandle(Executor(), streaming_runtime=native, state_scope="stream", resettable=True)
    owner.load()
    policy = LoopbackPolicy(owner) if managed else None
    if policy:
        policy.authority = owner.bind_execution_policy(policy)
    failures = []

    def open_stream(context):
        with pytest.raises(ExecutionFailure) as caught:
            getattr(owner, operation)(ExecutionContext("reentry", Deadline.after(0.05)))
        failures.append(caught.value)
        return {"count": 0}

    native.open_stream = open_stream
    try:
        stream = owner.open_stream(ExecutionContext("open"))
        assert failures[0].code == "runtime_reentrant"
        assert failures[0].evidence.state is OutcomeState.NOT_STARTED
        assert owner.local_ready
        assert owner.step(stream, ModelRequest(), ExecutionContext("next")).outputs == {"count": 1}
        if policy:
            assert policy.calls == ["open_stream", "step"]
    finally:
        dispose(owner, policy) if policy else owner.close()


@pytest.mark.parametrize("source", ["direct", "local"])
@pytest.mark.parametrize("failure_at", ["submit", "future"])
@pytest.mark.parametrize("proof", [None, "not_started", "completed"])
def test_local_frame_failure_requires_native_evidence(source, failure_at, proof):
    error = RuntimeError("frame failed")
    if proof is not None:
        error.evidence = getattr(OutcomeEvidence, proof)("native")

    def submit(request, *, context):
        if failure_at == "submit":
            raise error
        future = Future()
        future.set_exception(error)
        return future

    owner = ModelRuntimeHandle(SimpleNamespace(submit_frame=submit))
    owner.load()
    policy = LoopbackPolicy(owner) if source == "local" else None
    if policy:
        policy.authority = owner.bind_execution_policy(policy)
        policy.submit_frame = lambda request, context: policy.local("submit_frame", request, context)
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.submit_frame(ModelRequest(), ExecutionContext("frame")).result(timeout=1)
        evidence = caught.value.evidence
        assert evidence.outcome_known is (proof is not None)
        assert (
            evidence.state
            is {
                None: OutcomeState.STARTED,
                "not_started": OutcomeState.NOT_STARTED,
                "completed": OutcomeState.COMPLETED,
            }[proof]
        )
        assert owner.local_ready is (proof is not None)
        assert owner.health.failure_count == 1
    finally:
        dispose(owner, policy) if policy else owner.close()
