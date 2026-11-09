"""Stream execution shares native admission and the runtime drain barrier."""

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

from inference_service.backends import BackendCapabilities, ResourceDomainAdmissions
from inference_service.backends.types import BackendAdmissionEvidence
from inference_service.unified_runtime import (
    Deadline,
    ExecutionContext,
    ExecutionFailure,
    LifecycleState,
    ModelRequest,
    ModelRuntimeHandle,
    OutcomeEvidence,
    OutcomeState,
    RuntimeAssembly,
    RuntimeProviders,
)


def stream_owner(native, *, provider=None, limit=2, bank="per_stream", resettable=True, native_limit=1):
    capabilities = BackendCapabilities(
        stateful=True,
        resettable=resettable,
        max_in_flight_per_instance=native_limit,
        thread_safe=native_limit > 1,
        supports_multiple_instances=True,
        resource_domain="stream-device",
        admission_evidence=BackendAdmissionEvidence(
            sdk_initialization=True,
            multi_instance_execution=True,
            failure_isolation=True,
            independent_close=True,
            overlapping_calls=True,
            output_isolation=True,
            deterministic_cleanup=True,
        ),
    )
    owner = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=native,
            streaming_runtime=native,
            session=SimpleNamespace(name="stream-model", capabilities=capabilities),
            providers=RuntimeProviders(object(), provider) if provider is not None else None,
            stateful=True,
            resettable=resettable,
            state_scope="stream",
            state_bank_mode=bank,
            max_open_streams=1 if bank == "runtime_exclusive" else 2,
            max_active_executions=limit,
        )
    )
    owner.load()
    return owner


def stream_runtime(**operations):
    result = SimpleNamespace(
        open_stream=lambda _: object(),
        step=lambda *_: {},
        close_stream=lambda *_: None,
        reset_stream=lambda *_: None,
        close=lambda: None,
    )
    result.__dict__.update(operations)
    return result


@pytest.mark.parametrize("missing", ["open_stream", "step", "reset_stream", "close_stream"])
def test_stream_contract_rejects_missing_control_before_load(missing):
    loaded = []
    runtime = stream_runtime(load=lambda *_: loaded.append(True))
    delattr(runtime, missing)
    with pytest.raises(TypeError, match="streaming runtime requires"):
        stream_owner(runtime)
    assert loaded == []


@pytest.mark.parametrize("native_reset_fails", [False, True])
def test_host_reset_keeps_exclusive_capacity_until_native_reset_finishes(native_reset_fails):
    from inference_service.backends.errors import BackendInferenceError

    def unknown(*_):
        raise BackendInferenceError("completion unknown", operation_started=True, outcome_known=False)

    provider = ResourceDomainAdmissions()
    events = []
    native = stream_runtime(step=unknown, reset_host_stream=lambda *_: events.append("host-reset"))
    owner = stream_owner(native, provider=provider, bank="runtime_exclusive")
    peer = stream_owner(stream_runtime(step=lambda *_: events.append("peer") or {}), provider=provider)
    stream = owner.open_stream(ExecutionContext("open"))
    other = peer.open_stream(ExecutionContext("peer-open"))

    def reset(_):
        assert events == ["host-reset"]
        with pytest.raises(ExecutionFailure) as blocked:
            peer.step(other, ModelRequest(), ExecutionContext("during-reset", Deadline.after(0.02)))
        assert blocked.value.evidence.state is OutcomeState.NOT_STARTED
        events.append("native-reset")
        if native_reset_fails:
            raise RuntimeError("native reset failed")

    owner._assembly.session.reset = reset
    try:
        with pytest.raises(ExecutionFailure):
            owner.step(stream, ModelRequest(), ExecutionContext("unknown"))
        if native_reset_fails:
            with pytest.raises(ExecutionFailure, match="native reset failed"):
                owner.reset()
            with pytest.raises(ExecutionFailure):
                peer.step(other, ModelRequest(), ExecutionContext("after-failure", Deadline.after(0.02)))
            owner.close()
        else:
            owner.reset()
        assert events == ["host-reset", "native-reset"]
        peer.step(other, ModelRequest(), ExecutionContext("released"))
        assert events[-1] == "peer"
    finally:
        owner.close()
        peer.close()


@pytest.mark.parametrize("bank", ["per_stream", "runtime_exclusive"])
@pytest.mark.parametrize("known", [False, True])
@pytest.mark.parametrize("action", ["close_stream", "reload"])
@pytest.mark.parametrize("policy_failure", [False, True])
def test_explicit_stream_recovery_cannot_be_ignored_or_reset(bank, known, action, policy_failure):
    from inference_service.unified_runtime import RecoveryRequirement, StreamState

    def step(*_):
        raise ExecutionFailure(
            "bank_failed",
            "bank requires replacement",
            evidence=OutcomeEvidence.started("device", outcome_known=known, state_mutated=True),
            recovery=RecoveryRequirement("stream", action),
            recoverable=True,
        )

    owner = stream_owner(stream_runtime(step=step), bank=bank)
    stream = owner.open_stream(ExecutionContext("open"))
    if policy_failure:
        policy = SimpleNamespace(ready=True, cancel=lambda *_args, **_kwargs: None)
        for operation in ("execute", "reset", "close", "open_stream", "step", "reset_stream", "close_stream"):
            setattr(
                policy,
                operation,
                lambda *args, name=operation: getattr(owner, f"_{name}_local")(*args, _dispatch_key=authority),
            )
        policy.step = step
        authority = owner.bind_execution_policy(policy)
    try:
        with pytest.raises(ExecutionFailure) as failure:
            owner.step(stream, ModelRequest(), ExecutionContext("failed"))
        # Unknown exclusive work needs runtime recovery as well as a bank
        # replacement; a known result or isolated bank preserves its action.
        if known or bank == "per_stream":
            assert failure.value.recovery == RecoveryRequirement("stream", action)
            assert stream.state is StreamState.FAILED
        else:
            assert failure.value.recovery.scope.value == "runtime"
            assert failure.value.recovery.action.value == "reload"
        with pytest.raises(ExecutionFailure):
            owner.step(stream, ModelRequest(), ExecutionContext("blocked"))
        with pytest.raises(ExecutionFailure):
            owner.reset_stream(stream, ExecutionContext("too-weak"))
        if known or bank == "per_stream":
            with pytest.raises(ExecutionFailure):
                owner.reset()
        owner.close_stream(stream)
        assert stream.closed
    finally:
        owner.close()


@pytest.mark.parametrize("constraint", ["instance", "domain", "handle"])
def test_steps_obey_instance_domain_and_handle_capacity(constraint):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def step(stream, request, context):
        calls.append(context.request_id)
        entered.set()
        assert release.wait(2)
        return {}

    provider = ResourceDomainAdmissions()
    native = stream_runtime(step=step)
    owner = stream_owner(native, provider=provider, limit=1 if constraint == "handle" else 2)
    second_owner = stream_owner(native, provider=provider) if constraint == "domain" else owner
    first = owner.open_stream(ExecutionContext("open-first"))
    second = second_owner.open_stream(ExecutionContext("open-second"))
    try:
        with ThreadPoolExecutor(1) as pool:
            active = pool.submit(owner.step, first, ModelRequest(), ExecutionContext("first"))
            try:
                assert entered.wait(1)
                with pytest.raises(ExecutionFailure) as rejected:
                    second_owner.step(second, ModelRequest(), ExecutionContext("second", Deadline.after(0.02)))
                assert rejected.value.evidence.state is OutcomeState.NOT_STARTED
                assert calls == ["first"]
            finally:
                release.set()
            active.result(1)
        assert second_owner.step(second, ModelRequest(), ExecutionContext("retry")).evidence.outcome_known
    finally:
        release.set()
        owner.close()
        if second_owner is not owner:
            second_owner.close()


@pytest.mark.parametrize("operation", ["reset_stream", "close_stream"])
@pytest.mark.parametrize("drain_first", [False, True])
@pytest.mark.parametrize("bank", ["per_stream", "runtime_exclusive"])
def test_managed_drain_and_stream_controls_cannot_overlap(operation, drain_first, bank, monkeypatch):
    from contextlib import contextmanager

    entered, release, other_entered = (threading.Event() for _ in range(3))
    attempted = threading.Event()

    def blocked(*_):
        entered.set()
        assert release.wait(2)

    def following(*_):
        other_entered.set()

    def native_drain():
        assert not owner.local_ready
        (blocked if drain_first else following)()

    native = stream_runtime(
        **{
            operation: following if drain_first else blocked,
            "drain_uncertain_operations": native_drain,
        }
    )
    owner = stream_owner(native, bank=bank)
    owner._assembly.session.drain_uncertain_operations = native_drain
    stream = owner.open_stream(ExecutionContext("open"))
    policy = SimpleNamespace(ready=True, cancel=lambda *_args, **_kwargs: None)
    for name in ("execute", "reset", "close", "open_stream", "step", "reset_stream", "close_stream"):
        setattr(policy, name, lambda *args, name=name: getattr(owner, f"_{name}_local")(*args, _dispatch_key=authority))
    authority = owner.bind_execution_policy(policy)
    guard = owner._control_guard

    @contextmanager
    def observed_guard(*args, **kwargs):
        if entered.is_set():
            attempted.set()
        with guard(*args, **kwargs):
            yield

    monkeypatch.setattr(owner, "_control_guard", observed_guard)

    def drain():
        owner.drain_managed(ExecutionContext("drain"), _dispatch_key=authority)

    def control():
        getattr(owner, operation)(stream, ExecutionContext("control"))

    try:
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(drain if drain_first else control)
            try:
                assert entered.wait(1)
                second = pool.submit(control if drain_first else drain)
                assert attempted.wait(1)
                assert not other_entered.wait(0.05)
            finally:
                release.set()
            first.result(1)
            if drain_first and operation == "reset_stream":
                with pytest.raises(ExecutionFailure, match="recovery_required"):
                    second.result(1)
                assert not other_entered.is_set()
            else:
                second.result(1)
                assert other_entered.is_set()
    finally:
        release.set()
        owner.close()


@pytest.mark.parametrize(
    "recovery,resettable",
    [
        ("close", False),
        ("close", True),
        ("reset_stream", True),
        ("close_stream", False),
        ("close_stream", True),
    ],
)
def test_unknown_work_keeps_shared_capacity_until_owner_recovers(recovery, resettable):
    from inference_service.backends.errors import BackendInferenceError, BackendLifecycleError

    provider = ResourceDomainAdmissions()
    blocked = [True]
    calls = []

    def unknown(*_):
        raise BackendInferenceError("sync failed", operation_started=True, outcome_known=False)

    def recover(*_):
        if blocked[0]:
            raise BackendLifecycleError("still active", close_pending=True)

    owner = stream_owner(
        stream_runtime(step=unknown, reset_stream=recover, close_stream=recover, close=recover),
        provider=provider,
        resettable=resettable,
    )
    peer = stream_owner(stream_runtime(step=lambda *_: calls.append("peer") or {}), provider=provider)
    first = owner.open_stream(ExecutionContext("open"))
    second = peer.open_stream(ExecutionContext("peer-open"))
    try:
        with pytest.raises(ExecutionFailure):
            owner.step(first, ModelRequest(), ExecutionContext("unknown"))
        for attempt in range(2):
            with pytest.raises(ExecutionFailure) as pending:
                peer.step(second, ModelRequest(), ExecutionContext(f"peer-{attempt}", Deadline.after(0.02)))
            assert pending.value.evidence.state is OutcomeState.NOT_STARTED and calls == []
            if attempt == 0 and recovery == "close":
                with pytest.raises(ExecutionFailure):
                    owner.close()
            elif attempt == 0 and recovery == "close_stream":
                with pytest.raises(ExecutionFailure):
                    owner.close_stream(first)
        blocked[0] = False
        if recovery == "close":
            owner.close()
        elif recovery == "close_stream":
            owner.close_stream(first)
            assert first.closed and owner.ready
        else:
            owner.reset_stream(first, ExecutionContext("reset"))
            assert owner.ready
        assert peer.step(second, ModelRequest(), ExecutionContext("released")).evidence.outcome_known
        assert calls == ["peer"]
    finally:
        blocked[0] = False
        owner.close()
        peer.close()


@pytest.mark.parametrize("entry", ["execute", "step"])
@pytest.mark.parametrize("explicit", [False, True])
def test_admission_uses_the_same_evidence_as_public_execution(entry, explicit):
    provider = ResourceDomainAdmissions()
    execution = ExecutionContext("failure")
    evidence = OutcomeEvidence.completed("native", state_mutated=True)

    def fail(*_):
        if explicit:
            error = RuntimeError("native output invalid")
            error.evidence = evidence
            raise error
        execution.token.cancel("after native submit")
        execution.check("device_wait")

    native = stream_runtime(step=fail, execute=fail)
    owner = stream_owner(native, provider=provider)
    if entry == "execute":
        # The same resource capabilities, through the public request entry.
        owner.close()
        session = SimpleNamespace(
            name="request-model", capabilities=replace(owner._assembly.session.capabilities, stateful=False)
        )
        owner = ModelRuntimeHandle(
            RuntimeAssembly(
                runtime_executor=native,
                session=session,
                providers=RuntimeProviders(object(), provider),
                stateful=False,
                resettable=True,
            )
        )
        owner.load()
    peer = stream_owner(stream_runtime(), provider=provider)
    stream = owner.open_stream(ExecutionContext("open")) if entry == "step" else None
    other = peer.open_stream(ExecutionContext("peer-open"))
    try:
        with pytest.raises(ExecutionFailure) as caught:
            if stream is None:
                owner.execute(ModelRequest(), execution)
            else:
                owner.step(stream, ModelRequest(), execution)
        if explicit:
            assert caught.value.evidence == evidence
            assert peer.step(other, ModelRequest(), ExecutionContext("peer")).evidence.outcome_known
        else:
            assert caught.value.evidence.state is OutcomeState.STARTED
            assert not caught.value.evidence.outcome_known
            with pytest.raises(ExecutionFailure) as blocked:
                peer.step(other, ModelRequest(), ExecutionContext("peer", Deadline.after(0.02)))
            assert blocked.value.evidence.state is OutcomeState.NOT_STARTED
    finally:
        owner.close()
        peer.close()


@pytest.mark.parametrize("recovery", ["close", "reset"])
@pytest.mark.parametrize("gate", ["instance", "domain"])
def test_control_drains_its_own_native_admission_waiters(recovery, gate, monkeypatch):
    from contextlib import contextmanager

    from inference_service.backends.errors import BackendInferenceError

    entered, finish, waiting = (threading.Event() for _ in range(3))
    calls = []

    def step(*_):
        calls.append("native")
        entered.set()
        assert finish.wait(2)
        raise BackendInferenceError("unknown native work", operation_started=True, outcome_known=False)

    owner = stream_owner(
        stream_runtime(step=step), provider=ResourceDomainAdmissions(), native_limit=2 if gate == "domain" else 1
    )
    first = owner.open_stream(ExecutionContext("open-first"))
    second = owner.open_stream(ExecutionContext("open-second"))
    context = ExecutionContext("queued")
    admit = owner._admission.admit

    @contextmanager
    def observe(*args, **kwargs):
        if kwargs["context"] is context:
            waiting.set()
        with admit(*args, **kwargs):
            yield

    monkeypatch.setattr(owner._admission, "admit", observe)
    try:
        with ThreadPoolExecutor(2) as pool:
            active = pool.submit(owner.step, first, ModelRequest(), ExecutionContext("first"))
            try:
                assert entered.wait(1)
                queued = pool.submit(owner.step, second, ModelRequest(), context)
                assert waiting.wait(1)
                finish.set()
                with pytest.raises(ExecutionFailure):
                    active.result(1)
                getattr(owner, recovery)(ExecutionContext("control", Deadline.after(1)))
                with pytest.raises(ExecutionFailure) as stopped:
                    queued.result(1)
                assert stopped.value.evidence.state is OutcomeState.NOT_STARTED
                assert calls == ["native"]
                assert owner.closed if recovery == "close" else owner.ready
            finally:
                finish.set()
                context.token.cancel("test cleanup")
    finally:
        owner.close()


@pytest.mark.parametrize("closing", [False, True])
@pytest.mark.parametrize("action", ["reset_runtime", "reload"])
@pytest.mark.parametrize("scope", ["runtime", "request"])
def test_per_stream_failure_preserves_explicit_runtime_recovery(closing, action, scope):
    from inference_service.unified_runtime import RecoveryAction, RecoveryRequirement, RecoveryScope

    def step(*_):
        if closing:
            owner._streams[stream.stream_id].closing = True
        raise ExecutionFailure(
            "runtime_fault",
            "shared executor failed",
            evidence=OutcomeEvidence.started("device", outcome_known=False),
            recovery=RecoveryRequirement(RecoveryScope(scope), RecoveryAction(action)),
            recoverable=True,
        )

    provider = ResourceDomainAdmissions()
    owner = stream_owner(stream_runtime(step=step), provider=provider)
    peer = stream_owner(stream_runtime(), provider=provider)
    stream = owner.open_stream(ExecutionContext("open"))
    other = peer.open_stream(ExecutionContext("peer-open"))
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.step(stream, ModelRequest(), ExecutionContext("failed"))
        assert caught.value.recovery.scope is RecoveryScope.RUNTIME
        assert caught.value.recovery.action is RecoveryAction(action)
        assert not owner.ready
        owner.close_stream(stream)
        with pytest.raises(ExecutionFailure):
            peer.step(other, ModelRequest(), ExecutionContext("still-blocked", Deadline.after(0.02)))
        owner.close()
        peer.step(other, ModelRequest(), ExecutionContext("released"))
    finally:
        owner.close()
        peer.close()


def test_logical_exclusive_stream_close_does_not_prove_native_completion():
    from inference_service.backends.errors import BackendInferenceError

    def unknown(*_):
        raise BackendInferenceError("sync failed", operation_started=True, outcome_known=False)

    provider = ResourceDomainAdmissions()
    owner = stream_owner(stream_runtime(step=unknown), provider=provider, bank="runtime_exclusive")
    peer = stream_owner(stream_runtime(), provider=provider)
    stream = owner.open_stream(ExecutionContext("open"))
    other = peer.open_stream(ExecutionContext("peer-open"))
    try:
        with pytest.raises(ExecutionFailure):
            owner.step(stream, ModelRequest(), ExecutionContext("unknown"))
        owner.close_stream(stream)
        with pytest.raises(ExecutionFailure, match="requires a native reset"):
            owner.reset()
        assert not owner.ready
        with pytest.raises(ExecutionFailure):
            peer.step(other, ModelRequest(), ExecutionContext("blocked", Deadline.after(0.02)))
        owner.close()
        peer.step(other, ModelRequest(), ExecutionContext("released"))
    finally:
        owner.close()
        peer.close()


@pytest.mark.parametrize("closing", [False, True])
@pytest.mark.parametrize("provider_enabled", [False, True])
def test_exclusive_unknown_requires_runtime_recovery_even_while_closing(closing, provider_enabled, monkeypatch):
    from inference_service.backends.errors import BackendInferenceError
    from inference_service.unified_runtime import RecoveryAction, RecoveryRequirement, RecoveryScope

    entered, release, waiting = (threading.Event() for _ in range(3))

    def fail(*_):
        entered.set()
        assert release.wait(2)
        if closing:
            raise BackendInferenceError("completion unknown", operation_started=True, outcome_known=False)
        raise ExecutionFailure(
            "device_failed",
            "runtime reset required",
            evidence=OutcomeEvidence.started(outcome_known=False),
            recovery=RecoveryRequirement(RecoveryScope.RUNTIME, RecoveryAction.RESET_RUNTIME),
            recoverable=True,
        )

    provider = ResourceDomainAdmissions() if provider_enabled else None
    resets = []
    native = stream_runtime(
        step=fail, reset=lambda *_: resets.append("runtime"), reset_stream=lambda *_: resets.append("stream")
    )
    owner = stream_owner(native, provider=provider, bank="runtime_exclusive")
    stream = owner.open_stream(ExecutionContext("open"))
    original_wait = owner._wait_for_stream_quiescence_locked

    def wait(record, context):
        waiting.set()
        original_wait(record, context)

    monkeypatch.setattr(owner, "_wait_for_stream_quiescence_locked", wait)
    try:
        with ThreadPoolExecutor(2) as pool:
            step = pool.submit(owner.step, stream, ModelRequest(), ExecutionContext("fail"))
            assert entered.wait(1)
            close = pool.submit(owner.close_stream, stream) if closing else None
            if closing:
                assert waiting.wait(1)
            release.set()
            with pytest.raises(ExecutionFailure) as caught:
                step.result(1)
            if close is not None:
                close.result(1)
        assert caught.value.recovery.scope is RecoveryScope.RUNTIME
        assert caught.value.recovery.action is RecoveryAction.RESET_RUNTIME
        assert owner.state is LifecycleState.RESET_REQUIRED and not owner.ready
        if not closing:
            with pytest.raises(ExecutionFailure, match="recovery_required"):
                owner.reset_stream(stream, ExecutionContext("insufficient-recovery"))
            assert owner.state is LifecycleState.RESET_REQUIRED
        else:
            with pytest.raises(ExecutionFailure):
                owner.open_stream(ExecutionContext("blocked"))
        native.step = lambda *_: {}
        owner.reset()
        assert resets == (["runtime"] if closing else ["stream"])
        if closing:
            stream = owner.open_stream(ExecutionContext("reopen"))
        assert owner.step(
            stream, ModelRequest(), ExecutionContext("recovered", Deadline.after(0.1))
        ).evidence.outcome_known
    finally:
        release.set()
        owner.close()


def test_known_retry_request_failure_does_not_reset_a_healthy_stream():
    from inference_service.unified_runtime import OutcomeEvidence, RecoveryAction, RecoveryRequirement, RecoveryScope

    failure = ExecutionFailure(
        "retryable",
        "retry this request",
        evidence=OutcomeEvidence.not_started("backend"),
        recovery=RecoveryRequirement(RecoveryScope.REQUEST, RecoveryAction.RETRY_REQUEST),
        recoverable=True,
    )
    native = stream_runtime(step=lambda *_: failure)
    owner = stream_owner(native)
    stream = owner.open_stream(ExecutionContext("open"))
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.step(stream, ModelRequest(), ExecutionContext("retryable"))
        assert caught.value.recovery.action is RecoveryAction.RETRY_REQUEST
        native.step = lambda *_: {}
        assert owner.step(stream, ModelRequest(), ExecutionContext("next")).evidence.outcome_known
    finally:
        owner.close()


def test_stream_reset_waiting_for_step_rejects_the_next_input(monkeypatch):
    entered, finish_step, reset_waiting, finish_wait = (threading.Event() for _ in range(4))
    calls = []

    def step(*_):
        calls.append("step")
        entered.set()
        assert finish_step.wait(2)
        return {}

    owner = stream_owner(stream_runtime(step=step, reset_stream=lambda *_: calls.append("reset")))
    stream = owner.open_stream(ExecutionContext("open"))
    wait = owner._wait_for_stream_quiescence_locked

    def paused_wait(record, context):
        reset_waiting.set()
        while not finish_wait.is_set():
            owner._condition.wait(0.01)
        wait(record, context)

    monkeypatch.setattr(owner, "_wait_for_stream_quiescence_locked", paused_wait)
    try:
        with ThreadPoolExecutor(2) as pool:
            active = pool.submit(owner.step, stream, ModelRequest(), ExecutionContext("first"))
            try:
                assert entered.wait(1)
                reset = pool.submit(owner.reset_stream, stream, ExecutionContext("reset"))
                assert reset_waiting.wait(1)
                finish_step.set()
                active.result(1)
                with pytest.raises(ExecutionFailure, match="control operation"):
                    owner.step(stream, ModelRequest(), ExecutionContext("second"))
                assert calls == ["step"]
            finally:
                finish_step.set()
                finish_wait.set()
            reset.result(1)
        assert calls == ["step", "reset"]
        owner.step(stream, ModelRequest(), ExecutionContext("after-reset"))
    finally:
        finish_step.set()
        finish_wait.set()
        owner.close()


@pytest.mark.parametrize("with_evidence", [False, True])
def test_close_stream_reports_close_phase_and_preserves_native_evidence(with_evidence):
    original = OutcomeEvidence.started("device_release", outcome_known=False, state_mutated=True)

    def close(*_):
        if with_evidence:
            raise ExecutionFailure("device_pending", "pending", evidence=original)
        raise RuntimeError("pending")

    native = stream_runtime(close_stream=close)
    owner = stream_owner(native)
    stream = owner.open_stream(ExecutionContext("open"))
    try:
        with pytest.raises(ExecutionFailure) as failure:
            owner.close_stream(stream)
        assert failure.value.evidence.phase == ("device_release" if with_evidence else "close_stream")
        assert failure.value.close_pending and not stream.closed
    finally:
        native.close_stream = lambda *_: None
        owner.close()


@pytest.mark.parametrize("bank", ["per_stream", "runtime_exclusive"])
def test_reset_from_active_step_does_not_wait_for_close_owner(bank):
    entered, release = threading.Event(), threading.Event()

    def step(*_):
        entered.set()
        assert release.wait(2)
        with pytest.raises(ExecutionFailure, match="cannot reset a stream from its active step"):
            owner.reset_stream(stream, ExecutionContext("self-reset", Deadline.after(0.1)))
        return {}

    owner = stream_owner(stream_runtime(step=step), bank=bank)
    stream = owner.open_stream(ExecutionContext("open"))
    try:
        with ThreadPoolExecutor(2) as pool:
            active = pool.submit(owner.step, stream, ModelRequest(), ExecutionContext("step"))
            try:
                assert entered.wait(1)
                closing = pool.submit(owner.close)
                with owner._condition:
                    assert owner._condition.wait_for(lambda: owner._state is LifecycleState.CLOSING, timeout=1)
            finally:
                release.set()
            active.result(1)
            closing.result(1)
        assert owner.closed
    finally:
        release.set()
        owner.close()


@pytest.mark.parametrize("bank", ["per_stream", "runtime_exclusive"])
@pytest.mark.parametrize("known", [False, True])
@pytest.mark.parametrize("action", ["close_stream", "reload"])
def test_waiting_reset_preserves_new_bank_recovery(monkeypatch, bank, known, action):
    from inference_service.unified_runtime import RecoveryRequirement, StreamState

    entered, release, waiting = (threading.Event() for _ in range(3))
    resets = []

    def step(*_):
        entered.set()
        assert release.wait(2)
        raise ExecutionFailure(
            "bank_failed",
            "replace this bank",
            evidence=OutcomeEvidence.started(outcome_known=known, state_mutated=True),
            recovery=RecoveryRequirement("stream", action),
        )

    provider = ResourceDomainAdmissions()
    owner = stream_owner(
        stream_runtime(step=step, reset_stream=lambda *_: resets.append(True)), bank=bank, provider=provider
    )
    peer = stream_owner(stream_runtime(), provider=provider)
    stream = owner.open_stream(ExecutionContext("open"))
    other = peer.open_stream(ExecutionContext("peer-open"))
    wait = owner._wait_for_stream_quiescence_locked

    def observe(record, context):
        waiting.set()
        wait(record, context)

    monkeypatch.setattr(owner, "_wait_for_stream_quiescence_locked", observe)
    try:
        with ThreadPoolExecutor(2) as pool:
            active = pool.submit(owner.step, stream, ModelRequest(), ExecutionContext("step"))
            try:
                assert entered.wait(1)
                reset = pool.submit(owner.reset_stream, stream, ExecutionContext("reset", Deadline.after(1)))
                assert waiting.wait(1)
            finally:
                release.set()
            with pytest.raises(ExecutionFailure):
                active.result(1)
            with pytest.raises(ExecutionFailure):
                reset.result(1)
        record = owner._streams[stream.stream_id]
        assert stream.state is StreamState.FAILED and not record.resetting and resets == []
        assert record.recovery_requirement.action.value == (action if known or bank == "per_stream" else "reload")
        if not known:
            with pytest.raises(ExecutionFailure):
                peer.step(other, ModelRequest(), ExecutionContext("quarantined", Deadline.after(0.02)))
        owner.close()
        peer.step(other, ModelRequest(), ExecutionContext("released"))
    finally:
        release.set()
        owner.close()
        peer.close()


@pytest.mark.parametrize("operation", ["reset_stream", "close_stream"])
@pytest.mark.parametrize("action", ["reload", "reset_runtime"])
@pytest.mark.parametrize("policy_bound", [False, True])
@pytest.mark.parametrize("bank", ["per_stream", "runtime_exclusive"])
def test_local_stream_control_applies_runtime_recovery(operation, action, policy_bound, bank):
    from inference_service.unified_runtime import RecoveryRequirement

    def fail(*_):
        raise ExecutionFailure(
            "control_failed",
            "runtime recovery required",
            evidence=OutcomeEvidence.started("control", outcome_known=False, state_mutated=True),
            recovery=RecoveryRequirement("runtime", action),
        )

    native = stream_runtime(**{operation: fail})
    owner = stream_owner(native, bank=bank)
    stream = owner.open_stream(ExecutionContext("open"))
    other = owner.open_stream(ExecutionContext("other")) if bank == "per_stream" else stream
    if policy_bound:
        policy = SimpleNamespace(ready=True, cancel=lambda *_args, **_kwargs: None)
        for name in ("execute", "reset", "close", "open_stream", "step", "reset_stream", "close_stream"):
            setattr(
                policy, name, lambda *args, name=name: getattr(owner, f"_{name}_local")(*args, _dispatch_key=authority)
            )
        authority = owner.bind_execution_policy(policy)
    try:
        with pytest.raises(ExecutionFailure) as caught:
            getattr(owner, operation)(stream, ExecutionContext("control"))
        assert caught.value.recovery == RecoveryRequirement("runtime", action)
        assert owner.state is (LifecycleState.FAILED if action == "reload" else LifecycleState.RESET_REQUIRED)
        assert not owner.ready and not owner._admission_open
        assert owner._streams[stream.stream_id].recovery_requirement == caught.value.recovery
        with pytest.raises(ExecutionFailure):
            owner.step(other, ModelRequest(), ExecutionContext("must-not-run"))
    finally:
        setattr(native, operation, lambda *_: None)
        owner.close()


@pytest.mark.parametrize("known", [False, True])
def test_unclassified_reset_failure_keeps_stream_failed(known):
    from inference_service.unified_runtime import StreamState

    def fail(*_):
        raise ExecutionFailure(
            "reset_failed",
            "reset did not finish",
            evidence=(OutcomeEvidence.not_started() if known else OutcomeEvidence.started(outcome_known=False)),
        )

    owner = stream_owner(stream_runtime(reset_stream=fail))
    stream = owner.open_stream(ExecutionContext("open"))
    try:
        with pytest.raises(ExecutionFailure):
            owner.reset_stream(stream, ExecutionContext("reset"))
        assert stream.state is StreamState.FAILED
        assert not owner._streams[stream.stream_id].resetting
        with pytest.raises(ExecutionFailure):
            owner.step(stream, ModelRequest(), ExecutionContext("rejected"))
    finally:
        owner.close()


@pytest.mark.parametrize("reset_mode", ["unsupported", "success", "failure"])
def test_closed_exclusive_bank_requires_native_barrier_with_aliased_session(reset_mode):
    from inference_service.backends.errors import BackendInferenceError

    def unknown(*_):
        raise BackendInferenceError("native outcome unknown", operation_started=True, outcome_known=False)

    domains = ResourceDomainAdmissions()
    native = stream_runtime(step=unknown)
    native.name = "exclusive"
    native.capabilities = BackendCapabilities(stateful=True, resettable=True, resource_domain="closed-bank")
    owner = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=native,
            streaming_runtime=native,
            session=native,
            providers=RuntimeProviders(object(), domains),
            state_scope="stream",
            stateful=True,
            resettable=True,
            state_bank_mode="runtime_exclusive",
            max_open_streams=1,
        )
    )
    peer = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=SimpleNamespace(execute=lambda *_: {}),
            session=SimpleNamespace(name="peer", capabilities=BackendCapabilities(resource_domain="closed-bank")),
            providers=RuntimeProviders(object(), domains),
        )
    )
    owner.load()
    peer.load()
    resets = []

    def reset(*_):
        resets.append("native")
        with pytest.raises(ExecutionFailure):
            peer.execute(ModelRequest(), ExecutionContext("during-reset", Deadline.after(0.02)))
        if reset_mode == "failure":
            raise RuntimeError("reset did not complete")

    if reset_mode != "unsupported":
        native.reset = reset
    try:
        stream = owner.open_stream(ExecutionContext("open"))
        with pytest.raises(ExecutionFailure):
            owner.step(stream, ModelRequest(), ExecutionContext("unknown"))
        owner.close_stream(stream)
        if reset_mode == "success":
            owner.reset()
            assert owner.ready and resets == ["native"]
            peer.execute(ModelRequest(), ExecutionContext("released", Deadline.after(0.1)))
        else:
            with pytest.raises(ExecutionFailure) as caught:
                owner.reset()
            assert caught.value.code == ("reset_unsupported" if reset_mode == "unsupported" else "reset_failed")
            assert resets == ([] if reset_mode == "unsupported" else ["native"])
            assert not owner.ready
            with pytest.raises(ExecutionFailure):
                peer.execute(ModelRequest(), ExecutionContext("blocked", Deadline.after(0.02)))
            owner.close()
            peer.execute(ModelRequest(), ExecutionContext("released-by-close", Deadline.after(0.1)))
    finally:
        owner.close()
        peer.close()
