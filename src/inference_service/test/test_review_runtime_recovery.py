"""Regression coverage for the PR03/PR05 ownership review."""

from types import SimpleNamespace

import pytest

from inference_service.backends import BackendCapabilities, ResourceDomainAdmissions
from inference_service.backends.errors import BackendInferenceError
from inference_service.unified_runtime import (
    Deadline,
    ExecutionContext,
    ExecutionFailure,
    LifecycleState,
    ModelRequest,
    ModelRuntimeHandle,
    OutcomeState,
    RuntimeAssembly,
    RuntimeProviders,
)
from inference_service.unified_runtime.errors import CancellationRequested, DeadlineExceeded
from test.test_runtime_boundary_races import streaming
from test.test_runtime_execution_policy import dispose, managed_handle


@pytest.mark.parametrize("initial_state", ["ready", "reset_required"])
@pytest.mark.parametrize("failure", ["lock_timeout", "native_close"])
def test_runtime_reset_finishes_pending_close_before_resetting_streams(initial_state, failure):
    blocked, closed, reset = [True], [], []

    def close(*_):
        if blocked[0] and failure == "native_close":
            raise RuntimeError("close pending")
        closed.append(True)

    owner = streaming(close_stream=close, reset_stream=lambda *_: reset.append(True))
    stream = owner.open_stream(ExecutionContext("open"))
    record = owner._streams[stream.stream_id]
    if initial_state == "reset_required":
        owner.mark_reset_required()
    try:
        if failure == "lock_timeout":
            record.control_lock.acquire()
        try:
            with pytest.raises(ExecutionFailure):
                owner.close_stream(stream, deadline=Deadline.after(0.01))
        finally:
            if failure == "lock_timeout":
                record.control_lock.release()
        assert not stream.closed and owner.open_stream_count == 1
        blocked[0] = False
        owner.reset()
        assert owner.ready and stream.closed and owner.open_stream_count == 0
        assert closed == [True] and reset == []
        owner.open_stream(ExecutionContext("next"))
    finally:
        blocked[0] = False
        owner.close()


def test_failed_pending_close_retry_does_not_turn_runtime_into_failed():
    blocked = [True]

    def close(*_):
        if blocked[0]:
            raise RuntimeError("still closing")

    owner = streaming(close_stream=close)
    stream = owner.open_stream(ExecutionContext("open"))
    try:
        with pytest.raises(ExecutionFailure):
            owner.close_stream(stream)
        with pytest.raises(ExecutionFailure):
            owner.reset()
        assert owner.state is LifecycleState.READY
        blocked[0] = False
        owner.reset()
        assert owner.ready and stream.closed
    finally:
        blocked[0] = False
        owner.close()


@pytest.mark.parametrize("skipped", [False, True])
def test_request_reset_cannot_release_capacity_without_native_reset(skipped):
    calls = []

    def execute(*_):
        raise BackendInferenceError("uncertain", operation_started=True, outcome_known=False)

    native = SimpleNamespace(name="unresettable", capabilities=BackendCapabilities(resource_domain="review"))
    if skipped:
        native.reset = lambda: calls.append("reset")
    provider = ResourceDomainAdmissions()
    owner = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=SimpleNamespace(execute=execute),
            session=native,
            providers=RuntimeProviders(object(), provider),
            resettable=True,
            retry_failed_reset=skipped,
        )
    )
    owner.load()
    peer = provider.register("review", 1)
    try:
        with pytest.raises(ExecutionFailure):
            owner.execute(ModelRequest(), ExecutionContext("uncertain"))
        with pytest.raises(ExecutionFailure, match="reset_unsupported"):
            owner.reset()
        assert owner._admission.has_retained and not calls
        with pytest.raises(DeadlineExceeded):
            peer.acquire(context=ExecutionContext("peer", Deadline.after(0.01)))
    finally:
        owner.close()
        peer.close()


def test_policy_close_without_local_cleanup_stays_retryable():
    owner, executor, policy = managed_handle()
    close = policy.close
    policy.close = lambda _: None
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.close()
        assert caught.value.close_pending
        assert owner.state is LifecycleState.CLOSING and not owner._policy_close_complete
        policy.close = close
        owner.close()
        assert owner.closed and owner._policy_close_complete
    finally:
        policy.close = close
        dispose(owner, policy)


def test_unsupported_drain_preserves_healthy_nonresettable_runtime():
    owner, _, policy = managed_handle()
    owner._resettable = False
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.drain_managed(ExecutionContext("drain"), _dispatch_key=policy.authority)
        assert caught.value.code == "backend_drain_unsupported"
        assert caught.value.evidence.state is OutcomeState.NOT_STARTED
        assert owner.ready
    finally:
        dispose(owner, policy)


def test_drain_cancellation_after_native_progress_reports_started():
    owner, executor, policy = managed_handle()
    context = ExecutionContext("drain")
    executor.drain_uncertain_operations = lambda: context.token.cancel()
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.drain_managed(context, _dispatch_key=policy.authority)
        assert caught.value.evidence.state is OutcomeState.STARTED
        assert not caught.value.evidence.outcome_known and not owner.ready
    finally:
        dispose(owner, policy)


@pytest.mark.parametrize("error_type", [DeadlineExceeded, CancellationRequested])
@pytest.mark.parametrize("started", [False, True])
def test_policy_honors_explicit_operation_started(error_type, started):
    owner, _, policy = managed_handle()

    def execute(*_):
        raise error_type(phase="policy_queue", operation_started=started)

    policy.execute = execute
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest(), ExecutionContext("queue"))
        assert caught.value.evidence.state is (OutcomeState.STARTED if started else OutcomeState.NOT_STARTED)
        assert owner.ready is not started
    finally:
        dispose(owner, policy)


@pytest.mark.parametrize("resettable", [False, True])
def test_direct_stateless_host_failure_does_not_poison_handle(resettable):
    blocked = [True]

    def execute(*_):
        if blocked[0]:
            raise ValueError("invalid host input")
        return {}

    owner = ModelRuntimeHandle(SimpleNamespace(execute=execute), resettable=resettable)
    owner.load()
    try:
        with pytest.raises(ExecutionFailure) as caught:
            owner.execute(ModelRequest(), ExecutionContext("invalid"))
        assert caught.value.evidence.outcome_known and owner.ready
        blocked[0] = False
        owner.execute(ModelRequest(), ExecutionContext("valid"))
    finally:
        owner.close()
