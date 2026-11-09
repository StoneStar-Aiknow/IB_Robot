"""Completion evidence distinguishes device errors from unknown outcomes."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from inference_service.backends.errors import BackendInferenceError
from inference_service.model_sessions.torch_completion import torch_execution_scope


@pytest.mark.parametrize("device_name", ["cpu", "cuda", "npu", "mps"])
@pytest.mark.parametrize("failure", [None, "execute", "synchronize"])
def test_torch_completion_reports_only_proven_outcomes(device_name, failure):
    events = []

    @contextmanager
    def device_scope(device):
        events.append(("enter", device))
        yield
        events.append(("leave", device))

    def synchronize(*args):
        events.append(("sync", args))
        if failure == "synchronize":
            raise RuntimeError("device still busy")

    torch = SimpleNamespace(
        **{
            device_name: SimpleNamespace(
                device=device_scope,
                synchronize=synchronize,
                current_stream=lambda device: SimpleNamespace(synchronize=lambda: synchronize(device)),
            )
        }
    )
    error = ValueError("execute failed")

    def execute():
        with torch_execution_scope(torch, device_name, "device:1", complete_on_success=True):
            events.append("execute")
            if failure == "execute":
                raise error

    if failure == "execute":
        with pytest.raises(ValueError) as result:
            execute()
        assert result.value is error and error.operation_started and error.outcome_known
    elif failure is not None and device_name != "cpu":
        with pytest.raises(BackendInferenceError) as result:
            execute()
        assert result.value.operation_started
        assert result.value.outcome_known is (failure == "execute")
    else:
        execute()
    syncs = [event for event in events if isinstance(event, tuple) and event[0] == "sync"]
    assert syncs == ([] if device_name == "cpu" else [("sync", () if device_name == "mps" else ("device:1",))])
    if device_name in {"cuda", "npu"}:
        assert events[0] == ("enter", "device:1")


def test_failed_device_execution_with_failed_drain_stays_unknown():
    @contextmanager
    def device_scope(_device):
        yield

    def synchronize(_device):
        raise RuntimeError("drain failed")

    torch = SimpleNamespace(cuda=SimpleNamespace(device=device_scope, synchronize=synchronize))
    with pytest.raises(ValueError) as result, torch_execution_scope(torch, "cuda", "cuda:0"):
        raise ValueError("execute failed")
    assert result.value.operation_started and not result.value.outcome_known
    assert result.value.details["completion_error"] == "drain failed"


@pytest.mark.parametrize("device_name", ["cuda", "npu"])
def test_success_waits_only_for_the_entry_stream(device_name):
    from contextlib import nullcontext

    completed = []
    entry = SimpleNamespace(synchronize=lambda: completed.append("entry"))
    current = [entry]
    runtime = SimpleNamespace(
        device=lambda _: nullcontext(),
        current_stream=lambda _: current[0],
        synchronize=lambda *_: pytest.fail("unrelated stream must not be drained on success"),
    )
    torch = SimpleNamespace(**{device_name: runtime})
    with torch_execution_scope(torch, device_name, "device:0", complete_on_success=True):
        # A callback can restore/change the ambient stream; completion still
        # belongs to the entry stream onto which its work was submitted.
        current[0] = SimpleNamespace(synchronize=lambda: pytest.fail("wrong stream"))
    assert completed == ["entry"]


@pytest.mark.parametrize("device_name", ["cpu", "cuda", "npu"])
@pytest.mark.parametrize("normalized", [False, True])
def test_device_fence_preserves_explicit_rejection_and_cleanup_contracts(device_name, normalized):
    from contextlib import nullcontext

    from inference_service.unified_runtime import ExecutionFailure, OutcomeEvidence, RecoveryRequirement
    from inference_service.unified_runtime.errors import CancellationRequested

    calls = []
    torch = SimpleNamespace(
        **{device_name: SimpleNamespace(device=lambda _: nullcontext(), synchronize=lambda *_: calls.append("sync"))}
    )
    rejection = CancellationRequested(phase="before_callback")
    if normalized:
        rejection = ExecutionFailure(
            "request_canceled", "before callback", evidence=OutcomeEvidence.not_started("before_callback")
        )
    with pytest.raises(type(rejection)) as caught, torch_execution_scope(torch, device_name, "device:0"):
        raise rejection
    assert caught.value is rejection
    if normalized:
        assert calls == []
        assert rejection.evidence == OutcomeEvidence.not_started("before_callback")
    else:
        assert calls == ([] if device_name == "cpu" else ["sync"])
        assert rejection.evidence.outcome_known
        assert rejection.operation_started

    recovery = RecoveryRequirement("runtime", "reload")
    failure = BackendInferenceError("cleanup pending", operation_started=True, outcome_known=True)
    failure.close_pending = True
    failure.state_mutated = False
    failure.recovery = recovery
    failure.phase = "output_transfer"
    if normalized:
        failure = ExecutionFailure(
            "cleanup_pending",
            "cleanup pending",
            evidence=OutcomeEvidence.completed("output_transfer"),
            recovery=recovery,
            details={"close_pending": True},
        )
    with pytest.raises(type(failure)) as caught, torch_execution_scope(torch, device_name, "device:0"):
        raise failure
    assert caught.value is failure and failure.close_pending and failure.recovery is recovery
    assert (failure.evidence.phase if normalized else failure.phase) == "output_transfer"


@pytest.mark.parametrize("control", ["cancel", "deadline"])
@pytest.mark.parametrize("sync_fails", [False, True])
def test_control_error_after_device_submission_requires_completion(control, sync_fails):
    from contextlib import nullcontext

    from inference_service.unified_runtime import OutcomeState
    from inference_service.unified_runtime.errors import CancellationRequested, DeadlineExceeded, execution_evidence

    calls = []

    def synchronize(_device):
        calls.append("sync")
        if sync_fails:
            raise RuntimeError("device busy")

    torch = SimpleNamespace(cuda=SimpleNamespace(device=lambda _: nullcontext(), synchronize=synchronize))
    error = CancellationRequested() if control == "cancel" else DeadlineExceeded()
    with pytest.raises(type(error)), torch_execution_scope(torch, "cuda", "cuda:0"):
        calls.append("submitted")
        raise error
    assert calls == ["submitted", "sync"]
    evidence = execution_evidence(error, started=True, stateful=False, unknown_on_error=True)
    assert evidence.state is not OutcomeState.NOT_STARTED
    assert evidence.outcome_known is not sync_fails
