"""Device completion evidence shared by Torch model sessions."""

import logging
from contextlib import contextmanager, nullcontext
from dataclasses import replace

from inference_service.backends.errors import BackendInferenceError, BackendLifecycleError
from inference_service.unified_runtime import ExecutionFailureFactory, OutcomeEvidence
from inference_service.unified_runtime.contracts import OutcomeState
from inference_service.unified_runtime.errors import CancellationRequested, DeadlineExceeded, clear_exception_frames

LOGGER = logging.getLogger(__name__)

# A device stuck in a sticky error state never reports completion. Release
# retains resources for this many failed drain attempts, then abandons the
# wait so close can terminate (see drain_torch_device_before_release).
RELEASE_DRAIN_ATTEMPTS = 2


def resolve_torch_device(torch_module, device_name):
    """Freeze implicit accelerator placement while the model is being loaded."""
    if device_name in {"cuda", "npu"}:
        return torch_module.device(f"{device_name}:{getattr(torch_module, device_name).current_device()}")
    return torch_module.device(device_name)


def torch_device_scope(torch_module, device_name, device):
    """Bind SDK calls using an implicit device to the admitted physical device."""
    if device_name in {"cuda", "npu"}:
        return getattr(torch_module, device_name).device(device)
    return nullcontext()


def drain_torch_device(torch_module, device_name, device):
    if device_name == "cpu":
        return
    runtime = getattr(torch_module, device_name)
    if device_name == "mps":
        runtime.synchronize()
    else:
        runtime.synchronize(device)


def drain_torch_device_before_release(torch_module, device_name, device, *, failed_attempts: int, owner: str) -> None:
    """Wait for device completion before releasing model references.

    While ``failed_attempts`` is below ``RELEASE_DRAIN_ATTEMPTS`` a failed drain
    raises ``BackendLifecycleError(close_pending=True)``: completion is unknown,
    so the caller must retain the references and retry the release later. Once
    the bound is reached the device is presumed stuck in a sticky error state
    that never completes; the wait is abandoned so the caller can drop the
    references and let close terminate, and the abandoned completion is logged.
    """
    if failed_attempts >= RELEASE_DRAIN_ATTEMPTS:
        LOGGER.warning(
            "%s abandons the device drain after %d failed attempts; releasing references with device completion unknown",
            owner,
            failed_attempts,
        )
        return
    try:
        drain_torch_device(torch_module, device_name, device)
    except Exception as exc:
        raise BackendLifecycleError(str(exc), close_pending=True) from exc


def check_torch_completed_context(context, phase, torch_module, device_name, device, *, managed, stateful):
    """Fence a discarded direct result before claiming device completion."""
    try:
        context.check(phase)
    except (DeadlineExceeded, CancellationRequested) as exc:
        evidence = OutcomeEvidence.completed(phase, state_mutated=stateful)
        details = {}
        if not managed:
            try:
                drain_torch_device(torch_module, device_name, device)
            except Exception as completion_error:
                evidence = OutcomeEvidence.started(phase, outcome_known=False, state_mutated=stateful)
                details["completion_error"] = str(completion_error)
        raise ExecutionFailureFactory().from_exception(exc, evidence=evidence, details=details) from exc


@contextmanager
def torch_execution_scope(
    torch_module, device_name, device, *, complete_on_success=False, force_completion_on_error=False, stateful=False
):
    """Fence managed success on the caller's stream; conservatively drain errors.

    CUDA/NPU callbacks must submit to the entry stream or join child streams
    back to it before returning. Unjoined background work is unsupported.
    ``force_completion_on_error`` also fences exceptions labelled NOT_STARTED
    by a later processor, which cannot describe earlier work in this scope.
    It does not control the successful-return fence. ``stateful`` marks a
    session whose callback may mutate model state before a control exception.
    """
    stream = None
    try:
        with torch_device_scope(torch_module, device_name, device):
            if complete_on_success and device_name in {"cuda", "npu"}:
                stream = getattr(torch_module, device_name).current_stream(device)
            yield
    except Exception as exc:
        complete_torch_failure(
            exc, torch_module, device_name, device, discarded_result=force_completion_on_error, stateful=stateful
        )
        raise
    else:
        if complete_on_success:
            try:
                if stream is None:
                    drain_torch_device(torch_module, device_name, device)
                else:
                    stream.synchronize()
            except Exception as exc:
                raise BackendInferenceError(
                    str(exc), code="device_completion_unknown", operation_started=True, outcome_known=False
                ) from exc


def complete_torch_failure(exc, torch_module, device_name, device, *, discarded_result=False, stateful=False):
    """Fence execution errors and direct results discarded after Session return.

    ``stateful`` marks a session whose callback may have mutated model state
    before the exception; a bare control exception's default fields cannot
    prove it did not (see ``execution_evidence``'s ``started and stateful``).
    """
    evidence = getattr(exc, "evidence", None)
    if not discarded_result and (
        (
            not isinstance(exc, CancellationRequested | DeadlineExceeded)
            and getattr(exc, "operation_started", None) is False
        )
        or (evidence is not None and evidence.state is OutcomeState.NOT_STARTED)
    ):
        clear_exception_frames(exc)
        return
    synchronized = device_name == "cpu"
    try:
        drain_torch_device(torch_module, device_name, device)
        synchronized = True
    except Exception as completion_error:
        exc.details = {**getattr(exc, "details", {}), "completion_error": str(completion_error)}
    clear_exception_frames(exc)
    if evidence is not None:
        exc.evidence = replace(
            evidence,
            state=(
                OutcomeState.STARTED
                if discarded_result and (not synchronized or evidence.state is OutcomeState.NOT_STARTED)
                else evidence.state
            ),
            outcome_known=synchronized if discarded_result else evidence.outcome_known or synchronized,
        )
    elif isinstance(exc, CancellationRequested | DeadlineExceeded):
        # Defaults on a control exception say nothing about callback progress.
        exc.operation_started = True
        exc.outcome_known = synchronized
        exc.evidence = OutcomeEvidence(
            OutcomeState.COMPLETED if synchronized else OutcomeState.STARTED,
            synchronized,
            exc.state_mutated or stateful,
            exc.phase,
        )
    else:
        exc.operation_started = True if discarded_result else getattr(exc, "operation_started", True)
        exc.outcome_known = synchronized if discarded_result else getattr(exc, "outcome_known", False) or synchronized
