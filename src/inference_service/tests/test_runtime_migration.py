from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from inference_manifest import TorchRuntimeProfile
from inference_service import generic_runtime
from inference_service._legacy_named_tensor import NamedTensorResult, RuntimeErrorInfo
from inference_service._runtime_compat import build_session_runtime_handle
from inference_service.distributed import StructuredError, structured_error_from_exception
from inference_service.pipeline.runtime import _create_unified_policy_handle
from inference_service.unified_runtime import (
    ExecutionContext,
    ModelRequest,
    ModelResult,
    ModelRuntimeHandle,
    OutcomeEvidence,
    OutcomeState,
    RecoveryAction,
    RecoveryRequirement,
    RecoveryScope,
    RuntimeProviders,
)


def _providers() -> RuntimeProviders:
    return RuntimeProviders.create(SimpleNamespace(), SimpleNamespace())


def _context(model_type: str, operation: str = "predict") -> SimpleNamespace:
    identity = SimpleNamespace(
        interface="policy" if model_type in {"pi05", "smolvla", "diffusion"} else "tensor_model",
        model_type=model_type,
        operation=operation,
    )
    return SimpleNamespace(
        identity=identity,
        model_type=model_type,
        backend="torch",
        backend_profile=TorchRuntimeProfile(device="cpu"),
        target_runtime="torch",
        runtime_abi=None,
        deployment=SimpleNamespace(
            execution_contract=SimpleNamespace(
                state_scope="request",
                execution_structure="iterative",
                orchestration_visibility="executor",
            )
        ),
        deployment_name="cpu",
    )


class _IterativeExecutor:
    stages = (SimpleNamespace(plan=(), state_adapter=object()),)

    def __init__(self) -> None:
        self.loaded = False
        self.closed = False
        self.calls: list[str] = []

    def load(self, _context: object) -> None:
        self.loaded = True

    def execute(self, request, *, deadline, control):
        del deadline
        control.raise_if_canceled("test")
        self.calls.append(request.request_id)
        return {"value": request.inner.inputs["value"]}

    def cancel(self, _request_id: str, deadline=None) -> None:
        del deadline

    def reset(self, deadline=None) -> None:
        del deadline

    def health(self):
        return SimpleNamespace(ready=True, state=SimpleNamespace(value="ready"))

    def close(self) -> None:
        self.closed = True


class _Session:
    capabilities = SimpleNamespace(stateful=False, resettable=False)

    def __init__(self) -> None:
        self.loaded = False
        self.closed = False
        self.load_count = 0
        self.close_count = 0

    def load(self, _context: object) -> None:
        self.load_count += 1
        self.loaded = True

    def infer(self, request) -> ModelResult:
        return ModelResult(
            outputs={"audio": np.asarray(request.inputs["audio"])},
            latency=1.0,
            evidence=OutcomeEvidence.completed("backend"),
        )

    def health(self):
        return SimpleNamespace(
            ready=True,
            state=SimpleNamespace(value="ready"),
            failure_count=0,
            reason_code=None,
            message=None,
            recoverable=False,
        )

    def close(self) -> None:
        self.close_count += 1
        self.closed = True


def test_named_results_are_compatibility_only() -> None:
    assert generic_runtime.__all__ == ["NamedTensorRequest"]
    assert NamedTensorResult.__module__.endswith("_legacy_named_tensor")
    assert RuntimeErrorInfo.__module__.endswith("_legacy_named_tensor")
    assert not hasattr(generic_runtime, "NamedTensorResult")


def test_policy_iterative_path_uses_factory_handle_and_model_result() -> None:
    executor = _IterativeExecutor()
    context = _context("pi05")
    handle = _create_unified_policy_handle(context, executor, _providers(), resettable=False)

    assert isinstance(handle, ModelRuntimeHandle)
    assert handle.assembly.execution_contract.name == "request-iterative"
    assert handle.assembly.execution_contract.orchestration_visibility == "executor"

    handle.load()
    result = handle.execute(ModelRequest({"value": 7}), ExecutionContext("pi05-request"))

    assert isinstance(result, ModelResult)
    assert result.outputs["value"] == 7
    assert result.evidence.state is OutcomeState.COMPLETED
    handle.close()
    assert executor.closed


def test_session_visible_iterative_path_publishes_unified_result() -> None:
    session = _Session()
    context = _context("zipvoice", "synthesize")
    handle = build_session_runtime_handle(
        session,
        context,
        _providers(),
        execution_structure="iterative",
        orchestration_visibility="session",
        runtime_id="zipvoice-test",
    )

    assert handle.assembly.execution_contract.name == "request-iterative"
    assert handle.assembly.execution_contract.orchestration_visibility == "session"
    assert handle.assembly.session is session
    handle.load()
    assert session.load_count == 1
    result = handle.execute(ModelRequest({"audio": np.ones(2, dtype=np.float32)}), ExecutionContext("tts-1"))

    assert isinstance(result, ModelResult)
    np.testing.assert_array_equal(result.outputs["audio"], np.ones(2, dtype=np.float32))
    handle.close()
    assert session.closed
    assert session.close_count == 1


def test_structured_error_remains_the_policy_distributed_wire_value() -> None:
    error = StructuredError(code="cancel_failed", message="cancel failed", stage="cancel")

    assert error.code == "cancel_failed"
    assert error.stage == "cancel"


def test_distributed_mapping_keeps_unified_evidence_and_recovery_in_details() -> None:
    from inference_service.unified_runtime import ExecutionFailure

    failure = ExecutionFailure(
        "backend_async_failure",
        "device outcome is unknown",
        recoverable=True,
        recovery=RecoveryRequirement(RecoveryScope.REQUEST, RecoveryAction.RESET_RUNTIME),
        evidence=OutcomeEvidence.started("acl_async", outcome_known=False, state_mutated=True),
    )

    wire_error = structured_error_from_exception(failure, "backend")

    assert wire_error.details["evidence"]["state"] == "started"
    assert wire_error.details["recovery"] == {"scope": "request", "action": "reset_runtime"}
