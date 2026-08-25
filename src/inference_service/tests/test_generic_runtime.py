from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from inference_service._legacy_named_tensor import (
    DeploymentIdentity,
    NamedTensorResult,
    RuntimeErrorInfo,
    RuntimeLatency,
)
from inference_service.backends import BackendState
from inference_service.generic_runtime import NamedTensorRequest


def _deployment() -> DeploymentIdentity:
    return DeploymentIdentity(
        bundle="rampp",
        bundle_uuid="bundle-uuid",
        bundle_revision=1,
        deployment="torch_cpu",
        deployment_uuid="deployment-uuid",
        deployment_revision=2,
        deployment_fingerprint="sha256:fingerprint",
        backend="torch",
    )


def test_named_request_and_result_snapshot_mappings() -> None:
    inputs = {"image": np.zeros((1, 3, 8, 8), dtype=np.float32)}
    metadata = {"source": "test"}
    request = NamedTensorRequest("request-1", inputs, metadata=metadata)
    inputs["other"] = object()
    metadata["changed"] = True

    assert isinstance(request.inputs, MappingProxyType)
    assert "other" not in request.inputs
    assert request.metadata == {"source": "test"}
    with pytest.raises(TypeError):
        request.inputs["new"] = object()  # type: ignore[index]

    result = NamedTensorResult(
        outputs={"tag_logits": np.ones((1, 4), dtype=np.float32), "boxes": []},
        deployment=_deployment(),
        latency=RuntimeLatency(total_ms=3.0, backend_ms=2.0),
        metadata={"model_family": "ram++"},
    )
    assert set(result.outputs) == {"tag_logits", "boxes"}
    assert result.ready


@pytest.mark.parametrize(
    "factory",
    [
        lambda: NamedTensorRequest("", {"image": object()}),
        lambda: NamedTensorRequest("request", {}),
        lambda: RuntimeLatency(total_ms=-1.0, backend_ms=0.0),
        lambda: RuntimeErrorInfo("", "failed"),
        lambda: NamedTensorResult({}, _deployment(), RuntimeLatency(1.0, 1.0)),
    ],
)
def test_generic_contract_validation(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_failed_result_has_structured_error_and_cannot_be_adapted() -> None:
    result = NamedTensorResult(
        outputs={},
        deployment=_deployment(),
        latency=RuntimeLatency(total_ms=1.0, backend_ms=0.0),
        state=BackendState.FAILED,
        error=RuntimeErrorInfo("not_ready", "deployment is unavailable", recoverable=True),
    )

    assert result.ready is False
    assert result.error.details == {}
    with pytest.raises(ValueError, match="failed result"):
        result.to_backend_result()


def test_named_result_converts_to_existing_policy_backend_result() -> None:
    result = NamedTensorResult(
        outputs={"action": np.ones((1, 2, 6), dtype=np.float32)},
        deployment=_deployment(),
        latency=RuntimeLatency(total_ms=4.0, backend_ms=2.5, preprocess_ms=1.0, postprocess_ms=0.5),
        metadata={"request_id": "request-1"},
    )

    backend_result = result.to_backend_result(actual_chunk_size=2)

    np.testing.assert_array_equal(backend_result.action, result.outputs["action"])
    assert backend_result.actual_chunk_size == 2
    assert backend_result.backend_latency_ms == 2.5
    assert backend_result.metadata["deployment_fingerprint"] == "sha256:fingerprint"
    assert backend_result.metadata["latency_ms"]["total"] == 4.0
