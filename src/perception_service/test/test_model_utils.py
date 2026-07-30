import pytest

from perception_service.model_utils import inspect_backend, resolve_model_path


def test_resolve_model_path_uses_explicit_base(tmp_path):
    model = tmp_path / "model.om"
    model.write_bytes(b"model")

    assert resolve_model_path("model.om", tmp_path) == model


def test_inspect_backend_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend must be"):
        inspect_backend("unknown")


def test_cpu_backend_reports_runtime_status():
    status = inspect_backend("cpu")

    assert status.backend == "cpu"
    assert status.ready
    assert status.runtime_version
