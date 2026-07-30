import pytest

from perception_service.ascend_om_contracts import inspect_om_adapter, require_om_adapter_ready


@pytest.mark.parametrize(
    ("model", "ready"),
    [("ram_plus", True), ("sam2", False), ("siglip2", False), ("grounding_dino", False)],
)
def test_local_om_readiness_is_explicit(model, ready):
    result = inspect_om_adapter(model)

    assert result.ready is ready
    if not ready:
        assert result.reason


@pytest.mark.parametrize("model", ["sam2", "siglip2", "grounding_dino"])
def test_unfinalized_om_adapters_fail_closed(model):
    with pytest.raises(RuntimeError, match=f"{model} Ascend OM adapter is not ready"):
        require_om_adapter_ready(model)


def test_unknown_om_model_is_rejected():
    with pytest.raises(ValueError, match="model must be"):
        inspect_om_adapter("unknown")
