import pytest

from semantic_mapping.semantic_mapping_node import validate_mapping_backend


def test_mapping_backend_is_explicit_and_rejects_silent_fallback():
    assert validate_mapping_backend("embedded") == "embedded"
    assert validate_mapping_backend("service") == "service"
    with pytest.raises(ValueError, match="embedded.*service"):
        validate_mapping_backend("automatic")
