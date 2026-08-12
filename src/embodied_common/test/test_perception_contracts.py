import subprocess
import sys

import pytest

from embodied_common.perception_contracts import validate_result_schema


def test_perception_contracts_do_not_import_visual_game_registry():
    script = (
        "import sys; import embodied_common.perception_contracts; "
        "assert 'embodied_common.visual_game_contracts' not in sys.modules"
    )

    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize(
    ("contract", "result", "message"),
    [
        ({"field": "score", "kind": "number"}, {"score": float("inf")}, "finite number"),
        ({"field": "score", "kind": "number"}, {"score": True}, "finite number"),
        ({"field": "tags", "kind": "string_array"}, {"tags": ["ok", 1]}, "array of strings"),
        ({"field": "tags", "kind": "string_array"}, {"tags": "single"}, "array of strings"),
        ({"field": "x", "kind": "bogus"}, {"x": 1}, "unsupported result contract kind"),
        ("not-a-mapping", {"x": 1}, "response contract must be a mapping"),
        ({"field": "", "kind": "string"}, {"x": 1}, "non-empty string"),
        ({"field": "missing", "kind": "string"}, {}, "missing contract field"),
    ],
)
def test_validate_result_schema_branches(contract, result, message):
    error = validate_result_schema(contract, result)

    assert error is not None
    assert message in error


def test_validate_result_schema_accepts_supported_values():
    assert validate_result_schema({"field": "house", "kind": "enum", "allowed_values": ["a"]}, {"house": "a"}) is None
    assert validate_result_schema({"field": "score", "kind": "number"}, {"score": 0.9}) is None
    assert validate_result_schema({"field": "tags", "kind": "string_array"}, {"tags": ["a", "b"]}) is None
    assert validate_result_schema({"field": "name", "kind": "string"}, {"name": "ok"}) is None
