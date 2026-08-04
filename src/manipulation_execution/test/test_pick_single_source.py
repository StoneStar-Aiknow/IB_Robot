import ast
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_supervised_script_is_only_a_canonical_action_client_wrapper():
    script_path = _REPOSITORY_ROOT / "scripts" / "test_banana_handeye_pick.py"
    source = script_path.read_text(encoding="utf-8")
    module = ast.parse(source)

    imports = [node for node in module.body if isinstance(node, ast.ImportFrom)]
    assert len(imports) == 1
    assert imports[0].module == "manipulation_execution.pick_action_client"
    assert [alias.name for alias in imports[0].names] == ["main"]
    assert len(source.splitlines()) <= 10


def test_legacy_supervised_executor_does_not_exist():
    legacy_path = _REPOSITORY_ROOT / "scripts" / "test_banana_handeye_pick_legacy.py"

    assert not legacy_path.exists()


def test_production_executor_does_not_import_script_implementations():
    package_dir = _REPOSITORY_ROOT / "src" / "manipulation_execution" / "manipulation_execution"

    for module_path in package_dir.rglob("*.py"):
        source = module_path.read_text(encoding="utf-8")
        assert "scripts.test_banana_handeye_pick" not in source, module_path
