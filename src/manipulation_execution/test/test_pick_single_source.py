from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_legacy_supervised_script_does_not_exist():
    script_path = _REPOSITORY_ROOT / "scripts" / "test_banana_handeye_pick.py"

    assert not script_path.exists()


def test_direct_pick_action_client_does_not_exist():
    package_dir = _REPOSITORY_ROOT / "src" / "manipulation_execution"

    assert not (package_dir / "manipulation_execution" / "pick_action_client.py").exists()
    setup_source = (package_dir / "setup.py").read_text(encoding="utf-8")
    assert "pick_action_client" not in setup_source


def test_legacy_supervised_executor_does_not_exist():
    legacy_path = _REPOSITORY_ROOT / "scripts" / "test_banana_handeye_pick_legacy.py"

    assert not legacy_path.exists()


def test_production_executor_does_not_import_script_implementations():
    package_dir = _REPOSITORY_ROOT / "src" / "manipulation_execution" / "manipulation_execution"

    for module_path in package_dir.rglob("*.py"):
        source = module_path.read_text(encoding="utf-8")
        assert "scripts.test_banana_handeye_pick" not in source, module_path
