from pathlib import Path


def test_mcp_dependency_is_managed_by_workspace_and_package():
    package_root = Path(__file__).resolve().parents[1]
    workspace_root = Path(__file__).resolve().parents[3]
    requirement = "mcp>=1.26.0,<2"

    base_requirements = (workspace_root / "requirements" / "base.txt").read_text(encoding="utf-8").splitlines()
    setup_py = (package_root / "setup.py").read_text(encoding="utf-8")

    assert requirement in base_requirements
    assert f'"{requirement}"' in setup_py
