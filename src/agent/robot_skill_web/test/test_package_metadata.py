from __future__ import annotations

import ast
import xml.etree.ElementTree as ET
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_XML = PACKAGE_ROOT / "package.xml"
SETUP_PY = PACKAGE_ROOT / "setup.py"


def _package() -> ET.Element:
    return ET.parse(PACKAGE_XML).getroot()


def _setup_keyword(name: str):
    tree = ast.parse(SETUP_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setup":
            for keyword in node.keywords:
                if keyword.arg == name:
                    return ast.literal_eval(keyword.value)
    raise AssertionError(f"setup.py has no {name} keyword")


def test_package_and_setup_declare_the_same_license():
    package_license = _package().findtext("license")
    assert package_license
    assert package_license == _setup_keyword("license")


def test_package_xml_uses_the_ros_schema_namespace():
    text = PACKAGE_XML.read_text(encoding="utf-8")
    assert "schematypens=" in text
    assert "schematens=" not in text


def test_package_maintainer_is_real_and_matches_setup_metadata():
    maintainer = _package().find("maintainer")
    assert maintainer is not None
    name = (maintainer.text or "").strip()
    assert name not in {"", "maintainer"}
    assert name == _setup_keyword("maintainer")
    assert maintainer.get("email") == _setup_keyword("maintainer_email")


def test_runtime_dependencies_are_declared_in_the_package_manifest():
    root = _package()
    exec_dependencies = {node.text.strip() for node in root.findall("exec_depend") if node.text}
    dependencies = {node.text.strip() for node in root.findall("depend") if node.text}
    assert "robot_config" in exec_dependencies
    assert "robot_client" in dependencies | exec_dependencies


def test_ros_dependencies_are_not_published_as_pypi_requirements():
    requirements = set(_setup_keyword("install_requires"))
    assert "robot_client" not in requirements
    assert "robot_config" not in requirements
