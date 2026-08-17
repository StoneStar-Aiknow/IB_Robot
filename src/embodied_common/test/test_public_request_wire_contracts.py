import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ibrobot_msgs.action import PrimitiveCommand, SkillCommand
from ibrobot_msgs.srv import ValidatePrimitive, ValidateSkill

ROOT = Path(__file__).resolve().parents[3]

MANIPULATION_NODE_INIT_FILES = (
    ROOT / "src/manipulation_execution/manipulation_execution/pick_executor_node.py",
    ROOT / "src/manipulation_execution/manipulation_execution/placement_executor_node.py",
)

PUBLIC_REQUEST_IDLS = (
    ROOT / "src/ibrobot_msgs/action/SkillCommand.action",
    ROOT / "src/ibrobot_msgs/action/PrimitiveCommand.action",
    ROOT / "src/ibrobot_msgs/srv/ValidateSkill.srv",
    ROOT / "src/ibrobot_msgs/srv/ValidatePrimitive.srv",
)

PUBLIC_REQUEST_TYPES = (
    ("SkillCommand.Goal", SkillCommand.Goal),
    ("PrimitiveCommand.Goal", PrimitiveCommand.Goal),
    ("ValidateSkill.Request", ValidateSkill.Request),
    ("ValidatePrimitive.Request", ValidatePrimitive.Request),
)


def _first_request_field(interface_path: Path) -> str:
    request_source = interface_path.read_text(encoding="utf-8").partition("\n---\n")[0]
    declarations = (line.split("#", maxsplit=1)[0].strip() for line in request_source.splitlines())
    return next(line for line in declarations if line)


@pytest.mark.parametrize("interface_path", PUBLIC_REQUEST_IDLS, ids=lambda path: path.name)
def test_public_request_source_idl_starts_with_uint32_schema_version(interface_path):
    assert _first_request_field(interface_path) == "uint32 schema_version"


@pytest.mark.parametrize(("type_name", "request_type"), PUBLIC_REQUEST_TYPES, ids=lambda value: str(value))
def test_generated_public_request_starts_with_uint32_schema_version(type_name, request_type):
    fields = list(request_type.get_fields_and_field_types().items())

    assert fields[0] == ("schema_version", "uint32"), f"{type_name} has stale generated field metadata: {fields}"


def test_public_request_wire_contract_preflight_accepts_current_generated_types():
    from embodied_common.wire_contracts import validate_public_request_wire_contracts

    assert validate_public_request_wire_contracts() is None


@pytest.mark.parametrize("source_path", MANIPULATION_NODE_INIT_FILES, ids=lambda path: path.stem)
def test_manipulation_startup_wire_preflight_precedes_ros_resource_calls(source_path):
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    class_name = "PickExecutorNode" if source_path.stem == "pick_executor_node" else "PlacementExecutorNode"
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    init = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    calls = []
    for node in ast.walk(init):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            continue
        calls.append((node.lineno, name))
    calls.sort()

    preflight_lines = [line for line, name in calls if name == "validate_public_request_wire_contracts"]
    ros_resource_lines = [
        line
        for line, name in calls
        if name in {"create_client", "create_subscription", "ActionClient", "ActionServer", "wait_for_server"}
    ]

    assert preflight_lines, "manipulation startup must run the shared wire preflight"
    assert ros_resource_lines
    assert preflight_lines[0] < min(ros_resource_lines)


@pytest.mark.parametrize(
    "fixture_fields",
    [
        {"dispatch_binding": "ibrobot_msgs/DispatchBinding", "skill_name": "string"},
        {"schema_version": "int32", "dispatch_binding": "ibrobot_msgs/DispatchBinding"},
        {"dispatch_binding": "ibrobot_msgs/DispatchBinding", "schema_version": "uint32"},
    ],
    ids=("unversioned", "wrong-type", "wrong-order"),
)
def test_stale_generated_overlay_fails_preflight_before_startup_callback(tmp_path, fixture_fields):
    overlay = tmp_path / "stale_overlay"
    action_package = overlay / "ibrobot_msgs" / "action"
    service_package = overlay / "ibrobot_msgs" / "srv"
    action_package.mkdir(parents=True)
    service_package.mkdir(parents=True)
    (overlay / "ibrobot_msgs" / "__init__.py").write_text("", encoding="utf-8")

    generated_fixture = f"""
class _GeneratedRequest:
    @classmethod
    def get_fields_and_field_types(cls):
        return {fixture_fields!r}
"""
    action_package.joinpath("__init__.py").write_text(
        generated_fixture
        + "\nclass SkillCommand:\n    Goal = _GeneratedRequest\n"
        + "\nclass PrimitiveCommand:\n    Goal = _GeneratedRequest\n",
        encoding="utf-8",
    )
    service_package.joinpath("__init__.py").write_text(
        generated_fixture
        + "\nclass ValidateSkill:\n    Request = _GeneratedRequest\n"
        + "\nclass ValidatePrimitive:\n    Request = _GeneratedRequest\n",
        encoding="utf-8",
    )

    startup_marker = tmp_path / "startup-called"
    script = """
import sys
from pathlib import Path
from embodied_common.wire_contracts import validate_public_request_wire_contracts

def startup_callback():
    Path(sys.argv[1]).write_text("called", encoding="utf-8")

validate_public_request_wire_contracts()
startup_callback()
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(overlay), str(ROOT / "src/embodied_common")))

    result = subprocess.run(
        [sys.executable, "-c", script, str(startup_marker)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "schema_version" in result.stderr
    assert not startup_marker.exists()
