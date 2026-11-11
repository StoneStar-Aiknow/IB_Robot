"""The direct-control layer knows no platform, body, task or evaluation; it is imported only on demand."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "robot_runtime_sim" / "direct_control"
# Names that would mean a platform, a body or a task leaked into the shared layer.
FORBIDDEN = re.compile(
    r"robodojo|isaac|xpolicylab|curobo|arx|\bx5\b|left_arm|right_arm|general_pickup|cam_head|"
    r"ee_joint_state|arm_joint_state",
    re.IGNORECASE,
)
# Constants of the first RoboDojo body (case-sensitive: "arms" in prose is fine).
FORBIDDEN_CONSTANTS = re.compile(r"\bTICK_S\b|\bARM_DOF\b|\bARMS\b")
ALLOWED_IMPORT_ROOTS = {
    "__future__",
    "collections",
    "dataclasses",
    "importlib",
    "itertools",
    "json",
    "math",
    "sys",
    "threading",
    "time",
    "typing",
    "rclpy",
    "control_msgs",
    "geometry_msgs",
    "sensor_msgs",
    "std_msgs",
    "std_srvs",
    "ibrobot_msgs",
    "rosgraph_msgs",
    "robot_runtime",
    "robot_runtime_sim",  # only robot_runtime_sim.direct_control, checked below
}
# Evaluation is an optional layer on top (it attaches through the node's extensions): no
# identifier of this layer may name it.
EVALUATION_IDENTIFIER = re.compile(r"admi(t|ssion)|evaluat|verdict|benchmark", re.IGNORECASE)


def _sources():
    files = sorted(PACKAGE.glob("*.py"))
    assert files, PACKAGE
    return files


def test_no_platform_body_or_task_names_in_the_shared_layer():
    hits = []
    for path in _sources():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if FORBIDDEN.search(line) or FORBIDDEN_CONSTANTS.search(line):
                hits.append(f"{path.name}:{lineno}: {line.strip()}")
    assert hits == []


def test_imports_stay_within_the_runtime_contract_layer():
    bad = []
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            bad += [f"{path.name}: {n}" for n in names if n.split(".")[0] not in ALLOWED_IMPORT_ROOTS]
            bad += [
                f"{path.name}: {n}"
                for n in names
                if n.split(".")[0] == "robot_runtime_sim" and not n.startswith("robot_runtime_sim.direct_control")
            ]
    assert bad == []


def test_no_identifier_names_an_evaluation_concept():
    hits = []
    for path in _sources():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            name = getattr(node, "name", None) or getattr(node, "id", None) or getattr(node, "attr", None)
            if isinstance(name, str) and EVALUATION_IDENTIFIER.search(name):
                hits.append(f"{path.name}:{getattr(node, 'lineno', '?')}: {name}")
    assert hits == []


def test_the_engine_is_pure_and_the_package_does_not_load_this_layer_unasked():
    code = (
        "import sys\n"
        "import robot_runtime_sim, robot_runtime_sim.platforms\n"
        "assert 'robot_runtime_sim.direct_control' not in sys.modules, 'loaded with the package'\n"
        "import robot_runtime_sim.direct_control.engine, robot_runtime_sim.direct_control.host_step\n"
        "assert 'rclpy' not in sys.modules, 'the engine imported rclpy'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
