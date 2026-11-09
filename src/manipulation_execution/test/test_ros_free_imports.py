"""Fresh-interpreter import checks for the ROS-free modules (design D5/D7, task 2.7).

Each check imports one module in a clean child interpreter and asserts that
neither ``rclpy`` nor ``ibrobot_msgs`` (or, for the replay module, the ROS node
module) ended up in ``sys.modules``. A plain in-process import cannot prove
this: the test session itself loads rclpy.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

PROBE = """
import sys
import {module}
forbidden = {forbidden!r}
loaded = sorted(
    name
    for name in sys.modules
    if any(name == item or name.startswith(item + ".") for item in forbidden)
)
if loaded:
    print("FORBIDDEN_LOADED:" + ",".join(loaded))
else:
    print("CLEAN")
"""

ROS_FREE_MODULES = (
    "manipulation_execution.geometry",
    "manipulation_execution.shared.runtime_ownership",
    "manipulation_execution.place.containment",
    "manipulation_execution.hri.animation",
)


def _child_environment() -> dict[str, str]:
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    paths = [entry for entry in sys.path if entry]
    environment["PYTHONPATH"] = os.pathsep.join(paths + ([existing] if existing else []))
    return environment


def _probe(module: str, forbidden: tuple[str, ...]) -> str:
    script = PROBE.format(module=module, forbidden=forbidden)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=_child_environment(),
        timeout=60,
    )
    assert result.returncode == 0, f"import {module} failed: {result.stderr}"
    return result.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("module", ROS_FREE_MODULES)
def test_shared_and_extracted_modules_load_without_ros(module):
    assert _probe(module, ("rclpy", "ibrobot_msgs")) == "CLEAN"


def test_place_replay_loads_neither_ros_nor_the_node_module():
    outcome = _probe("manipulation_execution.place.replay", ("rclpy", "manipulation_execution.place.node"))
    assert outcome == "CLEAN"
