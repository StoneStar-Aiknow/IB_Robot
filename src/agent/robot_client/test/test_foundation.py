from __future__ import annotations

import subprocess
import sys

import robot_client
import robot_client.catalog
import robot_client.workflow


def test_lazy_robot_client_facade() -> None:
    from robot_client.client import RobotClient

    assert robot_client.RobotClient is RobotClient


def _isolated_import_probe(import_statement: str) -> list[str]:
    probe = rf"""
import sys

class ImportBlocker:
    blocked_roots = ("robot_skill_cli", "robot_config", "skill_catalog")

    def find_spec(self, name, path=None, target=None):
        if name in self.blocked_roots or name.startswith(tuple(root + "." for root in self.blocked_roots)):
            raise ImportError("blocked test import: " + name)
        return None

sys.meta_path.insert(0, ImportBlocker())
{import_statement}
print(*(name for name in sys.modules if name.startswith("robot_skill_cli")), sep="\n")
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return [name for name in result.stdout.splitlines() if name.startswith("robot_skill_cli")]


def test_catalog_import_isolation() -> None:
    assert _isolated_import_probe("import robot_client.catalog") == []


def test_package_import_isolation() -> None:
    assert _isolated_import_probe("import robot_client; robot_client.RobotClient") == []


def test_catalog_submodules_do_not_import_cli() -> None:
    assert (
        _isolated_import_probe(
            "import robot_client.catalog; import robot_client.workflow; "
            "from robot_client.transports.ros_bridge import RosBridge; "
            "from robot_client.transports.resident import ResidentBridge"
        )
        == []
    )


def test_legacy_ros_bridge_and_interactive_control_shims() -> None:
    from robot_client import interactive_control, ros_bridge, workflow
    from robot_client.transports import ros_bridge as transport

    assert ros_bridge.RosBridge is transport.RosBridge
    assert ros_bridge.BridgeError is transport.BridgeError
    assert ros_bridge.EXIT_ROS_UNAVAILABLE == 4
    assert ros_bridge.EXIT_TIMEOUT == 124
    assert interactive_control.InteractiveController is workflow.InteractiveController


def test_stop_contract_exit_codes_are_owned_by_embodied_common() -> None:
    from embodied_common import exit_codes
    from robot_client.stop_contract import EXIT_ROS_UNAVAILABLE, EXIT_TIMEOUT

    assert EXIT_ROS_UNAVAILABLE == 4
    assert EXIT_TIMEOUT == 124
    assert EXIT_ROS_UNAVAILABLE == exit_codes.EXIT_ROS_UNAVAILABLE
    assert EXIT_TIMEOUT == exit_codes.EXIT_TIMEOUT
