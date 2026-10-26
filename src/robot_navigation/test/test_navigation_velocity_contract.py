"""Navigation velocity-contract tests (robot_navigation's contribution).

Migrated from ``robot_config/test/test_navigation_velocity_contract.py``
(openspec change ``migrate-business-launch-builders``, task B.7): the
rejection case follows the navigation module (and its private validator) to
its owning package. The YAML-shape and cmd_vel cases stayed in robot_config
with the retained cmd_vel builder and the robot configs.
"""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]


def _lidar_config() -> dict:
    path = ROOT / "src/robot_config/config/robots/lekiwi_lidar.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["robot"]


def test_dynamic_navigation_rejects_a_bridge_topic_that_bypasses_collision_monitor():
    from robot_navigation.launch_builders.navigation import _validate_navigation_velocity_contract

    config = _lidar_config()
    navigation = config["nav_stages"]["navigation"]["navigation"]
    navigation["cmd_vel_bridge"]["cmd_vel_topic"] = "/cmd_vel"

    with pytest.raises(ValueError, match="cmd_vel_safe"):
        _validate_navigation_velocity_contract(config, navigation)
