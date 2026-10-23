"""Under a runtime provider, the runtime alone owns lidar odometry and the base.

(Migrated from ``robot_config/test/test_provider_navigation_ownership.py``
— openspec change ``migrate-business-launch-builders``, task B.7: the tests
follow the navigation module to its owning package; assertions are
verbatim, the robot config loads through robot_config's installed share
directory and only the module import path changed.)

Regressions for binding ``lekiwi_nav_grasp`` to ``runtime.provider: lekiwi_robot``:

- ``launch_builders.runtime`` handed each stage's ``navigation.fast_lio`` to the
  robot runtime while the generic navigation builder kept launching FAST-LIO and
  its odom bridge as well -- two lidar odometry chains and two
  ``odom -> base_link`` publishers.
- The generic ``cmd_vel_bridge`` was never told about the provider, so it and
  the runtime's base node both drove the wheels, the bridge behind its own
  navigation gate rather than the runtime's.
- With the bridge gone, a stage using the Nav2 collision monitor would feed
  unfiltered planner output to the base, so such a stage must refuse to launch.
"""

from copy import deepcopy
from pathlib import Path

import pytest
from ament_index_python.packages import get_package_share_directory

from robot_config.launch_builders.runtime import runtime_provider
from robot_config.loader import load_robot_config_dict
from robot_navigation.launch_builders.navigation import generate_navigation_nodes

CONFIG_PATH = Path(get_package_share_directory("robot_config")) / "config/robots/lekiwi_nav_grasp.yaml"
FAST_LIO_STAGES = ("mapping", "navigation", "hybrid")


def _executables(nodes) -> list[str]:
    return [f"{node.node_package}/{node.node_executable}" for node in nodes if hasattr(node, "node_package")]


def _stage(nav_stage: str) -> dict:
    return load_robot_config_dict(CONFIG_PATH, nav_stage=nav_stage, defer_interface_binding=True)


def _without_collision_monitor(config: dict) -> dict:
    config = deepcopy(config)
    config["navigation"].setdefault("nav2_bringup", {})["dyn_avoid_enabled"] = False
    return config


@pytest.mark.parametrize("nav_stage", FAST_LIO_STAGES)
def test_provider_stage_leaves_fast_lio_and_the_base_to_the_runtime(nav_stage):
    config = _without_collision_monitor(_stage(nav_stage))
    assert runtime_provider(config) == "lekiwi_robot"
    assert config["navigation"]["fast_lio"]["enabled"] is True
    assert config["navigation"]["cmd_vel_bridge"]["enabled"] is True

    launched = _executables(generate_navigation_nodes(config, use_sim=False, force_enable=True))

    assert not [name for name in launched if "fast_lio" in name or "fastlio" in name], launched
    assert "robot_navigation/cmd_vel_bridge_node" not in launched, launched


@pytest.mark.parametrize("nav_stage", ("navigation", "hybrid"))
def test_provider_stage_with_collision_monitor_refuses_to_launch(nav_stage):
    config = _stage(nav_stage)
    assert config["navigation"]["nav2_bringup"]["dyn_avoid_enabled"] is True

    with pytest.raises(ValueError, match=r"collision monitor.*/cmd_vel_safe.*lekiwi_robot"):
        generate_navigation_nodes(config, use_sim=False, force_enable=True)


@pytest.mark.parametrize("nav_stage", FAST_LIO_STAGES)
def test_provider_less_stage_still_launches_fast_lio_and_the_bridge(nav_stage):
    config = deepcopy(_stage(nav_stage))
    config.pop("runtime")

    launched = _executables(generate_navigation_nodes(config, use_sim=False, force_enable=True))

    assert launched.count("fast_lio/fastlio_mapping") == 1, launched
    assert launched.count("robot_navigation/fast_lio_odom_bridge") == 1, launched
    assert launched.count("robot_navigation/cmd_vel_bridge_node") == 1, launched
