"""Launch-builder tests for the navigation entry (robot_navigation's contribution).

Migrated from ``robot_config/test/test_launch_readiness.py`` (the
sim-style navigation node-generation cases) — openspec change
``migrate-business-launch-builders``, task B.7: the tests follow the
navigation module to its owning package. Assertions are verbatim; the EKF
config loads through robot_config's installed share directory.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from robot_navigation.launch_builders.navigation import generate_navigation_nodes

ekf_config = str(Path(get_package_share_directory("robot_config")) / "config" / "lekiwi" / "navigation" / "ekf.yaml")


def test_generate_navigation_nodes_for_lekiwi_mode():
    ekf_config = str(Path(__file__).resolve().parents[1] / "config" / "lekiwi" / "navigation" / "ekf.yaml")
    nodes = generate_navigation_nodes(
        {
            "navigation": {
                "enabled": True,
                "default_mode": "full",
                "modes": {"full": {"config": ekf_config}},
            }
        }
    )

    assert len(nodes) == 1
    assert isinstance(nodes[0], Node)


def test_generate_navigation_nodes_honors_force_enable_override():
    ekf_config = str(Path(__file__).resolve().parents[1] / "config" / "lekiwi" / "navigation" / "ekf.yaml")
    nodes = generate_navigation_nodes(
        {
            "navigation": {
                "enabled": False,
                "default_mode": "full",
                "modes": {"full": {"config": ekf_config}},
            }
        },
        force_enable=True,
    )

    assert len(nodes) == 1
    assert isinstance(nodes[0], Node)
