import pytest
from launch import LaunchContext
from launch_ros.utilities import evaluate_parameters


@pytest.mark.parametrize("use_sim", [False, True])
def test_enabled_slam_toolbox_uses_selected_clock(use_sim):
    from robot_navigation.launch_builders import localization

    nodes = localization.generate_localization_nodes({"slam_toolbox": {"enabled": True}}, use_sim=use_sim)
    assert len(nodes) == 1
    assert nodes[0].node_package == "slam_toolbox"
    parameters = evaluate_parameters(LaunchContext(), nodes[0]._Node__parameters)
    assert parameters[0]["use_sim_time"] is use_sim


def test_disabled_slam_toolbox_does_not_start_in_simulation():
    from robot_navigation.launch_builders import localization

    assert localization.generate_localization_nodes({"slam_toolbox": {"enabled": False}}, use_sim=True) == []


def test_simulation_does_not_start_rtabmap():
    from robot_navigation.launch_builders import localization

    assert localization.generate_localization_nodes({"ekf_rtabmap": {"enabled": True}}, use_sim=True) == []
