"""Resource locations must be supplied before preparation or simulation starts."""

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest
from launch import LaunchContext


@pytest.fixture
def simulation_launch():
    path = Path(__file__).resolve().parents[1] / "launch" / "simulation.launch.py"
    spec = importlib.util.spec_from_file_location("x2_simulation_launch_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launch_forwards_config_and_headless_arguments(simulation_launch, monkeypatch):
    node = Mock()
    monkeypatch.setattr(simulation_launch, "Node", node)
    context = LaunchContext()
    context.launch_configurations["assets"] = "/unused/x2-assets"
    context.launch_configurations["config_dir"] = "/unused/controllers"
    context.launch_configurations["headless"] = "true"

    for action in simulation_launch.generate_launch_description().entities:
        action.execute(context)

    arguments = ["--assets", "/unused/x2-assets"]
    arguments.extend(["--config-dir", "/unused/controllers", "--headless"])
    assert "controller_config" not in context.launch_configurations
    assert "motion" not in context.launch_configurations
    node.assert_called_once_with(package="aimdk_mujoco", executable="simulation", arguments=arguments, output="screen")
