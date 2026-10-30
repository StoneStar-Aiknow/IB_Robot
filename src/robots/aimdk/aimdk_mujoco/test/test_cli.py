"""Resource locations must be supplied before preparation or simulation starts."""

import importlib
import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from launch import LaunchContext


@pytest.mark.parametrize(
    "module,args",
    [
        ("assets", []),
        ("node", ["--headless"]),
        ("physics", ["--verify"]),
        ("verify_ros", ["--report", "/unused/report.json"]),
        ("verify_handoff", ["--report", "/unused/report.json"]),
    ],
)
def test_cli_requires_explicit_assets(module, args, monkeypatch, capsys):
    entry = importlib.import_module(f"aimdk_mujoco.{module}")
    monkeypatch.setattr(sys, "argv", [module, *args])
    with pytest.raises(SystemExit) as exc:
        entry.main()
    assert exc.value.code == 2
    assert "required: --assets" in capsys.readouterr().err


def test_handoff_requires_report(monkeypatch, capsys):
    from aimdk_mujoco import verify_handoff

    monkeypatch.setattr(sys, "argv", ["verify_handoff", "--assets", "/unused/assets"])
    with pytest.raises(SystemExit) as exc:
        verify_handoff.main()
    assert exc.value.code == 2
    assert "required: --report" in capsys.readouterr().err


@pytest.fixture
def simulation_launch():
    path = Path(__file__).resolve().parents[1] / "launch" / "simulation.launch.py"
    spec = importlib.util.spec_from_file_location("x2_simulation_launch_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launch_requires_explicit_assets(simulation_launch):
    context = LaunchContext()
    with pytest.raises(RuntimeError, match='Required launch argument "assets"'):
        for action in simulation_launch.generate_launch_description().entities:
            action.execute(context)


@pytest.mark.parametrize("config_dir", [None, "/unused/controllers"])
@pytest.mark.parametrize("headless", [None, "false", "true"])
def test_launch_forwards_optional_config_dir(simulation_launch, config_dir, headless, monkeypatch):
    node = Mock()
    monkeypatch.setattr(simulation_launch, "Node", node)
    context = LaunchContext()
    context.launch_configurations["assets"] = "/unused/x2-assets"
    if config_dir is not None:
        context.launch_configurations["config_dir"] = config_dir
    if headless is not None:
        context.launch_configurations["headless"] = headless

    for action in simulation_launch.generate_launch_description().entities:
        action.execute(context)

    arguments = ["--assets", "/unused/x2-assets"]
    if config_dir is not None:
        arguments.extend(["--config-dir", config_dir])
    else:
        assert context.launch_configurations["config_dir"] == ""
    if headless == "true":
        arguments.append("--headless")
    assert "controller_config" not in context.launch_configurations
    assert "motion" not in context.launch_configurations
    node.assert_called_once_with(package="aimdk_mujoco", executable="simulation", arguments=arguments, output="screen")
