"""Tests for robot_config's forwarding teleop_operator launch entry.

The real entry moved to robot_teleop with the teleop launch builder
(openspec change ``migrate-business-launch-builders``, design decision 5);
this file keeps the historical ``ros2 launch robot_config
teleop_operator.launch.py`` path working. The forwarder declares the same
arguments, passes them through and prints one deprecation notice, while
importing NO robot_teleop Python module — an environment without
robot_teleop must fail with a clear error instead of an ImportError at
description-generation time.
"""

import importlib.util
from pathlib import Path

import pytest
from ament_index_python.packages import PackageNotFoundError

_LAUNCH_PATH = Path(__file__).resolve().parents[1] / "launch" / "teleop_operator.launch.py"
_LAUNCH_SPEC = importlib.util.spec_from_file_location("teleop_operator_forwarding", _LAUNCH_PATH)
assert _LAUNCH_SPEC is not None
assert _LAUNCH_SPEC.loader is not None
forwarding_launch = importlib.util.module_from_spec(_LAUNCH_SPEC)
_LAUNCH_SPEC.loader.exec_module(forwarding_launch)


def _substitution_text(value) -> str:
    """Launch wraps raw strings in TextSubstitution; unwrap for assertions."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_substitution_text(item) for item in value)
    return str(getattr(value, "text", value))


def test_forwarder_declares_the_same_arguments_and_the_deprecation_notice():
    description = forwarding_launch.generate_launch_description()

    declared = {
        entity.name for entity in description.entities if hasattr(entity, "name") and isinstance(entity.name, str)
    }
    assert {"robot_config", "config_path"} <= declared
    messages = [_substitution_text(entity.msg) for entity in description.entities if hasattr(entity, "msg")]
    assert any("robot_teleop" in message for message in messages)


def test_forwarder_targets_robot_teleops_launch_file_through_the_share_directory(tmp_path, monkeypatch):
    """The include is located through robot_teleop's package share
    directory (design decision 5): whatever share directory
    ``get_package_share_directory`` resolves for robot_teleop, the
    forwarder includes ``launch/teleop_operator.launch.py`` from it —
    captured here without executing the target launch file, so the test
    needs neither robot_teleop installed nor its Python modules imported."""
    fake_share = tmp_path / "robot_teleop"
    (fake_share / "launch").mkdir(parents=True)
    target = fake_share / "launch" / "teleop_operator.launch.py"
    target.write_text("# robot_teleop operator launch\n", encoding="utf-8")

    monkeypatch.setattr("ament_index_python.packages.get_package_share_directory", lambda package_name: str(fake_share))
    captured = []
    real_source = forwarding_launch.PythonLaunchDescriptionSource

    def capturing_source(location):
        captured.append(location)
        return real_source(location)

    monkeypatch.setattr(forwarding_launch, "PythonLaunchDescriptionSource", capturing_source)

    forwarding_launch._operator_launch_source()

    assert captured == [str(target)]


def test_stale_robot_teleop_install_without_the_launch_file_fails_clearly(tmp_path, monkeypatch):
    """A robot_teleop install predating the migration (share directory
    present, launch file missing) fails with a rebuild hint instead of a
    confusing include error."""
    fake_share = tmp_path / "robot_teleop"
    fake_share.mkdir(parents=True)
    monkeypatch.setattr("ament_index_python.packages.get_package_share_directory", lambda package_name: str(fake_share))

    with pytest.raises(RuntimeError, match="Rebuild robot_teleop"):
        forwarding_launch._operator_launch_source()


def test_missing_robot_teleop_fails_with_a_clear_error(monkeypatch):
    def raise_missing(package_name):
        raise PackageNotFoundError(package_name)

    monkeypatch.setattr("ament_index_python.packages.get_package_share_directory", raise_missing)

    with pytest.raises(RuntimeError, match="robot_teleop package"):
        forwarding_launch.generate_launch_description()


def test_forwarder_imports_no_robot_teleop_python_module():
    """Design decision 5: the forwarding entry must not import robot_teleop
    code — loading it stays possible in an environment without the package
    (only the launch-time share lookup needs it)."""
    content = _LAUNCH_PATH.read_text(encoding="utf-8")

    assert "import robot_teleop" not in content
    assert "from robot_teleop" not in content
