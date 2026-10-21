"""runtime-peripheral-overrides D7: provider contract for the mock runtime.

The mock runtime is the contract layer's reference provider: its launch must
declare ``peripherals`` (and ``profile``) and pass both through to the node,
which merges the fragment against the profile at initialisation (D9) — name
and value errors surface as a node-initialisation failure, the provider
process exits, the waiter never succeeds and no bound consumer is released.

The per-provider ``peripherals`` declaration is checked by each provider
package's own suite against its real ``generate_launch_description()``; this
file covers the mock provider only.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import launch
import pytest
import yaml

_PACKAGE = Path(__file__).resolve().parents[1]
_MOCK_LAUNCH = _PACKAGE / "launch" / "mock_runtime.launch.py"


def _fixture_profile(tmp_path, peripherals):
    profile = {
        "runtime": {"name": "mock_runtime", "version": "0.1.0"},
        "use_sim": True,
        "modes": {"idle": {"controllers": [], "transitions": []}},
        "capabilities": {"joint.state": {}, "runtime.status": {}},
        "joints": ["j1"],
        "peripherals": peripherals,
    }
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(profile), encoding="utf-8")
    return path


def _fragment_file(tmp_path, content):
    path = tmp_path / "fragment.yaml"
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return path


def _mock_launch_module():
    spec = importlib.util.spec_from_file_location("mock_runtime_launch_under_test", _MOCK_LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _params(node):
    from launch import LaunchContext
    from launch_ros.utilities import evaluate_parameters

    params = {}
    for entry in evaluate_parameters(LaunchContext(), node._Node__parameters):
        params.update(entry)
    return params


def _render(**configurations):
    context = launch.LaunchContext()
    base = {"profile": "", "use_sim": "", "sim_platform": "", "simulated": "", "peripherals": ""}
    base.update(configurations)
    context.launch_configurations.update(base)
    return _mock_launch_module()._launch_setup(context)


def _declared_argument_names():
    from launch.actions import DeclareLaunchArgument

    def _text(value):
        return "".join(part.text if hasattr(part, "text") else str(part) for part in value)

    description = _mock_launch_module().generate_launch_description()
    return {
        _text(action._DeclareLaunchArgument__name)
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }


def test_the_mock_launch_declares_peripherals_and_profile():
    assert {"peripherals", "profile"} <= _declared_argument_names()


def test_the_default_profile_selection_is_unchanged():
    """An empty or ``mock`` profile argument keeps today's default: the
    in-process reference profile, with the node regenerating it itself."""
    (node,) = [action for action in _render() if type(action).__name__ == "Node"]
    params = _params(node)
    assert params["profile"] == ""
    assert params["use_sim"] is True  # default_profile(True)'s selection
    # the historical "mock" name (so101_mock_core_gate's runtime.profile)
    (node,) = [action for action in _render(profile="mock") if type(action).__name__ == "Node"]
    assert _params(node)["profile"] == ""


def test_a_profile_argument_selects_the_actual_profile(tmp_path):
    """A profile path is resolved once, drives the launch's simulation
    selection and is handed to the node verbatim (the node and the selection
    use the same actual profile)."""
    profile = _fixture_profile(tmp_path, [])
    (node,) = [action for action in _render(profile=str(profile)) if type(action).__name__ == "Node"]
    params = _params(node)
    assert params["profile"] == str(profile)
    assert params["use_sim"] is True  # the fixture profile's selection


def test_a_profile_argument_must_resolve(tmp_path):
    from robot_runtime.launch_support import RuntimeProfileError

    with pytest.raises(RuntimeProfileError, match="runtime profile not found"):
        _render(profile=str(tmp_path / "absent.yaml"))


def test_mock_launch_passes_the_fragment_through_to_the_node(tmp_path):
    fragment = _fragment_file(tmp_path, {"peripherals": [{"type": "camera", "name": "top", "disabled": True}]})
    (node,) = [action for action in _render(peripherals=str(fragment)) if type(action).__name__ == "Node"]
    assert _params(node)["peripherals"] == str(fragment), "the node receives the fragment path verbatim"


def _init_mock_runtime(tmp_path, profile, fragment):
    """Initialise the real mock node from launch-generated parameters."""
    import rclpy

    from robot_runtime.mock_runtime_node import MockRuntime

    (node,) = [
        action for action in _render(profile=str(profile), peripherals=str(fragment)) if type(action).__name__ == "Node"
    ]
    params = _params(node)
    arguments = ["--ros-args"]
    for key, value in params.items():
        arguments.append("-p")
        arguments.append(f"{key}:={value}")
    rclpy.init(args=arguments)
    try:
        return MockRuntime()
    except Exception:
        rclpy.shutdown()
        raise


def test_launch_profile_and_fragment_reach_the_node_description(tmp_path):
    """The full chain (M2): the launch receives a fixture profile with one
    camera plus a disabling fragment, the generated node parameters
    initialise the real mock node, and the camera is absent from the node's
    ROS graph (its synthetic publisher is never created)."""
    profile = _fixture_profile(
        tmp_path,
        [{"type": "camera", "name": "test_cam", "driver": "opencv", "width": 64, "height": 48, "fps": 5}],
    )
    camera_topic = "/camera/test_cam/image_raw"

    enabled = _init_mock_runtime(tmp_path, profile, _fragment_file(tmp_path, {"peripherals": []}))
    try:
        assert enabled.count_publishers(camera_topic) >= 1, "the enabled camera has a synthetic publisher"
    finally:
        enabled.destroy_node()
        import rclpy

        rclpy.shutdown()

    fragment = _fragment_file(tmp_path, {"peripherals": [{"type": "camera", "name": "test_cam", "disabled": True}]})
    disabled = _init_mock_runtime(tmp_path, profile, fragment)
    try:
        assert disabled.count_publishers(camera_topic) == 0, "the disabled camera must not be published"
    finally:
        disabled.destroy_node()
        import rclpy

        rclpy.shutdown()


def test_mock_node_init_fails_on_a_misspelled_partial_entry(tmp_path):
    """A misspelled partial fragment makes node initialisation fail (D9): the
    provider process exits, the waiter never succeeds, no consumer is bound."""
    import rclpy

    from robot_runtime.interface_description import InterfaceDescriptionError

    profile = _fixture_profile(
        tmp_path,
        [{"type": "camera", "name": "top", "driver": "opencv", "width": 64, "height": 48, "fps": 5}],
    )
    fragment = _fragment_file(tmp_path, {"peripherals": [{"type": "camera", "name": "tpo", "width": 128}]})
    rclpy.init(args=["--ros-args", "-p", f"profile:={profile}", "-p", f"peripherals:={fragment}"])
    try:
        with pytest.raises(InterfaceDescriptionError, match=r"\('camera', 'tpo'\) is not declared"):
            from robot_runtime.mock_runtime_node import MockRuntime

            MockRuntime()
    finally:
        rclpy.shutdown()
