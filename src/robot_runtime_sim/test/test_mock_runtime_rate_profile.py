"""The mock runtime node's capability path sees the traits' rate profile.

Code review 1, fix 3 (companion to ``test_facade_platform.py``): through the
REAL ``MockRuntime`` constructor, a camera-rate callable that reads
``profile["simulation"][platform]`` resolves, and one returning ``None``
applies no override instead of failing on ``float(None)``.

A separate module from ``test_facade_platform``: that module's
``platform_stack`` fixture keeps rclpy initialised without global arguments
for its whole lifetime, and ``MockRuntime`` reads its ``profile`` from the
rclpy global arguments (it takes no constructor argument), so this test must
own its own ``rclpy.init(args=...)`` the way ``test_mock_provider_peripherals``
does — the two cannot share a process lifetime.
"""

from __future__ import annotations

import sys
import types

import rclpy
import yaml

PLATFORM_PROFILE = {
    "runtime": {"name": "mock_rate_probe", "version": "0.0.1"},
    "controller_manager": "controller_manager",
    "use_sim": True,
    "sim_platforms": ["sdk", "mujoco"],
    "sim_platform": "mujoco",
    "modes": {"initial": "idle", "idle": {"controllers": [], "transitions": []}},
    "capabilities": {
        "joint.state": {"joint_count": 1},
        "runtime.stop": {"cancel_bound_s": 2.0, "idle_bound_s": 3.0, "torque_off_bound_s": 4.0},
    },
    "joints": ["1"],
    "peripherals": [
        {
            "type": "camera",
            "name": "front",
            "driver": "opencv",
            "width": 16,
            "height": 12,
            "fps": 30,
            "pixel_format": "mjpeg2rgb",
            "frame_id": "front_optical",
        }
    ],
    "simulation": {"mujoco": {"camera_publish_rate": 6.25}},
}


def _init_mock_runtime(path):
    rclpy.init(args=["--ros-args", "-p", f"profile:={path}"])
    from robot_runtime.mock_runtime_node import MockRuntime

    try:
        return MockRuntime()
    except Exception:
        rclpy.shutdown()
        raise


def test_the_mock_runtime_passes_the_profile_to_the_traits_rate_callable(tmp_path, monkeypatch):
    """Through the real mock node: the callable sees the profile the node
    holds, and a None return applies no override instead of crashing."""
    import robot_runtime.mock_runtime_node as mock_module
    from robot_runtime.sim_platforms import SimSensorTraits

    seen = {}

    def _reading_rate(profile):
        seen.update(profile)
        return None  # also proves the None return does not crash this path

    module = types.ModuleType("mock_rate_probe")
    module.reading_rate = _reading_rate
    monkeypatch.setitem(sys.modules, "mock_rate_probe", module)
    traits = SimSensorTraits(camera_rate="mock_rate_probe:reading_rate")
    monkeypatch.setattr(mock_module, "effective_sensor_traits", lambda *args: traits)

    path = tmp_path / "mock_runtime.yaml"
    path.write_text(yaml.safe_dump(PLATFORM_PROFILE), encoding="utf-8")
    node = _init_mock_runtime(path)
    try:
        capabilities = node._state.capabilities
    finally:
        node.destroy_node()
        rclpy.shutdown()
    assert capabilities["perception.camera"]["topics"] == ["/camera/front/image_raw", "/camera/front/camera_info"]
    assert seen.get("simulation") == {"mujoco": {"camera_publish_rate": 6.25}}, (
        "the mock node threaded its profile into the callable"
    )
