"""The world hooks and scene-task transports (openspec sim-platform-registry 3.1b/3.2).

The Gazebo hook reuses the provider hand-off's world start (the gz server
include plus the ``/clock`` bridge) and returns the gz world instance name;
the MuJoCo hook starts nothing and returns the scene identity. The
scene-task transports carry the platform-specific actions and parameters
around the task node.
"""

from __future__ import annotations

import pytest


def _gz_args(include) -> str:
    (value,) = [
        value
        for key, value in include.launch_arguments
        if str(key[0].text if hasattr(key[0], "text") else key) == "gz_args"
    ]
    return str(value[0].text if hasattr(value[0], "text") else value)


def test_gazebo_hook_starts_the_scene_world_and_clock_bridge():
    from launch.actions import IncludeLaunchDescription
    from launch_ros.actions import Node

    from robot_runtime_sim.world_hooks import gazebo_start_world

    startup = gazebo_start_world(
        {"name": "test_robot", "gazebo_world_name": "demo"}, scene="pick_banana", spawn_pose=(0.0, 0.3, -0.03, 0.0)
    )
    assert startup.world == "demo"
    includes = [a for a in startup.actions if isinstance(a, IncludeLaunchDescription)]
    assert len(includes) == 1
    assert "sim_models_pick_banana_" in _gz_args(includes[0])
    bridges = [n for n in startup.actions if isinstance(n, Node)]
    assert [b._Node__node_executable for b in bridges] == ["parameter_bridge"]
    assert "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock" in bridges[0]._Node__arguments
    # the embodiment is the provider's: no create, no joint-state bridge
    assert "create" not in {b._Node__node_executable for b in bridges}


def test_gazebo_hook_without_a_scene_uses_the_empty_world():
    from launch.actions import IncludeLaunchDescription

    from robot_runtime_sim.world_hooks import gazebo_start_world

    startup = gazebo_start_world({"name": "t"}, scene="", spawn_pose=(0.0, 0.0, 0.0, 0.0))
    include = next(a for a in startup.actions if isinstance(a, IncludeLaunchDescription))
    assert "sim_models_empty_" in _gz_args(include)


def test_gazebo_hook_scene_load_failure_falls_back_with_a_warning(caplog):
    import logging

    from launch.actions import IncludeLaunchDescription

    from robot_runtime_sim.world_hooks import gazebo_start_world

    with caplog.at_level(logging.WARNING, logger="robot_runtime_sim.world_hooks"):
        startup = gazebo_start_world({"name": "t"}, scene="no_such_scene", spawn_pose=(0.0, 0.0, 0.0, 0.0))
    assert "falling back to the empty scene world" in caplog.text
    include = next(a for a in startup.actions if isinstance(a, IncludeLaunchDescription))
    assert "sim_models_empty_" in _gz_args(include)


def test_mujoco_hook_starts_nothing_and_returns_the_scene():
    from robot_runtime_sim.world_hooks import mujoco_start_world

    startup = mujoco_start_world({"name": "t"}, scene="pick_banana", spawn_pose=(0.0, 0.3, -0.03, 0.0))
    assert startup.actions == []
    assert startup.world == "pick_banana"
    empty = mujoco_start_world({"name": "t"}, scene="", spawn_pose=(0.0, 0.0, 0.0, 0.0))
    assert empty.world == ""


# ---------------------------------------------------------------------------
# The scene-task transports (D5)
# ---------------------------------------------------------------------------


class _Startup:
    """A duck-typed SimulationStartup stand-in (the transport only reads .world)."""

    def __init__(self, world: str):
        self.world = world


def test_gazebo_transport_starts_the_object_bridge_and_writes_world_name():
    from launch_ros.actions import Node

    from robot_runtime_sim.scene_task import gazebo_scene_task_transport

    parameters: dict = {}
    actions = gazebo_scene_task_transport("pick_banana", _Startup("demo"), {"name": "t"}, parameters)
    assert parameters["world_name"] == "demo"
    (bridge,) = actions
    assert isinstance(bridge, Node) and bridge._Node__package == "ros_gz_bridge"
    assert "/world/demo/set_pose@ros_gz_interfaces/srv/SetEntityPose" in bridge._Node__arguments
    assert any("dynamic_pose/info" in arg and "gz.msgs.Pose_V" in arg for arg in bridge._Node__arguments)

    def text(value):
        if isinstance(value, list | tuple):
            return "".join(text(part) for part in value)
        return str(getattr(value, "text", value))

    remappings = {(text(s), text(t)) for s, t in bridge._Node__remappings}
    assert ("/world/demo/dynamic_pose/info", "/sim/pick_banana/object_poses") in remappings


def test_gazebo_transport_world_defaults_to_demo_when_the_startup_is_empty():
    from robot_runtime_sim.scene_task import gazebo_scene_task_transport

    parameters: dict = {}
    actions = gazebo_scene_task_transport("empty", _Startup(""), {"name": "t"}, parameters)
    assert parameters["world_name"] == "demo"
    assert len(actions) == 1


def test_mujoco_transport_adds_no_action_and_no_parameter():
    from robot_runtime_sim.scene_task import mujoco_scene_task_transport

    parameters: dict = {}
    actions = mujoco_scene_task_transport("pick_banana", _Startup("pick_banana"), {"name": "t"}, parameters)
    assert actions == []
    assert parameters == {}


@pytest.mark.parametrize(
    ("module_attr", "attr"),
    [
        ("robot_runtime_sim.world_hooks", "gazebo_start_world"),
        ("robot_runtime_sim.world_hooks", "mujoco_start_world"),
        ("robot_runtime_sim.scene_task", "gazebo_scene_task_transport"),
        ("robot_runtime_sim.scene_task", "mujoco_scene_task_transport"),
    ],
)
def test_the_specs_reference_the_hooks(module_attr, attr):
    """The platform specs name the hooks through resolvable module:attr strings."""
    from importlib import import_module

    from robot_runtime_sim.platforms import GAZEBO_SPEC, MUJOCO_SPEC

    for spec, field in (
        (GAZEBO_SPEC, "world"),
        (GAZEBO_SPEC, "scene_task"),
        (MUJOCO_SPEC, "world"),
        (MUJOCO_SPEC, "scene_task"),
    ):
        module_name, _, attribute = spec.__dict__[field].partition(":")
        module = import_module(module_name)
        assert hasattr(module, attribute), f"{module_name}:{attribute} must exist"
    assert GAZEBO_SPEC.world == "robot_runtime_sim.world_hooks:gazebo_start_world"
    assert MUJOCO_SPEC.world == "robot_runtime_sim.world_hooks:mujoco_start_world"
    assert GAZEBO_SPEC.scene_task == "robot_runtime_sim.scene_task:gazebo_scene_task_transport"
    assert MUJOCO_SPEC.scene_task == "robot_runtime_sim.scene_task:mujoco_scene_task_transport"
