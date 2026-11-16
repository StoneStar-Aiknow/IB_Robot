"""The per-platform scene-task transports (openspec sim-platform-registry D5).

The platform-neutral gates (the scene's declared ``platforms``, runtime
readiness, the bound trajectory interfaces) stay in
``robot_config.launch_builders.scene_task``; when they all pass, that builder
resolves the selected platform's ``SimPlatform.scene_task`` hook and calls it
as ``scene_task_transport(scene, startup, robot_config, parameters) ->
list``. The transport owns what the platform adds around the task node
itself and may add entries to ``parameters``:

- Gazebo adds the single ``ros_gz_bridge`` the backend needs (the set_pose
  service bridge plus the dynamic pose stream remapped to the scene's
  ``/sim/<scene>/object_poses`` — the spike T0 choice; the bridge is the
  only transport, there is no CLI fallback) and the ``world_name``
  parameter (the gz world instance of the started world).
- MuJoCo adds no transport: the runtime provider configures the
  FreeJointStatePublisherPlugin for the scene's dynamic bodies and the
  backend places through ``ResetWorld`` state overrides (issue #195).
"""

from __future__ import annotations

from typing import Any

from launch_ros.actions import Node


def gazebo_scene_task_transport(scene: str, startup, robot_config: dict, parameters: dict[str, Any]) -> list:
    """The Gazebo scene-task transport: the object bridge plus ``world_name``.

    ``startup`` is the :class:`SimulationStartup` the world bootstrap
    resolved; its ``world`` (the gz world instance name) names both the
    bridged set_pose service and the dynamic pose stream, and is written
    into ``parameters`` for the task node.
    """
    world_name = (getattr(startup, "world", "") or "").strip() or str(robot_config.get("gazebo_world_name", "demo"))
    parameters["world_name"] = world_name
    # One bridge carries both channels (D8): the set_pose service bridge
    # the backend places objects through, and the dynamic pose stream the
    # evaluation reads, remapped to the scene-namespaced topic.
    return [
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="scene_task_object_bridge",
            arguments=[
                f"/world/{world_name}/set_pose@ros_gz_interfaces/srv/SetEntityPose",
                f"/world/{world_name}/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
            ],
            remappings=[(f"/world/{world_name}/dynamic_pose/info", f"/sim/{scene}/object_poses")],
            output="screen",
        )
    ]


def mujoco_scene_task_transport(scene: str, startup, robot_config: dict, parameters: dict[str, Any]) -> list:
    """The MuJoCo scene-task transport: none (the provider owns both ends).

    The runtime provider configures the FreeJointStatePublisherPlugin for
    the scene's dynamic bodies (issue #195) and the backend places through
    ``ResetWorld`` state overrides — both endpoints resolve the shared
    topic/layout themselves, so the transport contributes no action and no
    parameter.
    """
    return []
