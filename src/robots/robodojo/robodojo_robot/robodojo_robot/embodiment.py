"""The ``robodojo`` platform's embodiment: the direct-control runtime with the RoboDojo adapter.

A ``SimEmbodiment`` (robot_runtime.sim_platforms) that needs no ros2_control: the robot's body
and its controllers live in RoboDojo's Isaac process, and IB-Robot's side is one process, the
``robot_runtime_sim`` direct-control runtime, which writes every tick's targets itself and
hosts RoboDojo's policy server through ``robodojo_robot.adapter``. That adapter reference is
the only place it is named: the adapter belongs to this platform, not to a registry of its own.

The runtime node is the simulated clock's source (it publishes ``/clock`` from RoboDojo's
ticks), so it runs on wall time itself; every consumer of the robot uses sim time.
"""

from __future__ import annotations

import json
from typing import Any

#: ``module:attr`` of the adapter factory the runtime instantiates.
ADAPTER = "robodojo_robot.adapter:create_adapter"
PLATFORM = "robodojo"
NODE_NAME = "robodojo_runtime"  # the historical node name, kept so logs and tools still match


class RoboDojoEmbodiment:
    """Description and launch actions of the RoboDojo X5 on the ``robodojo`` platform."""

    def render_description(
        self,
        profile: dict,
        profile_path: Any,
        *,
        peripherals: list | None = None,
        spawn_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        world: str = "",
    ) -> str:
        # No public model: RoboDojo owns the body (robot_config runtime.require_model: false).
        return ""

    def actions(
        self,
        profile: dict,
        profile_path: Any,
        peripherals_file: str = "",
        instance_id: str = "",
        *,
        robot_description: str | None = None,
        interface_description: dict | None = None,
        world: str = "",
        spawn_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    ) -> list:
        from launch_ros.actions import Node

        if peripherals_file:
            raise RuntimeError("the robodojo platform publishes RoboDojo's own cameras and accepts no peripherals")
        return [
            Node(
                package="robot_runtime_sim",
                executable="direct_control_runtime",
                name=NODE_NAME,
                output="screen",
                parameters=[runtime_parameters(profile, profile_path, instance_id)],
            )
        ]


def runtime_parameters(profile: dict, profile_path: Any, instance_id: str = "") -> dict:
    """The runtime node's parameters for ``profile`` (its adapter_config as the adapter overrides).

    A launch that changes the adapter configuration in memory (``policy_port:=``) passes the
    changed profile; the node re-reads the file and merges these overrides on top of it.
    """
    section = (profile.get("simulation") or {}).get(PLATFORM) or {}
    return {
        "profile": str(profile_path),
        "instance_id": str(instance_id or "").strip(),
        "platform": PLATFORM,
        "adapter": ADAPTER,
        "adapter_overrides_json": json.dumps(dict(section.get("adapter_config") or {})),
        "use_sim_time": False,
    }


ROBODOJO_EMBODIMENT = RoboDojoEmbodiment()
