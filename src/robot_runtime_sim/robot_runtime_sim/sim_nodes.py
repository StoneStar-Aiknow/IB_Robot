"""Simulation-only launch builders shared by every platform embodiment (D6).

``simulated_joint_current_actions`` moved from ``robot_runtime.launch_support``
together with its executable ``joint_current_publisher`` (node ownership:
a builder moves with its node). Both the Gazebo and the MuJoCo embodiment
compose it, so it lives in this shared module rather than in either of them.
"""

from __future__ import annotations

from typing import Any

from launch_ros.actions import Node


def simulated_joint_current_actions(profile: dict[str, Any], sim_platform: str = "gazebo") -> list:
    """The effort→JointCurrent republisher for a physics platform (D6b).

    Started only when the profile declares a ``joint.current`` interface whose
    runtime endpoint and joint names parameterize the robot-agnostic node; the
    effort→current scale comes from ``simulation.<sim_platform>.joint_current.scale``
    (default 1 per joint, matching the hardware adapter's effort-as-current
    semantics).
    """
    interface = (profile.get("interfaces") or {}).get("joint.current")
    if not interface or interface.get("message_type") != "ibrobot_msgs/msg/JointCurrent":
        return []
    joint_names = [str(name) for name in interface.get("joint_names") or []]
    if not joint_names:
        return []
    scale_section = (profile.get("simulation") or {}).get(sim_platform) or {}
    scale = scale_section.get("joint_current", {}).get("scale")
    if not isinstance(scale, list) or len(scale) != len(joint_names):
        scale = [1.0] * len(joint_names)
    return [
        Node(
            package="robot_runtime_sim",
            executable="joint_current_publisher",
            name="simulated_joint_current",
            output="screen",
            parameters=[
                {
                    "joint_names": joint_names,
                    "endpoint": str(interface.get("endpoint", "/joint_currents")),
                    "scale": [float(value) for value in scale],
                    "joint_states_topic": str(profile.get("joint_state_topic", "/joint_states")),
                    # physics platforms run every provider node on sim time
                    "use_sim_time": True,
                }
            ],
        )
    ]
