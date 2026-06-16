"""Virtual tool-frame static TF publishers for user-defined kinematics frames.

Translates the optional ``kinematics.frames`` SSOT YAML section into a list of
``tf2_ros/static_transform_publisher`` nodes — without touching URDF.

The robot_description package is currently ament_cmake (resource-only), so the
helper lives here in robot_config.launch_builders to reuse the existing launch
orchestration plumbing.

YAML format::

    kinematics:
      frames:
        tool0:
          parent: gripper
          xyz: [0.10, 0.0, 0.0]
          rpy: [0.0, 0.0, 0.0]
        tool1:
          parent: tool0
          xyz: [0.0, 0.0, 0.05]
          rpy: [0.0, 0.0, 0.0]
"""

from __future__ import annotations

from launch_ros.actions import Node

from robot_config.logger_utils import get_colored_logger

logger = get_colored_logger("robot_config.virtual_frames")


def generate_virtual_frame_nodes(
    frames: dict[str, dict] | None,
    use_sim: bool = False,
) -> list[Node]:
    """Generate static TF publisher nodes for user-defined virtual tool frames.

    Args:
        frames: Mapping ``{name: {parent, xyz, rpy}}``. ``None`` or empty
            returns an empty list (zero-regression default).
        use_sim: Currently unused — virtual frames are static and must publish
            in both sim and real launches.

    Returns:
        List of ``static_transform_publisher`` ``Node`` actions, one per frame.

    Raises:
        ValueError: When a frame entry is missing required fields or has
            malformed ``xyz`` / ``rpy`` (each must be a 3-tuple).
    """
    if not frames:
        return []

    nodes: list[Node] = []
    for name, spec in frames.items():
        if not isinstance(spec, dict):
            raise ValueError(f"kinematics.frames.{name}: expected dict, got {type(spec).__name__}")
        parent = spec.get("parent")
        xyz = spec.get("xyz", [0.0, 0.0, 0.0])
        rpy = spec.get("rpy", [0.0, 0.0, 0.0])

        if not parent:
            raise ValueError(f"kinematics.frames.{name}: missing 'parent'")
        if len(xyz) != 3:
            raise ValueError(f"kinematics.frames.{name}.xyz must have 3 elements")
        if len(rpy) != 3:
            raise ValueError(f"kinematics.frames.{name}.rpy must have 3 elements")

        nodes.append(
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name=f"static_tf_{name}",
                arguments=[
                    "--x",
                    str(float(xyz[0])),
                    "--y",
                    str(float(xyz[1])),
                    "--z",
                    str(float(xyz[2])),
                    # kinematics.frames.*.rpy is [roll, pitch, yaw] per contract.
                    "--roll",
                    str(float(rpy[0])),
                    "--pitch",
                    str(float(rpy[1])),
                    "--yaw",
                    str(float(rpy[2])),
                    "--frame-id",
                    str(parent),
                    "--child-frame-id",
                    str(name),
                ],
                output="screen",
            )
        )
        logger.info(f"virtual frame: {name} parent={parent} xyz={xyz} rpy={rpy}")
    return nodes


def collect_virtual_frame_names(frames: dict[str, dict] | None) -> list[str]:
    """Return the list of virtual frame names declared under ``kinematics.frames``.

    Used by ``launch_builders.teleop._validate_tool_frame`` to confirm the
    user-supplied ``tool_frame`` is either a URDF link, a declared virtual
    frame, or the base link.
    """
    if not frames:
        return []
    return list(frames.keys())
