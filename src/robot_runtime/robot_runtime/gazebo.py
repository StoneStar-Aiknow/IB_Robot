"""Robot-agnostic Gazebo sensor helpers shared by runtime providers and generic launch.

Moved from ``robot_config`` (openspec change ``so101-gazebo-runtime-transport``
D6.1): the camera-sensor URDF fragment builder, the gz camera bridge entries
and the override-aware pose resolver are robot-agnostic algorithms, so they
live in the contract layer and ``robot_config`` re-imports them — provider-less
generic Gazebo robots keep their existing camera topics, frames and poses
through the same call sites.

The functions here never import ``robot_config``; robot-specific values (the
generic preset table) are injected by the caller through ``default_pose_loader``.

Extensions for the provider path (D6, spec "Simulated cameras honour the
advertised camera contract") are opt-in so the re-imported generic behaviour
is unchanged:

- ``frame_contract=True`` adds the optical-frame link/joint and stamps the gz
  sensor with ``<gz_frame_id>`` so the published ``header.frame_id`` equals the
  advertised frame (gz Fortress otherwise publishes the scoped
  ``model/link/sensor`` name).
- ``full_streams=True`` bridges every stream the descriptor advertises
  (color, color info, native depth, aligned depth and their infos) instead of
  the legacy color + aligned-depth subset.
- ``spawn_pose`` carries the full spawn transform (x, y, z, yaw); world-parented
  sensors are placed by the inverse of that transform. With yaw 0 this
  degenerates to the historical xyz subtraction.

Pose resolution precedence (D6.3): user-saved override (``~/.ros/ibrobot/
sim_camera_overrides/<name>.yaml``, written by the sim_models camera adjuster)
> declared per-platform default (``use_default_transform`` preset via the
caller's loader, or the peripheral's ``simulation.<platform>.transform``) >
the plain ``transform`` field.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import yaml


def _override_base() -> Path:
    """The user override directory (resolved lazily so tests can redirect HOME)."""
    return Path.home() / ".ros" / "ibrobot" / "sim_camera_overrides"


# Standard body→optical rotation (x right, y down, z forward) as extrinsic
# XYZ rpy; identical to the static TF the physical driver path publishes.
_OPTICAL_RPY = (-math.pi / 2, 0.0, -math.pi / 2)


def _parse_bool(value, default: bool = False) -> bool:
    """Parse YAML-bool-ish values (mirrors ``robot_config.utils.parse_bool``)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off"):
        return False
    return default


def camera_requests_depth(periph: dict) -> bool:
    """Whether a camera peripheral requests any depth-capable stream.

    Mirrors ``robot_config.peripherals.camera.camera_dict_requests_depth``
    exactly so the shared builders keep the generic path's behaviour.
    """
    streams = periph.get("streams") or []
    if isinstance(streams, str):
        streams = [streams]
    stream_names = {str(stream).strip().lower() for stream in streams}
    return (
        "depth" in stream_names
        or "pointcloud" in stream_names
        or _parse_bool(periph.get("align_depth", False), default=False)
        or _parse_bool(periph.get("enable_depth", False), default=False)
        or _parse_bool(periph.get("enable_pointcloud", False), default=False)
    )


def load_camera_override(camera_name: str, *, fallback: dict | None = None) -> dict | None:
    """Load a user-saved Gazebo calibration override for one camera.

    Returns None when the override file is absent, unreadable, or still a
    placeholder stub; callers distinguish user-saved calibration from defaults.
    The file format is unchanged from the sim_models camera adjuster.
    ``fallback`` supplies the parent_frame/fovy defaults for override files
    that omit them (the generic path passes its preset table entry).
    """
    try:
        override_path = _override_base() / f"{camera_name}.yaml"
        if not override_path.exists():
            return None
        data = yaml.safe_load(override_path.read_text(encoding="utf-8")) or {}
        pose = data.get("pose", {}) or {}
        # Skip stub overrides (all-zero placeholder poses would silently break
        # the simulation); mirrors sim_camera_adjuster's handling.
        pose_keys = ("x", "y", "z", "roll", "pitch", "yaw")
        if bool(data.get("_is_stub")) or all(float(pose.get(k, 0.0)) == 0.0 for k in pose_keys):
            return None
        default = fallback or {}
        return {
            "parent_frame": data.get("parent_frame", default.get("parent_frame", "base")),
            "x": float(pose.get("x", 0.0)),
            "y": float(pose.get("y", 0.0)),
            "z": float(pose.get("z", 0.0)),
            "roll": float(pose.get("roll", 0.0)),
            "pitch": float(pose.get("pitch", 0.0)),
            "yaw": float(pose.get("yaw", 0.0)),
            "fovy": float(data.get("fovy_deg", default.get("fovy", 60))),
        }
    except Exception:
        return None


def resolve_camera_pose(periph: dict, platform: str = "gazebo", *, default_pose_loader=None) -> tuple[dict, Any]:
    """Resolve a camera's effective (transform, fovy) for a simulation platform.

    Precedence: user override > declared default > explicit transform. The
    declared default is either the caller's preset table (``use_default_transform``,
    the generic provider-less mechanism) or the peripheral's own
    ``simulation.<platform>.transform`` (+ ``fovy``), the runtime-profile
    mechanism. Overrides apply in Gazebo convention and only on the gazebo
    platform, exactly as before this change.
    """
    name = periph.get("name", "camera")
    transform = dict(periph.get("transform") or {})
    fovy = periph.get("fovy", 60)
    default: dict | None = None
    if periph.get("use_default_transform", False) and default_pose_loader is not None:
        default = default_pose_loader(platform, name)
    if default is None:
        section = (periph.get("simulation") or {}).get(platform) or {}
        declared = section.get("transform") if isinstance(section, dict) else None
        if isinstance(declared, dict) and declared:
            default = dict(declared)
            if section.get("fovy") is not None:
                default["fovy"] = section["fovy"]
    if default is not None:
        override = load_camera_override(name) if platform == "gazebo" else None
        if override is not None:
            return override, override.get("fovy", 60)
        transform = {key: default.get(key, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
        transform["parent_frame"] = default.get("parent_frame", "base")
        fovy = default.get("fovy", fovy)
    return transform, fovy


def _rotate_yaw_inverse(pose: dict, spawn_pose: tuple[float, float, float, float]) -> dict:
    """Place a world-parented sensor under the model-local URDF ``world`` link.

    The URDF ``world`` link sits at the robot's spawn pose in the simulator,
    not at the true world origin, so a camera calibrated in world coordinates
    must be transformed by the inverse of the full spawn transform
    (translation and yaw); the historical xyz subtraction is the yaw-0 case.
    """
    x, y, z, yaw = (float(value) for value in spawn_pose)
    if yaw == 0.0:
        return {**pose, "x": pose["x"] - x, "y": pose["y"] - y, "z": pose["z"] - z}
    cos, sin = math.cos(-yaw), math.sin(-yaw)
    dx, dy, dz = pose["x"] - x, pose["y"] - y, pose["z"] - z
    return {
        **pose,
        "x": cos * dx - sin * dy,
        "y": sin * dx + cos * dy,
        "z": dz,
        "yaw": pose["yaw"] - yaw,
    }


def advertised_camera_frame(periph: dict) -> str:
    """The frame id the runtime description advertises for this camera's images."""
    name = periph.get("name", "camera")
    if str(periph.get("driver", "opencv")) == "realsense":
        return str(periph.get("optical_frame_id") or f"camera_{name}_optical_frame")
    return str(periph.get("frame_id") or f"camera_{name}_frame")


def camera_sensor_urdf(
    peripherals: list,
    *,
    platform: str = "gazebo",
    spawn_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    default_pose_loader=None,
    frame_contract: bool = False,
) -> str:
    """Camera link/joint/<gazebo><sensor> XML for a peripheral list.

    Naming (unchanged from the generic path): URDF link = the YAML
    ``frame_id``, gz sensor = ``{name}_camera``, gz color topic =
    ``/{name}_camera/image`` (``rgbd_camera`` uses the bare sensor name and gz
    appends the stream suffixes). ``frame_contract=True`` additionally adds the
    optical link/joint and stamps ``<gz_frame_id>`` with the advertised frame.
    """
    parts: list[str] = []
    for periph in peripherals:
        if periph.get("type") != "camera":
            continue
        if periph.get("disabled", False) or periph.get("enabled") is False:
            continue
        if (periph.get("simulation") or {}).get("embedded_sensor", False):
            continue
        if periph.get("skip_urdf_without_transform", False) and not periph.get("transform"):
            continue
        name = periph["name"]
        frame_id = periph.get("frame_id", f"camera_{name}_frame")
        transform, cam_fovy = resolve_camera_pose(periph, platform, default_pose_loader=default_pose_loader)
        parent = transform.get("parent_frame", "world")
        pose = {key: float(transform.get(key, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
        # The URDF "world" link is model-local (it sits at the spawn pose in
        # the simulator): re-anchor world-calibrated cameras by the inverse of
        # the full spawn transform.
        if parent == "world":
            pose = _rotate_yaw_inverse(pose, spawn_pose)

        width = periph.get("width", 640)
        height = periph.get("height", 480)
        fps = periph.get("fps", 30)
        sensor_name = f"{name}_camera"
        has_depth = camera_requests_depth(periph)
        sensor_type = "rgbd_camera" if has_depth else "camera"
        topic_name = sensor_name if has_depth else f"{sensor_name}/image"

        # fovy (vertical, degrees) → horizontal_fov (radians)
        fovy_rad = math.radians(cam_fovy)
        hfov_rad = 2.0 * math.atan(math.tan(fovy_rad / 2.0) * width / height)

        frame_id_element = (
            f"\n            <gz_frame_id>{escape(advertised_camera_frame(periph))}</gz_frame_id>"
            if frame_contract
            else ""
        )
        parts.append(
            f"""
    <!-- {name} camera (generated from YAML peripherals) -->
    <link name="{frame_id}">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="0.01"/>
            <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
        </inertial>
    </link>
    <joint name="{name}_camera_joint" type="fixed">
        <parent link="{parent}"/>
        <child  link="{frame_id}"/>
        <origin xyz="{pose["x"]} {pose["y"]} {pose["z"]}" rpy="{pose["roll"]} {pose["pitch"]} {pose["yaw"]}"/>
    </joint>"""
        )
        if frame_contract:
            optical_frame = str(periph.get("optical_frame_id") or f"camera_{name}_optical_frame")
            parts.append(
                f"""
    <link name="{optical_frame}">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="0.001"/>
            <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
        </inertial>
    </link>
    <joint name="{name}_optical_joint" type="fixed">
        <parent link="{frame_id}"/>
        <child  link="{optical_frame}"/>
        <origin xyz="0 0 0" rpy="{_OPTICAL_RPY[0]} {_OPTICAL_RPY[1]} {_OPTICAL_RPY[2]}"/>
    </joint>"""
            )
        parts.append(
            f"""
    <gazebo reference="{frame_id}">
        <sensor type="{sensor_type}" name="{sensor_name}">{frame_id_element}
            <update_rate>{fps}</update_rate>
            <camera>
                <horizontal_fov>{hfov_rad:.4f}</horizontal_fov>
                <image>
                    <width>{width}</width>
                    <height>{height}</height>
                    <format>R8G8B8</format>
                </image>
                <clip><near>0.01</near><far>100</far></clip>
            </camera>
            <always_on>true</always_on>
            <visualize>true</visualize>
            <topic>{topic_name}</topic>
        </sensor>
    </gazebo>"""
        )
    return "\n".join(parts)


def camera_bridge_entries(periph: dict, *, full_streams: bool = False) -> list[dict]:
    """ros_gz bridge config entries for one camera peripheral.

    The ROS contract topics are unchanged (``/camera/{name}/image_raw`` etc.);
    gz topics are absolute sensor topics (``/{name}_camera/...``). With
    ``full_streams`` the entries cover every stream the runtime descriptor
    advertises: color, color info, native depth, aligned depth and their
    camera infos (gz ``rgbd_camera`` serves native and aligned depth from the
    same ``depth_image`` topic).
    """
    name = periph["name"]
    sensor_name = f"{name}_camera"
    gz_base = f"/{sensor_name}"

    def entry(ros_topic: str, gz_topic: str, ros_type: str, gz_type: str) -> dict:
        return {
            "ros_topic_name": ros_topic,
            "gz_topic_name": gz_topic,
            "ros_type_name": ros_type,
            "gz_type_name": gz_type,
            "direction": "GZ_TO_ROS",
            "lazy": False,
        }

    entries = [
        entry(f"/camera/{name}/image_raw", f"{gz_base}/image", "sensor_msgs/msg/Image", "gz.msgs.Image"),
        entry(
            f"/camera/{name}/camera_info", f"{gz_base}/camera_info", "sensor_msgs/msg/CameraInfo", "gz.msgs.CameraInfo"
        ),
    ]
    if not full_streams:
        # Legacy generic subset: aligned depth only, when any depth is requested.
        if camera_requests_depth(periph):
            entries.append(
                entry(
                    f"/camera/{name}/aligned_depth_to_color/image_raw",
                    f"{gz_base}/depth_image",
                    "sensor_msgs/msg/Image",
                    "gz.msgs.Image",
                )
            )
        return entries
    # Full contract: every stream the runtime descriptor advertises (the
    # descriptor's stream rules, not the looser depth-detection heuristic).
    # gz rgbd_camera serves native and aligned depth from the same
    # depth_image topic; it publishes no depth camera_info (the depth image
    # shares the color intrinsics, bridged as the color camera_info) and no
    # color-aligned pointcloud, which the descriptor drops on this transport.
    from robot_runtime.interface_description import camera_streams

    streams = camera_streams(periph)
    if "depth" not in streams:
        return entries
    entries.append(
        entry(
            f"/camera/{name}/depth/image_rect_raw", f"{gz_base}/depth_image", "sensor_msgs/msg/Image", "gz.msgs.Image"
        )
    )
    if periph.get("align_depth") and "color" in streams:
        entries.append(
            entry(
                f"/camera/{name}/aligned_depth_to_color/image_raw",
                f"{gz_base}/depth_image",
                "sensor_msgs/msg/Image",
                "gz.msgs.Image",
            )
        )
    return entries


def bridge_config_node(entries: list[dict], config_path: str | Path):
    """One ros_gz ``bridge_node`` reading a YAML config covering all entries."""
    from launch_ros.actions import Node

    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.dump(entries, stream, default_flow_style=False)
    return Node(
        package="ros_gz_bridge",
        executable="bridge_node",
        name="peripheral_bridge",
        parameters=[{"config_file": str(path)}],
        output="screen",
    )
