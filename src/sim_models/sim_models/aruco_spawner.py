"""
aruco_spawner.py — Gazebo dynamic spawn/despawn of the ArUco A4 calibration paper.

Public API (Phase 1):

    spawn_aruco_gazebo(pose_xyz=(0, 0, 0), world_name="demo") -> str
        Spawn the A4 paper into a running Gazebo simulation.
        Returns the model name (needed for despawn).

    despawn_aruco_gazebo(model_name, world_name="demo")
        Remove the model from Gazebo.

    verify_mujoco_model() -> bool
        Offline validation: load the MJCF fragment via mujoco.MjModel.from_xml_string.
        Does not require a running simulation.

Notes:
    - Gazebo spawn uses `ign service -s /world/{world}/create` with an SDF file.
      The SDF contains a file:// URI built from the ament_index-resolved install path.
    - MuJoCo runtime control (geom_rgba / mocap_pos) requires a custom bridge plugin;
      that is deferred to Phase 3. Phase 1 only validates file correctness offline.
    - Default pose (0, 0, 0) places the paper at the world origin; adjust after
      successful import using Phase 3 trackbar controls.
"""

import os
import subprocess

from ament_index_python.packages import get_package_share_directory

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PKG = "sim_models"
_PROP_REL = "props/calib/aruco_a4"

# Model name used inside Gazebo; also returned by spawn so the caller can despawn.
_MODEL_NAME = "aruco_a4"


def _get_prop_dir() -> str:
    """Return the absolute path to the installed aruco_a4 prop directory."""
    share = get_package_share_directory(_PKG)
    return os.path.join(share, _PROP_REL)


def _get_meshes_dir() -> str:
    share = get_package_share_directory(_PKG)
    return os.path.join(share, _PROP_REL, "meshes")


def _build_sdf(pose_xyz: tuple) -> str:
    """
    Read gazebo/model.sdf, substitute {{MESH_PATH}}, return final SDF string.
    """
    share = get_package_share_directory(_PKG)
    sdf_template = os.path.join(share, _PROP_REL, "gazebo", "model.sdf")
    with open(sdf_template) as f:
        content = f.read()
    mesh_path = os.path.join(share, _PROP_REL, "meshes")
    content = content.replace("{{MESH_PATH}}", mesh_path)
    return content


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def spawn_aruco_gazebo(
    pose_xyz: tuple = (-0.08, 0.34, 0.0),
    world_name: str = "demo",
) -> str:
    """
    Spawn the ArUco A4 paper into a running Gazebo simulation.

    Parameters
    ----------
    pose_xyz : (x, y, z) in metres; default (-0.08, 0.34, 0.0).
    world_name : Gazebo world name (check `ign service -l` for active world).

    Returns
    -------
    str : The model name spawned (use this to despawn).

    Raises
    ------
    subprocess.CalledProcessError : If `ign service` returns non-zero.
    FileNotFoundError : If `ign` binary is not found.
    """
    x, y, z = float(pose_xyz[0]), float(pose_xyz[1]), float(pose_xyz[2])
    sdf_content = _build_sdf(pose_xyz)

    # Use inline SDF (sdf: field) rather than sdf_filename. Ignition Gazebo's
    # UserCommands plugin queues the spawn command and replies immediately; the
    # actual file read happens in the next PreUpdate step, after subprocess.run
    # has already returned. A temp file would be deleted before Gazebo reads it.
    # The escaping here covers all proto text-format special chars in SDF output:
    # backslash, double-quote, and newlines. Carriage-returns are stripped.
    sdf_escaped = sdf_content.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "")

    req = (
        f'sdf: "{sdf_escaped}" '
        f'name: "{_MODEL_NAME}" '
        f"allow_renaming: true "
        f"pose {{ position {{ x: {x} y: {y} z: {z} }} }}"
    )
    subprocess.run(
        [
            "ign",
            "service",
            "-s",
            f"/world/{world_name}/create",
            "--reqtype",
            "ignition.msgs.EntityFactory",
            "--reptype",
            "ignition.msgs.Boolean",
            "--timeout",
            "5000",
            "-r",
            req,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    return _MODEL_NAME


def despawn_aruco_gazebo(
    model_name: str = _MODEL_NAME,
    world_name: str = "demo",
) -> None:
    """
    Remove the ArUco A4 model from a running Gazebo simulation.

    Parameters
    ----------
    model_name : Name returned by spawn_aruco_gazebo.
    world_name : Gazebo world name.

    Notes
    -----
    If the entity does not exist, the call is silently ignored (idempotent).
    Gazebo may still log a server-side [Err] for "entity not found"; this is
    expected and harmless.
    """
    # Entity.Type MODEL = 2
    req = f'name: "{model_name}" type: 2'
    result = subprocess.run(
        [
            "ign",
            "service",
            "-s",
            f"/world/{world_name}/remove",
            "--reqtype",
            "ignition.msgs.Entity",
            "--reptype",
            "ignition.msgs.Boolean",
            "--timeout",
            "5000",
            "-r",
            req,
        ],
        capture_output=True,
        text=True,
    )
    # ign service returns 0 even when the entity is not found; Gazebo logs
    # a server-side [Err]. Non-zero exit means a transport-level failure.
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)


# ---------------------------------------------------------------------------
# Phase 3: Alignment proxy camera — spawn / move / despawn
# ---------------------------------------------------------------------------


def _camera_name_from_topic(topic: str) -> str:
    """Extract camera name from a full ROS topic.

    Examples
    --------
    /camera_align/top/image_raw  -> top
    /camera/wrist/image_raw      -> wrist
    """
    parts = topic.strip("/").split("/")
    # Expect at least 2 path segments; the camera name is always index 1.
    if len(parts) >= 2:
        return parts[1]
    return topic.replace("/", "_")


def _build_proxy_camera_sdf(
    model_name: str,
    topic_base: str,
    fovy_rad: float,
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
) -> str:
    """Build an inline SDF string for a static proxy camera model.

    ``fovy_rad`` is the **vertical** FOV in radians.  Gazebo's SDF expects
    ``<horizontal_fov>``, so we convert here using the same formula as
    ``description.py`` so the proxy (adjuster preview) and the production
    camera (URDF-injected) always render the same scene.
    """
    import math as _math

    hfov_rad = 2.0 * _math.atan(_math.tan(fovy_rad / 2.0) * width / height)
    return f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{model_name}">
    <static>true</static>
    <link name="link">
      <sensor name="camera" type="camera">
        <camera>
          <horizontal_fov>{hfov_rad:.6f}</horizontal_fov>
          <image>
            <width>{width}</width>
            <height>{height}</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.01</near>
            <far>100</far>
          </clip>
        </camera>
        <always_on>true</always_on>
        <update_rate>{float(fps):.6f}</update_rate>
        <topic>{topic_base}</topic>
        <visualize>false</visualize>
      </sensor>
    </link>
  </model>
</sdf>"""


def spawn_alignment_camera_gazebo(
    topic: str,
    pose_xyz: tuple,
    pose_rpy: tuple,
    fovy_deg: float = 60.0,
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
    world_name: str = "demo",
) -> str:
    """Spawn an alignment-only proxy camera in Gazebo.

    Parameters
    ----------
    topic : Full ROS topic for the proxy camera image stream,
            e.g. ``/camera_align/top/image_raw``.
            The camera name is extracted from this string.
    pose_xyz : (x, y, z) in metres — initial position in world frame.
    pose_rpy : (roll, pitch, yaw) in radians — initial orientation.
    fovy_deg : Horizontal field-of-view in degrees.
    world_name : Gazebo world name.

    Returns
    -------
    str : The Gazebo model name (use this to despawn).
    """
    import math

    cam_name = _camera_name_from_topic(topic)
    model_name = f"alignment_cam_{cam_name}"

    # Use the full topic name (including /image_raw) as the Gazebo sensor topic
    # so ros_gz_bridge maps it to the exact same ROS2 topic path.
    fovy_rad = math.radians(fovy_deg)
    sdf_content = _build_proxy_camera_sdf(
        model_name,
        topic,
        fovy_rad,
        width=width,
        height=height,
        fps=fps,
    )

    x, y, z = float(pose_xyz[0]), float(pose_xyz[1]), float(pose_xyz[2])
    roll, pitch, yaw = float(pose_rpy[0]), float(pose_rpy[1]), float(pose_rpy[2])

    # Convert RPY to quaternion for the spawn pose
    import math as _math

    cr, cp, cy = _math.cos(roll / 2), _math.cos(pitch / 2), _math.cos(yaw / 2)
    sr, sp, sy = _math.sin(roll / 2), _math.sin(pitch / 2), _math.sin(yaw / 2)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    # See spawn_aruco_gazebo for why we use inline sdf: instead of sdf_filename.
    sdf_escaped = sdf_content.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "")

    req = (
        f'sdf: "{sdf_escaped}" '
        f'name: "{model_name}" '
        f"allow_renaming: false "
        f"pose {{ "
        f"  position {{ x: {x} y: {y} z: {z} }} "
        f"  orientation {{ x: {qx} y: {qy} z: {qz} w: {qw} }} "
        f"}}"
    )
    subprocess.run(
        [
            "ign",
            "service",
            "-s",
            f"/world/{world_name}/create",
            "--reqtype",
            "ignition.msgs.EntityFactory",
            "--reptype",
            "ignition.msgs.Boolean",
            "--timeout",
            "5000",
            "-r",
            req,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    print(f"[aruco_spawner] Proxy camera spawned: {model_name} at {pose_xyz} ({width}x{height}@{float(fps):g})")
    return model_name


def set_alignment_camera_pose_gazebo(
    topic: str,
    pose_xyz: tuple,
    pose_rpy: tuple,
    world_name: str = "demo",
) -> tuple[bool, str | None]:
    """Move the proxy camera to a new pose via gz service.

    Parameters
    ----------
    topic : Full ROS topic, used to derive the model name.
    pose_xyz : (x, y, z) in metres.
    pose_rpy : (roll, pitch, yaw) in radians.
    world_name : Gazebo world name.
    """
    import math

    cam_name = _camera_name_from_topic(topic)
    model_name = f"alignment_cam_{cam_name}"

    roll, pitch, yaw = pose_rpy
    # Convert RPY to quaternion for gz service
    cr, cp, cy = math.cos(roll / 2), math.cos(pitch / 2), math.cos(yaw / 2)
    sr, sp, sy = math.sin(roll / 2), math.sin(pitch / 2), math.sin(yaw / 2)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    x, y, z = float(pose_xyz[0]), float(pose_xyz[1]), float(pose_xyz[2])

    # ignition.msgs.Pose has flat fields: name, position, orientation
    # (no extra 'pose {}' wrapper — that's only in EntityFactory)
    req = f'name: "{model_name}" position {{ x: {x} y: {y} z: {z} }} orientation {{ x: {qx} y: {qy} z: {qz} w: {qw} }}'

    result = subprocess.run(
        [
            "ign",
            "service",
            "-s",
            f"/world/{world_name}/set_pose",
            "--reqtype",
            "ignition.msgs.Pose",
            "--reptype",
            "ignition.msgs.Boolean",
            "--timeout",
            "500",
            "-r",
            req,
        ],
        check=False,  # Non-fatal: pose update can be retried next trackbar tick
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True, None

    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    message = stderr or stdout or f"ign service exited with code {result.returncode}"
    return False, message


def despawn_alignment_camera_gazebo(
    topic: str,
    world_name: str = "demo",
) -> None:
    """Remove the proxy camera model from Gazebo.

    Parameters
    ----------
    topic : Full ROS topic, used to derive the model name.
    world_name : Gazebo world name.
    """
    cam_name = _camera_name_from_topic(topic)
    model_name = f"alignment_cam_{cam_name}"

    req = f'name: "{model_name}" type: 2'
    subprocess.run(
        [
            "ign",
            "service",
            "-s",
            f"/world/{world_name}/remove",
            "--reqtype",
            "ignition.msgs.Entity",
            "--reptype",
            "ignition.msgs.Boolean",
            "--timeout",
            "5000",
            "-r",
            req,
        ],
        capture_output=True,
        text=True,
    )
    print(f"[aruco_spawner] Proxy camera despawned: {model_name}")


def verify_mujoco_model() -> bool:
    """
    Offline validation: load the aruco_a4 MJCF fragment via MuJoCo.

    Wraps the fragment in a minimal standalone <mujoco> document,
    substitutes the absolute meshes path, and calls MjModel.from_xml_string.
    On success also writes /tmp/aruco_a4_verify.xml for manual viewer check.

    Returns True on success, False on failure (also prints the error).
    """
    try:
        import mujoco  # noqa: PLC0415 — optional dependency
    except ImportError:
        print("[aruco_spawner] ERROR: mujoco Python package not found.")
        return False

    meshes_dir = _get_meshes_dir()

    # Build a complete standalone MJCF document for validation.
    # Includes XYZ axis sites (spheres) at origin so you can see the coordinate
    # frame in the viewer. Press 'F' in the MuJoCo viewer to toggle body frames.
    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="aruco_a4_verify">
  <compiler meshdir="{meshes_dir}" texturedir="{meshes_dir}"/>

  <visual>
    <!-- Make sites visible by default -->
    <scale framewidth="0.005" framelength="0.1"/>
  </visual>

  <asset>
    <texture name="a4_tex" type="2d" file="a4_tex.png"/>
    <material name="a4_mat"
              texture="a4_tex"
              texrepeat="1 1"
              texuniform="true"
              shininess="0.05"
              specular="0.05"
              reflectance="0.0"/>
    <mesh name="a4_paper" file="a4_paper.obj" scale="0.001 0.001 0.001" inertia="shell"/>
  </asset>

  <worldbody>
    <light name="top_light" pos="0.105 0.5 0.15" dir="0 -1 0"
           diffuse="1 1 1" specular="0 0 0"/>
    <light name="bot_light" pos="0.105 -0.2 0.5" dir="0 0.5 -1"
           diffuse="0.6 0.6 0.6" specular="0 0 0"/>

    <!-- Coordinate axis markers at world origin (visible as coloured spheres).
         Red=X  Green=Y  Blue=Z  — each 10 cm from origin. -->
    <site name="origin"  pos="0 0 0"    size="0.012" rgba="1 1 1 1"/>
    <site name="axis_x"  pos="0.1 0 0"  size="0.008" rgba="1 0 0 1"/>
    <site name="axis_y"  pos="0 0.1 0"  size="0.008" rgba="0 1 0 1"/>
    <site name="axis_z"  pos="0 0 0.1"  size="0.008" rgba="0 0 1 1"/>

    <!-- euler="90 0 0": rotate 90° CCW around X axis so Z points up (out of paper face),
         matching Gazebo convention. Gazebo uses box geometry and is unaffected. -->
    <body name="aruco_a4" pos="0 0 0" euler="90 0 0">
      <geom name="aruco_a4_visual"
            type="mesh"
            mesh="a4_paper"
            material="a4_mat"
            contype="0"
            conaffinity="0"
            mass="0"/>
    </body>
  </worldbody>
</mujoco>
"""

    verify_path = "/tmp/aruco_a4_verify.xml"
    try:
        with open(verify_path, "w") as f:
            f.write(xml)
        model = mujoco.MjModel.from_xml_string(xml)
        print(f"[aruco_spawner] MuJoCo model OK — ngeom={model.ngeom}, nmesh={model.nmesh}, ntex={model.ntex}")
        print(f"[aruco_spawner] Verify XML written to {verify_path}")
        print(f"[aruco_spawner] Visual check: python3 -m mujoco.viewer --mjcf {verify_path}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[aruco_spawner] ERROR loading MuJoCo model: {exc}")
        return False
