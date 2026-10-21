"""Scene compiler for IB-Robot simulation.

Provides runtime path resolution for scene assets using a template
placeholder mechanism.  Template files (.world.template, .xml.template)
contain two placeholders that are substituted at launch time:

  {{MESHES_DIR}}      — absolute path to the scene's meshes/ directory
  {{ROBOT_XML_PATH}}  — absolute path to the robot MJCF XML (MuJoCo only)

The substituted content is written to a temporary directory so that Gazebo
and MuJoCo can reference mesh files by their installed absolute paths.

Public API
----------
  get_scene_file(scene_name, platform) -> Path
      Return the Path to the raw template file (for inspection or tests).

  get_gazebo_world_path(scene_name) -> Path
      Generate the resolved Gazebo world file.
      Used by GazeboAdapter.start_backend() when simulation.scene is set.

  randomization_grid(layout) -> list[(x, y, z, yaw)]
      The single randomisation generator shared by both physics platforms
      (so101-mujoco-runtime-platform D9): randomisation index *k* maps to
      the same banana placement on MuJoCo (keyframe ``random_KKK``) and on
      Gazebo (``SetEntityPose`` with the same index).

  composed_model_path(scene) -> Path
      The single composed-model location (D3), keyed by ``ROS_DOMAIN_ID`` so
      concurrent launches on different domains never overwrite each other's
      model. Used by the provider (writer) and the scene task (reader).

  compose_mujoco_model(robot_mjcf_path, scene, sdf_mesh_names) -> Path
      Robot-agnostic composition (D3): substitute the robot MJCF into the
      named scene (or compile it alone when ``scene`` is None), inject the
      shared keyframes, apply SDF collision to the given meshes, compile and
      save the ``.mjb``. Falls back to the ``.xml`` only when the ``mujoco``
      package is missing; the provider treats that as fatal.

  get_scene_layout(scene_name) -> dict
      The parsed layout.yaml of a scene (or {}).

  get_scene_task(scene_name) -> dict | None
      The scene's ``task:`` declaration (executable + platforms), or None.
"""

import logging
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory

_MESHES_PLACEHOLDER = "{{MESHES_DIR}}"
_ROBOT_XML_PLACEHOLDER = "{{ROBOT_XML_PATH}}"

_PLATFORM_EXT = {
    "gazebo": ".world.template",
    "mujoco": ".xml.template",
}


def randomization_grid(layout: dict) -> list[tuple[float, float, float, float]]:
    """The single randomisation generator shared by both physics platforms (D9).

    Index *k* of the returned list maps to the same banana placement on every
    platform: MuJoCo keyframe ``random_{k:03d}`` and the Gazebo backend's
    ``SetEntityPose`` with the same index. The order is stable: x outer,
    y middle, yaw inner, exactly as the previous keyframe-only generator
    produced, so existing randomisation indices keep their meaning.
    """
    rand = layout.get("randomization", {})
    banana_cfg = rand.get("banana", {})
    if not banana_cfg:
        return []
    x_min, x_max = banana_cfg.get("x_range", [-0.03, 0.10])
    y_min, y_max = banana_cfg.get("y_range", [-0.02, 0.10])
    z_spawn = float(banana_cfg.get("z_spawn", 0.05))
    grid_cols = int(banana_cfg.get("grid_cols", 5))
    grid_rows = int(banana_cfg.get("grid_rows", 7))
    yaw_steps = int(banana_cfg.get("yaw_steps", 6))
    xs = np.linspace(x_min, x_max, grid_cols)
    ys = np.linspace(y_min, y_max, grid_rows)
    yaws = [2.0 * math.pi * i / yaw_steps for i in range(yaw_steps)]
    return [(float(x), float(y), z_spawn, float(yaw)) for x in xs for y in ys for yaw in yaws]


def _quat_from_extrinsic_xyz(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Extrinsic XYZ (URDF rpy) rotation as a MuJoCo w-x-y-z quaternion."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    # q = qz(yaw) @ qy(pitch) @ qx(roll), scalar first
    return (
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    )


def _layout_dynamic_objects(layout: dict) -> list[tuple[str, list[float], list[float]]]:
    """The scene's dynamic objects as (name, position, rotation_euler_rad)."""
    out = []
    for obj in layout.get("objects", []) or []:
        if str(obj.get("type", "")).lower() != "dynamic":
            continue
        out.append(
            (
                str(obj.get("name", "")),
                [float(v) for v in (obj.get("position") or [0.0, 0.0, 0.0])],
                [float(v) for v in (obj.get("rotation_euler_rad") or [0.0, 0.0, 0.0])],
            )
        )
    return out


def _inject_random_keyframes(xml_content: str, layout: dict) -> str:
    """Inject the shared randomisation keyframes plus a named default (D3/D9).

    ``random_KKK`` keyframes come from ``randomization_grid`` (index k → pose);
    the ``default`` keyframe holds the layout's default object poses and
    ``arm_rest_qpos``, so a default reset and the trajectory controller's
    hold setpoint agree. Uses the mujoco Python API to resolve qpos addresses
    by joint name so the ordering is inferred from the model, not hardcoded.
    If mujoco is unavailable or the model cannot compile, returns
    xml_content unchanged.
    """
    try:
        import mujoco  # noqa: PLC0415
    except ImportError:
        logging.warning("mujoco not available; skipping keyframe injection")
        return xml_content

    grid = randomization_grid(layout)
    if not grid:
        return xml_content

    rand = layout.get("randomization", {})
    arm_joint_names = rand.get("arm_joint_names", [])
    arm_rest_qpos = rand.get("arm_rest_qpos", [])
    dynamics = _layout_dynamic_objects(layout)

    try:
        spec = mujoco.MjSpec.from_string(xml_content)
        model = spec.compile()
    except Exception as exc:  # noqa: BLE001
        logging.warning("Keyframe injection: model compile failed (%s); skipping", exc)
        return xml_content

    data = mujoco.MjData(model)

    def _adr(name: str) -> int:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(model.jnt_qposadr[jid]) if jid >= 0 else -1

    def _set_arm(qpos) -> None:
        for jname, qval in zip(arm_joint_names, arm_rest_qpos, strict=False):
            adr = _adr(jname)
            if adr >= 0:
                qpos[adr] = float(qval)

    def _set_freejoint(qpos, name: str, position: list[float], rpy: list[float]) -> None:
        adr = _adr(f"{name}_free")
        if adr < 0:
            return
        qw, qx, qy, qz = _quat_from_extrinsic_xyz(rpy[0], rpy[1], rpy[2])
        qpos[adr : adr + 7] = [position[0], position[1], position[2], qw, qx, qy, qz]

    banana = next((entry for entry in dynamics if entry[0] == "banana"), None)

    lines = ["  <keyframe>"]

    def _emit(name: str, qpos) -> None:
        qstr = " ".join(f"{v:.6f}" for v in qpos)
        lines.append(f'    <key name="{name}" qpos="{qstr}"/>')

    # named default: the arm at rest and every dynamic object at its layout pose
    qpos = data.qpos.copy()
    _set_arm(qpos)
    for obj_name, position, rpy in dynamics:
        _set_freejoint(qpos, obj_name, position, rpy)
    _emit("default", qpos)

    # randomisation grid: index k → random_KKK (banana at grid[k], arm at rest)
    for idx, (x, y, z, yaw) in enumerate(grid):
        qpos = data.qpos.copy()
        _set_arm(qpos)
        if banana is not None:
            # banana_free freejoint: [tx, ty, tz, qw, qx, qy, qz]
            _set_freejoint(qpos, banana[0], [x, y, z], [0.0, 0.0, yaw])
        _emit(f"random_{idx:03d}", qpos)
    lines.append("  </keyframe>")

    return xml_content.replace("</mujoco>", "\n".join(lines) + "\n</mujoco>")


def get_scene_file(scene_name: str, platform: str) -> Path:
    """Return the Path to a scene's template file.

    Args:
        scene_name: Scene name (e.g. ``"pick_banana"``, ``"empty"``).
        platform:   Simulator platform — ``"gazebo"`` or ``"mujoco"``.

    Returns:
        Absolute :class:`pathlib.Path` to the template file.

    Raises:
        ValueError:        If *platform* is not ``"gazebo"`` or ``"mujoco"``.
        FileNotFoundError: If the template file does not exist in the
                           installed sim_models share directory.
    """
    ext = _PLATFORM_EXT.get(platform)
    if ext is None:
        raise ValueError(f"Unknown platform: {platform!r}. Supported: {list(_PLATFORM_EXT)}")
    pkg_share = Path(get_package_share_directory("sim_models"))
    path = pkg_share / "scenes" / scene_name / f"{scene_name}{ext}"
    if not path.exists():
        raise FileNotFoundError(
            f"Scene template not found: {path}\n  Did you run 'colcon build --packages-select sim_models'?"
        )
    return path


def get_gazebo_world_path(scene_name: str) -> Path:
    """Generate a Gazebo world file with resolved mesh paths.

    Reads ``{scene_name}.world.template``, replaces ``{{MESHES_DIR}}``
    with the absolute path to the installed ``meshes/`` directory, and
    writes the result to ``/tmp/sim_models_{scene_name}.world``.

    Args:
        scene_name: Scene name (e.g. ``"pick_banana"``).

    Returns:
        :class:`pathlib.Path` to the generated ``.world`` file in ``/tmp/``.
    """
    tmpl = get_scene_file(scene_name, "gazebo")
    meshes_dir = tmpl.parent / "meshes"
    content = tmpl.read_text().replace(_MESHES_PLACEHOLDER, str(meshes_dir))
    out = Path(f"/tmp/sim_models_{scene_name}.world")
    out.write_text(content)
    return out


def composed_model_path(scene: str | None) -> Path:
    """The single composed-model location (D3), keyed by ``ROS_DOMAIN_ID``.

    Resolves to ``<tempdir>/ibrobot_sim_models/<ROS_DOMAIN_ID or 0>/<scene>.mjb``
    (``robot`` when no scene is composed), so concurrent launches on different
    domains (separate worktrees, e2e domains) do not overwrite each other's
    model. The provider (writer) and the scene task (reader) both resolve the
    composed model through this function.
    """
    domain = str(os.environ.get("ROS_DOMAIN_ID", "") or "0")
    name = scene if scene else "robot"
    return Path(tempfile.gettempdir()) / "ibrobot_sim_models" / domain / f"{name}.mjb"


def compose_mujoco_model(robot_mjcf_path: str, scene: str | None, sdf_mesh_names) -> Path:
    """Robot-agnostic robot+scene composition for the MuJoCo provider (D3).

    Substitutes the robot MJCF into the named scene template (``scene=None``
    compiles the robot MJCF alone), injects the shared keyframes when the
    layout has randomisation, enables SDF collision on the given mesh names,
    compiles and saves the ``.mjb`` at ``composed_model_path(scene)``. The
    intermediate XML is kept beside it for debugging. Falls back to the
    ``.xml`` path only when the ``mujoco`` package is missing; the provider
    treats that as fatal.

    Args:
        robot_mjcf_path:  Absolute path to the fully substituted robot MJCF.
        scene:            Scene name (e.g. ``"pick_banana"``, ``"empty"``),
                          or None for a robot-only model.
        sdf_mesh_names:   Mesh asset names that need SDF non-convex collision
                          (robot-specific input, e.g. the gripper fingers).

    Returns:
        :class:`pathlib.Path` to the compiled ``.mjb`` (or the ``.xml``
        fallback when the ``mujoco`` package is unavailable).
    """
    robot_path = Path(robot_mjcf_path)
    if scene:
        tmpl = get_scene_file(scene, "mujoco")
        meshes_dir = tmpl.parent / "meshes"
        content = tmpl.read_text().replace(_MESHES_PLACEHOLDER, str(meshes_dir))
        content = content.replace(_ROBOT_XML_PLACEHOLDER, str(robot_path))
        layout = get_scene_layout(scene)
    else:
        content = robot_path.read_text()
        layout = {}
    if layout.get("randomization"):
        content = _inject_random_keyframes(content, layout)

    xml_out = composed_model_path(scene).with_suffix(".xml")
    xml_out.parent.mkdir(parents=True, exist_ok=True)
    xml_out.write_text(content)

    try:
        import mujoco  # noqa: PLC0415
    except ImportError:
        logging.warning("mujoco not available; composed model left as XML at %s", xml_out)
        return xml_out

    try:
        spec = mujoco.MjSpec.from_string(content)
        sdf_names = {str(name) for name in sdf_mesh_names}
        for mesh in spec.meshes:
            if mesh.name in sdf_names:
                mesh.needsdf = True
        model = spec.compile()
        mjb_out = composed_model_path(scene)
        mujoco.mj_saveModel(model, str(mjb_out))
        return mjb_out
    except Exception as exc:  # noqa: BLE001
        logging.warning("MuJoCo compile failed, falling back to XML: %s", exc)
        return xml_out


def get_scene_layout(scene_name: str) -> dict:
    """Return the parsed layout.yaml for a scene, or an empty dict if not found.

    Args:
        scene_name: Scene name (e.g. ``"pick_banana"``).

    Returns:
        Parsed YAML dict.  Keys include ``robot_spawn`` (optional) and ``objects``.
    """
    pkg_share = Path(get_package_share_directory("sim_models"))
    layout_path = pkg_share / "scenes" / scene_name / "layout.yaml"
    if not layout_path.exists():
        return {}
    return yaml.safe_load(layout_path.read_text()) or {}


def get_scene_task(scene_name: str) -> dict | None:
    """Return the scene's ``task:`` declaration, or None when it declares none.

    A scene task declaration names the executable the world owner starts
    after runtime readiness and the physics platforms it supports::

        task:
          executable: pick_banana_task_node
          platforms: [mujoco, gazebo]
    """
    task = get_scene_layout(scene_name).get("task")
    return task if isinstance(task, dict) else None
