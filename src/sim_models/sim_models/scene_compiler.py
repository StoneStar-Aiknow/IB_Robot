"""Scene compiler for IB-Robot simulation.

Provides runtime path resolution for scene assets using a template
placeholder mechanism.  Template files (.world.template, .xml.template)
contain two placeholders that are substituted at launch time:

  {{MESHES_DIR}}      — absolute path to the scene's meshes/ directory
  {{ROBOT_XML_PATH}}  — absolute path to the robot MJCF XML (MuJoCo only)

The substituted content is written to /tmp/ so that Gazebo and MuJoCo
can reference mesh files by their installed absolute paths.

Public API
----------
  get_scene_file(scene_name, platform) -> Path
      Return the Path to the raw template file (for inspection or tests).

  get_gazebo_world_path(scene_name) -> Path
      Generate /tmp/sim_models_{scene}.world with {{MESHES_DIR}} resolved.
      Used by GazeboAdapter.start_backend() when simulation.scene is set.

  get_mujoco_scene_path(scene_name, robot_xml_path="") -> Path
      Generate /tmp/sim_models_{scene}.mjb with both placeholders resolved,
      SDF non-convex collision enabled for meshes listed in _SDF_MESH_NAMES,
      and the result saved as a binary .mjb for mujoco_ros2_control.
      Pass robot_xml_path for a full robot+scene file (T6);
      omit it for a standalone scene file (MuJoCo Viewer testing).
"""

import logging
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory

_MESHES_PLACEHOLDER = "{{MESHES_DIR}}"
_ROBOT_XML_PLACEHOLDER = "{{ROBOT_XML_PATH}}"

_PLATFORM_EXT = {
    "gazebo": ".world.template",
    "mujoco": ".xml.template",
}

# Mesh asset names that require SDF-based non-convex collision.
# MuJoCo normally convex-hulls all mesh geoms; needsdf=True enables the
# octree SDF so the exact (non-convex) surface is used for contact.
# The gripper finger meshes are strongly non-convex; without SDF, MuJoCo
# replaces them with convex hulls that distort fingertip contact.
_SDF_MESH_NAMES: frozenset[str] = frozenset(
    {
        "wrist_roll_follower",  # fixed jaw finger
        "moving_jaw",  # moving jaw finger
    }
)


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


def get_mujoco_scene_path(scene_name: str, robot_xml_path: str = "") -> Path:
    """Generate a MuJoCo scene binary (.mjb) with resolved paths and SDF collision.

    Steps:
    1. Read ``{scene_name}.xml.template``, substitute ``{{MESHES_DIR}}`` and
       optionally ``{{ROBOT_XML_PATH}}``.
    2. Write the intermediate XML to ``/tmp/sim_models_{scene_name}.xml``
       (kept for debugging / MuJoCo Viewer standalone use).
    3. Load the XML via ``mujoco.MjSpec``, set ``needsdf=True`` on every mesh
       listed in ``_SDF_MESH_NAMES``, compile, and save as
       ``/tmp/sim_models_{scene_name}.mjb``.
    4. Return the ``.mjb`` path so ``mujoco_ros2_control`` loads it with
       ``mj_loadModel`` (binary path), which preserves the SDF octree data.

    If the ``mujoco`` Python package is not available (e.g. build-time import),
    the function falls back to returning the plain ``.xml`` path without SDF.

    Args:
        scene_name:      Scene name (e.g. ``"pick_banana"``).
        robot_xml_path:  Absolute path to the robot MJCF (e.g. ``so101.xml``).
                         Leave empty to generate a standalone scene file.

    Returns:
        :class:`pathlib.Path` to the generated ``.mjb`` file in ``/tmp/``
        (or ``.xml`` if mujoco is unavailable).
    """
    tmpl = get_scene_file(scene_name, "mujoco")
    meshes_dir = tmpl.parent / "meshes"
    content = tmpl.read_text().replace(_MESHES_PLACEHOLDER, str(meshes_dir))
    if robot_xml_path:
        content = content.replace(_ROBOT_XML_PLACEHOLDER, robot_xml_path)
    else:
        # Remove the robot <include> line so the file is self-contained.
        content = "\n".join(line for line in content.splitlines() if _ROBOT_XML_PLACEHOLDER not in line)
    xml_out = Path(f"/tmp/sim_models_{scene_name}.xml")
    xml_out.write_text(content)

    try:
        import mujoco  # noqa: PLC0415
    except ImportError:
        return xml_out

    try:
        spec = mujoco.MjSpec.from_string(content)
        for mesh in spec.meshes:
            if mesh.name in _SDF_MESH_NAMES:
                mesh.needsdf = True
        model = spec.compile()
        mjb_out = Path(f"/tmp/sim_models_{scene_name}.mjb")
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
