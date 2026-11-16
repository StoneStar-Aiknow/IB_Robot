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

  randomization_placements(layout) -> list[(x, y, z, yaw)]
      The single randomisation generator shared by both physics platforms
      (so101-mujoco-runtime-platform D9 / issue #195): randomisation index
      *k* maps to the same banana placement on MuJoCo (``ResetWorld`` state
      overrides) and on Gazebo (``SetEntityPose`` with the same index).
      ``mode: grid`` (default) is bit-identical to the historical keyframe
      grid; ``mode: uniform`` samples deterministically from ``seed``.
      Every input is validated (finite ordered ranges, positive-integer
      divisions, integer seed/count with count >= 1) — ValueError names the
      offending key and fires in the consumer's constructor.

  arm_rest_pairs(layout) -> list[(joint_name, rest_qpos)]
      The validated arm rest pairs (``arm_joint_names`` /
      ``arm_rest_qpos``): both fields arrays, unique non-empty names,
      equal-length arrays, finite values. Shared by the scene task's rest
      trajectories and the MuJoCo backend's ``ResetWorld`` joint override.

  scene_dynamic_objects(scene_name) -> list[str]
      Names of the scene's dynamic objects (``type: dynamic`` in layout.yaml).

  free_joint_states_topic(scene) -> str
      The scene-namespaced free-joint observation topic the MuJoCo
      FreeJointStatePublisherPlugin publishes to (and the scene task's
      MuJoCo backend subscribes to); the Gazebo counterpart is
      ``/sim/<scene>/object_poses``.

  composed_model_path(scene) -> Path
      The single composed-model location (D3), keyed by ``ROS_DOMAIN_ID`` so
      concurrent launches on different domains never overwrite each other's
      model. Used by the provider (writer); the scene task no longer reads
      the composed model since placements moved to ``ResetWorld`` state
      overrides (issue #195).

  mjcf_engine_extensions(source) -> list[str]
      The engine-extension names an MJCF document declares
      (``<extension><plugin plugin="..."/>``), from a path or a string.

  load_mujoco_engine_plugins(required_extensions) -> None
      Load every library registered under the ament ``mujoco_plugins``
      resource index (D5.1): ``ctypes.CDLL(..., RTLD_GLOBAL)`` once per
      process, then verify each required extension with a probe parse of a
      minimal MJCF declaring it. Missing library or unregistered extension
      raises naming the apt package to install.

  compose_mujoco_model(robot_mjcf_path, scene, sdf_mesh_names) -> Path
      Robot-agnostic composition (D3): substitute the robot MJCF into the
      named scene (or compile it alone when ``scene`` is None), apply SDF
      collision to the given meshes, compile and save the ``.mjb``. Falls
      back to the ``.xml`` only when the ``mujoco`` package is missing; the
      provider treats that as fatal. When the composed MJCF declares engine
      extensions (lekiwi-mujoco-runtime-platform D5.1) the Python compiler
      loads them and fully compiles as validation — compile errors raise the
      original MuJoCo error — but the DELIVERED artifact is the XML
      (``MjSpec.to_xml()``): ``mj_saveModel`` serialises the compiling
      process's global plugin slot ids into the ``.mjb``, which the C++
      node (its own registration order) cannot read back (spike evidence
      /tmp/lekiwi-mujoco-evidence/spike/SUMMARY.md). Extension-free models
      keep the ``.mjb`` path byte-for-byte.

  get_scene_layout(scene_name) -> dict
      The parsed layout.yaml of a scene (or {}).

  resolve_robot_spawn(simulation, scene) -> (x, y, z, yaw)
      The one provider-path spawn-pose rule (D7): simulation.robot_spawn
      when it is a non-empty mapping (with or without a scene), else the
      scene layout's robot_spawn, else the origin. Generic launch resolves
      it once, before any world hook, so every platform shares one rule.

  get_scene_task(scene_name) -> dict | None
      The scene's ``task:`` declaration (executable + platforms), or None.

  mobile_manipulation_spec(layout) -> dict
      The validated ``mobile_manipulation`` section of a scene layout (#205
      grasp stage D1): the grasp object (a ``type: dynamic`` entry), the
      table (a ``static_geometry`` entry), the Nav2 staging pose and the
      success thresholds. ValueError names the offending key.

  rasterize_static_geometry(layout, *, resolution, origin, size) -> ndarray
      The occupancy grid of a layout's ``static_geometry`` (#205 grasp
      stage D2): yawed boxes by their exact cell overlap (the yawed box by
      its axis-aligned bounding box), cylinders by the closest point of each
      cell rectangle to the axis. Rows are in PGM order (row 0 = max y).
"""

import logging
import math
import os
import random
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


def scene_template_platforms() -> tuple[str, ...]:
    """The simulation platforms sim_models ships scene templates for."""
    return tuple(_PLATFORM_EXT)


def _finite_number(value, label: str) -> float:
    """A finite YAML number (bools are ints in Python but not valid here)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def _positive_int(value, label: str) -> int:
    """A strictly positive integer (rejects floats, bools and zero)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{label} must be >= 1, got {value}")
    return value


def _validated_range(value, label: str) -> tuple[float, float]:
    """A finite [min, max] pair with min <= max."""
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"{label} must be a [min, max] pair, got {value!r}")
    low = _finite_number(value[0], f"{label}[0]")
    high = _finite_number(value[1], f"{label}[1]")
    if low > high:
        raise ValueError(f"{label} must satisfy min <= max, got {value!r}")
    return low, high


def randomization_placements(layout: dict) -> list[tuple[float, float, float, float]]:
    """The single randomisation generator shared by both physics platforms (D9, issue #195).

    Index *k* of the returned list maps to the same banana placement on every
    platform: the MuJoCo backend's ``ResetWorld`` state overrides and the
    Gazebo backend's ``SetEntityPose`` both consume ``placements[k]``.

    ``randomization.banana.mode`` selects the generator:

    - ``grid`` (default, also when the key is absent): the historical
      keyframe-era grid — x outer, y middle, yaw inner, values from
      ``numpy.linspace`` — reproduced bit-identically, so existing
      randomisation indices keep their meaning.
    - ``uniform``: ``count`` poses sampled deterministically from ``seed``
      across the x/y/yaw ranges (``z`` stays ``z_spawn``); the same seed and
      count give the same sequence on every platform, so index *k* is still
      platform-independent.

    Every input is validated here so a bad layout fails in the consumer's
    constructor — before any motion is commanded —
    with a message naming the offending key: ranges must be finite ordered
    pairs, ``z_spawn`` finite, grid divisions positive integers, uniform
    ``seed``/``count`` integers (bools and floats rejected) with
    ``count >= 1``.
    """
    rand = layout.get("randomization", {}) or {}
    banana_cfg = rand.get("banana", {}) or {}
    if not banana_cfg:
        return []
    x_min, x_max = _validated_range(banana_cfg.get("x_range", [-0.03, 0.10]), "randomization.banana.x_range")
    y_min, y_max = _validated_range(banana_cfg.get("y_range", [-0.02, 0.10]), "randomization.banana.y_range")
    z_spawn = _finite_number(banana_cfg.get("z_spawn", 0.05), "randomization.banana.z_spawn")
    mode = str(banana_cfg.get("mode", "grid")).lower()
    if mode == "grid":
        grid_cols = _positive_int(banana_cfg.get("grid_cols", 5), "randomization.banana.grid_cols")
        grid_rows = _positive_int(banana_cfg.get("grid_rows", 7), "randomization.banana.grid_rows")
        yaw_steps = _positive_int(banana_cfg.get("yaw_steps", 6), "randomization.banana.yaw_steps")
        xs = np.linspace(x_min, x_max, grid_cols)
        ys = np.linspace(y_min, y_max, grid_rows)
        yaws = [2.0 * math.pi * i / yaw_steps for i in range(yaw_steps)]
        return [(float(x), float(y), z_spawn, float(yaw)) for x in xs for y in ys for yaw in yaws]
    if mode == "uniform":
        if "seed" not in banana_cfg or "count" not in banana_cfg:
            raise ValueError("randomization.banana uniform mode requires 'seed' and 'count'")
        seed = banana_cfg["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"randomization.banana.seed must be an integer, got {seed!r}")
        count = _positive_int(banana_cfg["count"], "randomization.banana.count")
        rng = random.Random(seed)
        return [
            (
                rng.uniform(x_min, x_max),
                rng.uniform(y_min, y_max),
                z_spawn,
                rng.uniform(0.0, 2.0 * math.pi),
            )
            for _ in range(count)
        ]
    raise ValueError(f"randomization.banana mode must be 'grid' or 'uniform', got {mode!r}")


def arm_rest_pairs(layout: dict) -> list[tuple[str, float]]:
    """The validated (arm joint name, rest qpos) pairs from the layout.

    The randomisation section's ``arm_joint_names`` / ``arm_rest_qpos`` feed
    both the scene task's rest trajectories and the MuJoCo backend's
    ``ResetWorld`` joint override, so the two arrays are parsed and validated
    once here: both fields must be arrays (a string would otherwise be
    iterated character by character), names must be unique non-empty
    strings, the arrays equally long, every rest value a finite number.
    ``ValueError`` surfaces in the consumer's constructor — before any
    motion is commanded.
    """
    rand = (layout or {}).get("randomization", {}) or {}
    names = rand.get("arm_joint_names")
    qpos = rand.get("arm_rest_qpos")
    names = [] if names is None else names
    qpos = [] if qpos is None else qpos
    for field, value in (("randomization.arm_joint_names", names), ("randomization.arm_rest_qpos", qpos)):
        if not isinstance(value, list | tuple):
            raise ValueError(f"{field} must be a list, got {value!r}")
    if len(names) != len(qpos):
        raise ValueError(
            f"randomization.arm_joint_names ({len(names)}) and arm_rest_qpos ({len(qpos)}) must have the same length"
        )
    pairs = []
    for name, value in zip(names, qpos, strict=True):
        if not isinstance(name, str) or not name:
            raise ValueError(f"randomization.arm_joint_names entries must be non-empty strings, got {name!r}")
        pairs.append((name, _finite_number(value, "randomization.arm_rest_qpos entries")))
    if len({name for name, _ in pairs}) != len(pairs):
        raise ValueError("randomization.arm_joint_names must be unique")
    return pairs


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


def scene_dynamic_objects(scene_name: str) -> list[str]:
    """Names of the scene's dynamic objects (``type: dynamic`` in layout.yaml).

    The MuJoCo runtime provider publishes exactly these bodies through the
    FreeJointStatePublisherPlugin, and the scene task's MuJoCo backend
    observes them on the resulting topic. Unknown scenes (or scenes without
    dynamic objects) return an empty list — no plugin is configured for them.
    """
    return [name for name, _position, _rpy in _layout_dynamic_objects(get_scene_layout(scene_name))]


def free_joint_states_topic(scene: str) -> str:
    """The scene-namespaced free-joint observation topic on the MuJoCo platform.

    ``/sim/<scene>/free_joint_states`` mirrors Gazebo's
    ``/sim/<scene>/object_poses``: one scene-namespaced absolute stream with
    every dynamic object's live pose. Both the runtime provider (which
    configures the plugin's ``topic`` parameter) and the scene task's
    ``MujocoBackend`` (which subscribes) resolve the name through this
    function, so the two ends can never drift apart.
    """
    return f"/sim/{scene}/free_joint_states"


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
    writes the result to a unique file under the temp directory
    (``sim_models_<scene_name>_<unique>.world``). The content embeds the
    installation's absolute mesh path, so one fixed name would let two
    worktrees or domains overwrite each other's world (sim-platform-registry
    D9); each construction owns its file.

    Args:
        scene_name: Scene name (e.g. ``"pick_banana"``).

    Returns:
        :class:`pathlib.Path` to the generated ``.world`` file.
    """
    tmpl = get_scene_file(scene_name, "gazebo")
    meshes_dir = tmpl.parent / "meshes"
    content = tmpl.read_text().replace(_MESHES_PLACEHOLDER, str(meshes_dir))
    fd, out_str = tempfile.mkstemp(prefix=f"sim_models_{scene_name}_", suffix=".world")
    out = Path(out_str)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(content)
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


def mjcf_engine_extensions(source: str | Path) -> list[str]:
    """The engine-extension names an MJCF document declares (D5.1).

    ``source`` is a file path or the document text; the result is the sorted
    ``plugin`` attributes of every ``<extension><plugin>`` element. Includes
    are followed (the composed scene template declares nothing itself — the
    robot it includes does); a document without an ``<extension>`` section
    declares none.
    """
    import xml.etree.ElementTree as ET

    seen: set[Path] = set()
    names: set[str] = set()

    def _collect(text: str, origin: Path | None) -> None:
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return
        if root.tag != "mujoco":
            return
        extension = root.find("extension")
        if extension is not None:
            names.update(str(plugin.get("plugin")) for plugin in extension.findall("plugin") if plugin.get("plugin"))
        for include in root.findall("include"):
            target = include.get("file")
            if not target:
                continue
            resolved = (Path(target) if os.path.isabs(target) else (origin.parent / target)) if origin else Path(target)
            resolved = resolved.resolve()
            if resolved in seen or not resolved.is_file():
                continue
            seen.add(resolved)
            _collect(resolved.read_text(encoding="utf-8"), resolved)

    if isinstance(source, Path) or (isinstance(source, str) and os.path.isfile(source)):
        origin = Path(source).resolve()
        if origin.is_file():
            seen.add(origin)
            _collect(origin.read_text(encoding="utf-8"), origin)
    else:
        _collect(str(source), None)
    return sorted(names)


#: Libraries already loaded into this process by :func:`load_mujoco_engine_plugins`
#: (one RTLD_GLOBAL load per path, per process).
_LOADED_ENGINE_LIBRARIES: set[str] = set()


def load_mujoco_engine_plugins(required_extensions: list[str] | tuple[str, ...]) -> None:
    """Load the ament-registered MuJoCo engine extensions into this process (D5.1).

    Mirrors what the C++ simulator does at startup: every library under the
    ``mujoco_plugins`` ament resource index is ``ctypes.CDLL``-loaded with
    ``RTLD_GLOBAL`` (once per process), then each required extension name is
    verified with a probe parse of a minimal MJCF declaring it — no plugin-table
    API is assumed. A missing resource index or an extension that stays
    unregistered raises naming the apt package to install.
    """
    import ctypes

    try:
        from ament_index_python import get_resource, get_resources
    except ImportError as exc:  # pragma: no cover - the runtime image always has it
        raise RuntimeError(
            "ament_index_python is needed to load the MuJoCo engine extensions "
            "(pip install ament-index-python or run inside the ROS environment)"
        ) from exc

    resources = get_resources("mujoco_plugins")
    if not resources:
        raise RuntimeError(
            "no mujoco_plugins ament resources found; the MuJoCo engine extensions need them "
            "(install ros-humble-mujoco-3d-lidar for mujoco.plugin.lidar)"
        )
    for package in sorted(resources):
        content, prefix = get_resource("mujoco_plugins", package)
        for library in sorted((Path(prefix) / content).glob("*.so")):
            if str(library) in _LOADED_ENGINE_LIBRARIES:
                continue
            try:
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
            except OSError as exc:
                raise RuntimeError(f"cannot load the MuJoCo engine extension library {library}: {exc}") from exc
            _LOADED_ENGINE_LIBRARIES.add(str(library))
    if not required_extensions:
        return
    import mujoco  # noqa: PLC0415

    for name in required_extensions:
        probe = f'<mujoco><extension><plugin plugin="{name}"/></extension><worldbody/></mujoco>'
        try:
            mujoco.MjSpec.from_string(probe)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"the MuJoCo engine extension {name!r} is not registered after loading "
                f"{sorted(_LOADED_ENGINE_LIBRARIES)}; install the package providing it "
                "(ros-humble-mujoco-3d-lidar provides mujoco.plugin.lidar)"
            ) from exc


def compose_mujoco_model(robot_mjcf_path: str, scene: str | None, sdf_mesh_names) -> Path:
    """Robot-agnostic robot+scene composition for the MuJoCo provider (D3).

    Substitutes the robot MJCF into the named scene template (``scene=None``
    compiles the robot MJCF alone), enables SDF collision on the given mesh
    names, compiles and saves the ``.mjb`` at ``composed_model_path(scene)``.
    The intermediate XML is kept beside it for debugging. Falls back to the
    ``.xml`` path only when the ``mujoco`` package is missing; the provider
    treats that as fatal. Issue #195: no keyframes are injected any more —
    scene placements travel through ``ResetWorld`` state overrides built
    from the layout, not through compiled-in ``random_KKK`` keys.

    When the composed MJCF declares engine extensions (D5.1): the extensions
    are loaded first, the full compile validates the model (compile errors
    raise the original MuJoCo error — nothing is silently swallowed) and the
    DELIVERED artifact is the compiled spec's XML (``MjSpec.to_xml()``),
    written to the ``.xml`` path beside the would-be ``.mjb``. A ``.mjb``
    bakes the compiling process's GLOBAL plugin slot ids, which the C++ node
    (which registers the extensions in its own order) cannot read back — it
    segfaults on the mismatched slot (spike evidence
    /tmp/lekiwi-mujoco-evidence/spike/SUMMARY.md). Extension-free models
    keep the ``.mjb`` path byte-for-byte. SDF mesh names combined with an
    engine extension fail closed: ``to_xml()`` cannot carry ``needsdf``.

    Args:
        robot_mjcf_path:  Absolute path to the fully substituted robot MJCF.
        scene:            Scene name (e.g. ``"pick_banana"``, ``"empty"``),
                          or None for a robot-only model.
        sdf_mesh_names:   Mesh asset names that need SDF non-convex collision
                          (robot-specific input, e.g. the gripper fingers).

    Returns:
        :class:`pathlib.Path` to the compiled ``.mjb`` (or the delivered
        ``.xml`` when the model declares engine extensions; the ``.xml``
        fallback when the ``mujoco`` package is unavailable).
    """
    robot_path = Path(robot_mjcf_path)
    if scene:
        tmpl = get_scene_file(scene, "mujoco")
        meshes_dir = tmpl.parent / "meshes"
        content = tmpl.read_text().replace(_MESHES_PLACEHOLDER, str(meshes_dir))
        content = content.replace(_ROBOT_XML_PLACEHOLDER, str(robot_path))
    else:
        content = robot_path.read_text()

    xml_out = composed_model_path(scene).with_suffix(".xml")
    xml_out.parent.mkdir(parents=True, exist_ok=True)
    xml_out.write_text(content)
    engine_extensions = mjcf_engine_extensions(content)

    try:
        import mujoco  # noqa: PLC0415
    except ImportError:
        if engine_extensions:
            # D5.1 (review round 1): an extension-declaring model is only
            # valid after a real compile in this process; returning the
            # unvalidated XML would hand the C++ node a model whose
            # extension parameters were never checked.
            raise ImportError(
                "the composed model declares engine extensions ("
                + ", ".join(sorted(engine_extensions))
                + ") and cannot be delivered uncompiled: the mujoco Python package is required"
            ) from None
        logging.warning("mujoco not available; composed model left as XML at %s", xml_out)
        return xml_out

    sdf_names = {str(name) for name in sdf_mesh_names}
    if engine_extensions:
        # D5.1: the extension must be registered before the parse that
        # declares it, and the compile validates the model for real.
        if sdf_names:
            raise ValueError(
                "an MJCF declaring engine extensions cannot also use SDF mesh collision: "
                "MjSpec.to_xml() does not serialise needsdf, so the delivered XML would "
                "silently drop it (no current model combines the two; fail closed)"
            )
        load_mujoco_engine_plugins(engine_extensions)
        spec = mujoco.MjSpec.from_string(content)
        model = spec.compile()
        # the delivered artifact: the compiled spec's XML, recompilable by the
        # C++ node with ITS OWN (correct) extension registration
        xml_out.write_text(spec.to_xml())
        del model  # the compile was validation; the artifact is the XML
        return xml_out

    try:
        spec = mujoco.MjSpec.from_string(content)
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


def resolve_robot_spawn(simulation: dict, scene: str | None) -> tuple[float, float, float, float]:
    """The one provider-path spawn-pose rule (sim-platform-registry D7).

    Precedence, shared by every physics platform:

    1. ``simulation.robot_spawn`` when it is a non-empty mapping — with or
       without a scene. Absent, ``null`` and ``{}`` mean unspecified, and a
       non-empty mapping defaults missing coordinates (and ``yaw``) to 0.
    2. otherwise the scene layout's ``robot_spawn`` when a scene is set and
       its layout loads (a scene that fails to load is never consulted: the
       caller has already fallen back to the ``empty`` scene);
    3. otherwise ``(0, 0, 0, 0)``.

    Returns the full pose ``(x, y, z, yaw)``; generic launch resolves it once
    before any world hook, so every platform shares one spawn rule.
    """
    spawn = (simulation or {}).get("robot_spawn")
    if not isinstance(spawn, dict) or not spawn:
        spawn = None
    if spawn is None and scene:
        try:
            layout_spawn = get_scene_layout(scene).get("robot_spawn")
        except Exception:  # noqa: BLE001 - a scene that failed to load (the caller has fallen back to empty) is never consulted for its layout
            layout_spawn = None
        if isinstance(layout_spawn, dict) and layout_spawn:
            spawn = layout_spawn
    if spawn is None:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        float(spawn.get("x", 0.0)),
        float(spawn.get("y", 0.0)),
        float(spawn.get("z", 0.0)),
        float(spawn.get("yaw", 0.0)),
    )


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


def mobile_manipulation_spec(layout: dict) -> dict:
    """The validated ``mobile_manipulation`` section of a scene layout (D1).

    The single definition of the grasp task data, read by the scene tests
    and (Phase 2) the application support through this validator only::

        mobile_manipulation:
          object: grasp_object             # a `type: dynamic` object
          table: table                     # a static_geometry entry
          object_prompt: magenta cube      # the grounded_sam2 prompt (R5)
          staging_pose: {x: .., y: .., yaw: ..}
          success: {lift_m: 0.05, hold_s: 3.0, max_object_rotation_in_gripper_deg: 10}

    Args:
        layout: The parsed layout.yaml of the scene.

    Returns:
        The section as a plain dict (``object``, ``table``, ``staging_pose``
        with float x/y/yaw, ``success`` with float lift_m/hold_s/
        max_object_rotation_in_gripper_deg).

    Raises:
        ValueError: Naming the offending key — the section or a sub-key is
            missing or of the wrong type; the object is not a dynamic object
            of the layout; the table is not a static_geometry entry; a value
            is not finite; or the staging pose lies inside a static element.
    """
    section = layout.get("mobile_manipulation")
    if not isinstance(section, dict):
        raise ValueError("mobile_manipulation: the section is missing or not a mapping")

    object_name = section.get("object")
    if not isinstance(object_name, str) or not object_name:
        raise ValueError("mobile_manipulation.object: missing or not a name")
    dynamic = [name for name, _position, _rotation in _layout_dynamic_objects(layout)]
    if object_name not in dynamic:
        raise ValueError(
            f"mobile_manipulation.object: {object_name!r} is not a `type: dynamic` object "
            f"of this layout (dynamic objects: {dynamic or 'none'})"
        )

    prompt = section.get("object_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("mobile_manipulation.object_prompt: missing or not a non-empty string")
    object_prompt = prompt.strip()

    table_name = section.get("table")
    if not isinstance(table_name, str) or not table_name:
        raise ValueError("mobile_manipulation.table: missing or not a name")
    static_names = [str(element.get("name", "")) for element in layout.get("static_geometry", []) or []]
    if table_name not in static_names:
        raise ValueError(
            f"mobile_manipulation.table: {table_name!r} is not a static_geometry entry "
            f"of this layout (entries: {static_names or 'none'})"
        )

    pose = section.get("staging_pose")
    if not isinstance(pose, dict):
        raise ValueError("mobile_manipulation.staging_pose: missing or not a mapping")
    staging = {
        key: _finite_number(pose.get(key), f"mobile_manipulation.staging_pose.{key}") for key in ("x", "y", "yaw")
    }

    success_in = section.get("success")
    if not isinstance(success_in, dict):
        raise ValueError("mobile_manipulation.success: missing or not a mapping")
    success = {}
    for key in ("lift_m", "hold_s", "max_object_rotation_in_gripper_deg"):
        value = _finite_number(success_in.get(key), f"mobile_manipulation.success.{key}")
        if value <= 0.0:
            raise ValueError(f"mobile_manipulation.success.{key}: must be positive, got {value}")
        success[key] = value

    # the staging pose must not sit inside (or on) a static element: Nav2
    # would goal-check inside an obstacle
    for element in layout.get("static_geometry", []) or []:
        name = str(element.get("name", ""))
        ex, ey = float(element["pose"][0]), float(element["pose"][1])
        if element.get("primitive") == "box":
            half_x, half_y = float(element["size"][0]) / 2.0, float(element["size"][1]) / 2.0
            inside = abs(staging["x"] - ex) <= half_x and abs(staging["y"] - ey) <= half_y
        else:
            inside = math.hypot(staging["x"] - ex, staging["y"] - ey) <= float(element["size"][0])
        if inside:
            raise ValueError(
                f"mobile_manipulation.staging_pose: ({staging['x']}, {staging['y']}) lies inside "
                f"the static element {name!r}"
            )

    return {
        "object": object_name,
        "object_prompt": object_prompt,
        "table": table_name,
        "staging_pose": staging,
        "success": success,
    }


def rasterize_static_geometry(
    layout: dict,
    *,
    resolution: float,
    origin: tuple[float, float],
    size: tuple[int, int],
) -> "np.ndarray":
    """The occupancy grid of a layout's ``static_geometry`` (D2).

    Cell rules (spike 1.2's chosen rules, pinned by the fidelity test
    against the shipped ``sim_map.pgm``):

    * **Box, yaw 0** — a cell is occupied when the box's exact interval
      ``[min, max]`` on an axis overlaps the cell's interval on that axis
      (computed in exact rational arithmetic, so cells never flip on
      floating-point representation error).
    * **Box, yaw != 0** — the same interval rule applied to the yawed box's
      axis-aligned bounding box (the map is planar; the yawed footprint's
      conservative envelope).
    * **Cylinder** — a cell is occupied when the closest point of the
      cell's rectangle to the cylinder axis lies within the radius.

    Args:
        layout:     The parsed layout.yaml (only ``static_geometry`` is read).
        resolution: The grid resolution in metres per cell (positive).
        origin:     The world coordinate of the grid's lower-left corner
                    ``(x, y)``.
        size:       The grid dimensions ``(width, height)`` in cells.

    Returns:
        A boolean :class:`numpy.ndarray` of shape ``(height, width)`` in
        PGM row order — row 0 is the MAXIMUM y, as ``map_server`` reads PGM
        rasters — so the array serialises directly into the map file.
    """
    if resolution <= 0.0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    width, height = size
    _positive_int(width, "size (width)")
    _positive_int(height, "size (height)")

    from fractions import Fraction

    res = Fraction(resolution).limit_denominator(10**9)
    origin_x = Fraction(origin[0]).limit_denominator(10**9)
    origin_y = Fraction(origin[1]).limit_denominator(10**9)
    # world-order grid (row 0 = min y); flipped to PGM order on return
    grid = np.zeros((int(height), int(width)), dtype=bool)

    def interval_cells(lo: Fraction, hi: Fraction) -> tuple[int, int]:
        """The half-open cell-index range [lo_cell, hi_cell) overlapped by
        the exact world interval [lo, hi]."""
        first = math.floor((lo - origin_axis) / res)
        last = math.ceil((hi - origin_axis) / res)
        return int(first), int(last)

    for element in layout.get("static_geometry", []) or []:
        pose = element.get("pose") or []
        if len(pose) < 6:
            raise ValueError(f"static_geometry entry {element.get('name')!r}: pose needs 6 values")
        px, py, _pz, _r, _p, yaw = (float(value) for value in pose)
        primitive = str(element.get("primitive", ""))
        if primitive == "box":
            sizes = element.get("size") or []
            if len(sizes) < 2:
                raise ValueError(f"static_geometry entry {element.get('name')!r}: box size needs >= 2 values")
            half_x, half_y = float(sizes[0]) / 2.0, float(sizes[1]) / 2.0
            if yaw == 0.0:
                x_lo, x_hi = px - half_x, px + half_x
                y_lo, y_hi = py - half_y, py + half_y
            else:
                cos_a, sin_a = abs(math.cos(yaw)), abs(math.sin(yaw))
                extent_x = cos_a * half_x + sin_a * half_y
                extent_y = sin_a * half_x + cos_a * half_y
                x_lo, x_hi = px - extent_x, px + extent_x
                y_lo, y_hi = py - extent_y, py + extent_y
            origin_axis = origin_x
            j0, j1 = interval_cells(Fraction(x_lo).limit_denominator(10**9), Fraction(x_hi).limit_denominator(10**9))
            origin_axis = origin_y
            i0, i1 = interval_cells(Fraction(y_lo).limit_denominator(10**9), Fraction(y_hi).limit_denominator(10**9))
            i0, i1 = max(i0, 0), min(i1, int(height))
            j0, j1 = max(j0, 0), min(j1, int(width))
            if i0 < i1 and j0 < j1:
                grid[i0:i1, j0:j1] = True
        elif primitive == "cylinder":
            sizes = element.get("size") or []
            if len(sizes) < 1:
                raise ValueError(f"static_geometry entry {element.get('name')!r}: cylinder size needs the radius")
            radius = float(sizes[0])
            radius_sq = radius * radius
            for i in range(int(height)):
                cell_y_lo = origin[1] + i * resolution
                ny = min(max(py, cell_y_lo), cell_y_lo + resolution) - py
                for j in range(int(width)):
                    cell_x_lo = origin[0] + j * resolution
                    nx = min(max(px, cell_x_lo), cell_x_lo + resolution) - px
                    if nx * nx + ny * ny <= radius_sq + 1e-12:
                        grid[i, j] = True
        else:
            raise ValueError(f"static_geometry entry {element.get('name')!r}: unknown primitive {primitive!r}")
    return grid[::-1, :]


def write_layout_map(
    layout: dict,
    *,
    out_prefix: str | Path,
    resolution: float,
    origin: tuple[float, float],
    size: tuple[int, int],
) -> tuple[Path, Path]:
    """Write a layout's occupancy grid as the trinary map pair (D2).

    The PGM is binary (``P5``) with 0 = occupied and 254 = free, rows
    top-down (row 0 = max y); the YAML mirrors ``sim_map.yaml``'s fields
    (``mode: trinary``, ``negate: 0``, ``occupied_thresh: 0.65``,
    ``free_thresh: 0.25``) so ``map_server`` reads it identically.

    Returns:
        ``(pgm_path, yaml_path)`` — ``{out_prefix}.pgm`` and
        ``{out_prefix}.yaml``.
    """
    grid = rasterize_static_geometry(layout, resolution=resolution, origin=origin, size=size)
    width, height = size
    prefix = Path(out_prefix)
    pgm_path = prefix.with_suffix(".pgm")
    pgm_path.parent.mkdir(parents=True, exist_ok=True)
    raster = np.where(grid, 0, 254).astype(np.uint8)
    pgm_path.write_bytes(f"P5\n{width} {height}\n255\n".encode("ascii") + raster.tobytes())
    yaml_path = prefix.with_suffix(".yaml")
    yaml_path.write_text(
        f"image: {pgm_path.name}\n"
        "mode: trinary\n"
        f"resolution: {resolution}\n"
        f"origin: [{origin[0]}, {origin[1]}, 0]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.25\n",
        encoding="ascii",
    )
    return pgm_path, yaml_path
