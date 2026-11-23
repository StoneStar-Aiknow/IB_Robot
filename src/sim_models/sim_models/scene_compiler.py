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
import xml.etree.ElementTree as ET
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


#: mujoco_ros2_control copies ``mujoco_model`` into a buffer of simulate's
#: ``kMaxFilenameLength`` bytes, so it keeps at most one byte less.
_SIMULATOR_MAX_FILENAME_BYTES = 1000


def mujoco_model_camera_names(model_path: str) -> list[str]:
    """The camera names of a composed model, in ``mjModel`` id order.

    Loaded the way mujoco_ros2_control 0.1.2 loads ``mujoco_model``: as a
    binary when the path is longer than four characters and ends in
    ``.mjb``, otherwise as MJCF, after registering the engine extensions the
    MJCF declares (``load_mujoco_engine_plugins``; even a parse needs them).
    An unnamed camera is ``""``. Raises when the model cannot be read; a
    missing file is reported before MuJoCo would log it to
    ``MUJOCO_LOG.TXT`` in the working directory. A path of 1000 bytes or
    more raises too: the simulator copies it into simulate's 1000-byte
    filename buffer (``kMaxFilenameLength``) and would load the truncated
    path, a different file from the one read here.
    """
    import mujoco

    if len(os.fsencode(model_path)) >= _SIMULATOR_MAX_FILENAME_BYTES:
        raise ValueError(
            f"MuJoCo model path is {len(os.fsencode(model_path))} bytes; the simulator truncates paths to "
            f"{_SIMULATOR_MAX_FILENAME_BYTES - 1} bytes and would load a different file"
        )
    if not os.path.isfile(model_path):  # unlike Path.is_file, never raises OSError
        raise FileNotFoundError(f"MuJoCo model {model_path!r} does not exist")
    if len(model_path) > 4 and model_path.endswith(".mjb"):
        model = mujoco.MjModel.from_binary_path(model_path)
    else:
        extensions = mjcf_engine_extensions(Path(model_path))
        if extensions:
            load_mujoco_engine_plugins(extensions)
        model = mujoco.MjModel.from_xml_path(model_path)
    return [model.camera(i).name for i in range(model.ncam)]


class ViewerViewError(ValueError):
    """The scene's ``mujoco_viewer`` view declaration is malformed."""


#: MuJoCo's default free-camera field of view (``visual/global/fovy``); the
#: layout ``distance`` is "what this fovy would show from that distance".
_VIEWER_REFERENCE_FOVY_DEG = 45.0
#: mjv_defaultFreeCamera places the free camera at 1.5 * stat.extent.
_FREE_CAMERA_DISTANCE_PER_EXTENT = 1.5
#: The complete key set of a layout ``mujoco_viewer`` view (fail closed).
_VIEWER_VIEW_KEYS = frozenset({"lookat", "distance", "azimuth", "elevation"})
#: mjv_moveCamera clamps the free camera elevation to this range.
_VIEWER_MIN_ELEVATION_DEG, _VIEWER_MAX_ELEVATION_DEG = -89.0, 89.0
_FLOAT32_MAX = float(np.finfo(np.float32).max)
#: Relative headroom for float32 rounding in a few products and sums.
_FLOAT32_ROUNDING_MARGIN = 1e-5


def scene_viewer_view(scene: str | None) -> dict | None:
    """The MuJoCo viewer's initial free-camera view, from the scene layout.

    ``layout.yaml`` may declare::

        mujoco_viewer:
          lookat: [x, y, z]     # world point the view is centred on (m)
          distance: 0.72        # apparent viewing distance (m), > 0
          azimuth: -110         # degrees, MuJoCo free-camera convention
          elevation: -55        # degrees, in [-89, 89]

    Returns the validated view (floats; ``lookat`` a 3-tuple), or ``None``
    without a scene or without the key; a key without a value is an error. See ``_apply_viewer_view`` for how it
    lands in the model without touching any recorded camera.
    """
    if not scene:
        return None
    layout = get_scene_layout(scene)
    if "mujoco_viewer" not in layout:
        return None
    # an explicit empty ``mujoco_viewer:`` (YAML null) is a malformed view,
    # not "no view": it goes through the mapping check below
    raw = layout["mujoco_viewer"]
    where = f"scene {scene!r}: mujoco_viewer"
    if not isinstance(raw, dict):
        raise ViewerViewError(f"{where} must be a mapping, got {raw!r}")
    missing = _VIEWER_VIEW_KEYS - raw.keys()
    if missing:
        raise ViewerViewError(f"{where} is missing {sorted(missing)}")
    unknown = set(raw) - _VIEWER_VIEW_KEYS
    if unknown:
        raise ViewerViewError(f"{where} has unknown keys {sorted(map(str, unknown))}")

    def number(key, value):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ViewerViewError(f"{where}.{key} must be a finite number, got {value!r}")
        try:
            converted = float(value)  # a huge Python int overflows here
        except OverflowError:
            converted = math.inf
        if not math.isfinite(converted):
            raise ViewerViewError(f"{where}.{key} must be a finite number, got {value!r}")
        return converted

    lookat = raw["lookat"]
    if not isinstance(lookat, list | tuple) or len(lookat) != 3:
        raise ViewerViewError(f"{where}.lookat must be [x, y, z], got {lookat!r}")
    view = {
        "lookat": tuple(number("lookat", v) for v in lookat),
        "distance": number("distance", raw["distance"]),
        "azimuth": number("azimuth", raw["azimuth"]),
        "elevation": number("elevation", raw["elevation"]),
    }
    # stat.center is a double, but the free camera's GL position is a C float
    # computed from it: a lookat beyond the float32 range renders from inf
    if any(abs(v) >= _FLOAT32_MAX for v in view["lookat"]):
        raise ViewerViewError(f"{where}.lookat must fit a float32, got {lookat!r}")
    if view["distance"] <= 0:
        raise ViewerViewError(f"{where}.distance must be > 0, got {view['distance']}")
    # visual/global azimuth and elevation are C floats. A finite double
    # azimuth such as 1e39 would overflow to inf (and a large one loses whole
    # degrees), so it is reduced to one turn first: [-180, 180], the range
    # mjv_moveCamera wraps it to.
    view["azimuth"] = float(np.float32(math.remainder(view["azimuth"], 360.0)))
    # mjv_moveCamera clamps the elevation to [-89, 89] on every mouse action
    # (rotate, pan or zoom), so a start view outside it would jump on the
    # first touch. The range is checked on the double, before the cast (which
    # would overflow); +-89 are exact floats and rounding is monotonic, so the
    # stored value stays in range.
    if not _VIEWER_MIN_ELEVATION_DEG <= view["elevation"] <= _VIEWER_MAX_ELEVATION_DEG:
        raise ViewerViewError(
            f"{where}.elevation must be in [{_VIEWER_MIN_ELEVATION_DEG:g}, {_VIEWER_MAX_ELEVATION_DEG:g}] "
            f"(the viewer's own range), got {view['elevation']!r}"
        )
    view["elevation"] = float(np.float32(view["elevation"]))
    return view


def _viewer_fovy(view: dict, extent: float, znear: float) -> float:
    """The free-camera fovy that makes the fixed-distance camera frame ``view``.

    The free camera sits at ``1.5 * stat.extent``, and the extent cannot be
    changed for the viewer: znear/zfar are ``vis.map.znear/zfar * extent``
    for EVERY camera, the recorded ones included. The apparent distance is
    therefore set through ``visual/global/fovy``, which only the free camera
    uses (MJCF cameras carry their own fovy).

    ``znear`` is the model's ``vis.map.znear``: the fovy must also give the
    viewer a usable projection, which MuJoCo builds in C floats.
    """
    if not (math.isfinite(extent) and extent > 0):
        raise ViewerViewError(f"mujoco_viewer: the compiled model extent {extent!r} cannot frame a view")
    camera_distance = _FREE_CAMERA_DISTANCE_PER_EXTENT * extent
    half = math.atan(math.tan(math.radians(_VIEWER_REFERENCE_FOVY_DEG) / 2) * view["distance"] / camera_distance)
    fovy = math.degrees(2 * half)
    # visual/global/fovy is a C float: check the value MuJoCo will hold, so a
    # distance that is valid as a double cannot round to a 0 or 180 deg fovy
    stored = float(np.float32(fovy))
    if not (math.isfinite(stored) and 0.0 < stored < 180.0):
        raise ViewerViewError(
            f"mujoco_viewer.distance {view['distance']!r} m needs a free-camera fovy of {stored!r} deg at "
            f"extent {extent!r}; it must lie strictly between 0 and 180 deg"
        )
    # mjv_updateScene builds the free camera's frustum in C floats: near =
    # znear * extent, top = -bottom = near * tan(fovy / 2), and the GL
    # projection scales by 2 near / (top - bottom). A tiny fovy underflows
    # top to 0 (inf scale); a wide fovy at a large near overflows top or
    # top - bottom to inf (zero scale). Every one must be finite and > 0.
    # (The horizontal extent also scales with the window aspect, a runtime
    # value this compile cannot know.)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        near = np.float32(znear * extent)
        top = np.float32(float(near) * math.tan(math.radians(stored) / 2))
        height = top + top
        scale = (near + near) / height if height > 0 else np.float32(np.inf)

    def finite_positive(value) -> bool:
        return bool(np.isfinite(value) and value > 0)

    if not all(finite_positive(v) for v in (near, top, height, scale)):
        raise ViewerViewError(
            f"mujoco_viewer.distance {view['distance']!r} m needs a free-camera fovy of {stored!r} deg at extent "
            f"{extent!r}, which cannot form a finite positive float32 projection (near {float(near)!r}, "
            f"top {float(top)!r}, height {float(height)!r}, scale {float(scale)!r})"
        )
    return stored


def _check_free_camera_position(view: dict, extent: float, ipd: float) -> None:
    """The free camera's GL eye positions and view transform must stay finite C floats.

    mjv_updateScene places each eye at ``lookat - 1.5 * extent * forward``
    (forward from azimuth/elevation), shifted by ``ipd / 2`` sideways, and
    stores it as a float. The renderer then builds the modelview translation
    in floats as dot products of the eye with the unit camera axes
    (``mjr_lookAt``): a lookat that fits a float32 can still overflow once
    the camera offset is added, and finite eye components can still overflow
    in that dot product. ``|lookat_i| + distance * |forward_i| + ipd / 2``
    bounds each eye component; their sum bounds every component, every dot
    product with a unit axis and every partial sum of it. A small margin
    covers float rounding of those operations.
    """
    distance = _FREE_CAMERA_DISTANCE_PER_EXTENT * extent
    azimuth, elevation = math.radians(view["azimuth"]), math.radians(view["elevation"])
    forward = (
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    )
    bound = [abs(c) + distance * abs(f) + abs(ipd) / 2 for c, f in zip(view["lookat"], forward, strict=True)]
    total = math.fsum(bound) * (1 + _FLOAT32_ROUNDING_MARGIN)
    if not (math.isfinite(total) and total < _FLOAT32_MAX):
        raise ViewerViewError(
            f"mujoco_viewer.lookat {list(view['lookat'])!r} at extent {extent!r}: the free camera's position or "
            f"view transform would leave the float32 range (|eye| components up to {bound!r})"
        )


def _check_free_camera_mode(model, where: str) -> None:
    """Reject a model whose viewer would not show ``mujoco_viewer`` as defined.

    The view is a perspective free-camera start view. Two ``visual/global``
    settings change what that means, so neither may be combined with it:

    - ``orthographic``: MuJoCo then reads the free camera's fovy as the
      half-height of the view volume (``fovy / 2``, a length), not as an
      angle, so the fovy derived from ``distance`` would frame something
      else;
    - ``cameraid`` naming one of the model's cameras: simulate's
      ``AlignAndScaleView`` then starts on that fixed camera, and Ctrl+A
      returns to it, so the start view would never be shown. (An id outside
      ``[0, ncam)`` falls back to the free camera and is accepted.)
    """
    vis_global = model.vis.global_
    if vis_global.orthographic:
        raise ViewerViewError(
            f"{where}: the model sets visual/global orthographic, but the view's distance sets a "
            "perspective fovy; an orthographic free camera reads fovy as a length"
        )
    if 0 <= vis_global.cameraid < model.ncam:
        raise ViewerViewError(
            f"{where}: the model sets visual/global cameraid={vis_global.cameraid}, so the viewer starts "
            "on that fixed camera (and Ctrl+A returns to it); the free-camera start view would never show"
        )


def _apply_viewer_view(view: dict, stat, vis_global, extent: float, znear: float) -> None:
    """Write ``view`` into a compiled model's (or a spec's) statistic/visual.

    The caller first rejects an orthographic free camera or a fixed start
    camera (``_check_free_camera_mode``). Scope of the compile-time numeric
    checks. A view failing one of them
    raises ``ViewerViewError``, and this function validates all of them
    before it writes any model field (the exact read-back of a delivered
    extension XML runs later, in ``compose_mujoco_model``):

    - azimuth is reduced to one turn and stored as a C float; elevation is
      in [-89, 89] (the range the viewer clamps it to) and stored as a C
      float; every lookat component (stored in the
      double ``stat.center``) is below the float32 maximum;
    - the fovy derived from ``distance`` is a C float in (0, 180), and the
      free camera's frustum ``near``, ``top``, ``top - bottom`` and
      ``2 near / (top - bottom)`` are finite and positive (far is the
      model's own, not changed here);
    - a bound on the free camera's GL eye components (both stereo eyes) and
      on its modelview translation stays within the float32 range.

    Not checked: renderer values computed further downstream, such as the
    squared eye-to-geom distance used to sort transparent geoms, infinite
    plane placement, shadow matrices, and the horizontal frustum, which
    depends on the runtime window aspect. The supported domain is scenes
    whose camera and geom coordinates stay within about 1e18 m. This is a
    support statement, not a rejection threshold: a view outside it may pass
    the checks above and still render wrongly.

    Viewer-only: ``stat.center``, the global azimuth/elevation and the global
    fovy are read by ``mjv_defaultFreeCamera`` (the viewer's start view and
    its Ctrl+A "Align") and by nothing that renders the recorded cameras.
    The extent is not written: MuJoCo's auto-computed extent does not depend
    on a user-set center, so it (and every camera's znear/zfar) stays as the
    plain compile yields it.
    """
    fovy = _viewer_fovy(view, extent, znear)
    _check_free_camera_position(view, extent, float(vis_global.ipd))
    stat.center[:] = view["lookat"]
    vis_global.azimuth = view["azimuth"]
    vis_global.elevation = view["elevation"]
    vis_global.fovy = fovy


def _view_exact_xml(xml: str, stat, vis_global) -> str:
    """``xml`` with the start view written at full precision.

    ``MjSpec.to_xml()`` prints 6 significant digits, so the view the C++
    node recompiles could differ from the one validated here: a valid
    float32 elevation of -89.99999 became -90, and a lookat of 1.23456789
    became 1.23457, a large error for a small framing. ``repr`` of a double
    round-trips exactly, and the float32 fields are exact doubles.
    """
    root = ET.fromstring(xml)
    statistic = root.find("statistic")
    if statistic is None:
        statistic = ET.SubElement(root, "statistic")
    statistic.set("center", " ".join(repr(float(value)) for value in stat.center))
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_ = visual.find("global")
    if global_ is None:
        global_ = ET.SubElement(visual, "global")
    for key in ("azimuth", "elevation", "fovy"):
        global_.set(key, repr(float(getattr(vis_global, key))))
    return ET.tostring(root, encoding="unicode")


def _check_delivered_view(delivered, stat, vis_global, where: str) -> None:
    """The delivered model must hold exactly the validated start view."""
    expected = (
        tuple(map(float, stat.center)),
        *(float(getattr(vis_global, k)) for k in ("azimuth", "elevation", "fovy")),
    )
    got = (
        tuple(map(float, delivered.stat.center)),
        *(float(getattr(delivered.vis.global_, k)) for k in ("azimuth", "elevation", "fovy")),
    )
    if got != expected:
        raise ViewerViewError(f"{where}: the delivered XML carries view {got!r}, not the validated {expected!r}")


def _view_not_applied(viewer_view: dict | None) -> str:
    """Suffix for a fallback warning: the uncompiled XML carries no start view."""
    return " (the scene's mujoco_viewer start view was NOT applied)" if viewer_view else ""


def compose_mujoco_model(robot_mjcf_path: str, scene: str | None, sdf_mesh_names) -> Path:
    """Robot-agnostic robot+scene composition for the MuJoCo provider (D3).

    Substitutes the robot MJCF into the named scene template (``scene=None``
    compiles the robot MJCF alone), enables SDF collision on the given mesh
    names, compiles and saves the ``.mjb`` at ``composed_model_path(scene)``.
    The intermediate XML is kept beside it for debugging. A layout
    ``mujoco_viewer`` view becomes the viewer's initial free camera
    (``scene_viewer_view``); a malformed view raises ``ViewerViewError``
    and never falls back. Falls back to the uncompiled ``.xml`` path when
    the ``mujoco`` package is missing or the extension-free compile fails
    (logged as a warning); the MuJoCo runtime provider rejects that
    fallback XML (``RuntimeProfileError``) instead of launching it. Issue #195: no keyframes are injected any more —
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
        ``.xml`` when the model declares engine extensions; the uncompiled
        ``.xml`` fallback when the ``mujoco`` package is missing or the
        extension-free compile fails, which the runtime provider rejects).
    """
    robot_path = Path(robot_mjcf_path)
    if scene:
        tmpl = get_scene_file(scene, "mujoco")
        meshes_dir = tmpl.parent / "meshes"
        content = tmpl.read_text().replace(_MESHES_PLACEHOLDER, str(meshes_dir))
        content = content.replace(_ROBOT_XML_PLACEHOLDER, str(robot_path))
    else:
        content = robot_path.read_text()

    viewer_view = scene_viewer_view(scene)
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
        logging.warning(
            "mujoco not available; composed model left as XML at %s%s", xml_out, _view_not_applied(viewer_view)
        )
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
        if not viewer_view:
            delivered = spec.to_xml()
        else:
            _check_free_camera_mode(model, f"scene {scene!r}: mujoco_viewer")
            stat, vis_global = spec.stat, spec.visual.global_
            _apply_viewer_view(viewer_view, stat, vis_global, float(model.stat.extent), float(model.vis.map.znear))
            delivered = _view_exact_xml(spec.to_xml(), stat, vis_global)
            # check what the node will read back
            _check_delivered_view(
                mujoco.MjModel.from_xml_string(delivered), stat, vis_global, f"scene {scene!r}: mujoco_viewer"
            )
        xml_out.write_text(delivered)
        del model  # the compile was validation; the artifact is the XML
        return xml_out

    try:
        spec = mujoco.MjSpec.from_string(content)
        for mesh in spec.meshes:
            if mesh.name in sdf_names:
                mesh.needsdf = True
        model = spec.compile()
        if viewer_view:
            _check_free_camera_mode(model, f"scene {scene!r}: mujoco_viewer")
            extent, znear = float(model.stat.extent), float(model.vis.map.znear)
            _apply_viewer_view(viewer_view, model.stat, model.vis.global_, extent, znear)
            # the debug XML beside the .mjb carries the same start view; best
            # effort only, it must never turn a good .mjb into the fallback
            try:
                _apply_viewer_view(viewer_view, spec.stat, spec.visual.global_, extent, znear)
                xml_out.write_text(_view_exact_xml(spec.to_xml(), spec.stat, spec.visual.global_))
            except Exception as exc:  # noqa: BLE001
                logging.warning("debug XML %s left without the viewer start view: %s", xml_out, exc)
        mjb_out = composed_model_path(scene)
        mujoco.mj_saveModel(model, str(mjb_out))
        return mjb_out
    except ViewerViewError:
        raise  # a layout error, never the compile fallback
    except Exception as exc:  # noqa: BLE001
        logging.warning("MuJoCo compile failed, falling back to XML: %s%s", exc, _view_not_applied(viewer_view))
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
