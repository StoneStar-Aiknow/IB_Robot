"""Scene composition tests (so101-mujoco-runtime-platform tasks 2.1/2.2).

Covers the robot-agnostic composition API, the shared randomisation
placements (grid bit-compatibility with the historical keyframe grid, seeded
uniform sampling, and the issue #195 requirement that the composed model
carries no injected keyframes), the domain-keyed composed-model path, the
layout task declaration, the shared scene helpers and the dependency
direction (sim_models never imports robot_runtime).
"""

from __future__ import annotations

import ast
import importlib
import math
import os
import pkgutil
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sim_models.scene_compiler import (
    ViewerViewError,
    arm_rest_pairs,
    compose_mujoco_model,
    composed_model_path,
    free_joint_states_topic,
    get_gazebo_world_path,
    get_scene_layout,
    get_scene_task,
    load_mujoco_engine_plugins,
    mjcf_engine_extensions,
    mujoco_model_camera_names,
    randomization_placements,
    resolve_robot_spawn,
    scene_dynamic_objects,
    scene_viewer_view,
)

mujoco = pytest.importorskip("mujoco")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PKG_ROOT = Path(__file__).resolve().parents[1]
SCENE_MESHES = PKG_ROOT / "scenes" / "pick_banana" / "meshes"

# A minimal robot MJCF: six position joints (the arm joints the layout rest
# pose addresses) and one mesh asset (the SDF probe). The scene's
# banana/plate freejoints come from the pick_banana template include.
ROBOT_MJCF_TEMPLATE = """<mujoco model="robot_probe">
  <asset>
    <mesh name="probe_mesh" file="{meshes_dir}/plate_col_0.stl" scale="1 1 1"/>
  </asset>
  <worldbody>
    <body name="base">
      <geom type="mesh" mesh="probe_mesh"/>
      {joints_and_geoms}
    </body>
  </worldbody>
  <actuator>
    {actuators}
  </actuator>
</mujoco>
"""


@pytest.fixture
def robot_mjcf(tmp_path):
    joints = []
    actuators = []
    for j in range(1, 7):
        joints.append(f'<joint name="{j}" type="hinge" axis="0 0 1" range="-2 2" damping="1.5" armature="0.005"/>')
        actuators.append(f'<position name="{j}" joint="{j}" kp="100"/>')
    path = tmp_path / "robot.xml"
    path.write_text(
        ROBOT_MJCF_TEMPLATE.format(
            meshes_dir=SCENE_MESHES,
            joints_and_geoms="\n      ".join(joints),
            actuators="\n    ".join(actuators),
        ),
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture
def clean_domain(monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "123")
    yield "123"


def _model_of(path: Path):
    return mujoco.MjModel.from_binary_path(str(path))


# ---------------------------------------------------------------------------
# 2.1: randomization_placements
# ---------------------------------------------------------------------------


def _historical_grid(banana_cfg: dict) -> list[tuple[float, float, float, float]]:
    """The pre-issue-195 keyframe grid computation, inlined verbatim."""
    x_min, x_max = banana_cfg["x_range"]
    y_min, y_max = banana_cfg["y_range"]
    z_spawn = float(banana_cfg.get("z_spawn", 0.05))
    xs = np.linspace(x_min, x_max, int(banana_cfg.get("grid_cols", 5)))
    ys = np.linspace(y_min, y_max, int(banana_cfg.get("grid_rows", 7)))
    yaws = [2.0 * math.pi * i / int(banana_cfg.get("yaw_steps", 6)) for i in range(int(banana_cfg.get("yaw_steps", 6)))]
    return [(float(x), float(y), z_spawn, float(yaw)) for x in xs for y in ys for yaw in yaws]


def test_grid_placements_are_bit_identical_to_the_historical_grid():
    layout = get_scene_layout("pick_banana")
    placements = randomization_placements(layout)
    assert placements == _historical_grid(layout["randomization"]["banana"]), (
        "grid mode must reproduce the historical keyframe grid exactly (old indices keep their meaning)"
    )


def test_grid_size_and_order_are_stable():
    layout = get_scene_layout("pick_banana")
    placements = randomization_placements(layout)
    # 5 x divisions × 7 y divisions × 6 yaw steps
    assert len(placements) == 5 * 7 * 6
    rand = layout["randomization"]["banana"]
    x_min, x_max = rand["x_range"]
    y_min, y_max = rand["y_range"]
    yaw_step = 2.0 * math.pi / 6
    # index 0 → (x_min, y_min, z_spawn, 0); the innermost loop is yaw
    assert placements[0] == pytest.approx((x_min, y_min, 0.05, 0.0), abs=1e-9)
    assert placements[1] == pytest.approx((x_min, y_min, 0.05, yaw_step), abs=1e-9)
    # after yaw_steps indices the y advances
    assert placements[6] == pytest.approx((x_min, y_min + (y_max - y_min) / 6, 0.05, 0.0), abs=1e-9)
    # the last entry is (x_max, y_max, last yaw)
    assert placements[-1] == pytest.approx((x_max, y_max, 0.05, 5 * yaw_step), abs=1e-9)


def test_placements_are_empty_without_randomization():
    assert randomization_placements({}) == []
    assert randomization_placements({"randomization": {}}) == []
    assert randomization_placements({"randomization": {"banana": {}}}) == []


def test_uniform_mode_is_deterministic_and_seeded():
    base = {
        "x_range": [-0.03, 0.10],
        "y_range": [-0.02, 0.10],
        "z_spawn": 0.05,
        "mode": "uniform",
        "seed": 42,
        "count": 25,
    }
    first = randomization_placements({"randomization": {"banana": dict(base)}})
    second = randomization_placements({"randomization": {"banana": dict(base)}})
    assert len(first) == 25
    assert first == second, "same seed + count must give the same sequence"

    # an independent reference implementation with the same seed agrees
    rng = random.Random(42)
    expected = [
        (
            rng.uniform(-0.03, 0.10),
            rng.uniform(-0.02, 0.10),
            0.05,
            rng.uniform(0.0, 2.0 * math.pi),
        )
        for _ in range(25)
    ]
    assert first == expected

    other_seed = randomization_placements({"randomization": {"banana": {**base, "seed": 43}}})
    assert other_seed != first, "a different seed changes the sequence"


def test_uniform_mode_rejects_missing_seed_or_count():
    with pytest.raises(ValueError, match="seed"):
        randomization_placements({"randomization": {"banana": {"mode": "uniform", "count": 5}}})
    with pytest.raises(ValueError, match="seed"):
        randomization_placements({"randomization": {"banana": {"mode": "uniform"}}})


# ---------------------------------------------------------------------------
# input validation fires in the shared parser
# ---------------------------------------------------------------------------


def _uniform_banana(**overrides):
    config = {
        "mode": "uniform",
        "seed": 42,
        "count": 10,
        "x_range": [-0.03, 0.10],
        "y_range": [-0.02, 0.10],
        "z_spawn": 0.05,
    }
    config.update(overrides)
    return {"randomization": {"banana": config}}


@pytest.mark.parametrize(
    "field,value",
    [("count", 0), ("count", -3), ("count", 2.9), ("count", True), ("seed", 42.9), ("seed", True)],
)
def test_uniform_mode_rejects_non_positive_or_non_integer_seed_and_count(field, value):
    # count=0 used to be accepted and crashed later with ZeroDivisionError in
    # randomize(); floats and bools were silently truncated.
    with pytest.raises(ValueError, match=f"randomization.banana.{field}"):
        randomization_placements(_uniform_banana(**{field: value}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("x_range", [0.10, -0.03]),  # min > max
        ("x_range", [float("nan"), 0.10]),
        ("x_range", [-0.03, float("inf")]),
        ("x_range", [-0.03]),  # wrong length
        ("x_range", "wide"),  # not a pair at all
        ("y_range", [0.10, -0.02]),
        ("z_spawn", float("nan")),
        ("z_spawn", float("inf")),
        ("z_spawn", "high"),
    ],
)
def test_ranges_and_spawn_must_be_finite_ordered_numbers(field, value):
    # NaN ranges used to silently generate NaN coordinates.
    with pytest.raises(ValueError, match=f"randomization.banana.{field}"):
        randomization_placements(_uniform_banana(**{field: value}))


@pytest.mark.parametrize("field", ["grid_cols", "grid_rows", "yaw_steps"])
@pytest.mark.parametrize("value", [0, -1, 2.5, True])
def test_grid_divisions_must_be_positive_integers(field, value):
    layout = {"randomization": {"banana": {"x_range": [0, 0.1], "y_range": [0, 0.1], field: value}}}
    with pytest.raises(ValueError, match=f"randomization.banana.{field}"):
        randomization_placements(layout)


def test_grid_division_validation_also_applies_to_the_default_mode():
    # grid is the default mode, so the validation must fire without `mode:`.
    layout = {"randomization": {"banana": {"x_range": [0, 0.1], "y_range": [0, 0.1], "grid_cols": 0}}}
    with pytest.raises(ValueError, match="grid_cols"):
        randomization_placements(layout)


# ---------------------------------------------------------------------------
# arm_rest_pairs
# ---------------------------------------------------------------------------


def test_arm_rest_pairs_returns_the_validated_pairs():
    layout = {
        "randomization": {
            "arm_joint_names": ["1", "2", "3"],
            "arm_rest_qpos": [0.0, -1.58, 2],
        }
    }
    assert arm_rest_pairs(layout) == [("1", 0.0), ("2", -1.58), ("3", 2.0)]
    assert arm_rest_pairs({}) == []
    assert arm_rest_pairs({"randomization": {}}) == []


@pytest.mark.parametrize(
    "names,qpos,match",
    [
        ("12", [0.0, 1.0], "arm_joint_names must be a list"),  # a string would iterate per character
        (["1", "2"], 0.0, "arm_rest_qpos must be a list"),
        (["1", "2"], "01", "arm_rest_qpos must be a list"),
        (["1", "2"], [0.0, -1.0, 1.5], "same length"),
        (["1", "1"], [0.0, -1.0], "unique"),
        (["1", ""], [0.0, -1.0], "non-empty strings"),
        (["1", 2], [0.0, -1.0], "non-empty strings"),
        (["1", "2"], [0.0, float("nan")], "arm_rest_qpos entries"),
        (["1", "2"], [0.0, True], "arm_rest_qpos entries"),
        (["1", "2"], [0.0, "down"], "arm_rest_qpos entries"),
    ],
)
def test_arm_rest_pairs_rejects_invalid_layouts(names, qpos, match):
    with pytest.raises(ValueError, match=match):
        arm_rest_pairs({"randomization": {"arm_joint_names": names, "arm_rest_qpos": qpos}})


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        randomization_placements({"randomization": {"banana": {"mode": "spiral", "seed": 1, "count": 2}}})


# ---------------------------------------------------------------------------
# 2.1: the shared scene helpers (issue #195)
# ---------------------------------------------------------------------------


def test_scene_dynamic_objects_lists_dynamic_bodies_only():
    assert scene_dynamic_objects("pick_banana") == ["plate", "banana"]
    assert scene_dynamic_objects("empty") == []
    assert scene_dynamic_objects("does_not_exist") == []


def test_free_joint_states_topic_is_scene_namespaced():
    assert free_joint_states_topic("pick_banana") == "/sim/pick_banana/free_joint_states"
    assert free_joint_states_topic("other_scene") == "/sim/other_scene/free_joint_states"


# ---------------------------------------------------------------------------
# 2.1: compose_mujoco_model / composed_model_path
# ---------------------------------------------------------------------------


def test_composed_model_path_is_keyed_by_domain(monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "31")
    a = composed_model_path("pick_banana")
    monkeypatch.setenv("ROS_DOMAIN_ID", "32")
    b = composed_model_path("pick_banana")
    assert a != b
    assert a == Path(tempfile.gettempdir()) / "ibrobot_sim_models" / "31" / "pick_banana.mjb"
    monkeypatch.delenv("ROS_DOMAIN_ID")
    assert composed_model_path("pick_banana") == Path(tempfile.gettempdir()) / "ibrobot_sim_models" / "0" / (
        "pick_banana.mjb"
    )
    # no scene composes the robot-only model under a distinct name
    assert composed_model_path(None).name == "robot.mjb"


# ---------------------------------------------------------------------------
# D9: the generated Gazebo world is unique per construction
# ---------------------------------------------------------------------------


def test_gazebo_world_path_is_unique_per_construction(monkeypatch, tmp_path):
    """Two constructions of the same scene, under different domain ids, give
    two distinct files, each carrying its own content (D9: the world embeds
    the installation's absolute mesh path, so a fixed name let two worktrees
    overwrite each other)."""
    mesh_sentinel = "unique-mesh-sentinel"
    scene_dir = tmp_path / "pick_banana"
    scene_dir.mkdir()
    (scene_dir / "meshes").mkdir()
    (scene_dir / "pick_banana.world.template").write_text(
        f'<world name="demo">{{{{MESHES_DIR}}}}/{mesh_sentinel}</world>', encoding="utf-8"
    )

    def fake_scene_file(scene_name, platform):
        assert platform == "gazebo"
        assert scene_name == "pick_banana"
        return scene_dir / "pick_banana.world.template"

    monkeypatch.setattr("sim_models.scene_compiler.get_scene_file", fake_scene_file)
    monkeypatch.setenv("ROS_DOMAIN_ID", "41")
    first = get_gazebo_world_path("pick_banana")
    monkeypatch.setenv("ROS_DOMAIN_ID", "42")
    second = get_gazebo_world_path("pick_banana")
    assert first != second, "each construction must own its world file"
    assert first.name.startswith("sim_models_pick_banana_") and first.suffix == ".world"
    assert second.name.startswith("sim_models_pick_banana_") and second.suffix == ".world"
    content_first = first.read_text(encoding="utf-8")
    content_second = second.read_text(encoding="utf-8")
    assert f"/meshes/{mesh_sentinel}</world>" in content_first
    assert f"/meshes/{mesh_sentinel}</world>" in content_second
    assert content_first == content_second, "same scene and installation render identical content"


def test_compose_with_pick_banana_compiles_without_injected_keyframes(robot_mjcf, clean_domain):
    """Issue #195: placements travel through ResetWorld state overrides, not
    compiled-in keyframes — the composed model must carry no injected keys."""
    out = compose_mujoco_model(robot_mjcf, "pick_banana", ())
    assert out == composed_model_path("pick_banana")
    assert out.exists() and out.suffix == ".mjb"
    model = _model_of(out)
    assert model.nkey == 0, "no random_KKK/default keyframes may be injected any more"
    # the placement targets themselves are still part of the composed world
    for body in ("banana", "plate"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) >= 0
    grid = randomization_placements(get_scene_layout("pick_banana"))
    assert len(grid) == 5 * 7 * 6


def test_compose_with_empty_scene_includes_the_robot(robot_mjcf, clean_domain):
    out = compose_mujoco_model(robot_mjcf, "empty", ())
    assert out.exists() and out.suffix == ".mjb"
    model = _model_of(out)
    # the robot joints are present (the empty template now includes the robot)
    for j in range(1, 7):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(j)) >= 0


def test_compose_robot_only(robot_mjcf, clean_domain):
    out = compose_mujoco_model(robot_mjcf, None, ())
    assert out == composed_model_path(None)
    model = _model_of(out)
    for j in range(1, 7):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(j)) >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "banana_free") < 0


def test_compose_applies_sdf_to_the_given_meshes_only(robot_mjcf, clean_domain):
    plain = compose_mujoco_model(robot_mjcf, "pick_banana", ())
    plain_buffer = _model_of(plain).nbuffer
    with_sdf = compose_mujoco_model(robot_mjcf, "pick_banana", ("probe_mesh",))
    # SDF octrees are serialized into the model buffer: the probe mesh grows it
    assert _model_of(with_sdf).nbuffer > plain_buffer


# ---------------------------------------------------------------------------
# 2.2: the scene task declaration
# ---------------------------------------------------------------------------


def test_get_scene_task_returns_the_declaration():
    task = get_scene_task("pick_banana")
    assert task == {"executable": "pick_banana_task_node", "platforms": ["mujoco", "gazebo"]}


def test_get_scene_task_returns_none_without_a_declaration():
    # the empty scene has no layout.yaml at all
    assert get_scene_task("empty") is None
    assert get_scene_task("does_not_exist") is None


# ---------------------------------------------------------------------------
# sim-platform-registry D7: the one provider-path spawn-pose rule
# ---------------------------------------------------------------------------


def test_robot_spawn_unspecified_without_a_scene_or_explicit_spawn():
    # absent / null / {} all mean unspecified: the origin
    assert resolve_robot_spawn({}, None) == (0.0, 0.0, 0.0, 0.0)
    assert resolve_robot_spawn({"robot_spawn": None}, None) == (0.0, 0.0, 0.0, 0.0)
    assert resolve_robot_spawn({"robot_spawn": {}}, None) == (0.0, 0.0, 0.0, 0.0)


def test_robot_spawn_partial_coordinates_default_to_zero():
    assert resolve_robot_spawn({"robot_spawn": {"z": -0.03}}, None) == (0.0, 0.0, -0.03, 0.0)
    assert resolve_robot_spawn({"robot_spawn": {"yaw": 0.5}}, "pick_banana") == (0.0, 0.0, 0.0, 0.5)


def test_robot_spawn_non_zero_yaw_is_kept():
    spawn = resolve_robot_spawn({"robot_spawn": {"x": 1.0, "y": 2.0, "z": -0.03, "yaw": 1.25}}, None)
    assert spawn == (1.0, 2.0, -0.03, 1.25)


def test_robot_spawn_scene_layout_applies_when_not_configured():
    # pick_banana's layout declares (0.0, 0.30, -0.03, no yaw)
    assert resolve_robot_spawn({}, "pick_banana") == (0.0, 0.30, -0.03, 0.0)


def test_robot_spawn_empty_mapping_with_a_scene_uses_the_layout():
    # robot_spawn: {} is unspecified, not an override — the layout applies
    assert resolve_robot_spawn({"robot_spawn": {}}, "pick_banana") == (0.0, 0.30, -0.03, 0.0)
    assert resolve_robot_spawn({"robot_spawn": None}, "pick_banana") == (0.0, 0.30, -0.03, 0.0)


def test_robot_spawn_explicit_without_a_scene_is_honoured():
    # the D7 behaviour change: Gazebo without a scene now honours robot_spawn
    assert resolve_robot_spawn({"robot_spawn": {"x": 0.25, "y": -0.1, "z": -0.03, "yaw": 0.7}}, None) == (
        0.25,
        -0.1,
        -0.03,
        0.7,
    )


def test_robot_spawn_explicit_overrides_the_scene_layout():
    assert resolve_robot_spawn({"robot_spawn": {"x": 0.25}}, "pick_banana") == (0.25, 0.0, 0.0, 0.0)


def test_robot_spawn_scene_load_failure_falls_back_to_the_origin(monkeypatch):
    # a scene that fails to load is never consulted: the caller has already
    # fallen back to the empty scene, whose layout has no robot_spawn
    def _raise(scene_name):
        raise RuntimeError(f"layout explosion: {scene_name}")

    monkeypatch.setattr("sim_models.scene_compiler.get_scene_layout", _raise)
    assert resolve_robot_spawn({}, "no_such_scene") == (0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Dependency direction (D3): sim_models never imports robot_runtime
# ---------------------------------------------------------------------------


def test_no_sim_models_module_imports_robot_runtime():
    import sim_models

    problems = []
    for module_info in pkgutil.walk_packages(sim_models.__path__, prefix="sim_models."):
        module = importlib.import_module(module_info.name)
        path = Path(getattr(module, "__file__", "") or "")
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "robot_runtime" or name.startswith("robot_runtime.") for name in names):
                problems.append(f"{module_info.name} imports robot_runtime")
    assert not problems, problems


def test_package_xml_declares_no_robot_runtime_dependency():
    content = (PKG_ROOT / "package.xml").read_text(encoding="utf-8")
    assert "robot_runtime" not in content


# ---------------------------------------------------------------------------
# lekiwi-mujoco-runtime-platform D5.1: engine extensions in the compiler and
# the XML artifact for extension-declaring models
# ---------------------------------------------------------------------------

LIDAR_ROBOT_TEMPLATE = """<mujoco model="lidar_probe">
  <extension><plugin plugin="mujoco.plugin.lidar"/></extension>
  <worldbody>
    <body name="base_link" pos="0 0 0.05">
      <freejoint name="floating_base_joint"/>
      <site name="probe_lidar_site" pos="0 0 0.2"/>
      <geom type="box" size="0.1 0.1 0.03"/>
    </body>
  </worldbody>
  <sensor>
    <plugin name="probe_lidar" plugin="mujoco.plugin.lidar" objtype="site" objname="probe_lidar_site">
      <config key="resolution" value="8 2"/>
      <config key="azimuth_range" value="-3.14159265 3.14159265"/>
      <config key="elevation_range" value="0 0.5"/>
      <config key="max_range" value="20"/>
    </plugin>
    <framequat name="probe_imu_quat" objtype="site" objname="probe_lidar_site"/>
    <gyro name="probe_imu_gyro" site="probe_lidar_site"/>
    <accelerometer name="probe_imu_accel" site="probe_lidar_site"/>
  </sensor>
</mujoco>
"""


@pytest.fixture
def lidar_robot_mjcf(tmp_path):
    path = tmp_path / "lidar_robot.xml"
    path.write_text(LIDAR_ROBOT_TEMPLATE, encoding="utf-8")
    return str(path)


def test_mjcf_engine_extensions_reads_path_and_content(lidar_robot_mjcf):
    assert mjcf_engine_extensions(lidar_robot_mjcf) == ["mujoco.plugin.lidar"]
    assert mjcf_engine_extensions(LIDAR_ROBOT_TEMPLATE) == ["mujoco.plugin.lidar"]
    # no <extension> section: none declared; unparseable text is not a model
    assert mjcf_engine_extensions(ROBOT_MJCF_TEMPLATE.format(meshes_dir="x", joints_and_geoms="", actuators="")) == []
    assert mjcf_engine_extensions("<not-mujoco/>") == []


def test_compose_delivers_xml_for_engine_extension_models(lidar_robot_mjcf, clean_domain):
    """D5.1: an extension-declaring model is validated by a full compile and
    delivered as the compiled spec's XML (never a .mjb — the binary bakes the
    compiling process's global plugin slot ids)."""
    # earlier tests in this domain may have left a robot .mjb behind
    composed_model_path("empty").unlink(missing_ok=True)
    out = compose_mujoco_model(lidar_robot_mjcf, "empty", ())
    assert out.suffix == ".xml"
    assert out == composed_model_path("empty").with_suffix(".xml")
    delivered = out.read_text(encoding="utf-8")
    assert mjcf_engine_extensions(delivered) == ["mujoco.plugin.lidar"]
    # no .mjb artifact is written next to it
    assert not composed_model_path("empty").exists()
    # the delivered XML recompiles in THIS process (extension already loaded)
    spec = mujoco.MjSpec.from_string(delivered)
    model = spec.compile()
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "probe_lidar_site") >= 0
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(model.nsensor)]
    assert "probe_lidar" in names and "probe_imu_gyro" in names


def test_extension_free_models_keep_the_mjb_artifact(robot_mjcf, clean_domain):
    """The rule, pinned: the same composition call produces a .mjb without an
    extension declaration and an .xml with one (D5.1)."""
    plain = compose_mujoco_model(robot_mjcf, "empty", ())
    assert plain.suffix == ".mjb" and plain.exists()


def test_extension_with_sdf_meshes_fails_closed(lidar_robot_mjcf, clean_domain):
    with pytest.raises(ValueError, match="cannot also use SDF mesh collision"):
        compose_mujoco_model(lidar_robot_mjcf, "empty", ("some_mesh"))


def test_extension_model_compile_error_raises(tmp_path, clean_domain):
    """D5.1 compile-error policy: an extension-declaring model that fails to
    compile RAISES the original MuJoCo error — the silent XML fallback stays
    reserved for extension-free models (there it is debugging support, here
    it would hand the node an unvalidated model)."""
    bad = LIDAR_ROBOT_TEMPLATE.replace(
        '<gyro name="probe_imu_gyro" site="probe_lidar_site"/>',
        '<gyro name="probe_imu_gyro" site="no_such_site"/>',
    )
    path = tmp_path / "bad_lidar_robot.xml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="no_such_site"):
        compose_mujoco_model(str(path), "empty", ())


def test_extension_model_without_the_mujoco_package_raises(robot_mjcf, tmp_path, clean_domain, monkeypatch):
    """Review round 1: with the mujoco package unavailable, an
    extension-declaring model must RAISE — returning the unvalidated XML
    would defeat the D5.1 compile-error policy through the ImportError path.
    An extension-free model keeps the documented XML fallback."""
    import sys

    monkeypatch.setitem(sys.modules, "mujoco", None)  # import mujoco now fails
    lidar = tmp_path / "lidar_robot.xml"
    lidar.write_text(LIDAR_ROBOT_TEMPLATE, encoding="utf-8")
    with pytest.raises(ImportError, match="engine extensions.*mujoco Python package"):
        compose_mujoco_model(str(lidar), "empty", ())
    out = compose_mujoco_model(robot_mjcf, "empty", ())
    assert out.suffix == ".xml"


def test_load_mujoco_engine_plugins_names_the_apt_package_for_unknown_extensions():
    with pytest.raises(RuntimeError, match="ros-humble-mujoco-3d-lidar"):
        load_mujoco_engine_plugins(("mujoco.plugin.not_a_thing",))


def test_fresh_process_compose_and_recompile(lidar_robot_mjcf, clean_domain, tmp_path):
    """The real compose path in a FRESH process produces the XML, and a second
    fresh process (its own plugin registration state) recompiles it — the
    artifact carries no compile-process state (D5.1)."""
    import subprocess
    import sys

    compose_script = (
        "from sim_models.scene_compiler import compose_mujoco_model, composed_model_path\n"
        f"out = compose_mujoco_model({lidar_robot_mjcf!r}, 'empty', ())\n"
        "assert out.suffix == '.xml', out\n"
        "assert out == composed_model_path('empty').with_suffix('.xml'), out\n"
        "print(out)\n"
    )
    first = subprocess.run(
        [sys.executable, "-c", compose_script], capture_output=True, text=True, timeout=120, check=False
    )
    assert first.returncode == 0, first.stderr
    delivered = Path(first.stdout.strip().splitlines()[-1])
    assert delivered.exists() and delivered.suffix == ".xml"

    recompile_script = (
        "import ctypes\n"
        "from pathlib import Path\n"
        "import mujoco\n"
        "from ament_index_python import get_resource\n"
        "content, prefix = get_resource('mujoco_plugins', 'mujoco_3d_lidar')\n"
        "ctypes.CDLL(str(Path(prefix) / content / 'libmujoco_3d_lidar.so'), mode=ctypes.RTLD_GLOBAL)\n"
        f"spec = mujoco.MjSpec.from_string(Path({str(delivered)!r}).read_text())\n"
        "model = spec.compile()\n"
        "names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(model.nsensor)]\n"
        "assert 'probe_lidar' in names and 'probe_imu_accel' in names, names\n"
        "assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'probe_lidar_site') >= 0\n"
        "print('RECOMPILED', model.nbody, model.nsensor, model.nplugin)\n"
    )
    second = subprocess.run(
        [sys.executable, "-c", recompile_script], capture_output=True, text=True, timeout=120, check=False
    )
    assert second.returncode == 0, second.stderr
    assert "RECOMPILED" in second.stdout


# ---------------------------------------------------------------------------
# MuJoCo viewer start view (layout mujoco_viewer)
# ---------------------------------------------------------------------------


def _free_camera(model):
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    return camera


def test_compose_pick_banana_sets_the_viewer_free_camera_without_touching_the_extent(robot_mjcf, clean_domain):
    view = scene_viewer_view("pick_banana")
    assert view is not None
    model = _model_of(compose_mujoco_model(robot_mjcf, "pick_banana", ()))
    camera = _free_camera(model)
    assert model.vis.global_.cameraid == -1, "the viewer starts on the (draggable) free camera"
    assert tuple(camera.lookat) == pytest.approx(view["lookat"])
    assert camera.azimuth == pytest.approx(view["azimuth"])
    assert camera.elevation == pytest.approx(view["elevation"])
    # the apparent distance comes from the fovy: same frame half-height as the
    # reference 45 deg fovy at the declared distance
    frame = camera.distance * math.tan(math.radians(model.vis.global_.fovy) / 2)
    assert frame == pytest.approx(view["distance"] * math.tan(math.radians(45.0) / 2), rel=1e-5)
    # the extent (znear/zfar scale of every camera) is the auto-computed one
    recompiled = mujoco.MjModel.from_xml_path(str(composed_model_path("pick_banana").with_suffix(".xml")))
    assert model.stat.extent == pytest.approx(recompiled.stat.extent)


def test_compose_without_a_viewer_view_keeps_the_default_free_camera(robot_mjcf, clean_domain):
    assert scene_viewer_view(None) is None
    assert scene_viewer_view("empty") is None
    model = _model_of(compose_mujoco_model(robot_mjcf, None, ()))
    assert model.vis.global_.fovy == pytest.approx(45.0)


@pytest.mark.parametrize(
    "value",
    [
        None,  # an explicit empty "mujoco_viewer:" is not "no view"
        "front",
        {"lookat": [0, 0, 0], "distance": 1.0, "azimuth": 0},
        {"lookat": [0, 0], "distance": 1.0, "azimuth": 0, "elevation": -30},
        {"lookat": [0, 0, 0], "distance": 0.0, "azimuth": 0, "elevation": -30},
        {"lookat": [0, 0, 0], "distance": 1.0, "azimuth": "x", "elevation": -30},
        {"lookat": [0, 0, 0], "distance": 1.0, "azimuth": 0, "elevation": -90},
        {"lookat": [0, 0, 0], "distnace": 1.0, "distance": 1.0, "azimuth": 0, "elevation": -30},
        {"lookat": [0, 0, 0], "distance": 1.0, "azimuth": 0, "elevation": -30, "fovy": 30},
    ],
)
def test_compose_rejects_a_malformed_viewer_view(robot_mjcf, clean_domain, monkeypatch, value):
    layout = {**get_scene_layout("pick_banana"), "mujoco_viewer": value}
    monkeypatch.setattr("sim_models.scene_compiler.get_scene_layout", lambda _scene: layout)
    with pytest.raises(ViewerViewError):
        compose_mujoco_model(robot_mjcf, "pick_banana", ())


_TEST_VIEW = {"lookat": [0.1, 0.06, 0.0], "distance": 0.72, "azimuth": -110, "elevation": -55}


def test_extension_model_carries_the_viewer_view_with_an_unchanged_extent(lidar_robot_mjcf, clean_domain, monkeypatch):
    """The delivered XML (engine-extension path) carries the start view, and the
    extent the C++ node recompiles is exactly the one the plain path yields."""
    composed_model_path("empty").unlink(missing_ok=True)
    plain_xml = compose_mujoco_model(lidar_robot_mjcf, "empty", ()).read_text(encoding="utf-8")
    plain = mujoco.MjModel.from_xml_string(plain_xml)

    layout = {**get_scene_layout("empty"), "mujoco_viewer": _TEST_VIEW}
    monkeypatch.setattr("sim_models.scene_compiler.get_scene_layout", lambda _scene: layout)
    out = compose_mujoco_model(lidar_robot_mjcf, "empty", ())
    assert out.suffix == ".xml"
    model = mujoco.MjModel.from_xml_string(out.read_text(encoding="utf-8"))
    assert model.stat.extent == plain.stat.extent
    assert model.vis.map.znear == plain.vis.map.znear and model.vis.map.zfar == plain.vis.map.zfar
    camera = _free_camera(model)
    assert tuple(camera.lookat) == pytest.approx(_TEST_VIEW["lookat"])
    assert camera.azimuth == pytest.approx(-110) and camera.elevation == pytest.approx(-55)


def test_compile_fallback_warns_that_the_viewer_view_was_not_applied(robot_mjcf, clean_domain, monkeypatch, caplog):
    layout = {**get_scene_layout("pick_banana"), "mujoco_viewer": _TEST_VIEW}
    monkeypatch.setattr("sim_models.scene_compiler.get_scene_layout", lambda _scene: layout)

    def broken(_content):
        raise RuntimeError("probe compile failure")

    monkeypatch.setattr(mujoco.MjSpec, "from_string", staticmethod(broken))
    with caplog.at_level("WARNING"):
        out = compose_mujoco_model(robot_mjcf, "pick_banana", ())
    assert out.suffix == ".xml"
    assert "mujoco_viewer start view was NOT applied" in caplog.text


# The out-of-range numbers must fail closed with ViewerViewError on every
# delivery path: never an OverflowError, never a 0 / 180 deg fovy, never the
# silent compile-fallback XML.
_NUMERIC_EDGE_VIEWS = [
    pytest.param({**_TEST_VIEW, "distance": 10**400}, id="huge-int-distance"),
    pytest.param({**_TEST_VIEW, "lookat": [10**400, 0, 0]}, id="huge-int-lookat"),
    pytest.param({**_TEST_VIEW, "azimuth": -(10**400)}, id="huge-int-azimuth"),
    pytest.param({**_TEST_VIEW, "distance": 1e308}, id="huge-distance-fovy-180"),
    pytest.param({**_TEST_VIEW, "distance": 5e-324}, id="tiny-distance-fovy-0"),
    pytest.param({**_TEST_VIEW, "elevation": 89.000001}, id="elevation-above-viewer-clamp"),
    pytest.param({**_TEST_VIEW, "elevation": -89.5}, id="elevation-below-viewer-clamp"),
    pytest.param({**_TEST_VIEW, "lookat": [1.0e39, 0, 0]}, id="lookat-beyond-float32"),
    pytest.param({**_TEST_VIEW, "distance": 1e-44}, id="tiny-distance-zero-frustum"),
    pytest.param({**_TEST_VIEW, "distance": 1e-39}, id="tiny-distance-inf-projection"),
]

# Valid views that MjSpec.to_xml()'s 6 significant digits would have changed:
# a float32 elevation just inside the bound (printed -89), a fovy just below
# 180 deg (printed 180), and a lookat / angles finer than 6 digits, which at a
# small framing moved the view by more than the frame itself.
_FINE_VIEWS = [
    pytest.param({**_TEST_VIEW, "elevation": -88.99999}, id="elevation-just-inside"),
    pytest.param(None, id="fovy-just-below-180"),
    pytest.param(
        {
            "lookat": [1.23456789, -0.98765432, 0.0123456789],
            "distance": 1e-6,
            "azimuth": -110.123456789,
            "elevation": -55.123456789,
        },
        id="fine-lookat-and-angles",
    ),
]


def _fine_view(view, extent):
    if view is not None:
        return view
    import numpy as np

    fovy = float(np.nextafter(np.float32(180.0), np.float32(0.0)))  # 179.99998, a valid float32 fovy
    return {**_TEST_VIEW, "distance": 1.5 * extent * math.tan(math.radians(fovy / 2)) / math.tan(math.radians(22.5))}


def _expected_free_camera(view):
    import numpy as np

    return tuple(view["lookat"]), float(np.float32(view["azimuth"])), float(np.float32(view["elevation"]))


@pytest.mark.parametrize("view", _FINE_VIEWS)
def test_plain_model_keeps_a_fine_view_exactly(robot_mjcf, clean_domain, monkeypatch, view):
    composed_model_path("pick_banana").unlink(missing_ok=True)
    extent = float(_model_of(compose_mujoco_model(robot_mjcf, "pick_banana", ())).stat.extent)
    view = _fine_view(view, extent)
    layout = {**get_scene_layout("pick_banana"), "mujoco_viewer": view}
    monkeypatch.setattr("sim_models.scene_compiler.get_scene_layout", lambda _scene: layout)
    model = _model_of(compose_mujoco_model(robot_mjcf, "pick_banana", ()))
    camera = _free_camera(model)
    assert (tuple(camera.lookat), camera.azimuth, camera.elevation) == _expected_free_camera(view)
    # the debug XML beside the .mjb carries the same view
    debug = mujoco.MjModel.from_xml_path(str(composed_model_path("pick_banana").with_suffix(".xml")))
    assert tuple(debug.stat.center) == tuple(model.stat.center)
    assert debug.vis.global_.fovy == model.vis.global_.fovy


@pytest.mark.parametrize("view", _FINE_VIEWS)
def test_extension_model_delivers_a_fine_view_exactly(lidar_robot_mjcf, clean_domain, monkeypatch, view):
    composed_model_path("empty").unlink(missing_ok=True)
    plain = mujoco.MjModel.from_xml_string(compose_mujoco_model(lidar_robot_mjcf, "empty", ()).read_text("utf-8"))
    view = _fine_view(view, float(plain.stat.extent))
    layout = {**get_scene_layout("empty"), "mujoco_viewer": view}
    monkeypatch.setattr("sim_models.scene_compiler.get_scene_layout", lambda _scene: layout)
    delivered = mujoco.MjModel.from_xml_string(compose_mujoco_model(lidar_robot_mjcf, "empty", ()).read_text("utf-8"))
    camera = _free_camera(delivered)
    assert (tuple(camera.lookat), camera.azimuth, camera.elevation) == _expected_free_camera(view)
    assert 0.0 < delivered.vis.global_.fovy < 180.0
    assert delivered.stat.extent == plain.stat.extent, "the exact view rewrite leaves the rest of the model alone"


def test_extension_model_rejects_a_delivered_view_that_differs(lidar_robot_mjcf, clean_domain, monkeypatch):
    """The read-back check itself: a delivered XML that lost the view fails closed."""
    layout = {**get_scene_layout("empty"), "mujoco_viewer": _TEST_VIEW}
    monkeypatch.setattr("sim_models.scene_compiler.get_scene_layout", lambda _scene: layout)
    monkeypatch.setattr(
        "sim_models.scene_compiler._view_exact_xml", lambda xml, _stat, _global: xml.replace("-110", "-111")
    )
    with pytest.raises(ViewerViewError, match="delivered XML"):
        compose_mujoco_model(lidar_robot_mjcf, "empty", ())


def _with_extent(mjcf_path, extent):
    """The robot MJCF with a declared model extent (scales every znear/zfar)."""
    path = Path(mjcf_path)
    text = path.read_text(encoding="utf-8")
    head, sep, rest = text.partition(">")  # end of the <mujoco ...> tag
    assert head.lstrip().startswith("<mujoco"), "the robot template opens with <mujoco>"
    out = path.with_name(f"extent_{path.name}")
    out.write_text(f'{head}{sep}<statistic extent="{extent!r}"/>{rest}', encoding="utf-8")
    return str(out)


def _with_visual_global(mjcf_path, attrs):
    """The robot MJCF with extra ``visual/global`` attributes."""
    path = Path(mjcf_path)
    text = path.read_text(encoding="utf-8")
    head, sep, rest = text.partition(">")  # end of the <mujoco ...> tag
    assert head.lstrip().startswith("<mujoco"), "the robot template opens with <mujoco>"
    out = path.with_name(f"global_{path.name}")
    out.write_text(f"{head}{sep}<visual><global {attrs}/></visual>{rest}", encoding="utf-8")
    return str(out)


@pytest.mark.parametrize(
    ("attrs", "match"),
    [('orthographic="true"', "orthographic"), ('cameraid="0"', "cameraid=0")],
)
def test_compose_rejects_a_view_the_viewer_would_not_show(robot_mjcf, clean_domain, monkeypatch, attrs, match):
    """An orthographic free camera reads fovy as a length, and a valid cameraid
    starts simulate (and Ctrl+A) on a fixed camera: either breaks the view."""
    robot = _with_visual_global(robot_mjcf, attrs)
    without_view = {k: v for k, v in get_scene_layout("pick_banana").items() if k != "mujoco_viewer"}
    monkeypatch.setattr("sim_models.scene_compiler.get_scene_layout", lambda _scene: without_view)
    plain = _model_of(compose_mujoco_model(robot, "pick_banana", ()))
    assert plain.ncam > 0, "pick_banana has fixed cameras for cameraid to name"
    assert (plain.vis.global_.orthographic, plain.vis.global_.cameraid) != (0, -1), "the setting reaches the model"
    layout = {**without_view, "mujoco_viewer": _TEST_VIEW}
    monkeypatch.setattr("sim_models.scene_compiler.get_scene_layout", lambda _scene: layout)
    with pytest.raises(ViewerViewError, match=match):
        compose_mujoco_model(robot, "pick_banana", ())


def test_compose_accepts_a_cameraid_outside_the_model_cameras(robot_mjcf, clean_domain, monkeypatch):
    """simulate falls back to the free camera for an id outside [0, ncam)."""
    robot = _with_visual_global(robot_mjcf, 'cameraid="999"')
    layout = {**get_scene_layout("pick_banana"), "mujoco_viewer": _TEST_VIEW}
    monkeypatch.setattr("sim_models.scene_compiler.get_scene_layout", lambda _scene: layout)
    model = _model_of(compose_mujoco_model(robot, "pick_banana", ()))
    assert model.vis.global_.cameraid == 999 and model.ncam < 999
    assert model.vis.global_.azimuth == pytest.approx(_TEST_VIEW["azimuth"])


def test_extension_model_rejects_an_orthographic_free_camera(lidar_robot_mjcf, clean_domain, monkeypatch):
    robot = _with_visual_global(lidar_robot_mjcf, 'orthographic="true"')
    layout = {**get_scene_layout("empty"), "mujoco_viewer": _TEST_VIEW}
    monkeypatch.setattr("sim_models.scene_compiler.get_scene_layout", lambda _scene: layout)
    with pytest.raises(ViewerViewError, match="orthographic"):
        compose_mujoco_model(robot, "empty", ())


# A wide fovy at a large near: top = near * tan(fovy / 2) overflows a C float,
# which used to pass as a finite (zero) projection scale.
_OVERFLOW_TOP_VIEW = {**_TEST_VIEW, "lookat": [0.0, 0.0, 0.0], "distance": 1e42}


def test_compose_rejects_a_view_whose_frustum_overflows(robot_mjcf, clean_domain, monkeypatch, caplog):
    robot = _with_extent(robot_mjcf, 1e35)
    composed_model_path("pick_banana").unlink(missing_ok=True)
    plain = _model_of(compose_mujoco_model(robot, None, ()))
    assert plain.stat.extent == pytest.approx(1e35), "the declared extent reaches the composed model"
    layout = {**get_scene_layout("pick_banana"), "mujoco_viewer": _OVERFLOW_TOP_VIEW}
    monkeypatch.setattr("sim_models.scene_compiler.get_scene_layout", lambda _scene: layout)
    with caplog.at_level("WARNING"), pytest.raises(ViewerViewError, match="float32 projection"):
        compose_mujoco_model(robot, "pick_banana", ())
    assert "falling back to XML" not in caplog.text


def test_extension_model_rejects_a_view_whose_frustum_overflows(lidar_robot_mjcf, clean_domain, monkeypatch):
    robot = _with_extent(lidar_robot_mjcf, 1e35)
    layout = {**get_scene_layout("empty"), "mujoco_viewer": _OVERFLOW_TOP_VIEW}
    monkeypatch.setattr("sim_models.scene_compiler.get_scene_layout", lambda _scene: layout)
    with pytest.raises(ViewerViewError, match="float32 projection"):
        compose_mujoco_model(robot, "empty", ())


# lookat in float32 range, but the free camera sits 1.5 * extent away from it
# and its GL position (a C float) overflowed.
_OVERFLOW_POSITION_VIEW = {"lookat": [3.0e38, 0.0, 0.0], "distance": 1e38, "azimuth": 180, "elevation": -10}


# finite eye components at a normal model scale, whose modelview translation
# (a float dot product of the eye with a camera axis) overflowed
_OVERFLOW_MODELVIEW_VIEW = {"lookat": [3.0e38, 3.0e38, 0.0], "distance": 0.72, "azimuth": 45, "elevation": -10}


@pytest.mark.parametrize(
    ("extent", "view"),
    [(1e38, _OVERFLOW_POSITION_VIEW), (None, _OVERFLOW_MODELVIEW_VIEW)],
    ids=["eye-position", "modelview-translation"],
)
def test_compose_rejects_a_view_whose_camera_position_overflows(
    robot_mjcf, clean_domain, monkeypatch, caplog, extent, view
):
    robot = robot_mjcf if extent is None else _with_extent(robot_mjcf, extent)
    layout = {**get_scene_layout("pick_banana"), "mujoco_viewer": view}
    monkeypatch.setattr("sim_models.scene_compiler.get_scene_layout", lambda _scene: layout)
    composed_model_path("pick_banana").unlink(missing_ok=True)
    with caplog.at_level("WARNING"), pytest.raises(ViewerViewError, match="float32 range"):
        compose_mujoco_model(robot, "pick_banana", ())
    assert "falling back to XML" not in caplog.text
    assert not composed_model_path("pick_banana").exists()


@pytest.mark.parametrize(
    ("extent", "view"),
    [(1e38, _OVERFLOW_POSITION_VIEW), (None, _OVERFLOW_MODELVIEW_VIEW)],
    ids=["eye-position", "modelview-translation"],
)
def test_extension_model_rejects_a_view_whose_camera_position_overflows(
    lidar_robot_mjcf, clean_domain, monkeypatch, extent, view
):
    robot = lidar_robot_mjcf if extent is None else _with_extent(lidar_robot_mjcf, extent)
    layout = {**get_scene_layout("empty"), "mujoco_viewer": view}
    monkeypatch.setattr("sim_models.scene_compiler.get_scene_layout", lambda _scene: layout)
    with pytest.raises(ViewerViewError, match="float32 range"):
        compose_mujoco_model(robot, "empty", ())


# lookats: the origin, a normal scene point, and points near the float32 limit
# on one or several axes, where the camera offset (1.5 * extent, either sign)
# or the modelview dot product decides overflow
_SWEEP_LOOKATS = [
    (0.0, 0.0, 0.0),
    (1.0, -0.5, 0.2),
    (-3.0e38, 0.0, 0.0),
    (3.4e38, 0.0, 0.0),
    (3.0e38, 3.0e38, 0.0),
    (2.0e38, -2.0e38, 2.0e38),
    (1.0e38, 1.0e38, 1.0e38),
]


def _modelview_translation(gl):
    """mjr_lookAt's float modelview translation for one GL camera (eye).

    In MuJoCo's order: normalise forward, side = forward x up normalised,
    up = side x forward normalised, then the three dot products with the eye,
    every step (partial sums included) in floats.
    """
    import numpy as np

    f32 = np.float32
    eye, forward, up = (np.asarray(v, dtype=f32) for v in (gl.pos, gl.forward, gl.up))
    partials = []
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):

        def normalise(v):
            # MuJoCo: scl = 1.0f / length, then multiply each component
            scale = f32(f32(1.0) / f32(np.sqrt(f32(f32(v[0] * v[0]) + f32(v[1] * v[1]) + f32(v[2] * v[2])))))
            return (v * scale).astype(f32)

        def dot(a, b):
            first = f32(f32(a[0] * b[0]) + f32(a[1] * b[1]))
            partials.append(first)
            return f32(first + f32(a[2] * b[2]))

        forward = normalise(forward)
        side = normalise(np.cross(forward, up).astype(f32))
        up = normalise(np.cross(side, forward).astype(f32))
        translation = np.array([-dot(side, eye), -dot(up, eye), dot(forward, eye)], dtype=f32)
    return np.concatenate([translation, np.asarray(partials, dtype=f32)])


@pytest.mark.parametrize("extent", [1e-3, 1.0, 1e3, 1e30, 1e35, 1e38])
def test_free_camera_is_finite_for_every_accepted_view(extent):
    """Every view _apply_viewer_view accepts, across model scales (near =
    znear * extent, camera offset = 1.5 * extent), lookats, directions and
    distances, gives mjv_updateScene finite float32 eye positions, a finite
    modelview translation and a finite frustum with a finite, positive
    projection scale; the rest raise before anything is written."""
    import numpy as np

    from sim_models.scene_compiler import _apply_viewer_view

    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><statistic extent="{extent!r}"/><worldbody><geom size=".1"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    extent, znear = float(model.stat.extent), float(model.vis.map.znear)
    accepted = rejected = 0
    for distance in np.logspace(-50, 45, 191):
        for lookat in _SWEEP_LOOKATS:
            for azimuth, elevation in ((180.0, -10.0), (0.0, -10.0), (45.0, -10.0), (-110.0, -55.0)):
                view = {
                    "lookat": lookat,
                    "distance": float(distance),
                    "azimuth": azimuth,
                    "elevation": elevation,
                }
                before = (tuple(model.stat.center), float(model.vis.global_.fovy))
                try:
                    _apply_viewer_view(view, model.stat, model.vis.global_, extent, znear)
                except ViewerViewError:
                    rejected += 1
                    assert (tuple(model.stat.center), float(model.vis.global_.fovy)) == before, "nothing written"
                    continue
                accepted += 1
                scene = mujoco.MjvScene(model, 10)
                camera = _free_camera(model)
                mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
                where = f"extent {extent} lookat {lookat} distance {distance} az {azimuth}"
                for gl in scene.camera:
                    assert np.isfinite(np.asarray(gl.pos, dtype=np.float32)).all(), f"{where}: {gl.pos}"
                    assert np.isfinite(_modelview_translation(gl)).all(), f"{where}: modelview of {gl.pos}"
                    near, top, bottom = (np.float32(v) for v in (gl.frustum_near, gl.frustum_top, gl.frustum_bottom))
                    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                        height = top - bottom
                        scale = (near + near) / height
                    for value in (near, top, height, scale):
                        assert np.isfinite(value) and value > 0, f"{where}: {gl}"
    assert accepted and rejected


@pytest.mark.parametrize("elevation", [-89, 89, -89.0, 89.0])
def test_viewer_view_accepts_the_viewer_elevation_bounds(monkeypatch, elevation):
    # mjv_moveCamera clamps the free camera elevation to [-89, 89]: the bounds
    # themselves are views the viewer keeps on the first mouse action
    layout = {"mujoco_viewer": {**_TEST_VIEW, "elevation": elevation}}
    monkeypatch.setattr("sim_models.scene_compiler.get_scene_layout", lambda _scene: layout)
    assert scene_viewer_view("pick_banana")["elevation"] == float(elevation)


def test_viewer_view_rejects_a_huge_elevation_without_a_cast_warning(monkeypatch):
    import warnings

    layout = {"mujoco_viewer": {**_TEST_VIEW, "elevation": 1e39}}
    monkeypatch.setattr("sim_models.scene_compiler.get_scene_layout", lambda _scene: layout)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ViewerViewError):
            scene_viewer_view("pick_banana")


# Finite azimuths that a C float cannot hold as given: reduced to one turn
# before MuJoCo stores them (expected free-camera azimuth in degrees).
_LARGE_AZIMUTHS = [
    pytest.param(1.0e39, -8.0, id="float32-overflow"),
    pytest.param(-110 + 360 * 10**6, -110.0, id="float32-precision-loss"),
    pytest.param(250, -110.0, id="one-turn-equivalent"),
]


@pytest.mark.parametrize(("azimuth", "expected"), _LARGE_AZIMUTHS)
def test_compose_reduces_a_large_azimuth_to_one_turn(robot_mjcf, clean_domain, monkeypatch, caplog, azimuth, expected):
    layout = {**get_scene_layout("pick_banana"), "mujoco_viewer": {**_TEST_VIEW, "azimuth": azimuth}}
    monkeypatch.setattr("sim_models.scene_compiler.get_scene_layout", lambda _scene: layout)
    with caplog.at_level("WARNING"):
        out = compose_mujoco_model(robot_mjcf, "pick_banana", ())
    assert out.suffix == ".mjb" and "falling back to XML" not in caplog.text
    model = _model_of(out)
    assert model.vis.global_.azimuth == pytest.approx(expected)
    assert _free_camera(model).azimuth == pytest.approx(expected)


@pytest.mark.parametrize(("azimuth", "expected"), _LARGE_AZIMUTHS)
def test_extension_model_reduces_a_large_azimuth_to_one_turn(
    lidar_robot_mjcf, clean_domain, monkeypatch, azimuth, expected
):
    layout = {**get_scene_layout("empty"), "mujoco_viewer": {**_TEST_VIEW, "azimuth": azimuth}}
    monkeypatch.setattr("sim_models.scene_compiler.get_scene_layout", lambda _scene: layout)
    out = compose_mujoco_model(lidar_robot_mjcf, "empty", ())
    model = mujoco.MjModel.from_xml_string(out.read_text(encoding="utf-8"))
    assert model.vis.global_.azimuth == pytest.approx(expected)
    assert _free_camera(model).azimuth == pytest.approx(expected)


@pytest.mark.parametrize("view", _NUMERIC_EDGE_VIEWS)
def test_compose_rejects_numeric_edge_views_without_the_compile_fallback(
    robot_mjcf, clean_domain, monkeypatch, caplog, view
):
    layout = {**get_scene_layout("pick_banana"), "mujoco_viewer": view}
    monkeypatch.setattr("sim_models.scene_compiler.get_scene_layout", lambda _scene: layout)
    composed_model_path("pick_banana").unlink(missing_ok=True)
    with caplog.at_level("WARNING"), pytest.raises(ViewerViewError):
        compose_mujoco_model(robot_mjcf, "pick_banana", ())
    assert "falling back to XML" not in caplog.text
    assert not composed_model_path("pick_banana").exists(), "no model artifact for a rejected view"


@pytest.mark.parametrize("view", _NUMERIC_EDGE_VIEWS)
def test_extension_model_rejects_numeric_edge_views(lidar_robot_mjcf, clean_domain, monkeypatch, view):
    layout = {**get_scene_layout("empty"), "mujoco_viewer": view}
    monkeypatch.setattr("sim_models.scene_compiler.get_scene_layout", lambda _scene: layout)
    with pytest.raises(ViewerViewError):
        compose_mujoco_model(lidar_robot_mjcf, "empty", ())


def test_viewer_fovy_is_checked_after_float32_rounding():
    """A distance whose double fovy is just below 180 deg rounds to 180.0 in
    MuJoCo's float field and must be rejected; a normal one is returned as
    the exact float32 value MuJoCo stores, strictly inside (0, 180)."""
    import numpy as np

    from sim_models.scene_compiler import _viewer_fovy

    extent = 2.437745426586574  # the pick_banana composition's extent
    camera_distance = 1.5 * extent
    fovy = 180.0 - 1e-6
    assert fovy < 180.0 and float(np.float32(fovy)) == 180.0
    distance = camera_distance * math.tan(math.radians(fovy / 2)) / math.tan(math.radians(22.5))
    with pytest.raises(ViewerViewError):
        _viewer_fovy({**_TEST_VIEW, "distance": distance}, extent, 0.01)
    stored = _viewer_fovy(_TEST_VIEW, extent, 0.01)
    assert 0.0 < stored < 180.0 and stored == float(np.float32(stored))
    for bad_extent in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ViewerViewError):
            _viewer_fovy(_TEST_VIEW, bad_extent, 0.01)


# ---------------------------------------------------------------------------
# mujoco_model_camera_names: the cameras the simulator's model carries
# ---------------------------------------------------------------------------

_CAMERAS_MJCF = (
    '<mujoco><worldbody><camera name="first"/><body name="b"><camera/><camera name="last"/></body></worldbody></mujoco>'
)


def test_model_camera_names_follow_the_model_id_order(tmp_path):
    """XML and binary models list their cameras in mjModel id order; an
    unnamed camera is ''."""
    xml = tmp_path / "model.xml"
    xml.write_text(_CAMERAS_MJCF, encoding="utf-8")
    binary = tmp_path / "model.mjb"
    mujoco.mj_saveModel(mujoco.MjModel.from_xml_path(str(xml)), str(binary), None)
    assert mujoco_model_camera_names(str(xml)) == ["first", "", "last"]
    assert mujoco_model_camera_names(str(binary)) == ["first", "", "last"]


def test_model_camera_names_load_like_mujoco_ros2_control(tmp_path, monkeypatch):
    """A file named exactly '.mjb' is MJCF (the binary rule needs more than
    four characters); a .mjb name over XML text is read as a binary and fails."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".mjb").write_text(_CAMERAS_MJCF, encoding="utf-8")
    assert mujoco_model_camera_names(".mjb") == ["first", "", "last"]
    (tmp_path / "x.mjb").write_text(_CAMERAS_MJCF, encoding="utf-8")
    with pytest.raises(Exception):  # noqa: B017 - any load error; the binary reader's type is not part of the rule
        mujoco_model_camera_names("x.mjb")


def test_model_camera_names_report_unusable_paths_without_mujoco(tmp_path, monkeypatch):
    """A missing file or a path the OS rejects (ENAMETOOLONG: a 300-byte name)
    raises FileNotFoundError before MuJoCo would log it to MUJOCO_LOG.TXT in
    the cwd."""
    monkeypatch.chdir(tmp_path)
    for path in (str(tmp_path / "missing.xml"), str(tmp_path / ("x" * 300))):
        with pytest.raises(FileNotFoundError):
            mujoco_model_camera_names(path)
    assert not (tmp_path / "MUJOCO_LOG.TXT").exists()


def _model_path_of(directory: Path, size: int) -> Path:
    """An .xml path of exactly ``size`` bytes under ``directory``, nested so
    every component stays under the 255-byte name limit."""
    path = directory
    while size - len(os.fsencode(str(path))) > 200:
        path = path / ("d" * 100)
    return path / ("m" * (size - len(os.fsencode(str(path))) - len("/.xml")) + ".xml")


def test_model_camera_names_reject_paths_the_simulator_truncates(tmp_path):
    """mujoco_ros2_control copies mujoco_model into simulate's 1000-byte
    buffer, so from 1000 bytes on it would load a truncated path, a different
    file. Bytes count, not characters: the copy is a C string."""
    fits = _model_path_of(tmp_path, 999)
    fits.parent.mkdir(parents=True, exist_ok=True)
    fits.write_text(_CAMERAS_MJCF, encoding="utf-8")
    assert len(os.fsencode(str(fits))) == 999
    assert mujoco_model_camera_names(str(fits)) == ["first", "", "last"]

    too_long = _model_path_of(tmp_path, 1000)
    too_long.parent.mkdir(parents=True, exist_ok=True)
    too_long.write_text(_CAMERAS_MJCF, encoding="utf-8")
    with pytest.raises(ValueError, match="truncates"):
        mujoco_model_camera_names(str(too_long))

    # fewer than 1000 characters, 1000 bytes: the two-byte name crosses the limit
    multibyte = fits.with_name(fits.name[: -len("m.xml")] + "é.xml")
    multibyte.write_text(_CAMERAS_MJCF, encoding="utf-8")
    assert len(str(multibyte)) < 1000 <= len(os.fsencode(str(multibyte)))
    with pytest.raises(ValueError, match="truncates"):
        mujoco_model_camera_names(str(multibyte))


def test_model_camera_names_read_engine_extension_models(tmp_path):
    """An extension-declaring model (the XML the compiler delivers) parses
    only with its engine plugins registered; the helper registers them. Run
    in a fresh process: other tests here register the lidar plugin in this one."""
    path = tmp_path / "lidar_robot.xml"
    path.write_text(
        LIDAR_ROBOT_TEMPLATE.replace(
            '<site name="probe_lidar_site"', '<camera name="probe_cam"/><site name="probe_lidar_site"'
        ),
        encoding="utf-8",
    )
    probe = (
        "import sys\n"
        "from sim_models.scene_compiler import mujoco_model_camera_names\n"
        "print(mujoco_model_camera_names(sys.argv[1]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe, str(path)], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "['probe_cam']"
