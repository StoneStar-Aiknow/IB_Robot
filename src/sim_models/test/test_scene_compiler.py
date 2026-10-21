"""Scene composition tests (so101-mujoco-runtime-platform tasks 2.1/2.2).

Covers the robot-agnostic composition API, the shared randomisation grid, the
domain-keyed composed-model path, the layout task declaration and the
dependency direction (sim_models never imports robot_runtime).
"""

from __future__ import annotations

import ast
import importlib
import math
import pkgutil
import tempfile
from pathlib import Path

import pytest

from sim_models.scene_compiler import (
    compose_mujoco_model,
    composed_model_path,
    get_scene_layout,
    get_scene_task,
    randomization_grid,
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


def _keyframe_id(model, name: str) -> int:
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    assert kid >= 0, f"keyframe {name!r} missing (nkey={model.nkey})"
    return kid


def _free_qpos(model, key_id: int, joint: str):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    adr = int(model.jnt_qposadr[jid])
    return model.key_qpos[key_id][adr : adr + 7]


def _scalar_qpos(model, key_id: int, joint: str):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    adr = int(model.jnt_qposadr[jid])
    return model.key_qpos[key_id][adr]


# ---------------------------------------------------------------------------
# 2.1: randomization_grid
# ---------------------------------------------------------------------------


def test_grid_size_and_order_are_stable():
    layout = get_scene_layout("pick_banana")
    grid = randomization_grid(layout)
    # 5 x divisions × 7 y divisions × 6 yaw steps
    assert len(grid) == 5 * 7 * 6
    rand = layout["randomization"]["banana"]
    x_min, x_max = rand["x_range"]
    y_min, y_max = rand["y_range"]
    yaw_step = 2.0 * math.pi / 6
    # index 0 → (x_min, y_min, z_spawn, 0); the innermost loop is yaw
    assert grid[0] == pytest.approx((x_min, y_min, 0.05, 0.0), abs=1e-9)
    assert grid[1] == pytest.approx((x_min, y_min, 0.05, yaw_step), abs=1e-9)
    # after yaw_steps indices the y advances
    assert grid[6] == pytest.approx((x_min, y_min + (y_max - y_min) / 6, 0.05, 0.0), abs=1e-9)
    # the last entry is (x_max, y_max, last yaw)
    assert grid[-1] == pytest.approx((x_max, y_max, 0.05, 5 * yaw_step), abs=1e-9)


def test_grid_is_empty_without_randomization():
    assert randomization_grid({}) == []
    assert randomization_grid({"randomization": {}}) == []
    assert randomization_grid({"randomization": {"banana": {}}}) == []


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


def test_compose_with_pick_banana_compiles_and_injects_keyframes(robot_mjcf, clean_domain):
    out = compose_mujoco_model(robot_mjcf, "pick_banana", ())
    assert out == composed_model_path("pick_banana")
    assert out.exists() and out.suffix == ".mjb"
    model = _model_of(out)
    assert model.nkey == 1 + 5 * 7 * 6  # default + random grid

    grid = randomization_grid(get_scene_layout("pick_banana"))
    for k in (0, 1, 17, 209):
        kid = _keyframe_id(model, f"random_{k:03d}")
        banana = _free_qpos(model, kid, "banana_free")
        x, y, z, yaw = grid[k]
        assert list(banana[:3]) == pytest.approx([x, y, z], abs=1e-5)
        assert list(banana[3:]) == pytest.approx([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], abs=1e-5)


def test_compose_default_keyframe_holds_layout_poses_and_rest(robot_mjcf, clean_domain):
    out = compose_mujoco_model(robot_mjcf, "pick_banana", ())
    model = _model_of(out)
    kid = _keyframe_id(model, "default")
    layout = get_scene_layout("pick_banana")
    objects = {obj["name"]: obj for obj in layout["objects"]}
    banana = _free_qpos(model, kid, "banana_free")
    assert list(banana[:3]) == pytest.approx(objects["banana"]["position"], abs=1e-5)
    plate = _free_qpos(model, kid, "plate_free")
    assert list(plate[:3]) == pytest.approx(objects["plate"]["position"], abs=1e-5)
    # plate rotation [pi/2, 0, 0] → quaternion w-x-y-z (0.7071, 0.7071, 0, 0)
    assert list(plate[3:]) == pytest.approx([0.7071068, 0.7071068, 0.0, 0.0], abs=1e-5)
    rest = layout["randomization"]["arm_rest_qpos"]
    for j in range(1, 7):
        assert _scalar_qpos(model, kid, str(j)) == pytest.approx(rest[j - 1], abs=1e-5)


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
