"""The nav_grasp_room scene (#205 grasp stage D1).

The mobile-manipulation sibling of nav_room: the same navigation geometry
plus the grasp table and one dynamic magenta cube. This module pins the
task data (`mobile_manipulation`, validated through
`sim_models.scene_compiler.mobile_manipulation_spec`), the dynamic object's
mass/friction/colour mirroring in BOTH simulator templates, the
scene-internal colour uniqueness of the magenta hue band, the MJCF compile
of the furniture, and the staging-pose clearance against the rasterised
layout map (D2).
"""

from __future__ import annotations

import colorsys
import math
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from sim_models.scene_compiler import (
    get_scene_layout,
    mobile_manipulation_spec,
    rasterize_static_geometry,
    scene_dynamic_objects,
)

_SCENE_DIR = Path(__file__).resolve().parents[1] / "scenes" / "nav_grasp_room"

#: The map parameters the committed navigation map is generated with (D2;
#: identical to sim_map.yaml's, the 4 m x 4 m rooms).
MAP_RESOLUTION = 0.05
MAP_ORIGIN = (-2.5, -2.5)
MAP_SIZE = (100, 100)

#: The cube's mask band (D1): hue 270-330 deg, S > 0.5, V > 0.25 — no other
#: scene element may fall in it (scene-internal uniqueness).
MASK_HUE_BAND_DEG = (270.0, 330.0)
MASK_MIN_SATURATION = 0.5
MASK_MIN_VALUE = 0.25

#: The Nav2 goal-check clearance the staging goal must clear (the pick-sim
#: overlay's final-approach inflation radius — embodied_bringup's test
#: pick_sim_staging.PICK_SIM_INFLATION_RADIUS_M; sim_models cannot import
#: it, so the value is pinned here and its staging helper comment names this
#: rule). The pre-tuning goal (0.72) cleared the shipped 0.28 footprint
#: radius exactly; the tuned goal (0.785) sits 0.215 m from the table's west
#: edge — clear of the 0.12-inflated costmap Nav2 goal-checks against.
STAGING_GOAL_CHECK_CLEARANCE_M = 0.12


def _layout() -> dict:
    return get_scene_layout("nav_grasp_room")


def _grasp_object() -> dict:
    return next(obj for obj in _layout()["objects"] if obj["name"] == "grasp_object")


# ---------------------------------------------------------------------------
# D1: the mobile_manipulation section, validated through one validator
# ---------------------------------------------------------------------------


def test_the_mobile_manipulation_spec_validates():
    spec = mobile_manipulation_spec(_layout())
    assert spec == {
        "object": "grasp_object",
        "object_prompt": "magenta cube",
        "table": "table",
        "staging_pose": {"x": 0.785, "y": 0.40, "yaw": 0.0},
        "success": {"lift_m": 0.05, "hold_s": 3.0, "max_object_rotation_in_gripper_deg": 10.0},
    }


def test_scene_dynamic_objects_lists_the_grasp_cube():
    assert scene_dynamic_objects("nav_grasp_room") == ["grasp_object"]


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda layout: layout.pop("mobile_manipulation"), r"mobile_manipulation: the section is missing"),
        (lambda layout: layout["mobile_manipulation"].pop("object"), r"mobile_manipulation.object: missing"),
        (
            lambda layout: layout["mobile_manipulation"].update(object="table"),
            r"mobile_manipulation.object: 'table' is not a `type: dynamic` object",
        ),
        (
            lambda layout: layout["mobile_manipulation"].pop("object_prompt"),
            r"mobile_manipulation.object_prompt: missing or not a non-empty string",
        ),
        (
            lambda layout: layout["mobile_manipulation"].update(object_prompt="   "),
            r"mobile_manipulation.object_prompt: missing or not a non-empty string",
        ),
        (lambda layout: layout["mobile_manipulation"].pop("table"), r"mobile_manipulation.table: missing"),
        (
            lambda layout: layout["mobile_manipulation"].update(table="grasp_object"),
            r"mobile_manipulation.table: 'grasp_object' is not a static_geometry entry",
        ),
        (
            lambda layout: layout["mobile_manipulation"].pop("staging_pose"),
            r"mobile_manipulation.staging_pose: missing",
        ),
        (
            lambda layout: layout["mobile_manipulation"].update(staging_pose={"x": 1.2, "y": 0.4, "yaw": 0.0}),
            r"staging_pose: \(1.2, 0.4\) lies inside the static element 'table'",
        ),
        (
            lambda layout: layout["mobile_manipulation"].update(staging_pose={"x": 0.0, "y": 0.0, "yaw": 0.0}),
            r"staging_pose: \(0.0, 0.0\) lies inside the static element 'obstacle_box'",
        ),
        (lambda layout: layout["mobile_manipulation"].pop("success"), r"mobile_manipulation.success: missing"),
        (
            lambda layout: layout["mobile_manipulation"]["success"].update(lift_m=0.0),
            r"mobile_manipulation.success.lift_m: must be positive",
        ),
        (
            lambda layout: layout["mobile_manipulation"]["success"].update(hold_s=-1.0),
            r"mobile_manipulation.success.hold_s: must be positive",
        ),
        (
            lambda layout: layout["mobile_manipulation"]["success"].update(max_object_rotation_in_gripper_deg=0.0),
            r"mobile_manipulation.success.max_object_rotation_in_gripper_deg: must be positive",
        ),
        (
            lambda layout: layout["mobile_manipulation"]["staging_pose"].update(x="not-a-number"),
            r"mobile_manipulation.staging_pose.x",
        ),
    ],
)
def test_the_mobile_manipulation_spec_rejects_invalid_sections(mutate, match):
    import yaml

    layout = yaml.safe_load(yaml.safe_dump(_layout()))  # a deep copy
    mutate(layout)
    with pytest.raises(ValueError, match=match):
        mobile_manipulation_spec(layout)


# ---------------------------------------------------------------------------
# D1: the dynamic object mirrors its layout data in BOTH templates
# ---------------------------------------------------------------------------


def _mjcf_grasp_geom() -> ET.Element:
    root = ET.parse(_SCENE_DIR / "nav_grasp_room.xml.template").getroot()
    body = next(element for element in root.iter("body") if element.get("name") == "grasp_object")
    return body.find("geom")


def test_the_mjcf_template_mirrors_the_grasp_object():
    obj = _grasp_object()
    root = ET.parse(_SCENE_DIR / "nav_grasp_room.xml.template").getroot()
    body = next(element for element in root.iter("body") if element.get("name") == "grasp_object")
    assert body.find("freejoint") is not None, "the grasp cube must be a free-joint dynamic body"
    assert [float(v) for v in body.get("pos").split()] == pytest.approx(obj["position"])
    geom = body.find("geom")
    assert geom.get("type") == "box"
    assert [float(v) for v in geom.get("size").split()] == pytest.approx([side / 2.0 for side in obj["size"]])
    assert float(geom.get("mass")) == pytest.approx(obj["mass_kg"])
    assert [float(v) for v in geom.get("friction").split()][0] == pytest.approx(obj["friction"])
    assert [float(v) for v in geom.get("rgba").split()] == pytest.approx(obj["rgba"])
    assert geom.get("condim") == "4"


def test_the_gazebo_template_mirrors_the_grasp_object():
    obj = _grasp_object()
    root = ET.parse(_SCENE_DIR / "nav_grasp_room.world.template").getroot()
    model = next(element for element in root.iter("model") if element.get("name") == "grasp_object")
    assert model.find("static") is None, "the grasp cube must be a dynamic model"
    pose = [float(v) for v in model.find("pose").text.split()]
    assert pose == pytest.approx([*obj["position"], 0.0, 0.0, obj["rotation_euler_rad"][2]])
    inertial = model.find("./link/inertial")
    assert float(inertial.find("mass").text) == pytest.approx(obj["mass_kg"])
    # a solid cube's inertia: I = m a^2 / 6 per axis
    expected = obj["mass_kg"] * obj["size"][0] ** 2 / 6.0
    inertia = inertial.find("inertia")
    for axis in ("ixx", "iyy", "izz"):
        assert float(inertia.find(axis).text) == pytest.approx(expected), axis
    size = [float(v) for v in model.find("./link/collision/geometry/box/size").text.split()]
    assert size == pytest.approx(obj["size"])
    mu = float(model.find("./link/collision/surface/friction/ode/mu").text)
    assert mu == pytest.approx(obj["friction"])
    for visual_material_tag in ("ambient", "diffuse"):
        rgba = [float(v) for v in model.find(f"./link/visual/material/{visual_material_tag}").text.split()]
        assert rgba == pytest.approx(obj["rgba"]), visual_material_tag


def test_the_gazebo_table_mirrors_the_layout_entry():
    element = next(item for item in _layout()["static_geometry"] if item["name"] == "table")
    root = ET.parse(_SCENE_DIR / "nav_grasp_room.world.template").getroot()
    model = next(item for item in root.iter("model") if item.get("name") == "table")
    assert model.find("static").text.strip() == "true"
    pose = [float(v) for v in model.find("pose").text.split()]
    assert pose == pytest.approx(element["pose"])
    size = [float(v) for v in model.find("./link/collision/geometry/box/size").text.split()]
    assert size == pytest.approx(element["size"])


# ---------------------------------------------------------------------------
# D1: the magenta hue band is unique inside the scene
# ---------------------------------------------------------------------------


def _in_mask_band(rgba: list[float]) -> bool:
    hue, saturation, value = colorsys.rgb_to_hsv(rgba[0], rgba[1], rgba[2])
    hue_deg = hue * 360.0
    return (
        MASK_HUE_BAND_DEG[0] <= hue_deg <= MASK_HUE_BAND_DEG[1]
        and saturation > MASK_MIN_SATURATION
        and value > MASK_MIN_VALUE
    )


def test_the_magenta_hue_band_is_scene_unique():
    """The cube's mask band (hue 270-330, S > 0.5, V > 0.25) contains the
    cube's colour and NO other colour of either scene template — the
    colour-based object perception stays unambiguous inside the scene."""
    obj = _grasp_object()
    assert _in_mask_band(obj["rgba"]), "the grasp cube itself must sit in the mask band"

    colours = []
    for template in sorted(_SCENE_DIR.glob("nav_grasp_room.*.template")):
        root = ET.parse(template).getroot()
        for tag in ("ambient", "diffuse", "specular"):
            colours.extend(
                [float(v) for v in element.text.split()]
                for element in root.iter(tag)
                if element.text and len(element.text.split()) == 4
            )
        colours.extend(
            [float(v) for v in element.get("rgba").split()] for element in root.iter("geom") if element.get("rgba")
        )
    in_band = [rgba for rgba in colours if _in_mask_band(rgba)]
    # the cube's own colour legitimately appears once per template slot
    # (Gazebo ambient + diffuse, MJCF rgba); every in-band colour must BE it
    assert in_band and all(rgba == pytest.approx(obj["rgba"]) for rgba in in_band), (
        f"only the grasp cube's colour may sit in the mask band; in-band colours: {in_band}"
    )


# ---------------------------------------------------------------------------
# D8 (mujoco counterpart): the composed scene compiles with the furniture
# ---------------------------------------------------------------------------


def test_nav_grasp_room_mjcf_compiles_with_the_furniture(tmp_path, monkeypatch):
    mujoco = pytest.importorskip("mujoco")
    from sim_models.scene_compiler import compose_mujoco_model

    monkeypatch.setenv("ROS_DOMAIN_ID", "nav_grasp_mjcf_test")
    stub = tmp_path / "stub_robot.xml"
    stub.write_text('<mujoco model="stub"><worldbody/></mujoco>', encoding="utf-8")
    out = compose_mujoco_model(str(stub), "nav_grasp_room", ())
    assert out.suffix == ".mjb" and out.exists()
    model = mujoco.MjModel.from_binary_path(str(out))

    obj = _grasp_object()
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "grasp_object")
    assert body_id >= 0, "the grasp cube body is missing from the compiled scene"
    assert model.body_jntnum[body_id] == 1, "the cube carries exactly its free joint"
    assert model.body_jntadr[body_id] >= 0
    assert list(model.body_pos[body_id]) == pytest.approx(obj["position"])
    geom_id = model.body_geomadr[body_id]
    assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
    assert [value * 2.0 for value in model.geom_size[geom_id]] == pytest.approx(obj["size"])
    assert model.body_mass[body_id] == pytest.approx(obj["mass_kg"])
    assert model.geom_friction[geom_id][0] == pytest.approx(obj["friction"])
    assert model.geom_condim[geom_id] == 4

    table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    assert table_id >= 0 and model.body_jntnum[table_id] == 0, "the table is a static body"
    table_geom = model.body_geomadr[table_id]
    assert model.geom_priority[table_geom] == 1, "the table follows the support-surface priority rule"
    assert model.geom_friction[table_geom][0] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# D1/D2: the staging pose clears the rasterised layout map
# ---------------------------------------------------------------------------


def test_the_staging_pose_clears_the_layout_map():
    """No lethal (occupied) cell of the rasterised layout map lies within
    the goal-check clearance of the mobile_manipulation goal — the inflated
    costmap radius Nav2 goal-checks the staging goal against (the pick-sim
    overlay's final-approach inflation)."""
    spec = mobile_manipulation_spec(_layout())
    grid = rasterize_static_geometry(_layout(), resolution=MAP_RESOLUTION, origin=MAP_ORIGIN, size=MAP_SIZE)
    height, width = grid.shape
    sx, sy = spec["staging_pose"]["x"], spec["staging_pose"]["y"]
    clearance = STAGING_GOAL_CHECK_CLEARANCE_M
    nearest = math.inf
    for row in range(height):
        for column in range(width):
            if not grid[row, column]:
                continue
            x_lo = MAP_ORIGIN[0] + column * MAP_RESOLUTION
            y_lo = MAP_ORIGIN[1] + (height - row - 1) * MAP_RESOLUTION  # row 0 = max y
            dx = max(x_lo - sx, 0.0, sx - (x_lo + MAP_RESOLUTION))
            dy = max(y_lo - sy, 0.0, sy - (y_lo + MAP_RESOLUTION))
            nearest = min(nearest, math.hypot(dx, dy))
    assert nearest + 1e-9 >= clearance, (
        f"a lethal map cell lies {nearest:.3f} m from the staging pose ({sx}, {sy}) — inside the "
        f"{clearance} m goal-check clearance"
    )


def test_a_staging_pose_on_the_table_fails_the_clearance():
    """Negative fixture: a staging pose inside the table's cells violates
    the clearance rule (the cell under the goal is lethal)."""
    grid = rasterize_static_geometry(_layout(), resolution=MAP_RESOLUTION, origin=MAP_ORIGIN, size=MAP_SIZE)
    height, _width = grid.shape
    # the table occupies x [1.0, 1.4], y [0.15, 0.65]: the cell containing
    # its centre (1.2, 0.4) is lethal
    column = int((1.2 - MAP_ORIGIN[0]) / MAP_RESOLUTION)
    row = height - 1 - int((0.4 - MAP_ORIGIN[1]) / MAP_RESOLUTION)
    assert grid[row, column], "the fixture expects a lethal cell at the table centre"


# ---------------------------------------------------------------------------
# setup.py installs the scene
# ---------------------------------------------------------------------------


def test_the_scene_installs_through_the_setup_globs():
    from ament_index_python.packages import get_package_share_directory

    scene_share = Path(get_package_share_directory("sim_models")) / "scenes" / "nav_grasp_room"
    for name in ("layout.yaml", "nav_grasp_room.world.template", "nav_grasp_room.xml.template"):
        assert (scene_share / name).is_file(), f"{name} is not installed with the scene"


def _gazebo_available() -> bool:
    try:
        return subprocess.run(["ign", "gazebo", "--version"], capture_output=True, timeout=10).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.mark.skipif(not _gazebo_available(), reason="Gazebo (ign gazebo) is not available")
def test_nav_grasp_room_world_loads_in_gazebo():
    """The dynamic cube model loads in a real (headless) Gazebo server —
    the explicit inertia and surface friction parse cleanly."""
    import os
    import uuid

    from sim_models.scene_compiler import get_gazebo_world_path

    world = get_gazebo_world_path("nav_grasp_room")
    environment = dict(os.environ)
    environment["GZ_PARTITION"] = f"nav_grasp_load_{uuid.uuid4().hex[:8]}"
    environment["IGN_PARTITION"] = environment["GZ_PARTITION"]
    result = subprocess.run(
        ["ign", "gazebo", "-s", "-r", "--iterations", "10", str(world)],
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
    )
    assert result.returncode == 0, f"gazebo failed to load {world}:\n{result.stdout}\n{result.stderr}"
