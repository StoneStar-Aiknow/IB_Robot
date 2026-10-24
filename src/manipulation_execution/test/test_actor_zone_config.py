"""Reading and checking the standing-zone selector's two files.

``actor_zone.yaml`` (design 9.1) and ``actor_tuning.yaml`` (design 9.2) are
read once at launch by the builder and by the node through the same
``load_zone`` / ``load_tuning``. A wrong file must stop the launch with the
file and the field named, never fall back to whole-frame selection: a robot
that starts with the wrong zone looks like it works while picking the wrong
people.

Dev plan section 3, table A: A1 and A3-A13 (A2, the homography, is tested with
the core). Every expected value here comes from the design and the plan, not
from the code: the defaults and ranges are the ones written in design 9.2, and
the zone rules the ones listed under "校验" in design 9.1.

Files are written into ``tmp_path``; nothing under the tool kit's data/ is read.
A12 runs the command-line check in a subprocess pointed at the same copy of the
package this process imported, so it cannot pass against another install.
"""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from manipulation_execution import actor_zone_config
from manipulation_execution.actor_zone_config import ZoneConfigError, load_tuning, load_zone

# Design 9.2, in file order.
DESIGN_DEFAULTS = {
    "acquire_updates": 2,
    "release_updates": 3,
    "release_hold_s": 3.0,
    "track_keep_updates": 2,
    "foot_half_width": 0.12,
    "center_weight": 0.6,
    "front_weight": 0.4,
    "occlude_overlap": 0.3,
    "occlude_dy": 0.05,
    "gate_dx": None,
    "gate_dy": None,
    "gate_h_ratio": None,
}

# The design 9.1 layout, comments included, as the zone tool writes it. Near
# edge at 930 on a 972 px image: 42 px from the bottom, inside the default 30.
GOOD_ZONE_TEXT = """\
version: 1
# 画区域时用的图像尺寸。运行时帧尺寸不同，就按比例缩放，并 warn 一次。
image_size: [1344, 972]
# 画面带：框整个落在这个区间外面的人，完全不管。
crop_x: [270, 1074]
# 地面区域：图上 4 个脚的位置，顺序固定。
#   近左、近右、远右、远左（近 = 靠机器人 = 图上偏下）
floor_quad:
  - [420, 930]   # near_left
  - [920, 930]   # near_right
  - [820, 640]   # far_right
  - [520, 640]   # far_left
# 谁、什么时候、对着哪张图画的
note: "test zone"
"""

GOOD_QUAD = [[420, 930], [920, 930], [820, 640], [520, 640]]


def good_zone():
    return {
        "version": 1,
        "image_size": [1344, 972],
        "crop_x": [270, 1074],
        "floor_quad": [list(p) for p in GOOD_QUAD],
        "note": "test zone",
    }


def write(tmp_path: Path, name: str, data) -> Path:
    path = tmp_path / name
    path.write_text(data if isinstance(data, str) else yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def zone_error(tmp_path: Path, data) -> str:
    with pytest.raises(ZoneConfigError) as info:
        load_zone(write(tmp_path, "actor_zone.yaml", data))
    return str(info.value)


def tuning_error(tmp_path: Path, data) -> str:
    with pytest.raises(ZoneConfigError) as info:
        load_tuning(write(tmp_path, "actor_tuning.yaml", data))
    return str(info.value)


def tuning_dict(tuning) -> dict:
    return {key: getattr(tuning, key) for key in DESIGN_DEFAULTS}


def with_quad(*points):
    zone = good_zone()
    zone["floor_quad"] = [list(p) for p in points]
    return zone


# ----- A1: a good zone file -----


def test_good_zone_file_reads_back_every_field(tmp_path):
    path = write(tmp_path, "actor_zone.yaml", GOOD_ZONE_TEXT)
    zone = load_zone(path)
    assert tuple(zone.image_size) == (1344, 972)
    assert tuple(zone.crop_x) == (270, 1074)
    assert [list(p) for p in zone.floor_quad] == GOOD_QUAD
    assert zone.note == "test zone"
    assert zone.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_zone_file_without_a_margin_uses_30_px(tmp_path):
    zone = load_zone(write(tmp_path, "actor_zone.yaml", GOOD_ZONE_TEXT))
    assert zone.near_edge_margin_px == 30


def test_zone_file_with_a_margin_keeps_it(tmp_path):
    data = good_zone()
    data["near_edge_margin_px"] = 40
    assert load_zone(write(tmp_path, "actor_zone.yaml", data)).near_edge_margin_px == 40


# ----- A3: corners in the wrong order -----


@pytest.mark.parametrize(
    "points",
    [
        # near and far swapped: the "near" corners are the upper ones
        ([520, 640], [820, 640], [920, 930], [420, 930]),
        # left and right swapped
        ([920, 930], [420, 930], [520, 640], [820, 640]),
    ],
    ids=["near_far_swapped", "left_right_swapped"],
)
def test_corners_in_the_wrong_order_name_floor_quad(tmp_path, points):
    assert "floor_quad" in zone_error(tmp_path, with_quad(*points))


# ----- A4: not a convex quadrilateral -----


@pytest.mark.parametrize(
    "points",
    [
        # far_left pushed in past the far_right - near_left diagonal: a dent
        ([400, 900], [800, 900], [700, 600], [650, 800]),
        # the near edge crosses the far edge: a bow tie
        ([400, 800], [800, 700], [750, 650], [500, 790]),
        # all four on the line y = 1100 - x / 2, still left-of-right and near-below-far
        ([400, 900], [600, 800], [700, 750], [500, 850]),
    ],
    ids=["concave", "self_intersecting", "four_collinear"],
)
def test_a_quad_that_is_not_convex_is_refused(tmp_path, points):
    assert "floor_quad" in zone_error(tmp_path, with_quad(*points))


# ----- A5: corners outside the image or the band -----


@pytest.mark.parametrize(
    "points, corner",
    [
        (([420, 930], [920, 930], [820, 640], [520, -10]), "far_left"),
        (([420, 930], [920, 930], [820, 980], [520, 640]), "far_right"),
        (([260, 930], [920, 930], [820, 640], [520, 640]), "near_left"),
        (([420, 930], [1080, 930], [820, 640], [520, 640]), "near_right"),
    ],
    ids=["above_the_image", "below_the_image", "left_of_the_band", "right_of_the_band"],
)
def test_a_corner_outside_the_image_or_the_band_is_refused(tmp_path, points, corner):
    message = zone_error(tmp_path, with_quad(*points))
    assert "floor_quad" in message
    assert corner in message


@pytest.mark.parametrize(
    "crop_x",
    [[600, 600], [700, 600], [-1, 1074], [270, 1345]],
    ids=["empty", "reversed", "negative", "wider_than_image"],
)
def test_a_band_outside_0_to_width_is_refused(tmp_path, crop_x):
    data = good_zone()
    data["crop_x"] = crop_x
    assert "crop_x" in zone_error(tmp_path, data)


# ----- A6: the near edge and the bottom margin -----


@pytest.mark.parametrize(
    "points, margin, corner",
    [
        # default margin 30: the near edge may go down to 972 - 30 = 942
        (([420, 950], [920, 930], [820, 640], [520, 640]), None, "near_left"),
        (([420, 930], [920, 943], [820, 640], [520, 640]), None, "near_right"),
        # a larger margin moves the limit up: 972 - 50 = 922
        (([420, 930], [920, 930], [820, 640], [520, 640]), 50, "near_left"),
    ],
    ids=["near_left_in_default_margin", "near_right_1px_in_default_margin", "inside_a_larger_margin"],
)
def test_a_near_corner_inside_the_bottom_margin_is_refused(tmp_path, points, margin, corner):
    data = with_quad(*points)
    if margin is not None:
        data["near_edge_margin_px"] = margin
    assert corner in zone_error(tmp_path, data)


def test_a_near_edge_exactly_at_the_margin_is_accepted(tmp_path):
    zone = load_zone(write(tmp_path, "actor_zone.yaml", with_quad([420, 942], [920, 942], [820, 640], [520, 640])))
    assert zone.floor_quad[0][1] == 942


@pytest.mark.parametrize("margin", [15, 0, -30])
def test_a_margin_below_16_px_is_refused(tmp_path, margin):
    data = good_zone()
    data["near_edge_margin_px"] = margin
    assert "near_edge_margin_px" in zone_error(tmp_path, data)


def test_a_fractional_margin_is_refused(tmp_path):
    # Design 9.1: the margin is a whole number of pixels.
    data = good_zone()
    data["near_edge_margin_px"] = 30.5
    assert "near_edge_margin_px" in zone_error(tmp_path, data)


def test_a_margin_of_16_px_is_accepted(tmp_path):
    data = good_zone()
    data["near_edge_margin_px"] = 16
    assert load_zone(write(tmp_path, "actor_zone.yaml", data)).near_edge_margin_px == 16


# ----- A7: malformed zone files name the field -----


@pytest.mark.parametrize("key", ["version", "image_size", "crop_x", "floor_quad"])
def test_a_missing_zone_key_is_named(tmp_path, key):
    data = good_zone()
    del data[key]
    assert key in zone_error(tmp_path, data)


def test_an_unknown_zone_key_is_named(tmp_path):
    data = good_zone()
    data["floor_quadd"] = GOOD_QUAD
    assert "floor_quadd" in zone_error(tmp_path, data)


@pytest.mark.parametrize("version", [2, 0, "1", None], ids=["2", "0", "string", "null"])
def test_a_zone_version_other_than_1_is_refused(tmp_path, version):
    data = good_zone()
    data["version"] = version
    assert "version" in zone_error(tmp_path, data)


@pytest.mark.parametrize(
    "field, value, named",
    [
        ("image_size", [1344, "972"], "image_size"),
        ("image_size", [1344.0, 972.0], "image_size"),
        ("image_size", [0, 972], "image_size"),
        ("image_size", [1344], "image_size"),
        ("crop_x", ["left", 1074], "crop_x"),
        ("crop_x", [270, math.nan], "crop_x"),
        ("floor_quad", [["x", 930], [920, 930], [820, 640], [520, 640]], "floor_quad"),
        ("floor_quad", [[math.nan, 930], [920, 930], [820, 640], [520, 640]], "floor_quad"),
        ("floor_quad", [[420, 930], [920, math.inf], [820, 640], [520, 640]], "floor_quad"),
        ("floor_quad", [[420, 930], [920, 930], [820, 640]], "floor_quad"),
        ("near_edge_margin_px", "thirty", "near_edge_margin_px"),
    ],
    ids=[
        "image_size_string",
        "image_size_float",
        "image_size_zero",
        "image_size_one_number",
        "crop_x_string",
        "crop_x_nan",
        "corner_string",
        "corner_nan",
        "corner_inf",
        "three_corners",
        "margin_string",
    ],
)
def test_a_bad_zone_value_names_its_field(tmp_path, field, value, named):
    data = good_zone()
    data[field] = value
    assert named in zone_error(tmp_path, data)


# ----- A8 / A9: tuning defaults -----


@pytest.mark.parametrize("path", [None, ""], ids=["none", "empty_string"])
def test_no_tuning_file_means_every_design_default(path):
    tuning, warnings = load_tuning(path)
    assert tuning_dict(tuning) == DESIGN_DEFAULTS
    assert warnings == []


def test_a_tuning_file_with_only_the_version_is_all_defaults(tmp_path):
    tuning, warnings = load_tuning(write(tmp_path, "actor_tuning.yaml", {"version": 1}))
    assert tuning_dict(tuning) == DESIGN_DEFAULTS
    assert warnings == []


@pytest.mark.parametrize(
    "written",
    [
        {"acquire_updates": 4},
        {"release_hold_s": 5.0, "occlude_dy": 0.1},
        {"center_weight": 1.0, "front_weight": 0.0, "gate_dx": 0.5},
    ],
    ids=["one_key", "two_keys", "three_keys"],
)
def test_a_partial_tuning_file_keeps_the_defaults_for_the_rest(tmp_path, written):
    tuning, _ = load_tuning(write(tmp_path, "actor_tuning.yaml", {"version": 1, **written}))
    assert tuning_dict(tuning) == {**DESIGN_DEFAULTS, **written}


def test_a_tuning_file_carries_its_sha256(tmp_path):
    path = write(tmp_path, "actor_tuning.yaml", {"version": 1, "acquire_updates": 4})
    tuning, _ = load_tuning(path)
    assert tuning.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


# ----- A10: bad tuning values name the key -----


def test_an_unknown_tuning_key_is_named(tmp_path):
    assert "aquire_updates" in tuning_error(tmp_path, {"version": 1, "aquire_updates": 2})


def test_a_tuning_version_other_than_1_is_refused(tmp_path):
    assert "version" in tuning_error(tmp_path, {"version": 2})


def test_a_tuning_file_without_a_version_is_refused(tmp_path):
    # Design 9.2: version: 1 must be written, as in the zone file.
    assert "version" in tuning_error(tmp_path, {"release_hold_s": 3.0})


@pytest.mark.parametrize(
    "key, value",
    [
        ("acquire_updates", 0),
        ("acquire_updates", 11),
        ("release_updates", 0),
        ("release_updates", 11),
        ("release_hold_s", -0.1),
        ("track_keep_updates", -1),
        ("track_keep_updates", 11),
        ("foot_half_width", 0.01),
        ("foot_half_width", 0.51),
        ("center_weight", -0.1),
        ("front_weight", -0.1),
        ("occlude_overlap", 0.0),
        ("occlude_overlap", 1.01),
        ("occlude_dy", -0.01),
        ("gate_dx", 0.0),
        ("gate_dy", -0.1),
        ("gate_h_ratio", 1.0),
    ],
)
def test_an_out_of_range_tuning_value_names_its_key(tmp_path, key, value):
    assert key in tuning_error(tmp_path, {"version": 1, key: value})


@pytest.mark.parametrize(
    "key, value",
    [
        ("acquire_updates", 1),
        ("acquire_updates", 10),
        ("release_updates", 1),
        ("release_updates", 10),
        ("track_keep_updates", 0),
        ("track_keep_updates", 10),
        ("foot_half_width", 0.02),
        ("foot_half_width", 0.5),
        ("center_weight", 0.0),
        ("front_weight", 0.0),
        ("occlude_overlap", 1.0),
        ("occlude_dy", 0.0),
        ("gate_h_ratio", 1.01),
    ],
)
def test_a_tuning_value_on_the_edge_of_its_range_is_accepted(tmp_path, key, value):
    tuning, _ = load_tuning(write(tmp_path, "actor_tuning.yaml", {"version": 1, key: value}))
    assert getattr(tuning, key) == value


@pytest.mark.parametrize("key", ["acquire_updates", "release_updates", "track_keep_updates"])
def test_an_integer_key_written_as_a_float_is_refused(tmp_path, key):
    assert key in tuning_error(tmp_path, {"version": 1, key: 2.0})


@pytest.mark.parametrize(
    "key, value",
    [
        ("release_hold_s", "3s"),
        ("foot_half_width", math.nan),
        ("occlude_dy", math.inf),
        ("acquire_updates", True),
        ("occlude_dy", None),
        ("release_hold_s", None),
    ],
    ids=["string", "nan", "inf", "bool", "null_not_allowed_occlude_dy", "null_not_allowed_release_hold_s"],
)
def test_a_tuning_value_that_is_not_a_usable_number_names_its_key(tmp_path, key, value):
    assert key in tuning_error(tmp_path, {"version": 1, key: value})


def test_both_pick_weights_zero_is_refused(tmp_path):
    message = tuning_error(tmp_path, {"version": 1, "center_weight": 0.0, "front_weight": 0})
    assert "center_weight" in message
    assert "front_weight" in message


# ----- A11: null turns a rule off, a number turns it on -----


@pytest.mark.parametrize("key", ["gate_dx", "gate_dy", "gate_h_ratio", "occlude_overlap"])
def test_null_turns_an_optional_rule_off(tmp_path, key):
    tuning, _ = load_tuning(write(tmp_path, "actor_tuning.yaml", {"version": 1, key: None}))
    assert getattr(tuning, key) is None


@pytest.mark.parametrize(
    "key, value", [("gate_dx", 0.5), ("gate_dy", 0.2), ("gate_h_ratio", 1.5), ("occlude_overlap", 0.5)]
)
def test_a_number_turns_an_optional_rule_on(tmp_path, key, value):
    tuning, _ = load_tuning(write(tmp_path, "actor_tuning.yaml", {"version": 1, key: value}))
    assert getattr(tuning, key) == value


def test_by_default_the_gates_are_off_and_the_occlusion_rule_is_on():
    tuning, _ = load_tuning(None)
    assert (tuning.gate_dx, tuning.gate_dy, tuning.gate_h_ratio) == (None, None, None)
    assert tuning.occlude_overlap == 0.3


# ----- A13: release_hold_s and the session's 2.0 s cancel confirm -----


@pytest.mark.parametrize("hold", [0, 1.0, 1.99])
def test_a_hold_shorter_than_2_s_loads_with_one_warning(tmp_path, hold):
    tuning, warnings = load_tuning(write(tmp_path, "actor_tuning.yaml", {"version": 1, "release_hold_s": hold}))
    assert tuning.release_hold_s == hold
    assert len(warnings) == 1
    assert "release_hold_s" in warnings[0]


@pytest.mark.parametrize("hold", [2.0, 30, 30.0])
def test_a_hold_of_2_to_30_s_loads_without_warning(tmp_path, hold):
    tuning, warnings = load_tuning(write(tmp_path, "actor_tuning.yaml", {"version": 1, "release_hold_s": hold}))
    assert tuning.release_hold_s == hold
    assert warnings == []


@pytest.mark.parametrize("hold", [31, 30.01])
def test_a_hold_longer_than_30_s_is_refused(tmp_path, hold):
    assert "release_hold_s" in tuning_error(tmp_path, {"version": 1, "release_hold_s": hold})


# ----- A12: the command-line check -----


def run_check(*args):
    # Point the child at the package this process imported, ahead of any install.
    package_root = str(Path(actor_zone_config.__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (package_root, env.get("PYTHONPATH", "")) if p)
    return subprocess.run(
        [sys.executable, "-m", "manipulation_execution.actor_zone_config", *map(str, args)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def test_check_prints_ok_and_the_zone_sha256(tmp_path):
    zone = write(tmp_path, "actor_zone.yaml", GOOD_ZONE_TEXT)
    result = run_check("--zone", zone)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout.splitlines()
    assert hashlib.sha256(zone.read_bytes()).hexdigest() in result.stdout


def test_check_prints_both_sha256_with_a_tuning_file(tmp_path):
    zone = write(tmp_path, "actor_zone.yaml", GOOD_ZONE_TEXT)
    tuning = write(tmp_path, "actor_tuning.yaml", {"version": 1, "acquire_updates": 3})
    result = run_check("--zone", zone, "--tuning", tuning)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout.splitlines()
    assert hashlib.sha256(zone.read_bytes()).hexdigest() in result.stdout
    assert hashlib.sha256(tuning.read_bytes()).hexdigest() in result.stdout


def test_check_names_the_field_of_a_bad_zone_and_fails(tmp_path):
    data = good_zone()
    data["near_edge_margin_px"] = 15
    result = run_check("--zone", write(tmp_path, "actor_zone.yaml", data))
    assert result.returncode != 0
    assert "near_edge_margin_px" in result.stderr
    assert "OK" not in result.stdout.splitlines()


def test_check_names_the_key_of_a_bad_tuning_file_and_fails(tmp_path):
    zone = write(tmp_path, "actor_zone.yaml", GOOD_ZONE_TEXT)
    tuning = write(tmp_path, "actor_tuning.yaml", {"version": 1, "release_hold_s": 31})
    result = run_check("--zone", zone, "--tuning", tuning)
    assert result.returncode != 0
    assert "release_hold_s" in result.stderr
    assert "OK" not in result.stdout.splitlines()


def test_check_fails_on_a_missing_zone_file(tmp_path):
    result = run_check("--zone", tmp_path / "absent.yaml")
    assert result.returncode != 0
    assert "absent.yaml" in result.stderr
