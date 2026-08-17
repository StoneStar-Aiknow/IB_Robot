import hashlib
import json
import struct
import uuid

import pytest

from embodied_common.skill_request import canonical_skill_payload, skill_goal_uuid, skill_payload_hash


def test_canonical_skill_payload_normalizes_strings_and_uses_default_timeout():
    omitted_timeout = canonical_skill_payload(
        "  move_relative_ee  ",
        schema_version=1,
        target_name="  cup  ",
        place_name=" tray ",
        motion_direction=" LEFT ",
        motion_distance=0.25,
        default_timeout_sec=12.0,
    )
    explicit_timeout = canonical_skill_payload(
        "move_relative_ee",
        schema_version=1,
        target_name="cup",
        place_name="tray",
        motion_direction="left",
        motion_distance=0.25,
        timeout_sec=12.0,
        default_timeout_sec=12.0,
    )

    assert omitted_timeout == {
        "schema_version": 1,
        "skill_name": "move_relative_ee",
        "target_name": "cup",
        "container_name": "",
        "place_name": "tray",
        "motion_direction": "left",
        "motion_distance": 0.25,
        "timeout_sec": 12.0,
    }
    assert skill_payload_hash(omitted_timeout) == skill_payload_hash(explicit_timeout)


def test_canonical_skill_payload_matches_float32_ros_wire_values():
    payload = canonical_skill_payload(
        "move_relative_ee",
        schema_version=1,
        motion_distance=0.1,
        timeout_sec=0.1,
        default_timeout_sec=1.0,
    )
    wire_value = struct.unpack("!f", struct.pack("!f", 0.1))[0]

    assert payload["motion_distance"] == wire_value
    assert payload["timeout_sec"] == wire_value


def test_canonical_skill_payload_has_exact_fields_and_normalizes_negative_zero():
    payload = canonical_skill_payload(
        "open_gripper_skill", schema_version=1, motion_distance=-0.0, default_timeout_sec=1.0
    )

    assert list(payload) == [
        "schema_version",
        "skill_name",
        "target_name",
        "container_name",
        "place_name",
        "motion_direction",
        "motion_distance",
        "timeout_sec",
    ]
    assert payload["motion_distance"] == 0.0
    assert str(payload["motion_distance"]) == "0.0"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_skill_payload_rejects_non_finite_numbers(value):
    with pytest.raises(ValueError, match="motion_distance"):
        canonical_skill_payload("move", schema_version=1, motion_distance=value, default_timeout_sec=1.0)
    with pytest.raises(ValueError, match="timeout_sec"):
        canonical_skill_payload("move", schema_version=1, timeout_sec=value, default_timeout_sec=1.0)


def test_canonical_skill_payload_rejects_negative_motion_distance():
    with pytest.raises(ValueError, match="motion_distance"):
        canonical_skill_payload("move", schema_version=1, motion_distance=-0.01, default_timeout_sec=1.0)


def test_canonical_skill_payload_rejects_numeric_overflow_as_value_error():
    with pytest.raises(ValueError, match="motion_distance"):
        canonical_skill_payload("move", schema_version=1, motion_distance=10**1000, default_timeout_sec=1.0)
    with pytest.raises(ValueError, match="timeout_sec"):
        canonical_skill_payload("move", schema_version=1, timeout_sec=10**1000, default_timeout_sec=1.0)


@pytest.mark.parametrize("skill_name", ["", "   ", None])
def test_canonical_skill_payload_requires_a_non_empty_skill_name(skill_name):
    with pytest.raises(ValueError, match="skill_name"):
        canonical_skill_payload(skill_name, schema_version=1, default_timeout_sec=1.0)


def test_canonical_skill_payload_requires_submitted_schema_version():
    with pytest.raises(TypeError, match="schema_version"):
        canonical_skill_payload("inspect_scene", default_timeout_sec=3.0)


def test_canonical_skill_payload_and_hash_distinguish_submitted_versions():
    version_1 = canonical_skill_payload("inspect_scene", schema_version=1, default_timeout_sec=3.0)
    version_2 = canonical_skill_payload("inspect_scene", schema_version=2, default_timeout_sec=3.0)

    assert version_1["schema_version"] == 1
    assert version_2["schema_version"] == 2
    assert skill_payload_hash(version_1) != skill_payload_hash(version_2)


def test_skill_payload_hash_uses_canonical_json_and_goal_uuid_is_deterministic():
    payload = canonical_skill_payload("inspect_scene", schema_version=1, default_timeout_sec=3.0)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode(
        "utf-8"
    )

    assert skill_payload_hash(payload) == hashlib.sha256(encoded).hexdigest()
    assert skill_goal_uuid("task-42") == uuid.uuid5(uuid.NAMESPACE_URL, "ibrobot:task-42")
    with pytest.raises(ValueError, match="task_id"):
        skill_goal_uuid(" ")


def test_canonical_navigation_payload_normalizes_direction_and_preserves_signed_coordinates():
    payload = canonical_skill_payload(
        "nav_abs_coordinate",
        schema_version=2,
        direction=" LEFT ",
        distance=1.25,
        degree=90.0,
        x=0.0,
        y=-2.5,
        yaw=-180.0,
        default_timeout_sec=30.0,
    )

    assert payload["direction"] == "left"
    assert payload["distance"] == 1.25
    assert payload["degree"] == 90.0
    assert (payload["has_x"], payload["x"]) == (True, 0.0)
    assert (payload["has_y"], payload["y"]) == (True, -2.5)
    assert (payload["has_yaw"], payload["yaw"]) == (True, -180.0)


def test_canonical_navigation_payload_distinguishes_absent_from_explicit_zero():
    absent = canonical_skill_payload("nav_abs_coordinate", schema_version=2, default_timeout_sec=30.0)
    explicit_zero = canonical_skill_payload(
        "nav_abs_coordinate",
        schema_version=2,
        x=0.0,
        y=0.0,
        yaw=0.0,
        default_timeout_sec=30.0,
    )

    assert (absent.get("has_x", False), absent.get("x", 0.0)) == (False, 0.0)
    assert (explicit_zero["has_x"], explicit_zero["x"]) == (True, 0.0)
    assert skill_payload_hash(absent) != skill_payload_hash(explicit_zero)


@pytest.mark.parametrize("field_name", ["distance", "degree", "x", "y", "yaw"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_navigation_payload_rejects_non_finite_numbers(field_name, value):
    with pytest.raises(ValueError, match=field_name):
        canonical_skill_payload("nav", schema_version=2, default_timeout_sec=1.0, **{field_name: value})


@pytest.mark.parametrize("field_name", ["distance", "degree"])
def test_canonical_navigation_payload_rejects_negative_magnitudes(field_name):
    with pytest.raises(ValueError, match=field_name):
        canonical_skill_payload("nav", schema_version=2, default_timeout_sec=1.0, **{field_name: -0.01})
