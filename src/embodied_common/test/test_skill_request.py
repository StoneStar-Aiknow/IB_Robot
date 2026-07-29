import hashlib
import json
import uuid

import pytest

from embodied_common.skill_request import canonical_skill_payload, skill_goal_uuid, skill_payload_hash


def test_canonical_skill_payload_normalizes_strings_and_uses_default_timeout():
    omitted_timeout = canonical_skill_payload(
        "  move_relative_ee  ",
        target_name="  cup  ",
        place_name=" tray ",
        motion_direction=" LEFT ",
        motion_distance=0.25,
        default_timeout_sec=12.0,
    )
    explicit_timeout = canonical_skill_payload(
        "move_relative_ee",
        target_name="cup",
        place_name="tray",
        motion_direction="left",
        motion_distance=0.25,
        timeout_sec=12.0,
        default_timeout_sec=12.0,
    )

    assert omitted_timeout == {
        "skill_name": "move_relative_ee",
        "target_name": "cup",
        "place_name": "tray",
        "motion_direction": "left",
        "motion_distance": 0.25,
        "timeout_sec": 12.0,
    }
    assert skill_payload_hash(omitted_timeout) == skill_payload_hash(explicit_timeout)


def test_canonical_skill_payload_has_exact_fields_and_normalizes_negative_zero():
    payload = canonical_skill_payload("open_gripper_skill", motion_distance=-0.0, default_timeout_sec=1.0)

    assert list(payload) == [
        "skill_name",
        "target_name",
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
        canonical_skill_payload("move", motion_distance=value, default_timeout_sec=1.0)
    with pytest.raises(ValueError, match="timeout_sec"):
        canonical_skill_payload("move", timeout_sec=value, default_timeout_sec=1.0)


def test_canonical_skill_payload_rejects_negative_motion_distance():
    with pytest.raises(ValueError, match="motion_distance"):
        canonical_skill_payload("move", motion_distance=-0.01, default_timeout_sec=1.0)


def test_canonical_skill_payload_rejects_numeric_overflow_as_value_error():
    with pytest.raises(ValueError, match="motion_distance"):
        canonical_skill_payload("move", motion_distance=10**1000, default_timeout_sec=1.0)
    with pytest.raises(ValueError, match="timeout_sec"):
        canonical_skill_payload("move", timeout_sec=10**1000, default_timeout_sec=1.0)


@pytest.mark.parametrize("skill_name", ["", "   ", None])
def test_canonical_skill_payload_requires_a_non_empty_skill_name(skill_name):
    with pytest.raises(ValueError, match="skill_name"):
        canonical_skill_payload(skill_name, default_timeout_sec=1.0)


def test_skill_payload_hash_uses_canonical_json_and_goal_uuid_is_deterministic():
    payload = canonical_skill_payload("inspect_scene", default_timeout_sec=3.0)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode(
        "utf-8"
    )

    assert skill_payload_hash(payload) == hashlib.sha256(encoded).hexdigest()
    assert skill_goal_uuid("task-42") == uuid.uuid5(uuid.NAMESPACE_URL, "ibrobot:task-42")
    with pytest.raises(ValueError, match="task_id"):
        skill_goal_uuid(" ")
