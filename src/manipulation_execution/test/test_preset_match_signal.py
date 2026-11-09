"""Only current actor/round geometry states may request or normally exit a preset."""

import json
from collections.abc import Mapping

import pytest

from manipulation_execution.preset_motions.match_signal import MatchSignal, decode_match_state


def message(**changes):
    return {
        "schema_version": 1,
        "template_name": "overhead_heart",
        "stamp": 20.0,
        "source_stamp_ns": 19_900_000_000,
        "session_epoch": 7,
        "sequence": 2,
        "actor_id": 3,
        "active": True,
        "input_valid": True,
        "reason": "holding",
    } | changes


def decode(value=None, **changes):
    arguments = {"now_ros_s": 20.0, "source_after_ns": 19_000_000_000, "actor_id": 3} | changes
    return decode_match_state(message() if value is None else value, **arguments)


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("reason", ["matched", "holding", "exit_confirming"])
def test_active_pose_requires_positive_current_round_geometry(as_json, reason):
    value = message(reason=reason)
    result = decode(json.dumps(value) if as_json else value)
    assert result == MatchSignal(True, True, 7, reason)
    assert decode(value, epoch=7) == result


@pytest.mark.parametrize(
    "reason",
    ["pose_not_matched", "confirming", "matched", "holding", "exit_confirming", "pose_exited", "confirmation_gap"],
)
def test_valid_inactive_pose_is_a_normal_exit_signal(reason):
    assert decode(message(active=False, reason=reason)) == MatchSignal(False, True, 7, reason)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("schema_version", True, "schema_invalid"),
        ("schema_version", 2, "schema_invalid"),
        ("template_name", "wave", "template_invalid"),
        ("stamp", True, "stamp_invalid"),
        ("stamp", "20", "stamp_invalid"),
        ("stamp", float("nan"), "stamp_invalid"),
        ("stamp", float("inf"), "stamp_invalid"),
        ("stamp", 19.64, "message_stale"),
        ("stamp", 20.11, "message_future"),
        ("source_stamp_ns", True, "source_stamp_invalid"),
        ("source_stamp_ns", 19_900_000_000.0, "source_stamp_invalid"),
        ("source_stamp_ns", -1, "source_stamp_invalid"),
        ("source_stamp_ns", 19_000_000_000, "source_before_round"),
        ("source_stamp_ns", 18_999_999_999, "source_before_round"),
        ("source_stamp_ns", 19_100_000_000, "source_stale"),
        ("source_stamp_ns", 20_110_000_000, "source_future"),
        ("session_epoch", True, "epoch_invalid"),
        ("session_epoch", -1, "epoch_invalid"),
        ("session_epoch", 7.0, "epoch_invalid"),
        ("sequence", True, "sequence_invalid"),
        ("sequence", -1, "sequence_invalid"),
        ("sequence", 2.0, "sequence_invalid"),
        ("actor_id", True, "actor_mismatch"),
        ("actor_id", -1, "actor_mismatch"),
        ("actor_id", 4, "actor_mismatch"),
        ("active", 1, "flags_invalid"),
        ("input_valid", 1, "flags_invalid"),
        ("input_valid", False, "input_invalid"),
        ("reason", "actor_lost", "reason_invalid"),
        ("reason", "source_stale", "reason_invalid"),
        ("reason", [], "reason_invalid"),
        ("reason", "pose_exited", "active_reason_invalid"),
        ("reason", "confirming", "active_reason_invalid"),
        ("reason", "pose_not_matched", "active_reason_invalid"),
        ("reason", "confirmation_gap", "active_reason_invalid"),
    ],
)
def test_invalid_input_never_becomes_a_normal_exit(field, value, reason):
    signal = decode(message(**{field: value}))
    assert signal == MatchSignal(False, False, None, reason)


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"now_ros_s": float("nan")}, "clock_invalid"),
        ({"now_ros_s": True}, "clock_invalid"),
        ({"source_after_ns": None}, "source_barrier_unavailable"),
        ({"source_after_ns": True}, "source_barrier_unavailable"),
        ({"source_after_ns": -1}, "source_barrier_unavailable"),
        ({"actor_id": None}, "actor_unavailable"),
        ({"actor_id": True}, "actor_unavailable"),
        ({"actor_id": -1}, "actor_unavailable"),
        ({"epoch": 8}, "epoch_mismatch"),
        ({"epoch": True}, "expected_epoch_invalid"),
        ({"epoch": -1}, "expected_epoch_invalid"),
        ({"max_message_age_s": 0.0}, "age_limit_invalid"),
        ({"max_source_age_s": float("inf")}, "age_limit_invalid"),
    ],
)
def test_missing_round_context_or_invalid_configuration_cannot_enable_pose(changes, reason):
    assert decode(**changes) == MatchSignal(False, False, None, reason)


@pytest.mark.parametrize("missing", list(message()))
def test_required_message_fields_cannot_be_omitted(missing):
    value = message()
    value.pop(missing)
    signal = decode(value)
    assert not signal.valid and not signal.active and signal.epoch is None


@pytest.mark.parametrize("value", ["{", "null", "[]", "true", [], object(), 3, b"{}"])
def test_malformed_objects_do_not_raise(value):
    signal = decode(value)
    assert not signal.valid and not signal.active


def test_none_message_is_invalid():
    signal = decode_match_state(None, now_ros_s=20.0, source_after_ns=19_000_000_000, actor_id=3)
    assert signal == MatchSignal(False, False, None, "message_not_mapping")


def test_a_misbehaving_mapping_does_not_raise():
    class Broken(Mapping):
        def __getitem__(self, key):
            raise RuntimeError("invalid external object")

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 0

    assert decode(Broken()) == MatchSignal(False, False, None, "message_invalid")


def test_age_limits_accept_exact_boundaries_but_reject_the_next_nanosecond():
    assert decode(message(stamp=19.65, source_stamp_ns=19_200_000_000)).valid
    assert decode(message(stamp=20.1, source_stamp_ns=20_100_000_000)).valid
    assert not decode(message(source_stamp_ns=19_199_999_999)).valid
    assert not decode(message(source_stamp_ns=20_100_000_001)).valid


def test_zero_epoch_sequence_and_actor_are_valid_integers():
    signal = decode(message(session_epoch=0, sequence=0, actor_id=0), actor_id=0, epoch=0)
    assert signal == MatchSignal(True, True, 0, "holding")


def test_round_barrier_is_strictly_newer_and_rejects_stale_human_exit():
    value = message(active=False, reason="pose_exited")
    assert not decode(value, source_after_ns=value["source_stamp_ns"]).valid
    assert decode(value, source_after_ns=value["source_stamp_ns"] - 1).valid
    assert not decode(value, epoch=8).valid


def test_input_invalid_cannot_be_relabelled_as_a_valid_geometry_exit():
    signal = decode(message(active=False, input_valid=False, reason="pose_exited"))
    assert signal == MatchSignal(False, False, None, "input_invalid")
