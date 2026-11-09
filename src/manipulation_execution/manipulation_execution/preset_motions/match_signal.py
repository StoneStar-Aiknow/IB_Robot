"""Decode fresh pose state without confusing a pose exit with broken input."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

_POSE_REASONS = frozenset(
    {"pose_not_matched", "confirming", "matched", "holding", "exit_confirming", "pose_exited", "confirmation_gap"}
)
_ACTIVE_REASONS = frozenset({"matched", "holding", "exit_confirming"})
_FUTURE_TOLERANCE_S = 0.1


@dataclass(frozen=True)
class MatchSignal:
    active: bool
    valid: bool
    epoch: int | None
    reason: str


def _integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _invalid(reason: str) -> MatchSignal:
    return MatchSignal(False, False, None, reason)


def decode_match_state(
    message,
    *,
    now_ros_s,
    source_after_ns,
    actor_id,
    epoch=None,
    max_message_age_s=0.35,
    max_source_age_s=0.8,
) -> MatchSignal:
    """Validate one live matcher message against the current round and actor.

    Message and source timestamps use ROS time. A missing round barrier or
    selected actor can never admit an old pose into a new mirror round.
    Invalid messages always return ``active=False, valid=False`` and do not
    supply an epoch for the caller to latch.
    """
    try:
        return _decode(message, now_ros_s, source_after_ns, actor_id, epoch, max_message_age_s, max_source_age_s)
    except Exception:
        return _invalid("message_invalid")


def _decode(message, now, source_after_ns, actor_id, expected_epoch, message_age, source_age):
    if not _finite(now):
        return _invalid("clock_invalid")
    if not _finite(message_age) or not _finite(source_age) or message_age <= 0.0 or source_age <= 0.0:
        return _invalid("age_limit_invalid")
    if not _integer(source_after_ns):
        return _invalid("source_barrier_unavailable")
    if not _integer(actor_id):
        return _invalid("actor_unavailable")
    if expected_epoch is not None and not _integer(expected_epoch):
        return _invalid("expected_epoch_invalid")
    if isinstance(message, str):
        try:
            message = json.loads(message)
        except (TypeError, ValueError):
            return _invalid("message_json_invalid")
    if not isinstance(message, Mapping):
        return _invalid("message_not_mapping")
    schema = message.get("schema_version")
    if not _integer(schema) or schema != 1:
        return _invalid("schema_invalid")
    if message.get("template_name") != "overhead_heart":
        return _invalid("template_invalid")
    stamp = message.get("stamp")
    if not _finite(stamp):
        return _invalid("stamp_invalid")
    if stamp > now + _FUTURE_TOLERANCE_S:
        return _invalid("message_future")
    if now > stamp + message_age:
        return _invalid("message_stale")
    source_ns = message.get("source_stamp_ns")
    if not _integer(source_ns):
        return _invalid("source_stamp_invalid")
    if source_ns <= source_after_ns:
        return _invalid("source_before_round")
    source_stamp = source_ns / 1_000_000_000
    if source_stamp > now + _FUTURE_TOLERANCE_S:
        return _invalid("source_future")
    if now > source_stamp + source_age:
        return _invalid("source_stale")
    session_epoch = message.get("session_epoch")
    if not _integer(session_epoch):
        return _invalid("epoch_invalid")
    if expected_epoch is not None and session_epoch != expected_epoch:
        return _invalid("epoch_mismatch")
    if not _integer(message.get("sequence")):
        return _invalid("sequence_invalid")
    selected = message.get("actor_id")
    if not _integer(selected) or selected != actor_id:
        return _invalid("actor_mismatch")
    active, input_valid = message.get("active"), message.get("input_valid")
    if not isinstance(active, bool) or not isinstance(input_valid, bool):
        return _invalid("flags_invalid")
    if not input_valid:
        return _invalid("input_invalid")
    reason = message.get("reason")
    if not isinstance(reason, str) or reason not in _POSE_REASONS:
        return _invalid("reason_invalid")
    if active and reason not in _ACTIVE_REASONS:
        return _invalid("active_reason_invalid")
    return MatchSignal(active, True, session_epoch, reason)
