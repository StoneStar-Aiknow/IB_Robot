"""Versioned perception events, independent from ROS and motion execution."""

import uuid

_VALID_POSE_REASONS = frozenset(
    {"pose_not_matched", "confirming", "matched", "holding", "exit_confirming", "pose_exited", "confirmation_gap"}
)


def match_state(result, now_ns, *, session_active):
    """A live matcher state is distinct from the once-per-activation photo event."""
    return {
        "schema_version": 1,
        "stamp": now_ns / 1e9,
        "template_name": result.template_name,
        "session_epoch": result.session_epoch,
        "actor_id": result.actor_id,
        "source_stamp_ns": result.source_stamp_ns,
        "sequence": result.sequence,
        "active": session_active is True and result.active is True,
        "input_valid": session_active is True
        and result.reason in _VALID_POSE_REASONS
        and result.source_stamp_ns is not None,
        "reason": result.reason,
    }


def stamp(ns):
    sec, nanosec = divmod(int(ns), 1_000_000_000)
    return {"sec": sec, "nanosec": nanosec}


def match_event(result, now_ns, event_id=None):
    """Emit only a newly confirmed match, never each held observation."""
    if not result.entered or result.source_stamp_ns is None:
        return None
    return {
        "schema_version": 1,
        "event_id": event_id or uuid.uuid4().hex,
        "template_name": result.template_name,
        "score": result.score,
        "source_stamp": stamp(result.source_stamp_ns),
        "matched_stamp": stamp(now_ns),
        "actor_id": result.actor_id,
    }
