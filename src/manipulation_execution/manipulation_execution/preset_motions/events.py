"""Versioned perception events, independent from ROS and motion execution."""

import uuid


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
