"""The X2 mirror photo's external trigger, shared by the photo node and whoever asks it for a photo.

During MIRRORING the photo node (manipulation_execution ``mirror_photo_node``)
keeps candidate frames in memory and, when the round ends, sends the best one
to the upper PC. Besides its own triggers (hand gestures, the person holding
still, a steady rhythm), any node may ask for a candidate by publishing one
JSON object on ``TRIGGER_TOPIC`` (``std_msgs/String``, reliable, depth
``TRIGGER_DEPTH``)::

    {"source": "preset_motions", "label": "heart", "score": 1.0, "stamp": 1762651234.5}

``source``  who asks (non-empty); reported with the photo
``label``   what was seen or done (non-empty); its weight is
            ``mirror_session.photo.triggers.event.labels.<label>``, else the
            event default weight
``score``   optional, 0 to 1 (default 1.0)
``stamp``   optional, seconds: the camera frame time the label belongs to
            (PearFrame / HandFrame ``header.stamp``); left out or 0, the
            newest frame the node holds

The sender needs nothing but this module: ``trigger_json`` builds the message.
A request outside MIRRORING, or one the node's limits refuse, is dropped
without an answer. Keep this module dependency-free: importing it must not
pull in any launch builder or ROS package.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

NODE_NAME = "mirror_photo"
TRIGGER_TOPIC = f"/{NODE_NAME}/trigger"
TRIGGER_DEPTH = 10


@dataclass(frozen=True)
class TriggerRequest:
    source: str
    label: str
    score: float = 1.0
    # Camera time in seconds; None: the newest frame.
    stamp: float | None = None


def trigger_json(label: str, *, source: str, score: float = 1.0, stamp: float | None = None) -> str:
    """The message a sender publishes on ``TRIGGER_TOPIC``; checked as the node reads it."""
    data: dict[str, Any] = {"source": source, "label": label, "score": score}
    if stamp is not None:
        data["stamp"] = stamp
    text = json.dumps(data, ensure_ascii=False, sort_keys=True)
    parse_trigger(text)
    return text


def parse_trigger(text: str) -> TriggerRequest:
    """One message from ``TRIGGER_TOPIC``; a ``ValueError`` says what is wrong with it."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"not JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise ValueError("must be a JSON object")
    unknown = sorted(set(data) - {"source", "label", "score", "stamp"})
    if unknown:
        raise ValueError(f"unknown keys {unknown}")
    for key in ("source", "label"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty string")
    score = data.get("score", 1.0)
    if isinstance(score, bool) or not isinstance(score, int | float) or not 0.0 <= score <= 1.0:
        raise ValueError("score must be a number from 0 to 1")
    stamp = data.get("stamp")
    if stamp is not None and (
        isinstance(stamp, bool) or not isinstance(stamp, int | float) or not math.isfinite(stamp)
    ):
        raise ValueError("stamp must be a number of seconds")
    return TriggerRequest(
        source=data["source"].strip(),
        label=data["label"].strip(),
        score=float(score),
        stamp=float(stamp) if stamp else None,
    )
