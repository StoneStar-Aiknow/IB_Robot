"""Parsing a QoS mapping from a runtime description into an rclpy profile.

This lives outside :mod:`robot_config.contract_utils` so that a node publishing on a single bound
endpoint can apply that interface's QoS without importing the contract datamodel, which reaches
``tensormsg`` and ``torch``. ``contract_utils`` re-exports this function, so callers that already
import it from there are unaffected.
"""

from __future__ import annotations

from typing import Any

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


def qos_profile_from_dict(d: dict[str, Any] | None) -> QoSProfile | None:
    """Build a :class:`QoSProfile` from a description's ``qos`` mapping; ``None`` when absent."""

    if not d:
        return None
    rel = str(d.get("reliability", "reliable")).lower()
    hist = str(d.get("history", "keep_last")).lower()
    dur = str(d.get("durability", "volatile")).lower()
    depth = int(d.get("depth", 10))
    return QoSProfile(
        reliability=(ReliabilityPolicy.BEST_EFFORT if rel == "best_effort" else ReliabilityPolicy.RELIABLE),
        history=(HistoryPolicy.KEEP_ALL if hist == "keep_all" else HistoryPolicy.KEEP_LAST),
        depth=depth,
        durability=(DurabilityPolicy.TRANSIENT_LOCAL if dur == "transient_local" else DurabilityPolicy.VOLATILE),
    )
