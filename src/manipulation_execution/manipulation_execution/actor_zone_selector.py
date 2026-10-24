"""The standing-zone actor selector on ROS messages.

``ZoneActorSelector`` is what the stream node builds instead of
``ActorSelector`` when ``actor_zone_file`` is set. It answers the same question
with the same ``update(frame, detections) -> ActorDecision``, so the node's
per-frame code does not change; the decision itself comes from
``actor_zone_core.ZoneCore``, which this module only feeds.

The node calls ``update`` for every frame PEAR runs on, but YOLOX refreshes
about once a second, so the same detections arrive several times. Only a new
detections stamp is a refresh and advances the core; the frames in between
reuse its answer. The staleness contract is ``ActorSelector``'s: missing
detections, or detections more than ``stale_sec`` older than the frame, are
unknown, never present, and release a locked actor.

Each refresh produces one report (the raw boxes plus every track and the
decision). It goes to ``on_report``, which the node wires to the
``actor_candidates`` publisher; the selector never touches ROS publishers
itself, so the node's frame path stays unchanged.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable

from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image

from ibrobot_msgs.msg import DetectionArray
from manipulation_execution.actor_selector import (
    NO_ACTOR,
    REASON_NO_DETECTIONS,
    REASON_STALE,
    ActorDecision,
    stamp_seconds,
)
from manipulation_execution.actor_zone_config import TuningConfig, ZoneConfig
from manipulation_execution.actor_zone_core import Box, ZoneCore


class ZoneActorSelector:
    """Lock one person standing inside the floor zone; release when they leave or are covered."""

    def __init__(
        self,
        zone: ZoneConfig,
        tuning: TuningConfig,
        person_confidence_threshold: float,
        stale_sec: float,
        on_report: Callable[[str], None] | None = None,
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        stale = float(stale_sec)
        if not math.isfinite(stale) or stale <= 0.0:
            raise ValueError("actor_stale_sec must be finite and greater than zero")
        self._stale_sec = stale
        self._core = ZoneCore(zone, tuning, person_confidence_threshold)
        self._on_report = on_report
        self._on_warning = on_warning

    @property
    def core(self) -> ZoneCore:
        return self._core

    def update(self, frame: Image, detections: DetectionArray | None) -> ActorDecision:
        """Decide the actor for ``frame`` from the newest YOLOX ``detections``."""
        frame_stamp = stamp_seconds(frame.header.stamp)
        if detections is None:
            self._core.unknown(frame_stamp)
            return self._unknown(frame, None, REASON_NO_DETECTIONS)
        detections_stamp = stamp_seconds(detections.header.stamp)
        if frame_stamp - detections_stamp > self._stale_sec:
            self._core.unknown(frame_stamp)
            return self._unknown(frame, detections.header.stamp, REASON_STALE)
        if detections_stamp != self._core.last_stamp:
            boxes = [
                Box(str(d.label), float(d.confidence), tuple(float(v) for v in d.bbox))
                for d in detections.detections
            ]
            report = self._core.step(detections_stamp, (int(frame.width), int(frame.height)), boxes)
            if report is not None:
                self._emit(report)
        core = self._core
        return ActorDecision(
            stamp=frame.header.stamp,
            detections_stamp=detections.header.stamp,
            persons_count=core.persons_count,
            actor_present=core.present,
            start_ready=core.present,
            actor_id=core.actor_id,
            bbox=core.bbox,
            confidence=core.confidence,
            reason=core.reason,
        )

    def _emit(self, report: dict) -> None:
        # The report is a record: failing to publish it must not cost the frame its decision.
        warnings = list(report["warnings"])
        if self._on_report is not None:
            try:
                self._on_report(json.dumps(report, separators=(",", ":"), allow_nan=False))
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"actor_candidates report not published: {type(exc).__name__}: {exc}")
        if self._on_warning is not None:
            for warning in warnings:
                self._on_warning(warning)

    @staticmethod
    def _unknown(frame: Image, detections_stamp: Time | None, reason: str) -> ActorDecision:
        return ActorDecision(
            stamp=frame.header.stamp,
            detections_stamp=detections_stamp,
            persons_count=0,
            actor_present=False,
            start_ready=False,
            actor_id=NO_ACTOR,
            bbox=(0.0, 0.0, 0.0, 0.0),
            confidence=0.0,
            reason=reason,
        )


__all__ = ["ZoneActorSelector"]
