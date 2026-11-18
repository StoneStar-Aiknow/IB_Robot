"""The ``motion_template`` config file (yaml) and everything built from it. No ROS.

::

    motion_template:
      enabled: false                    # false (or no file): nothing is built, the baseline
      segments_file: segments.json      # relative to this file; the data plane reads the same one
      label_topic: /preset_motions/state
      play_topic: /mirror_session/template_play
      status_topic: /aimdk_dataplane/template_status
      dataplane_status_topic: /aimdk_runtime/dataplane/status
      label_stale_s: 0.5
      keepalive_period_s: 0.02          # at most 0.05 (M0: every 50 ms); 0.02 = every tick
      keepalive_give_up_s: 10.0
      blend: {...}                      # BlendConfig, every duration and threshold
      templates:                        # label -> the segments it may play
        xsign: {enabled: false, segments: [xsign_left_front, xsign_right_front]}

A template whose segments did not all pass the whitelist checks is left out
(logged); the others stay.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import yaml

from manipulation_execution.template_motion.blender import BlendConfig
from manipulation_execution.template_motion.registry import Segment, load_segments

# The data plane's CBF arm limits (generated_constants.hpp kArmLimits) shrunk by 1 degree, arm_stream order.
_ONE_DEGREE = math.radians(1.0)
_CBF_ARM = (
    (-3.0805, 2.0403),
    (-0.05236, 3.1241),
    (-2.5307, 2.5307),
    (-2.3562, 0.0),
    (-2.5482, 2.5482),
    (-0.5236, 0.5236),
    (-0.6981, 0.6981),
    (-3.0805, 2.0403),
    (-3.1241, 0.05236),
    (-2.5307, 2.5307),
    (-2.3562, 0.0),
    (-2.5482, 2.5482),
    (-0.5236, 0.5236),
    (-0.6981, 0.6981),
)
ARM_LIMITS = tuple((low + _ONE_DEGREE, high - _ONE_DEGREE) for low, high in _CBF_ARM)

_TOP_KEYS = {
    "enabled",
    "segments_file",
    "label_topic",
    "play_topic",
    "status_topic",
    "dataplane_status_topic",
    "label_stale_s",
    "keepalive_period_s",
    "keepalive_give_up_s",
    "blend",
    "templates",
}


@dataclass(frozen=True)
class TemplateSetup:
    blend: BlendConfig
    segments: dict[str, Segment]
    templates: dict[str, tuple[str, ...]]  # the enabled labels with every segment loaded
    label_topic: str
    play_topic: str
    status_topic: str
    dataplane_status_topic: str
    label_stale_s: float
    keepalive_period_s: float
    keepalive_give_up_s: float
    notes: list[str] = field(default_factory=list)  # what was left out, and why


def _topic(raw: Mapping, key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError(f"motion_template.{key} must be an absolute topic name")
    return value


def _positive(raw: Mapping, key: str, default: float) -> float:
    value = raw.get(key, default)
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"motion_template.{key} must be a positive number")
    return float(value)


def _keepalive(raw: Mapping) -> float:
    value = _positive(raw, "keepalive_period_s", 0.02)
    if value > 0.05:
        raise ValueError("motion_template.keepalive_period_s must be at most 0.05 (the port wants one every 50 ms)")
    return value


def parse_config(
    raw: object,
    base_dir: str,
    hand_limits: tuple[Sequence[float], Sequence[float]] | None = None,
) -> TemplateSetup | None:
    """None when switched off; ValueError for a config that cannot be used at all."""
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or not isinstance(raw.get("motion_template"), Mapping):
        raise ValueError("the file needs a motion_template mapping")
    raw = raw["motion_template"]
    unknown = sorted(set(raw) - _TOP_KEYS)
    if unknown:
        raise ValueError(f"unknown motion_template keys: {', '.join(map(str, unknown))}")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("motion_template.enabled must be true or false")
    if not enabled:
        return None
    blend = BlendConfig.from_mapping(raw.get("blend"))
    segments_file = raw.get("segments_file")
    if not isinstance(segments_file, str) or not segments_file:
        raise ValueError("motion_template.segments_file must be a path")
    path = segments_file if os.path.isabs(segments_file) else os.path.join(base_dir, segments_file)
    segments, rejected = load_segments(path, arm_limits=ARM_LIMITS, hand_limits=hand_limits)
    notes = [f"segment {name} left out: {reason}" for name, reason in sorted(rejected.items())]
    entries = raw.get("templates")
    if not isinstance(entries, Mapping):
        raise ValueError("motion_template.templates must be a mapping")
    templates = {}
    for label, entry in entries.items():
        if not isinstance(entry, Mapping) or set(entry) - {"enabled", "segments"}:
            raise ValueError(f"motion_template.templates.{label} must have enabled and segments only")
        names = entry.get("segments")
        if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
            raise ValueError(f"motion_template.templates.{label}.segments must list segment names")
        if not isinstance(entry.get("enabled", False), bool):
            raise ValueError(f"motion_template.templates.{label}.enabled must be true or false")
        if not entry.get("enabled", False):
            continue
        missing = [name for name in names if name not in segments]
        if missing:
            notes.append(f"template {label} left out: segments {missing} not loaded")
            continue
        templates[str(label)] = tuple(names)
    return TemplateSetup(
        blend=blend,
        segments=segments,
        templates=templates,
        label_topic=_topic(raw, "label_topic"),
        play_topic=_topic(raw, "play_topic"),
        status_topic=_topic(raw, "status_topic"),
        dataplane_status_topic=_topic(raw, "dataplane_status_topic"),
        label_stale_s=_positive(raw, "label_stale_s", 0.5),
        keepalive_period_s=_keepalive(raw),
        keepalive_give_up_s=_positive(raw, "keepalive_give_up_s", 10.0),
        notes=notes,
    )


def load_config(path: str, hand_limits=None) -> TemplateSetup | None:
    """Empty path: off."""
    if not path:
        return None
    with open(path, encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    return parse_config(raw, os.path.dirname(os.path.abspath(path)), hand_limits)
