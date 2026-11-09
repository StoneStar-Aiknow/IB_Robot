"""The ``mirror_session.photo`` section, checked; shared by the launch builder and the photo node.

The builder checks the section at launch and, when it is wrong, warns and
starts no photo node (the mirror starts without it); the node receives the
checked section as one JSON parameter (``to_json``) and checks it again. No
ROS and no third-party imports: the builder imports this inside the registry
import budget.

A block accepts only its known keys, so a misspelt key fails at launch instead
of being ignored. Every error is a ``ValueError`` that starts with the path of
the bad value, e.g. ``mirror_session.photo.triggers.still.v_still``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, is_dataclass
from types import MappingProxyType
from typing import Any

SECTION_PATH = "mirror_session.photo"


@dataclass(frozen=True)
class UniformRule:
    """A candidate every ``period_sec`` of camera time: the round always has some."""

    enabled: bool = True
    period_sec: float = 3.0
    weight: float = 0.0


@dataclass(frozen=True)
class StillRule:
    """The person holding a pose: body speed (rad/s) below ``v_still`` for ``hold_sec``.

    The slowest frame of that stretch is the candidate. The next one needs the
    speed back above ``v_move`` and ``min_gap_sec`` since, so one pose gives
    one candidate.
    """

    enabled: bool = True
    v_still: float = 0.25
    v_move: float = 0.6
    hold_sec: float = 0.4
    min_gap_sec: float = 2.0
    weight: float = 1.5


_DEFAULT_GESTURES = MappingProxyType({"Victory": 3.0, "Thumb_Up": 3.0, "Pointing_Up": 1.5})


@dataclass(frozen=True)
class GestureRule:
    """A listed gesture on either hand at ``min_score`` or more for ``hold_sec``; per label at most every ``min_gap_sec``."""

    enabled: bool = True
    min_score: float = 0.6
    hold_sec: float = 0.2
    min_gap_sec: float = 1.5
    # label -> weight; read-only.
    labels: Mapping[str, float] = field(default_factory=lambda: _DEFAULT_GESTURES)


@dataclass(frozen=True)
class EventRule:
    """Requests on ``mirror_photo_contract.TRIGGER_TOPIC``; a label left out weighs ``default_weight``."""

    enabled: bool = True
    default_weight: float = 2.5
    labels: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class Limits:
    max_candidates_per_round: int = 24
    # Slots uniform candidates may not take: the triggered ones always find room.
    reserve_for_priority: int = 6
    # Triggered (non-uniform) candidates per minute of camera time.
    max_priority_per_minute: int = 20


@dataclass(frozen=True)
class Upload:
    # host[:port] or http://host[:port]; used only while target_file is missing or empty.
    default_target: str = ""
    timeout_sec: float = 3.0
    # The outbox keeps every photo not yet delivered, oldest dropped first past any of these.
    outbox_max_files: int = 200
    outbox_max_mb: float = 100.0
    min_free_gb: float = 2.0


@dataclass(frozen=True)
class PhotoSection:
    enabled: bool = False
    # Absolute; the outbox lives in <photos_dir>/outbox.
    photos_dir: str = ""
    # Absolute; one line host[:port], rewritten by scripts/x2_photo_capture/photo_target.sh.
    target_file: str = ""
    # No candidate in the first seconds of a round (the person is still stepping in).
    warmup_sec: float = 1.5
    uniform: UniformRule = field(default_factory=UniformRule)
    still: StillRule = field(default_factory=StillRule)
    gesture: GestureRule = field(default_factory=GestureRule)
    event: EventRule = field(default_factory=EventRule)
    # Added to a candidate's trigger weight: its rank in the round (0 to 1) times these.
    # Their sum (1.0) is below the gap between Victory/Thumb_Up and a held pose
    # (1.5), so a gesture beats a pose however blurred or moving it is.
    stillness_weight: float = 0.5
    sharpness_weight: float = 0.5
    # Past this the sharpness is left out for every candidate: the photo is never late.
    scoring_deadline_sec: float = 1.0
    limits: Limits = field(default_factory=Limits)
    upload: Upload = field(default_factory=Upload)


class _Reader:
    """One block's values, each checked and named by its path."""

    def __init__(self, block: Any, path: str, keys: tuple[str, ...]):
        if block is None:
            block = {}
        if not isinstance(block, Mapping):
            raise ValueError(f"{path}: must be a mapping")
        unknown = sorted(str(key) for key in set(block) - set(keys))
        if unknown:
            raise ValueError(f"{path}: unknown keys {unknown}; known: {list(keys)}")
        self.block = block
        self.path = path

    def get(self, key: str, default: Any) -> Any:
        return self.block.get(key, default)

    def flag(self, key: str, default: bool) -> bool:
        value = self.block.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(f"{self.path}.{key}: must be true or false")
        return value

    def number(self, key: str, default: float, *, minimum: float = 0.0, positive: bool = False) -> float:
        value = self.block.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            raise ValueError(f"{self.path}.{key}: must be a number")
        if value < minimum or (positive and value <= 0.0):
            raise ValueError(f"{self.path}.{key}: must be {'greater than' if positive else 'at least'} {minimum:g}")
        return float(value)

    def integer(self, key: str, default: int, *, minimum: int) -> int:
        value = self.block.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{self.path}.{key}: must be an integer of at least {minimum}")
        return value

    def text(self, key: str, default: str) -> str:
        value = self.block.get(key, default)
        if not isinstance(value, str):
            raise ValueError(f"{self.path}.{key}: must be a string")
        return value.strip()

    def labels(self, key: str, default: Mapping[str, float]) -> Mapping[str, float]:
        value = self.block.get(key)
        if value is None:
            return default
        if not isinstance(value, Mapping):
            raise ValueError(f"{self.path}.{key}: must map label to weight")
        checked = {}
        for label, weight in value.items():
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"{self.path}.{key}: labels must be non-empty strings")
            if isinstance(weight, bool) or not isinstance(weight, int | float) or not math.isfinite(weight):
                raise ValueError(f"{self.path}.{key}.{label}: must be a number")
            checked[label.strip()] = float(weight)
        return MappingProxyType(checked)


def _fields(cls) -> tuple[str, ...]:
    return tuple(cls.__dataclass_fields__)


def parse_photo_section(section: Mapping[str, Any] | None) -> PhotoSection:
    """The section over the defaults; ``None`` is a section left out (photo off)."""
    if section is None:
        return PhotoSection()
    top = _Reader(
        section,
        SECTION_PATH,
        (
            "enabled",
            "photos_dir",
            "target_file",
            "warmup_sec",
            "triggers",
            "weights",
            "scoring_deadline_sec",
            "limits",
            "upload",
        ),
    )
    enabled = top.flag("enabled", False)
    dirs = {}
    for key in ("photos_dir", "target_file"):
        dirs[key] = top.text(key, "")
        if enabled and not dirs[key].startswith("/"):
            raise ValueError(f"{SECTION_PATH}.{key}: must be an absolute path")
    if dirs["target_file"] and dirs["photos_dir"] and dirs["target_file"].rstrip("/") == dirs["photos_dir"].rstrip("/"):
        raise ValueError(f"{SECTION_PATH}.target_file: must not be photos_dir")

    triggers = _Reader(top.get("triggers", None), f"{SECTION_PATH}.triggers", ("uniform", "still", "gesture", "event"))
    uniform = _Reader(triggers.get("uniform", None), f"{triggers.path}.uniform", _fields(UniformRule))
    still = _Reader(triggers.get("still", None), f"{triggers.path}.still", _fields(StillRule))
    gesture = _Reader(triggers.get("gesture", None), f"{triggers.path}.gesture", _fields(GestureRule))
    event = _Reader(triggers.get("event", None), f"{triggers.path}.event", _fields(EventRule))
    weights = _Reader(top.get("weights", None), f"{SECTION_PATH}.weights", ("stillness", "sharpness"))
    limits = _Reader(top.get("limits", None), f"{SECTION_PATH}.limits", _fields(Limits))
    upload = _Reader(top.get("upload", None), f"{SECTION_PATH}.upload", _fields(Upload))

    defaults = PhotoSection()
    still_rule = StillRule(
        enabled=still.flag("enabled", True),
        v_still=still.number("v_still", StillRule.v_still, positive=True),
        v_move=still.number("v_move", StillRule.v_move, positive=True),
        hold_sec=still.number("hold_sec", StillRule.hold_sec),
        min_gap_sec=still.number("min_gap_sec", StillRule.min_gap_sec),
        weight=still.number("weight", StillRule.weight),
    )
    if still_rule.v_move < still_rule.v_still:
        raise ValueError(f"{still.path}.v_move: must be at least v_still")
    limit = Limits(
        max_candidates_per_round=limits.integer("max_candidates_per_round", Limits.max_candidates_per_round, minimum=1),
        reserve_for_priority=limits.integer("reserve_for_priority", Limits.reserve_for_priority, minimum=0),
        max_priority_per_minute=limits.integer("max_priority_per_minute", Limits.max_priority_per_minute, minimum=1),
    )
    if limit.reserve_for_priority >= limit.max_candidates_per_round:
        raise ValueError(f"{limits.path}.reserve_for_priority: must be less than max_candidates_per_round")
    default_target = upload.text("default_target", "")
    if default_target:
        try:
            normalize_target(default_target)
        except ValueError as exc:
            raise ValueError(f"{upload.path}.default_target: {exc}") from exc
    return PhotoSection(
        enabled=enabled,
        photos_dir=dirs["photos_dir"],
        target_file=dirs["target_file"],
        warmup_sec=top.number("warmup_sec", defaults.warmup_sec),
        uniform=UniformRule(
            enabled=uniform.flag("enabled", True),
            period_sec=uniform.number("period_sec", UniformRule.period_sec, positive=True),
            weight=uniform.number("weight", UniformRule.weight),
        ),
        still=still_rule,
        gesture=GestureRule(
            enabled=gesture.flag("enabled", True),
            min_score=gesture.number("min_score", GestureRule.min_score),
            hold_sec=gesture.number("hold_sec", GestureRule.hold_sec),
            min_gap_sec=gesture.number("min_gap_sec", GestureRule.min_gap_sec),
            labels=gesture.labels("labels", _DEFAULT_GESTURES),
        ),
        event=EventRule(
            enabled=event.flag("enabled", True),
            default_weight=event.number("default_weight", EventRule.default_weight),
            labels=event.labels("labels", MappingProxyType({})),
        ),
        stillness_weight=weights.number("stillness", defaults.stillness_weight),
        sharpness_weight=weights.number("sharpness", defaults.sharpness_weight),
        scoring_deadline_sec=top.number("scoring_deadline_sec", defaults.scoring_deadline_sec, positive=True),
        limits=limit,
        upload=Upload(
            default_target=default_target,
            timeout_sec=upload.number("timeout_sec", Upload.timeout_sec, positive=True),
            outbox_max_files=upload.integer("outbox_max_files", Upload.outbox_max_files, minimum=1),
            outbox_max_mb=upload.number("outbox_max_mb", Upload.outbox_max_mb, positive=True),
            min_free_gb=upload.number("min_free_gb", Upload.min_free_gb),
        ),
    )


DEFAULT_PORT = 8765


def normalize_target(text: str) -> str:
    """``host[:port]`` or ``http://host[:port]`` as the base URL ``http://host:port``; ``ValueError`` if neither."""
    value = text.strip()
    if value.startswith("http://"):
        value = value[len("http://") :]
    value = value.rstrip("/")
    if not value or any(ch in "/@?#" or ch.isspace() for ch in value):
        raise ValueError(f"{text!r} is not host[:port]")
    host, sep, port = value.rpartition(":")
    if not sep:
        host, port = value, str(DEFAULT_PORT)
    if not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise ValueError(f"{text!r} is not host[:port]")
    return f"http://{host}:{int(port)}"


def _plain(value: Any) -> Any:
    # Not dataclasses.asdict: it deep-copies, and the read-only label maps cannot be copied.
    if is_dataclass(value):
        return {name: _plain(getattr(value, name)) for name in _fields(type(value))}
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    return value


def to_json(section: PhotoSection) -> str:
    """The section as the photo node's one string parameter."""
    data = _plain(section)
    # Back to the yaml layout, so from_json reads it with the same checks.
    yaml_layout = {
        "enabled": data["enabled"],
        "photos_dir": data["photos_dir"],
        "target_file": data["target_file"],
        "warmup_sec": data["warmup_sec"],
        "triggers": {rule: data[rule] for rule in ("uniform", "still", "gesture", "event")},
        "weights": {"stillness": data["stillness_weight"], "sharpness": data["sharpness_weight"]},
        "scoring_deadline_sec": data["scoring_deadline_sec"],
        "limits": data["limits"],
        "upload": data["upload"],
    }
    return json.dumps(yaml_layout, ensure_ascii=False, sort_keys=True)


def from_json(text: str) -> PhotoSection:
    """``to_json`` read back, checked again as ``parse_photo_section`` does."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{SECTION_PATH}: not a JSON section ({exc})") from exc
    return parse_photo_section(data)
