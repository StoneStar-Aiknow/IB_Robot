"""Bounded, dependency-free contract for the optional X2 gesture model service.

The model service classifies a crop; it never owns a robot or changes a session.
Client context stays immutable through infer/probe/retire and is echoed in full.
No ROS, NumPy, model runtime, filesystem or network imports belong here.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, fields

SERVICE_TYPE = "ibrobot_msgs/srv/ClassifyHandGesture"
SERVICE_ENDPOINT = "/perception/hand_gesture/classify"
GESTURE_TOPIC = "/hri/hand/gesture"
SOURCE_TOPIC = "/hri/hand/gesture_source_state"
REQUEST_SCHEMA = "x2_hand_gesture_request/v1"
RESULT_SCHEMA = "x2_hand_gesture_result/v1"
SOURCE_SCHEMA = "x2_gesture_source/v1"
GESTURE_SCHEMA = "x2_hand_gesture/v1"
JSON_BYTES = 4096
SOURCE_BYTES = 2048
MAX_INT = (1 << 63) - 1
CLASSES = ("heart", "holy", "xsign", "none")
OPERATIONS = ("infer", "probe", "retire")
STATUSES = ("ok", "not_ready", "busy", "rejected", "retired", "error")


def integer(value, name, minimum=0):
    if type(value) is not int or not minimum <= value <= MAX_INT:
        raise ValueError(f"{name} must be an integer in [{minimum}, {MAX_INT}]")
    return value


def number(value, name, minimum=0.0, maximum=float("inf")):
    if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be finite and in range")
    return float(value)


def identifier(value, name, *, empty=False):
    if empty and value == "":
        return value
    if not isinstance(value, str) or len(value) != 36:
        raise ValueError(f"{name} must be a canonical UUID")
    try:
        valid = str(uuid.UUID(value)) == value
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a canonical UUID")
    return value


def digest(value, name, *, empty=False):
    if empty and value == "":
        return value
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return value


def _pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out


def _constant(_value):
    raise ValueError("non-finite JSON number")


def load_json(text, schema, limit=JSON_BYTES):
    if not isinstance(text, str) or len(text) > limit or len(text.encode("utf-8")) > limit:
        raise ValueError("JSON exceeds its UTF-8 byte limit")
    try:
        value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    except (TypeError, RecursionError) as exc:
        raise ValueError("invalid JSON") from exc
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError("unknown JSON schema")
    return value


def dump_json(value, limit=JSON_BYTES):
    try:
        text = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, RecursionError) as exc:
        raise ValueError("invalid JSON value") from exc
    if len(text.encode("utf-8")) > limit:
        raise ValueError("JSON exceeds its UTF-8 byte limit")
    return text


def _fields(value, cls):
    if set(value) != {"schema", *(f.name for f in fields(cls))}:
        raise ValueError("missing or unknown fields")


@dataclass(frozen=True)
class RequestMeta:
    requester_boot_id: str
    producer_start_mono_ns: int
    stream_generation: int
    request_id: int
    op: str = "probe"
    session_token: str = ""
    actor_id: int = -1
    stamp_ns: int = 0
    pear_stamp_ns: int = 0
    crop: str = ""
    roi: tuple[int, int, int, int] = (0, 0, 0, 0)
    source_size: tuple[int, int] = (0, 0)
    budget_ms: float = 0.0
    retire_through: int = 0

    def __post_init__(self):
        identifier(self.requester_boot_id, "requester_boot_id")
        integer(self.producer_start_mono_ns, "producer_start_mono_ns")
        integer(self.stream_generation, "stream_generation", 1)
        integer(self.request_id, "request_id", 1)
        integer(self.retire_through, "retire_through")
        if self.op not in OPERATIONS:
            raise ValueError("invalid operation")
        if self.op == "retire" and not 0 < self.retire_through < self.request_id:
            raise ValueError("retirement needs an earlier request watermark")
        if self.op != "retire" and self.retire_through:
            raise ValueError("unexpected retirement watermark")
        identifier(self.session_token, "session_token", empty=self.op != "infer")
        integer(self.actor_id, "actor_id", 0 if self.op == "infer" else -1)
        integer(self.stamp_ns, "stamp_ns")
        integer(self.pear_stamp_ns, "pear_stamp_ns")
        number(self.budget_ms, "budget_ms", maximum=800.0)
        if len(self.roi) != 4 or len(self.source_size) != 2:
            raise ValueError("invalid crop dimensions")
        for value in (*self.roi, *self.source_size):
            if type(value) is not int or not -MAX_INT <= value <= MAX_INT:
                raise ValueError("crop dimensions must be integers")
        object.__setattr__(self, "roi", tuple(self.roi))
        object.__setattr__(self, "source_size", tuple(self.source_size))
        if self.op == "infer":
            x0, y0, x1, y1 = self.roi
            if self.crop not in ("one", "both") or not 0 < x1 - x0 == y1 - y0 <= 8192:
                raise ValueError("invalid crop ROI")
            if min(self.source_size) <= 0 or self.stamp_ns == 0 or self.pear_stamp_ns > self.stamp_ns:
                raise ValueError("invalid source frame")
            if self.budget_ms <= 0:
                raise ValueError("inference needs a positive budget")

    @property
    def source_key(self):
        return self.producer_start_mono_ns, self.requester_boot_id, self.stream_generation

    def to_dict(self):
        return {"schema": REQUEST_SCHEMA, **asdict(self)}

    def to_json(self):
        return dump_json(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.get("schema") != REQUEST_SCHEMA:
            raise ValueError("invalid request metadata")
        _fields(value, cls)
        try:
            return cls(**{k: v for k, v in value.items() if k != "schema"})
        except TypeError as exc:
            raise ValueError("invalid request fields") from exc

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(load_json(text, REQUEST_SCHEMA))


@dataclass(frozen=True)
class ResultMeta:
    request: RequestMeta
    server_boot_id: str
    engine_sha256: str
    status: str = "ok"
    gesture: str = "none"
    score: float = 0.0
    idle: bool = True
    retired_through: int = 0

    def __post_init__(self):
        if not isinstance(self.request, RequestMeta):
            raise ValueError("missing echoed request")
        identifier(self.server_boot_id, "server_boot_id")
        digest(self.engine_sha256, "engine_sha256")
        if self.status not in STATUSES or self.gesture not in CLASSES:
            raise ValueError("invalid result")
        number(self.score, "score", maximum=1.0)
        integer(self.retired_through, "retired_through")
        if type(self.idle) is not bool:
            raise ValueError("idle must be boolean")

    def to_json(self):
        value = asdict(self)
        value["request"] = self.request.to_dict()
        return dump_json({"schema": RESULT_SCHEMA, **value})

    @classmethod
    def from_json(cls, text):
        value = load_json(text, RESULT_SCHEMA)
        _fields(value, cls)
        value.pop("schema")
        value["request"] = RequestMeta.from_dict(value.get("request"))
        try:
            return cls(**value)
        except TypeError as exc:
            raise ValueError("invalid result fields") from exc


@dataclass(frozen=True)
class SourceState:
    requester_boot_id: str
    producer_start_mono_ns: int
    stream_generation: int
    state_seq: int
    available: bool
    published_mono_ns: int
    actor_id: int = -1
    session_token: str = ""
    server_boot_id: str = ""
    engine_sha256: str = ""
    reason: str = ""

    def __post_init__(self):
        identifier(self.requester_boot_id, "requester_boot_id")
        integer(self.producer_start_mono_ns, "producer_start_mono_ns")
        integer(self.stream_generation, "stream_generation", 1)
        integer(self.state_seq, "state_seq", 1)
        integer(self.published_mono_ns, "published_mono_ns")
        if type(self.available) is not bool:
            raise ValueError("available must be boolean")
        integer(self.actor_id, "actor_id", 0 if self.available else -1)
        identifier(self.session_token, "session_token", empty=not self.available)
        identifier(self.server_boot_id, "server_boot_id", empty=not self.available)
        digest(self.engine_sha256, "engine_sha256", empty=not self.available)
        if self.published_mono_ns < self.producer_start_mono_ns:
            raise ValueError("source publication precedes process startup")
        if not isinstance(self.reason, str) or len(self.reason.encode("utf-8")) > 128:
            raise ValueError("source reason too long")

    @property
    def source_key(self):
        return self.producer_start_mono_ns, self.requester_boot_id, self.stream_generation

    def to_json(self):
        return dump_json({"schema": SOURCE_SCHEMA, **asdict(self)}, SOURCE_BYTES)

    @classmethod
    def from_json(cls, text):
        value = load_json(text, SOURCE_SCHEMA, SOURCE_BYTES)
        _fields(value, cls)
        value.pop("schema")
        try:
            return cls(**value)
        except TypeError as exc:
            raise ValueError("invalid source fields") from exc


def session_token(state):
    """Canonical RUNNING transition token, shared by independently started nodes."""
    if isinstance(state, str):
        if len(state) > JSON_BYTES or len(state.encode("utf-8")) > JSON_BYTES:
            raise ValueError("session state too long")
        state = json.loads(state, object_pairs_hook=_pairs, parse_constant=_constant)
    if not isinstance(state, dict) or state.get("state") != "MIRRORING" or state.get("phase") != "running":
        return ""
    transition = state.get("last_transition")
    if not isinstance(transition, dict) or transition.get("to") != "MIRRORING/running":
        return ""
    when = number(transition.get("t"), "transition time")
    return str(uuid.uuid5(uuid.NAMESPACE_OID, f"x2-mirror-running:{when:.17g}"))
