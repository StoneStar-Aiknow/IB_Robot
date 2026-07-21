# -----------------------------------------------------------------------------
# Contract schema + runtime processing.
# -----------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

# Bridge to tensormsg for global registry
import tensormsg.registry as tensormsg_registry

# ---------- Contract datamodel ----------


@dataclass(frozen=True, slots=True)
class AlignSpec:
    """Timestamp alignment/selection behavior for an observation stream.

    strategy:  "hold" | "asof" | "drop"
    tol_ms:    as-of tolerance in ms for 'asof' (ignored for others)
    max_age_ms: maximum live age measured from local receipt time
    stamp:     "receive" | "header"
    """

    strategy: str = "hold"
    tol_ms: int = 0
    max_age_ms: int = 0
    stamp: str = "receive"


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    """Observation stream description (image/vector), driven by AlignSpec."""

    key: str
    topic: str
    type: str
    selector: dict[str, Any] | None = None  # {names: [...]}
    image: dict[str, Any] | None = None  # {resize:[H,W], encoding:'rgb8'|'bgr8'|'mono8'...}
    align: AlignSpec | None = None
    # {reliability, history, depth, durability}
    qos: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """Action stream description (publisher settings + mapping)."""

    key: str
    publish_topic: str
    type: str
    selector: dict[str, Any] | None = None
    from_tensor: dict[str, Any] | None = None
    publish_qos: dict[str, Any] | None = None
    publish_strategy: dict[str, Any] | None = None
    safety_behavior: str = "zeros"  # "zeros" | "hold"


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Optional 'task' channels (e.g., prompts)."""

    key: str
    topic: str
    type: str
    qos: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class Contract:
    """Top-level contract describing a policy's ROS 2 I/O surface."""

    name: str
    version: int
    rate_hz: float
    max_duration_s: float
    observations: list[ObservationSpec]
    actions: list[ActionSpec]
    tasks: list[TaskSpec]
    recording: dict[str, Any]
    robot_type: str | None = None
    timestamp_source: str = "receive"
    process: dict[str, Any] = None


def _as_align(d: dict[str, Any] | None) -> AlignSpec | None:
    if not d:
        return None
    tol_ms = int(d.get("tol_ms", 0))
    max_age_ms = int(d.get("max_age_ms", 0))
    if tol_ms < 0:
        raise ValueError("align.tol_ms must be non-negative")
    if max_age_ms < 0:
        raise ValueError("align.max_age_ms must be non-negative")
    return AlignSpec(
        strategy=str(d.get("strategy", "hold")).lower(),
        tol_ms=tol_ms,
        max_age_ms=max_age_ms,
        stamp=str(d.get("stamp", "receive")).lower(),
    )


# ---------- Unified SpecView (runtime) ----------


@dataclass(frozen=True, slots=True)
class SpecView:
    """Normalized runtime view of a single stream (observation or action)."""

    key: str
    topic: str
    ros_type: str
    is_action: bool
    names: list[str]
    image_resize: tuple[int, int] | None
    image_encoding: str
    image_channels: int  # 1, 3, or 4
    resample_policy: str  # obs: align.strategy; actions: "hold"
    asof_tol_ms: int
    max_age_ms: int
    stamp_src: str
    clamp: tuple[float, float] | None  # actions only
    safety_behavior: str | None  # actions only: "zeros" | "hold"


def _num_channels_from_encoding(encoding: str) -> int:
    """Infer channel count from ROS image encoding."""
    enc = encoding.lower()
    if enc in ("mono8", "mono16"):
        return 1
    if enc in ("bgr8", "rgb8", "bgr16", "rgb16"):
        return 3
    if enc in ("bgra8", "rgba8", "bgra16", "rgba16"):
        return 4
    abstract_prefixes = ["8uc", "8sc", "16uc", "16sc", "32sc", "32fc", "64fc"]
    for prefix in abstract_prefixes:
        if enc.startswith(prefix):
            if len(enc) == len(prefix):
                return 1
            try:
                channel_str = enc[len(prefix) :]
                if channel_str.isdigit():
                    return int(channel_str)
            except (ValueError, IndexError):
                pass
    if enc == "yuv422":
        return 2
    return 3


def iter_specs(contract: Contract) -> Iterable[SpecView]:
    """Yield normalized runtime specs for observations and actions."""
    for o in contract.observations:
        resize = None
        default_enc = "bgr8"
        if "depth" in o.topic.lower() or "depth" in o.key.lower():
            default_enc = "32fc1"
        if o.image:
            r = o.image.get("resize")
            if r and len(r) == 2:
                resize = (int(r[0]), int(r[1]))
            if "encoding" in o.image:
                default_enc = str(o.image.get("encoding")).lower()
        channels = _num_channels_from_encoding(default_enc)
        if "depth" in o.topic.lower() or "depth" in o.key.lower():
            channels = 3
        if o.image and "channels" in o.image:
            channels = int(o.image["channels"])
        names = list((o.selector or {}).get("names", []))
        al = o.align or AlignSpec()
        yield SpecView(
            key=o.key,
            topic=o.topic,
            ros_type=o.type,
            is_action=False,
            names=names,
            image_resize=resize,
            image_encoding=default_enc,
            image_channels=channels,
            resample_policy=al.strategy,
            asof_tol_ms=int(al.tol_ms),
            max_age_ms=int(al.max_age_ms),
            stamp_src=al.stamp,
            clamp=None,
            safety_behavior=None,
        )
    for a in contract.actions:
        names = list((a.selector or {}).get("names", []))
        clamp: tuple[float, float] | None = None
        if a.from_tensor and "clamp" in a.from_tensor:
            lo, hi = a.from_tensor["clamp"]
            clamp = (float(lo), float(hi))
        yield SpecView(
            key=a.key,
            topic=a.publish_topic,
            ros_type=a.type,
            is_action=True,
            names=names,
            image_resize=None,
            image_encoding="bgr8",
            image_channels=3,
            resample_policy="hold",
            asof_tol_ms=0,
            max_age_ms=0,
            stamp_src=contract.timestamp_source,
            clamp=clamp,
            safety_behavior=(a.safety_behavior or "zeros").lower(),
        )


def feature_from_spec(spec: SpecView, use_videos: bool) -> tuple[str, dict[str, Any], bool]:
    if spec.image_resize:
        h, w = int(spec.image_resize[0]), int(spec.image_resize[1])
        dtype = "video" if use_videos else "image"
        return (
            spec.key,
            {
                "dtype": dtype,
                "shape": (h, w, spec.image_channels),
                "names": ["height", "width", "channel"],
            },
            True,
        )
    if spec.ros_type == "sensor_msgs/msg/PointCloud2":
        return (
            spec.key,
            {"dtype": "pointcloud", "shape": None},  # sentinel, not into LeRobot features
            False,
        )
    if not spec.names:
        raise ValueError(f"{spec.key}: vector features must specify selector.names")
    return (
        spec.key,
        {"dtype": "float32", "shape": (len(spec.names),), "names": list(spec.names)},
        False,
    )


# ---------- Decoder/Encoder registries (Bridged to tensormsg) ----------

# Maintain compatibility with local DECODERS/ENCODERS dictionaries
DECODERS = tensormsg_registry.DECODER_REGISTRY
ENCODERS = tensormsg_registry.ENCODER_REGISTRY

# register_decoder and register_encoder are already imported from tensormsg


# ---------- Time helpers ----------


def stamp_from_header_ns(msg) -> int | None:
    try:
        st = msg.header.stamp
        ts_ns = int(st.sec) * 1_000_000_000 + int(st.nanosec)
        if ts_ns == 0:
            return None
        return ts_ns
    except (AttributeError, TypeError, ValueError):
        return None


# ---------- Decoders (Bridged) ----------


def decode_value(ros_type: str, msg, spec) -> Any:
    """Decode a ROS message using tensormsg through the rosetta API."""
    from tensormsg.converter import TensorMsgConverter

    return TensorMsgConverter.decode(msg, spec)


# ---------- Resampling ----------


def resample_hold(ts_ns: np.ndarray, vals: list[Any], ticks_ns: np.ndarray) -> list[Any]:
    out: list[Any] = []
    j, last = 0, None
    if len(ts_ns) > 0 and len(ticks_ns) > 0 and ticks_ns[0] < ts_ns[0]:
        last = vals[0]
    for t in ticks_ns:
        while j + 1 < len(ts_ns) and ts_ns[j + 1] <= t:
            j += 1
        if j < len(vals) and ts_ns[j] <= t:
            last = vals[j]
        out.append(last)
    return out


def resample_asof(ts_ns: np.ndarray, vals: list[Any], ticks_ns: np.ndarray, tol_ns: int) -> list[Any | None]:
    if tol_ns <= 0:
        return resample_hold(ts_ns, vals, ticks_ns)
    out: list[Any | None] = []
    j = 0
    for t in ticks_ns:
        while j + 1 < len(ts_ns) and ts_ns[j + 1] <= t:
            j += 1
        ok = j < len(vals) and ts_ns[j] <= t and (t - ts_ns[j]) <= tol_ns
        out.append(vals[j] if ok else None)
    return out


def resample_drop(ts_ns: np.ndarray, vals: list[Any], ticks_ns: np.ndarray, step_ns: int) -> list[Any | None]:
    out: list[Any | None] = []
    j, n = -1, len(ts_ns)
    for t in ticks_ns:
        while j + 1 < n and ts_ns[j + 1] <= t:
            j += 1
        out.append(vals[j] if (j >= 0 and ts_ns[j] > t - step_ns) else None)
    return out


def resample(
    policy: str,
    ts_ns: np.ndarray,
    vals: list[Any],
    ticks_ns: np.ndarray,
    step_ns: int,
    tol_ms: int,
) -> list[Any]:
    if policy == "drop":
        return resample_drop(ts_ns, vals, ticks_ns, step_ns)
    if policy == "asof":
        return resample_asof(ts_ns, vals, ticks_ns, max(0, int(tol_ms)) * 1_000_000)
    return resample_hold(ts_ns, vals, ticks_ns)


class StreamBuffer:
    def __init__(self, policy: str, step_ns: int, tol_ns: int = 0):
        self.policy = policy
        self.step_ns = int(step_ns)
        self.tol_ns = int(tol_ns)
        self.last_ts: int | None = None
        self.last_val: Any | None = None

    def push(self, ts_ns: int, val: Any) -> None:
        if self.last_ts is None or ts_ns >= self.last_ts:
            self.last_ts, self.last_val = ts_ns, val

    def sample(self, tick_ns: int):
        if self.last_ts is None:
            return None
        if self.policy == "drop":
            return self.last_val if (self.last_ts > tick_ns - self.step_ns) else None
        if self.policy == "asof":
            return self.last_val if (tick_ns - self.last_ts <= self.tol_ns) else None
        if self.policy == "hold":
            return self.last_val
        return None


# ---------- Encoders (Bridged) ----------


def encode_value(
    ros_type: str,
    names: list[str],
    action_vec: Sequence[float],
    clamp: tuple[float, float] | None = None,
):
    """Encode an action vector using tensormsg."""
    from tensormsg.converter import TensorMsgConverter

    return TensorMsgConverter.encode(ros_type, action_vec, names, clamp)


# ---------- QoS utilities ----------


def qos_profile_from_dict(d: dict[str, Any] | None) -> QoSProfile | None:
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


# ---------- Data processing utilities ----------


def zero_pad(feature_meta: dict[str, Any]) -> Any:
    dtype = feature_meta["dtype"]
    shape = tuple(feature_meta.get("shape") or ())
    if dtype in ("video", "image"):
        return np.zeros(shape, dtype=np.float32)
    if dtype == "float32":
        return np.zeros(shape, dtype=np.float32)
    if dtype == "float64":
        return np.zeros(shape, dtype=np.float64)
    if dtype == "string":
        return ""
    return None


def contract_fingerprint(contract) -> str:
    if hasattr(contract, "__dataclass_fields__"):
        contract_dict = asdict(contract)
    else:
        contract_dict = contract
    fingerprint_data = {
        "name": contract_dict.get("name"),
        "observations": [],
        "actions": [],
    }
    for obs in contract_dict.get("observations", []):
        fingerprint_data["observations"].append(
            {
                "key": obs.get("key"),
                "topic": obs.get("topic"),
                "type": obs.get("type"),
                "selector": obs.get("selector"),
                "align": obs.get("align"),
            }
        )
    for act in contract_dict.get("actions", []):
        fingerprint_data["actions"].append(
            {
                "key": act.get("key"),
                "publish_topic": act.get("publish_topic"),
                "type": act.get("type"),
                "selector": act.get("selector"),
            }
        )
    json_str = json.dumps(fingerprint_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()[:16]
