"""Schema and persistence helpers for mHandPro retarget calibration."""

from __future__ import annotations

import json
import math
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from .mocap_retarget import (
    CHANNEL_NAMES,
    FEATURE_SCHEMA_AERO_COMPACT,
    FEATURE_SCHEMA_AERO_COMPACT_V1,
    FEATURE_SCHEMA_AERO_COMPACT_V2,
    FEATURE_SCHEMA_LEGACY,
    FEATURE_SCHEMA_SDK_VIRTUAL,
    MIN_CALIBRATION_SPAN,
)

SCHEMA_VERSION = 1
RAW_CAPTURE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RecordedGloveFrame:
    positions: list[list[float]]
    sequence: int
    timestamp: float
    quaternions: list[list[float]] | None = None
    virtual_positions: list[list[float]] | None = None
    sensor_states: list[int] | None = None


def calibration_document(
    side: str,
    low,
    high,
    *,
    sdk_version: str,
    persistence_verified: bool,
    feature_schema: str = FEATURE_SCHEMA_LEGACY,
    acquisition: dict | None = None,
    task_space: dict | None = None,
    thumb_neutral: dict | None = None,
    thumb_endpoints: dict | None = None,
    finger_endpoints: dict | None = None,
) -> dict:
    """Build a versioned calibration document after validating its endpoints."""
    calibration = {"low": list(low), "high": list(high)}
    _validate_endpoints(calibration, side)
    if feature_schema not in (
        FEATURE_SCHEMA_LEGACY,
        FEATURE_SCHEMA_SDK_VIRTUAL,
        FEATURE_SCHEMA_AERO_COMPACT_V1,
        FEATURE_SCHEMA_AERO_COMPACT_V2,
        FEATURE_SCHEMA_AERO_COMPACT,
    ):
        raise ValueError(f"Unsupported glove calibration feature_schema: {feature_schema!r}")
    document = {
        "schema_version": SCHEMA_VERSION,
        "side": side,
        "channel_order": list(CHANNEL_NAMES),
        "feature_unit": "radians",
        "target_unit": "normalized_0_1",
        "feature_schema": feature_schema,
        "low": [float(value) for value in calibration["low"]],
        "high": [float(value) for value in calibration["high"]],
        "sdk": {
            "name": "mHandPro",
            "version": str(sdk_version or "unknown"),
            "p_pose_cross_process_persistence_verified": bool(persistence_verified),
        },
    }
    if acquisition is not None:
        document["acquisition"] = acquisition
    if task_space is not None:
        document["task_space"] = {
            str(key): [float(item) for item in value] if isinstance(value, list | tuple) else float(value)
            for key, value in task_space.items()
        }
    if thumb_neutral is not None:
        document["thumb_neutral"] = _validated_thumb_neutral(thumb_neutral, side)
    if thumb_endpoints is not None:
        document["thumb_endpoints"] = _validated_thumb_endpoints(
            thumb_endpoints,
            side,
            require_mcp_ip=feature_schema == FEATURE_SCHEMA_AERO_COMPACT,
        )
    if finger_endpoints is not None:
        document["finger_endpoints"] = _validated_finger_endpoints(finger_endpoints, side)
    if feature_schema == FEATURE_SCHEMA_AERO_COMPACT:
        if "thumb_endpoints" not in document:
            raise ValueError(f"{side} {FEATURE_SCHEMA_AERO_COMPACT} calibration requires thumb_endpoints")
        if "finger_endpoints" not in document:
            raise ValueError(f"{side} {FEATURE_SCHEMA_AERO_COMPACT} calibration requires finger_endpoints")
    return document


def load_calibration(
    path: str | Path,
    side: str,
    *,
    require_persistence: bool,
) -> dict:
    """Load and validate calibration, returning only the retarget endpoints."""
    calibration_path = Path(path).expanduser()
    with calibration_path.open(encoding="utf-8") as stream:
        document = json.load(stream)

    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported glove calibration schema: {document.get('schema_version')!r}")
    if document.get("side") != side:
        raise ValueError(f"Glove calibration side is {document.get('side')!r}, expected {side!r}")
    if document.get("channel_order") != list(CHANNEL_NAMES):
        raise ValueError("Glove calibration channel_order does not match the runtime contract")
    if document.get("feature_unit") != "radians" or document.get("target_unit") != "normalized_0_1":
        raise ValueError("Glove calibration units do not match the runtime contract")
    if require_persistence and not document.get("sdk", {}).get("p_pose_cross_process_persistence_verified", False):
        raise ValueError(
            "mHandPro P-pose calibration has not been verified across SDK processes; "
            "rerun calibrate_glove without --skip-p-pose-persistence-check"
        )

    feature_schema = document.get("feature_schema", FEATURE_SCHEMA_LEGACY)
    if feature_schema not in (
        FEATURE_SCHEMA_LEGACY,
        FEATURE_SCHEMA_SDK_VIRTUAL,
        FEATURE_SCHEMA_AERO_COMPACT_V1,
        FEATURE_SCHEMA_AERO_COMPACT_V2,
        FEATURE_SCHEMA_AERO_COMPACT,
    ):
        raise ValueError(f"Unsupported glove calibration feature_schema: {feature_schema!r}")
    calibration = {
        "low": document.get("low"),
        "high": document.get("high"),
        "feature_schema": feature_schema,
    }
    if isinstance(document.get("task_space"), dict):
        calibration["task_space"] = document["task_space"]
    if feature_schema == FEATURE_SCHEMA_AERO_COMPACT_V1:
        calibration["thumb_neutral"] = _validated_thumb_neutral(document.get("thumb_neutral"), side)
    elif feature_schema in (FEATURE_SCHEMA_AERO_COMPACT_V2, FEATURE_SCHEMA_AERO_COMPACT):
        calibration["thumb_endpoints"] = _validated_thumb_endpoints(
            document.get("thumb_endpoints"),
            side,
            require_mcp_ip=feature_schema == FEATURE_SCHEMA_AERO_COMPACT,
        )
        if feature_schema == FEATURE_SCHEMA_AERO_COMPACT and "finger_endpoints" in document:
            calibration["finger_endpoints"] = _validated_finger_endpoints(document["finger_endpoints"], side)
    _validate_endpoints(calibration, side)
    return calibration


def _validated_thumb_neutral(values, side: str) -> dict[str, float]:
    required = ("root_yaw_rad", "root_pitch_rad", "mcp_flex_rad", "ip_flex_rad")
    if not isinstance(values, dict) or set(values) != set(required):
        raise ValueError(f"{side} thumb_neutral must contain exactly: {', '.join(required)}")
    converted = {key: float(values[key]) for key in required}
    if not all(math.isfinite(value) for value in converted.values()):
        raise ValueError(f"{side} thumb_neutral must contain finite values")
    return converted


def _validated_thumb_endpoints(values, side: str, *, require_mcp_ip: bool = False) -> dict[str, dict[str, float]]:
    required = ("root_yaw_rad", "root_pitch_rad", "mcp_flex_rad", "ip_flex_rad")
    if require_mcp_ip:
        required = (*required, "mcp_ip_flex_rad")
    if not isinstance(values, dict) or set(values) != set(required):
        raise ValueError(f"{side} thumb_endpoints must contain exactly: {', '.join(required)}")
    converted = {}
    for name in required:
        endpoint = values[name]
        if not isinstance(endpoint, dict) or set(endpoint) != {"neutral", "active"}:
            raise ValueError(f"{side} {name} endpoint must contain neutral and active")
        neutral = float(endpoint["neutral"])
        active = float(endpoint["active"])
        if not math.isfinite(neutral) or not math.isfinite(active):
            raise ValueError(f"{side} {name} endpoint values must be finite")
        if abs(active - neutral) < MIN_CALIBRATION_SPAN:
            raise ValueError(f"{side} {name} endpoint span is too small")
        converted[name] = {"neutral": neutral, "active": active}
    return converted


def _validated_finger_endpoints(values, side: str) -> dict[str, dict[str, float]]:
    required = ("index", "middle", "ring", "pinky")
    if not isinstance(values, dict) or set(values) != set(required):
        raise ValueError(f"{side} finger_endpoints must contain exactly: {', '.join(required)}")
    converted = {}
    for name in required:
        endpoint = values[name]
        if not isinstance(endpoint, dict) or set(endpoint) != {"neutral", "active"}:
            raise ValueError(f"{side} {name} finger endpoint must contain neutral and active")
        neutral = float(endpoint["neutral"])
        active = float(endpoint["active"])
        if not math.isfinite(neutral) or not math.isfinite(active) or active <= neutral:
            raise ValueError(f"{side} {name} finger endpoint values are invalid")
        if active - neutral < MIN_CALIBRATION_SPAN:
            raise ValueError(f"{side} {name} finger endpoint span is too small")
        converted[name] = {"neutral": neutral, "active": active}
    return converted


def write_calibration_atomic(path: str | Path, document: dict) -> Path:
    """Atomically replace a calibration JSON file in its destination directory."""
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output_path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise
    return output_path


def raw_capture_document(side: str, sdk_version: str, phase_frames: dict[str, list]) -> dict:
    """Build a replayable position/quaternion capture without altering calibration state."""
    if side not in ("left", "right"):
        raise ValueError("Raw glove capture side must be left or right")
    frames = []
    for phase, captured in phase_frames.items():
        if phase not in ("open", "sweep"):
            raise ValueError(f"Unknown raw glove capture phase: {phase!r}")
        for frame in captured:
            positions = [[float(value) for value in point] for point in frame.positions]
            if (
                len(positions) != 20
                or any(len(point) != 3 for point in positions)
                or not all(math.isfinite(value) for point in positions for value in point)
            ):
                raise ValueError("Raw glove capture positions must contain twenty xyz triples")
            timestamp = float(frame.timestamp)
            if not math.isfinite(timestamp):
                raise ValueError("Raw glove capture timestamp must be finite")
            item = {
                "phase": phase,
                "sequence": int(frame.sequence),
                "timestamp_monotonic": timestamp,
                "positions": positions,
            }
            if frame.quaternions is not None:
                quaternions = [[float(value) for value in quaternion] for quaternion in frame.quaternions]
                if (
                    len(quaternions) != 20
                    or any(len(quaternion) != 4 for quaternion in quaternions)
                    or not all(math.isfinite(value) for quaternion in quaternions for value in quaternion)
                ):
                    raise ValueError("Raw glove capture quaternions must contain twenty wxyz quadruples")
                item["quaternions_wxyz"] = quaternions
            if frame.virtual_positions is not None:
                virtual_positions = [[float(value) for value in point] for point in frame.virtual_positions]
                if (
                    len(virtual_positions) != 5
                    or any(len(point) != 3 for point in virtual_positions)
                    or not all(math.isfinite(value) for point in virtual_positions for value in point)
                ):
                    raise ValueError("Raw glove capture virtual_positions must contain five xyz triples")
                item["virtual_positions"] = virtual_positions
            if frame.sensor_states is not None:
                sensor_states = [int(value) for value in frame.sensor_states]
                if len(sensor_states) != 20:
                    raise ValueError("Raw glove capture sensor_states must contain twenty values")
                item["sensor_states"] = sensor_states
            frames.append(item)
    if not frames:
        raise ValueError("Raw glove capture requires at least one frame")
    return {
        "schema_version": RAW_CAPTURE_SCHEMA_VERSION,
        "side": side,
        "sdk": {"name": "mHandPro", "version": str(sdk_version or "unknown")},
        "position_unit": "sdk_world",
        "quaternion_order": "wxyz",
        "frames": frames,
    }


def write_raw_capture_atomic(path: str | Path, side: str, sdk_version: str, phase_frames: dict[str, list]) -> Path:
    """Atomically write a replayable raw glove capture."""
    return write_calibration_atomic(path, raw_capture_document(side, sdk_version, phase_frames))


def load_raw_capture(path: str | Path, side: str | None = None) -> dict[str, list[RecordedGloveFrame]]:
    """Load and validate a raw capture for deterministic offline analysis."""
    capture_path = Path(path).expanduser()
    with capture_path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("schema_version") != RAW_CAPTURE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported raw glove capture schema: {document.get('schema_version')!r}")
    capture_side = document.get("side")
    if capture_side not in ("left", "right") or (side is not None and capture_side != side):
        raise ValueError(f"Raw glove capture side is {capture_side!r}, expected {side!r}")
    if document.get("quaternion_order") != "wxyz":
        raise ValueError("Raw glove capture quaternion order must be wxyz")

    phases: dict[str, list[RecordedGloveFrame]] = {"open": [], "sweep": []}
    for item in document.get("frames", []):
        phase = item.get("phase")
        if phase not in phases:
            raise ValueError(f"Unknown raw glove capture phase: {phase!r}")
        positions = item.get("positions")
        if not isinstance(positions, list) or len(positions) != 20 or any(len(point) != 3 for point in positions):
            raise ValueError("Raw glove capture positions must contain twenty xyz triples")
        quaternions = item.get("quaternions_wxyz")
        if quaternions is not None and (
            not isinstance(quaternions, list)
            or len(quaternions) != 20
            or any(len(quaternion) != 4 for quaternion in quaternions)
        ):
            raise ValueError("Raw glove capture quaternions must contain twenty wxyz quadruples")
        virtual_positions = item.get("virtual_positions")
        if virtual_positions is not None and (
            not isinstance(virtual_positions, list)
            or len(virtual_positions) != 5
            or any(len(point) != 3 for point in virtual_positions)
        ):
            raise ValueError("Raw glove capture virtual_positions must contain five xyz triples")
        sensor_states = item.get("sensor_states")
        if sensor_states is not None and (not isinstance(sensor_states, list) or len(sensor_states) != 20):
            raise ValueError("Raw glove capture sensor_states must contain twenty values")
        frame = RecordedGloveFrame(
            positions=[[float(value) for value in point] for point in positions],
            sequence=int(item["sequence"]),
            timestamp=float(item["timestamp_monotonic"]),
            quaternions=(
                [[float(value) for value in quaternion] for quaternion in quaternions]
                if quaternions is not None
                else None
            ),
            virtual_positions=(
                [[float(value) for value in point] for point in virtual_positions]
                if virtual_positions is not None
                else None
            ),
            sensor_states=[int(value) for value in sensor_states] if sensor_states is not None else None,
        )
        phases[phase].append(frame)
    if not phases["open"] or not phases["sweep"]:
        raise ValueError("Raw glove capture requires both open and sweep frames")
    return phases


def _validate_endpoints(calibration: dict, side: str) -> None:
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    low = calibration.get("low")
    high = calibration.get("high")
    if not isinstance(low, list) or not isinstance(high, list) or len(low) != 7 or len(high) != 7:
        raise ValueError(f"{side} calibration must contain seven low/high endpoints")
    for name, start, end in zip(CHANNEL_NAMES, low, high, strict=True):
        start_value = float(start)
        end_value = float(end)
        if not math.isfinite(start_value) or not math.isfinite(end_value):
            raise ValueError(f"{side} {name} calibration endpoints must be finite")
        if abs(end_value - start_value) < MIN_CALIBRATION_SPAN:
            raise ValueError(f"{side} {name} calibration endpoints are too close")
