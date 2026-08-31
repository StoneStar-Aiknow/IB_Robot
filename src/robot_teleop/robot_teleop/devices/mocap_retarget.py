"""Map mHandPro hand-node positions to seven calibrated channels."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from .retarget_utils import percentile

CHANNEL_NAMES = ("thumb_abd", "thumb_opp", "thumb_mcp", "index", "middle", "ring", "pinky")
FEATURE_SCHEMA_LEGACY = "positions_v1"
FEATURE_SCHEMA_SDK_VIRTUAL = "sdk_virtual_tip_v1"
FEATURE_SCHEMA_AERO_COMPACT_V1 = "aero_compact_v1"
FEATURE_SCHEMA_AERO_COMPACT_V2 = "aero_compact_v2"
FEATURE_SCHEMA_AERO_COMPACT = "aero_compact_v3"
FINGER_COEFFICIENTS = (12.4912, 7.3211, 9.0)
MIN_CALIBRATION_SPAN = math.radians(5.0)


@dataclass(frozen=True, slots=True)
class ThumbKinematics:
    """Human thumb geometry expressed in a palm-local frame."""

    root_yaw: float
    root_pitch: float
    mcp_flex: float
    ip_flex: float


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _length(vector):
    return math.sqrt(_dot(vector, vector))


def _unit(vector):
    length = _length(vector)
    if length < 1e-9:
        raise ValueError("Hand nodes overlap; a stable direction cannot be computed")
    return [value / length for value in vector]


def _scale(vector, factor):
    return [value * factor for value in vector]


def _geometric_angle(first, second):
    first_unit = _unit(first)
    second_unit = _unit(second)
    return math.atan2(
        _length(_cross(first_unit, second_unit)),
        max(-1.0, min(1.0, _dot(first_unit, second_unit))),
    )


def _joint_flex(parent, joint, child):
    return _geometric_angle(_sub(joint, parent), _sub(child, joint))


def _palm_basis(positions, side):
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")

    longitudinal = _unit(_sub(positions[8], positions[0]))
    radial_hint = _sub(positions[4], positions[16])
    radial = _unit(_sub(radial_hint, _scale(longitudinal, _dot(radial_hint, longitudinal))))
    normal = _unit(_cross(radial, longitudinal))
    if side == "left":
        normal = _scale(normal, -1.0)
    return radial, longitudinal, normal


def extract_features(positions, side):
    """Extract seven geometric features in radians before user calibration."""
    if len(positions) < 20:
        raise ValueError("Twenty hand positions are required")
    if any(len(position) != 3 or not all(math.isfinite(value) for value in position) for position in positions[:20]):
        raise ValueError("Hand positions must be finite xyz triples")

    radial, longitudinal, normal = _palm_basis(positions, side)
    thumb = _unit(_sub(positions[2], positions[1]))
    radial_component = _dot(thumb, radial)
    longitudinal_component = _dot(thumb, longitudinal)
    normal_component = _dot(thumb, normal)

    thumb_abd = math.atan2(radial_component, longitudinal_component)
    thumb_opp = math.atan2(normal_component, math.hypot(radial_component, longitudinal_component))
    thumb_mcp = _joint_flex(positions[1], positions[2], positions[3])

    coefficient_sum = sum(FINGER_COEFFICIENTS)
    fingers = []
    for root, pip, dip, tip in ((4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15), (16, 17, 18, 19)):
        angles = (
            _joint_flex(positions[0], positions[root], positions[pip]),
            _joint_flex(positions[root], positions[pip], positions[dip]),
            _joint_flex(positions[pip], positions[dip], positions[tip]),
        )
        fingers.append(
            sum(coefficient * angle for coefficient, angle in zip(FINGER_COEFFICIENTS, angles, strict=True))
            / coefficient_sum
        )
    return [thumb_abd, thumb_opp, thumb_mcp, *fingers]


def extract_sdk_skeleton_features(positions, virtual_positions, side):
    """Extract channels from the complete SDK skeleton, including virtual fingertips."""
    if (
        not isinstance(virtual_positions, list | tuple)
        or len(virtual_positions) != 5
        or any(len(point) != 3 or not all(math.isfinite(value) for value in point) for point in virtual_positions)
    ):
        raise ValueError("Five finite SDK virtual fingertip xyz triples are required")
    thumb = extract_thumb_kinematics(positions, virtual_positions, side)
    features = extract_features(positions, side)
    # Nodes 2 and 3 are the thumb proximal/distal bones. The SDK virtual thumb
    # tip completes the distal direction that is missing from the 20-node path.
    features[2] = thumb.ip_flex
    return features


def extract_thumb_kinematics(positions, virtual_positions, side) -> ThumbKinematics:
    """Extract the two root directions and both thumb flexion angles."""
    if len(positions) < 20:
        raise ValueError("Twenty hand positions are required")
    if any(len(position) != 3 or not all(math.isfinite(value) for value in position) for position in positions[:20]):
        raise ValueError("Hand positions must be finite xyz triples")
    if (
        not isinstance(virtual_positions, list | tuple)
        or len(virtual_positions) != 5
        or any(len(point) != 3 or not all(math.isfinite(value) for value in point) for point in virtual_positions)
    ):
        raise ValueError("Five finite SDK virtual fingertip xyz triples are required")

    radial, longitudinal, normal = _palm_basis(positions, side)
    root_direction = _unit(_sub(positions[2], positions[1]))
    radial_component = _dot(root_direction, radial)
    longitudinal_component = _dot(root_direction, longitudinal)
    normal_component = _dot(root_direction, normal)
    return ThumbKinematics(
        root_yaw=math.atan2(radial_component, longitudinal_component),
        root_pitch=math.atan2(normal_component, math.hypot(radial_component, longitudinal_component)),
        mcp_flex=_joint_flex(positions[1], positions[2], positions[3]),
        ip_flex=_joint_flex(positions[2], positions[3], virtual_positions[0]),
    )


def extract_finger_flexions(positions) -> dict[str, tuple[float, float, float]]:
    """Return anatomical MCP/PIP/DIP flexion for the four non-thumb fingers."""
    if len(positions) < 20:
        raise ValueError("Twenty hand positions are required")
    if any(len(position) != 3 or not all(math.isfinite(value) for value in position) for position in positions[:20]):
        raise ValueError("Hand positions must be finite xyz triples")
    fingers = {}
    for name, (root, pip, dip, tip) in zip(
        ("index", "middle", "ring", "pinky"),
        ((4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15), (16, 17, 18, 19)),
        strict=True,
    ):
        fingers[name] = (
            _joint_flex(positions[0], positions[root], positions[pip]),
            _joint_flex(positions[root], positions[pip], positions[dip]),
            _joint_flex(positions[pip], positions[dip], positions[tip]),
        )
    return fingers


def median_features(frames, side):
    if not frames:
        raise ValueError(f"No calibration frames captured for {side}")
    values = [extract_features(frame, side) for frame in frames]
    return [statistics.median(row[index] for row in values) for index in range(7)]


def build_calibration(pose_frames, side):
    """Build seven independent endpoints from four calibration poses."""
    required = ("open", "fist", "thumb_abd", "thumb_opp")
    missing = [pose for pose in required if not pose_frames.get(pose)]
    if missing:
        raise ValueError(f"Incomplete {side} calibration: {', '.join(missing)}")

    medians = {pose: median_features(pose_frames[pose], side) for pose in required}
    low = list(medians["open"])
    high = [medians["thumb_abd"][0], medians["thumb_opp"][1], *medians["fist"][2:]]
    for index, (start, end) in enumerate(zip(low, high, strict=True)):
        if abs(end - start) < MIN_CALIBRATION_SPAN:
            raise ValueError(
                f"{side} {CHANNEL_NAMES[index]} calibration endpoints are too close: {start:.4f}, {end:.4f} rad"
            )
    return {"low": low, "high": high}


def build_sweep_calibration(
    open_frames,
    sweep_frames,
    side,
    *,
    minimum_spans=None,
    lower_quantile=0.02,
    upper_quantile=0.98,
):
    """Build directed endpoints from an open reference and a continuous motion sweep."""
    if not open_frames or not sweep_frames:
        raise ValueError(f"{side} sweep calibration requires open-reference and motion-sweep frames")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("Sweep calibration quantiles must satisfy 0 <= low < high <= 1")
    if minimum_spans is None:
        minimum_spans = [MIN_CALIBRATION_SPAN] * len(CHANNEL_NAMES)
    if len(minimum_spans) != len(CHANNEL_NAMES):
        raise ValueError("Sweep calibration minimum_spans must contain seven values")

    open_features = median_features(open_frames, side)
    sweep_features = [extract_features(frame, side) for frame in sweep_frames]
    endpoints = []
    for index, (name, start) in enumerate(zip(CHANNEL_NAMES, open_features, strict=True)):
        values = [features[index] for features in sweep_features]
        lower = percentile(values, lower_quantile)
        upper = percentile(values, upper_quantile)
        end = max((lower, upper), key=lambda candidate: abs(candidate - start))
        required_span = max(MIN_CALIBRATION_SPAN, float(minimum_spans[index]))
        observed_span = abs(end - start)
        if not math.isfinite(required_span) or observed_span < required_span:
            raise ValueError(
                f"{side} {name} sweep coverage is too small: "
                f"{math.degrees(observed_span):.1f} deg observed, "
                f"{math.degrees(required_span):.1f} deg required"
            )
        endpoints.append(end)
    return {"low": open_features, "high": endpoints}


def build_sdk_skeleton_sweep_calibration(
    open_frames,
    sweep_frames,
    side,
    *,
    minimum_spans=None,
    lower_quantile=0.02,
    upper_quantile=0.98,
):
    """Build endpoints from complete SDK frames containing virtual fingertips."""
    if not open_frames or not sweep_frames:
        raise ValueError(f"{side} SDK skeleton calibration requires open-reference and motion-sweep frames")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("SDK skeleton calibration quantiles must satisfy 0 <= low < high <= 1")
    if minimum_spans is None:
        minimum_spans = [MIN_CALIBRATION_SPAN] * len(CHANNEL_NAMES)
    if len(minimum_spans) != len(CHANNEL_NAMES):
        raise ValueError("SDK skeleton calibration minimum_spans must contain seven values")

    def features(frame):
        positions = getattr(frame, "positions", None)
        virtual_positions = getattr(frame, "virtual_positions", None)
        if positions is None:
            raise ValueError("SDK skeleton calibration requires complete glove frames")
        return extract_sdk_skeleton_features(positions, virtual_positions, side)

    open_rows = [features(frame) for frame in open_frames]
    open_features = [statistics.median(row[index] for row in open_rows) for index in range(7)]
    sweep_features = [features(frame) for frame in sweep_frames]
    endpoints = []
    for index, (name, start) in enumerate(zip(CHANNEL_NAMES, open_features, strict=True)):
        values = [row[index] for row in sweep_features]
        lower = percentile(values, lower_quantile)
        upper = percentile(values, upper_quantile)
        end = max((lower, upper), key=lambda candidate: abs(candidate - start))
        required_span = max(MIN_CALIBRATION_SPAN, float(minimum_spans[index]))
        observed_span = abs(end - start)
        if not math.isfinite(required_span) or observed_span < required_span:
            raise ValueError(
                f"{side} {name} SDK skeleton coverage is too small: "
                f"{math.degrees(observed_span):.1f} deg observed, "
                f"{math.degrees(required_span):.1f} deg required"
            )
        endpoints.append(end)
    return {"low": open_features, "high": endpoints}


def positions_to_normalized(positions, calibration, side):
    """Apply per-channel calibration and return seven values clamped to [0, 1]."""
    return _features_to_normalized(extract_features(positions, side), calibration, side)


def sdk_skeleton_to_normalized(positions, virtual_positions, calibration, side):
    """Normalize complete SDK skeleton features through versioned endpoints."""
    if calibration.get("feature_schema") != FEATURE_SCHEMA_SDK_VIRTUAL:
        raise ValueError(
            f"{side} calibration feature_schema must be {FEATURE_SCHEMA_SDK_VIRTUAL!r} for sdk_skeleton mode"
        )
    return _features_to_normalized(
        extract_sdk_skeleton_features(positions, virtual_positions, side),
        calibration,
        side,
    )


def _features_to_normalized(features, calibration, side):
    if not calibration or len(calibration.get("low", ())) != 7 or len(calibration.get("high", ())) != 7:
        raise ValueError(f"{side} calibration must contain seven low/high endpoints")
    normalized = []
    for index, value in enumerate(features):
        start = float(calibration["low"][index])
        end = float(calibration["high"][index])
        span = end - start
        if not math.isfinite(start) or not math.isfinite(end) or abs(span) < MIN_CALIBRATION_SPAN:
            raise ValueError(f"{side} {CHANNEL_NAMES[index]} calibration endpoints are invalid")
        normalized.append(max(0.0, min(1.0, (value - start) / span)))
    return normalized
