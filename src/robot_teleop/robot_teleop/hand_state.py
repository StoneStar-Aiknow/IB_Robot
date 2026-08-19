"""Vendor-neutral human-hand geometry extracted from mocap landmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .devices.mocap_retarget import extract_finger_flexions, extract_thumb_kinematics

HUMAN_HAND_SCHEMA = "human_hand_geometry_v1"
HUMAN_HAND_FEATURE_NAMES = (
    "thumb_root_yaw",
    "thumb_root_pitch",
    "thumb_mcp_flex",
    "thumb_ip_flex",
    "index_mcp_flex",
    "index_pip_flex",
    "index_dip_flex",
    "middle_mcp_flex",
    "middle_pip_flex",
    "middle_dip_flex",
    "ring_mcp_flex",
    "ring_pip_flex",
    "ring_dip_flex",
    "pinky_mcp_flex",
    "pinky_pip_flex",
    "pinky_dip_flex",
)


@dataclass(frozen=True, slots=True)
class HumanHandGeometry:
    feature_names: tuple[str, ...]
    features: tuple[float, ...]
    openness_score: float

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.feature_names, self.features, strict=True))


def extract_human_hand_geometry(positions, virtual_positions, side: str) -> HumanHandGeometry:
    """Extract scale-invariant anatomical angles from the complete hand skeleton."""
    if len(positions) < 20:
        raise ValueError("Twenty hand landmarks are required")
    if virtual_positions is None or len(virtual_positions) != 5:
        raise ValueError("Five virtual fingertips are required")

    thumb = extract_thumb_kinematics(positions, virtual_positions, side)
    values = [thumb.root_yaw, thumb.root_pitch, thumb.mcp_flex, thumb.ip_flex]
    for flexions in extract_finger_flexions(positions).values():
        values.extend(flexions)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Human hand geometry contains non-finite values")

    flexion = values[2:]
    openness_score = sum(flexion) / len(flexion)
    return HumanHandGeometry(HUMAN_HAND_FEATURE_NAMES, tuple(values), openness_score)


def detect_open_frames(frames, side: str, *, minimum_frames: int = 20):
    """Find the densest stable extended-hand cluster in an unlabeled motion sweep."""
    if len(frames) < minimum_frames:
        raise ValueError(f"At least {minimum_frames} frames are required to detect an open hand")

    rows = []
    for frame in frames:
        try:
            geometry = extract_human_hand_geometry(frame.positions, frame.virtual_positions, side)
        except (TypeError, ValueError, ArithmeticError):
            continue
        rows.append((frame, geometry))
    if len(rows) < minimum_frames:
        raise ValueError("Too few valid complete-skeleton frames to detect an open hand")

    rows.sort(key=lambda item: item[1].openness_score)
    extended = rows[: max(minimum_frames, math.ceil(len(rows) * 0.35))]
    angular_radius = math.radians(18.0)

    best_cluster = []
    for _, center in extended:
        center_values = center.features
        cluster = [
            item
            for item in extended
            if math.hypot(
                _angle_delta(item[1].features[0], center_values[0]),
                _angle_delta(item[1].features[1], center_values[1]),
            )
            <= angular_radius
        ]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster

    if len(best_cluster) < minimum_frames:
        raise ValueError("No stable open-hand cluster was found; include one clear full opening in the motion sweep")
    best_cluster.sort(key=lambda item: item[1].openness_score)
    return [item[0] for item in best_cluster]


def _angle_delta(first: float, second: float) -> float:
    return math.atan2(math.sin(first - second), math.cos(first - second))
