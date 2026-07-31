"""Semantic object to distinct navigation stand-off target resolution."""

import math
from dataclasses import dataclass

import numpy as np

from .association import ACTION_READY_STATES, SemanticTrack


@dataclass(frozen=True)
class StagingCandidate:
    position: np.ndarray
    yaw: float
    clearance_m: float


@dataclass(frozen=True)
class TargetResolution:
    ready: bool
    message: str
    object: SemanticTrack
    staging: StagingCandidate | None = None


def generate_staging_candidates(
    object_position: np.ndarray,
    robot_position: np.ndarray,
    stand_off_distance_m: float,
    *,
    candidate_count: int = 16,
) -> list[StagingCandidate]:
    if stand_off_distance_m <= 0.0 or candidate_count <= 0:
        raise ValueError("stand-off distance and candidate count must be positive")
    object_position = np.asarray(object_position, dtype=np.float64)
    robot_position = np.asarray(robot_position, dtype=np.float64)
    if object_position.shape != (3,) or robot_position.shape != (3,):
        raise ValueError("object and robot positions must contain three coordinates")
    preferred = math.atan2(robot_position[1] - object_position[1], robot_position[0] - object_position[0])
    offsets = [0.0]
    for index in range(1, candidate_count):
        step = (index + 1) // 2
        sign = 1.0 if index % 2 else -1.0
        offsets.append(sign * step * 2.0 * math.pi / candidate_count)
    candidates = []
    for offset in offsets:
        angle = preferred + offset
        position = object_position.copy()
        position[0] += stand_off_distance_m * math.cos(angle)
        position[1] += stand_off_distance_m * math.sin(angle)
        yaw = math.atan2(object_position[1] - position[1], object_position[0] - position[0])
        candidates.append(StagingCandidate(position, yaw, stand_off_distance_m))
    return candidates


def resolve_target(
    track: SemanticTrack,
    robot_position: np.ndarray,
    stand_off_distance_m: float,
    checker,
    *,
    require_manipulation_ready: bool = False,
) -> TargetResolution:
    if track.state not in ACTION_READY_STATES:
        return TargetResolution(False, f"object state {track.state} is not action-ready", track)
    if require_manipulation_ready and track.state != "observed":
        return TargetResolution(False, "manipulation requires a freshly observed object", track)
    for candidate in generate_staging_candidates(track.position, robot_position, stand_off_distance_m):
        accepted, reason = checker(candidate)
        if accepted:
            return TargetResolution(True, "target resolved", track, candidate)
    return TargetResolution(False, reason or "no reachable stand-off pose", track)
