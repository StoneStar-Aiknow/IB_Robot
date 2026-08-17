"""Pure geometry and admission logic for Nav2 dynamic target following."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FollowTarget:
    session_id: str
    state: int
    frame_id: str
    position: np.ndarray
    covariance_xy: np.ndarray
    measured: bool
    actionable: bool
    confidence: float
    stamp_ns: int


@dataclass(frozen=True)
class FollowGoal:
    position: np.ndarray
    yaw: float


def transform_planar_point(position: np.ndarray, translation: np.ndarray, yaw: float) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    if position.shape != (2,) or translation.shape != (2,):
        raise ValueError("planar positions must contain two coordinates")
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return translation + np.asarray(
        [cosine * position[0] - sine * position[1], sine * position[0] + cosine * position[1]]
    )


def stand_off_goal(object_position: np.ndarray, robot_position: np.ndarray, distance_m: float) -> FollowGoal:
    object_position = np.asarray(object_position, dtype=np.float64)
    robot_position = np.asarray(robot_position, dtype=np.float64)
    if object_position.shape != (2,) or robot_position.shape != (2,):
        raise ValueError("stand-off positions must contain two coordinates")
    if not math.isfinite(distance_m) or distance_m <= 0.0:
        raise ValueError("stand-off distance must be positive and finite")
    direction = robot_position - object_position
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        raise ValueError("robot and object positions must not coincide")
    goal = object_position + direction / norm * distance_m
    return FollowGoal(goal, math.atan2(object_position[1] - goal[1], object_position[0] - goal[0]))


def should_replan(
    previous: FollowGoal | None,
    current: FollowGoal,
    *,
    displacement_m: float,
    heading_delta_rad: float,
    elapsed_s: float,
    minimum_interval_s: float,
    prediction_only: bool = False,
) -> bool:
    if previous is None:
        return not prediction_only
    if prediction_only or elapsed_s < minimum_interval_s:
        return False
    distance = float(np.linalg.norm(current.position - previous.position))
    yaw_error = abs(math.atan2(math.sin(current.yaw - previous.yaw), math.cos(current.yaw - previous.yaw)))
    return distance >= displacement_m or yaw_error >= heading_delta_rad


class PathReplacementGate:
    """Serialize FollowPath replacement and keep only the newest pending path."""

    def __init__(self):
        self.active_path = None
        self.pending_path = None
        self.cancel_requested = False
        self.failed = False

    def activate(self, path):
        if self.failed or self.active_path is not None:
            raise RuntimeError("cannot activate a path while another path is authoritative")
        self.active_path = path

    def request_replacement(self, path):
        if self.failed:
            raise RuntimeError("path replacement gate is failed closed")
        self.pending_path = path
        self.cancel_requested = self.active_path is not None
        return self.cancel_requested

    def on_active_terminal(self):
        self.active_path = None
        self.cancel_requested = False
        replacement = self.pending_path
        self.pending_path = None
        return replacement

    def fail_closed(self):
        self.active_path = None
        self.pending_path = None
        self.cancel_requested = False
        self.failed = True
