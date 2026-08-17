import math
from enum import IntEnum


class CommandType(IntEnum):
    ABSOLUTE_POSE = 0
    FORWARD = 1
    BACKWARD = 2
    STRAFE_LEFT = 3
    STRAFE_RIGHT = 4
    TURN_LEFT = 5
    TURN_RIGHT = 6


class GoalValidationError(ValueError):
    pass


class StopVelocityGate:
    def __init__(self, *, linear_threshold: float, angular_threshold: float, stable_duration: float):
        self.linear_threshold = linear_threshold
        self.angular_threshold = angular_threshold
        self.stable_duration = stable_duration
        self._stable_since: float | None = None

    def reset(self) -> None:
        self._stable_since = None

    def observe(self, *, vx: float, vy: float, wz: float, now: float) -> bool:
        stopped = math.hypot(vx, vy) <= self.linear_threshold and abs(wz) <= self.angular_threshold
        if not stopped:
            self._stable_since = None
            return False
        if self._stable_since is None:
            self._stable_since = now
        return now - self._stable_since >= self.stable_duration


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _require_finite(*values: float) -> None:
    if not all(math.isfinite(value) for value in values):
        raise GoalValidationError("Navigation target values must be finite")


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    _require_finite(x, y, z, w)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9:
        raise GoalValidationError("Navigation target quaternion has zero norm")
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def resolve_navigation_target(
    *,
    command_type: int,
    value: float,
    target_frame: str,
    target_x: float,
    target_y: float,
    target_yaw: float,
    base_x: float,
    base_y: float,
    base_yaw: float,
) -> tuple[float, float, float]:
    try:
        command = CommandType(command_type)
    except ValueError as exc:
        raise GoalValidationError(f"Unsupported navigation command type: {command_type}") from exc

    if command == CommandType.ABSOLUTE_POSE:
        if target_frame != "map":
            raise GoalValidationError("Absolute navigation targets must use the map frame")
        _require_finite(target_x, target_y, target_yaw)
        return target_x, target_y, _normalize_angle(target_yaw)

    if not math.isfinite(value) or value <= 0.0:
        raise GoalValidationError("Relative command value must be positive finite")
    _require_finite(base_x, base_y, base_yaw)

    local_x = 0.0
    local_y = 0.0
    yaw_delta = 0.0
    if command == CommandType.FORWARD:
        local_x = value
    elif command == CommandType.BACKWARD:
        local_x = -value
    elif command == CommandType.STRAFE_LEFT:
        local_y = value
    elif command == CommandType.STRAFE_RIGHT:
        local_y = -value
    elif command == CommandType.TURN_LEFT:
        yaw_delta = value
    elif command == CommandType.TURN_RIGHT:
        yaw_delta = -value

    cos_yaw = math.cos(base_yaw)
    sin_yaw = math.sin(base_yaw)
    goal_x = base_x + cos_yaw * local_x - sin_yaw * local_y
    goal_y = base_y + sin_yaw * local_x + cos_yaw * local_y
    return goal_x, goal_y, _normalize_angle(base_yaw + yaw_delta)
