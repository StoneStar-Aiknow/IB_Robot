"""MoveIt Servo backend — wraps :class:`pymoveit2.MoveIt2Servo`.

Functional behaviour identical to the previous direct ``MoveIt2Servo`` usage
in ``xbox_controller.py`` / ``phone_device.py``. The only added concern is
the ``ToolAngularAdapter`` conversion from ``tool_frame`` to ``base`` before
the twist is published.
"""

from __future__ import annotations

from pymoveit2.moveit2_servo import MoveIt2Servo

from .base import CartesianBackend
from .frame_adapter import ToolAngularAdapter

Vec3 = tuple[float, float, float]


class MoveItServoBackend(CartesianBackend):
    def __init__(
        self,
        node,
        tf_buffer,
        base_link: str,
        tool_frame: str,
        linear_speed: float = 1.0,
        angular_speed: float = 1.0,
        stale_threshold_s: float = 0.2,
        input_mode: str = "velocity",
    ):
        if input_mode != "velocity":
            raise ValueError("MoveIt Servo only supports velocity input")
        self._node = node
        self._adapter = ToolAngularAdapter(
            node=node,
            tf_buffer=tf_buffer,
            base_link=base_link,
            tool_frame=tool_frame,
            stale_threshold_s=stale_threshold_s,
        )
        # MoveIt Servo always receives twists already in base — adapter converted.
        self._linear_speed = float(linear_speed)
        self._angular_speed = float(angular_speed)
        self._servo = MoveIt2Servo(
            node=node,
            frame_id=base_link,
            linear_speed=self._linear_speed,
            angular_speed=self._angular_speed,
            enable_at_init=False,
        )

    def enable(self) -> bool:
        return bool(self._servo.enable())

    def disable(self) -> bool:
        return bool(self._servo.disable())

    def servo(self, linear: Vec3, angular: Vec3) -> None:
        v_B, w_B = self._adapter.convert(linear, angular)
        self._servo.servo(linear=v_B, angular=w_B)

    def servo_pose(self, position: Vec3, orientation: tuple[float, float, float, float]) -> None:
        raise RuntimeError("MoveIt Servo does not support the relative-pose phone contract")

    def home(self) -> bool:
        return False

    def consume_home_result(self) -> bool | None:
        return None

    @property
    def is_enabled(self) -> bool:
        return bool(self._servo.is_enabled)

    @property
    def max_linear_speed(self) -> float:
        return self._linear_speed

    @property
    def max_angular_speed(self) -> float:
        return self._angular_speed
