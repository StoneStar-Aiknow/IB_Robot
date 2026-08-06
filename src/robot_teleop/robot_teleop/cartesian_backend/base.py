"""Abstract base class for Cartesian teleop backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


class CartesianBackend(ABC):
    """Common interface mirroring :class:`pymoveit2.MoveIt2Servo`."""

    @abstractmethod
    def enable(self) -> bool:
        """Request downstream enable; use ``is_enabled`` for confirmed state."""

    @abstractmethod
    def disable(self) -> bool:
        """Idempotently request downstream stop. Returns whether it was queued."""

    @abstractmethod
    def servo(self, linear: Vec3, angular: Vec3) -> None:
        """Send a single twist tick.

        Args:
            linear: 3-tuple in the **base** frame.
            angular: 3-tuple in the **tool_frame** frame (per device contract).
        """

    @abstractmethod
    def servo_pose(self, position: Vec3, orientation: Quat) -> None:
        """Send a clutch-relative base-frame pose command."""

    @abstractmethod
    def home(self) -> bool:
        """Request the backend home motion. Returns whether it was dispatched."""

    @abstractmethod
    def consume_home_result(self) -> bool | None:
        """Consume an asynchronous home result, or return ``None`` while pending."""

    def keepalive(self) -> None:
        """Refresh an optional downstream command lease."""
        return None

    @property
    @abstractmethod
    def is_enabled(self) -> bool: ...

    @property
    @abstractmethod
    def max_linear_speed(self) -> float:
        """Physical linear speed represented by a unit command."""

    @property
    @abstractmethod
    def max_angular_speed(self) -> float:
        """Physical angular speed represented by a unit command."""
