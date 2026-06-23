"""Abstract base class for Cartesian teleop backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

Vec3 = tuple[float, float, float]


class CartesianBackend(ABC):
    """Common interface mirroring :class:`pymoveit2.MoveIt2Servo`."""

    @abstractmethod
    def enable(self) -> bool:
        """Request downstream solver enable. Returns ``True`` if accepted."""

    @abstractmethod
    def disable(self) -> bool:
        """Disable downstream solver. Returns ``True`` on success."""

    @abstractmethod
    def servo(self, linear: Vec3, angular: Vec3) -> None:
        """Send a single twist tick.

        Args:
            linear: 3-tuple in the **base** frame.
            angular: 3-tuple in the **tool_frame** frame (per device contract).
        """

    @property
    @abstractmethod
    def is_enabled(self) -> bool: ...
