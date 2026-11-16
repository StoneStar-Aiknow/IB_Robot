"""Reusable robot client foundation for IB-Robot public workflows."""

from typing import TYPE_CHECKING, Any

from .transport import ENDPOINT_TYPES, GatewayTransport, gateway_transport

__version__ = "0.1.0"
__all__ = ["ENDPOINT_TYPES", "GatewayTransport", "RobotClient", "gateway_transport"]

if TYPE_CHECKING:
    from robot_client.client import RobotClient


def __getattr__(name: str) -> Any:
    if name == "RobotClient":
        from robot_client.client import RobotClient

        return RobotClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
