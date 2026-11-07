"""Subscribe the runtime's public joint-state views into a :class:`FeedbackHistory`.

Endpoints and QoS come from the bound public interface description, never from
vendor topic names: this package reads the robot exactly as any other runtime
consumer does. The views are taken on demand
(:mod:`robot_runtime.on_demand`): each reader queue holds
as many samples as the history, and the history fills when a reader drains them.
The runtime status keeps the receipt time the middleware recorded for it, so
the history can tell a live verdict from one whose publishing stopped
(:data:`aimdk_grasp.feedback.MAX_STATUS_AGE_S`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sensor_msgs.msg import JointState

from ibrobot_msgs.msg import RuntimeStatus
from robot_config.contract_utils import qos_profile_from_dict

from .feedback import FeedbackHistory

STATUS_INTERFACE = "runtime.status"


def _bound(interfaces: Mapping[str, Any], name: str, message_type: str) -> Mapping[str, Any]:
    interface = interfaces.get(name) or {}
    shape = (interface.get("kind"), interface.get("direction"), interface.get("message_type"))
    if shape != ("topic", "publish", message_type):
        raise ValueError(f"interface description must publish {name} as a {message_type} topic, got {shape}")
    return interface


def subscribe_feedback(subscriptions, description: Mapping[str, Any], history: FeedbackHistory) -> None:
    """Subscribe the views and the runtime status on ``subscriptions`` (an ``OnDemandSubscriptions``)."""
    interfaces = description.get("interfaces") or {}
    for view in history.views:
        interface = _bound(interfaces, view, "sensor_msgs/msg/JointState")

        def on_state(message: JointState, _received: float | None, view: str = view) -> None:
            stamp = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
            history.record(view, stamp, message.name, message.position)

        subscriptions.subscribe(
            JointState, str(interface["endpoint"]), qos_profile_from_dict(interface.get("qos")), history.depth, on_state
        )
    status = _bound(interfaces, STATUS_INTERFACE, "ibrobot_msgs/msg/RuntimeStatus")

    def on_status(message: RuntimeStatus, received: float | None) -> None:
        history.set_healthy(message.lifecycle == "ACTIVE" and not message.faults, received)

    subscriptions.subscribe(
        RuntimeStatus, str(status["endpoint"]), qos_profile_from_dict(status.get("qos")), 1, on_status
    )
