"""Shared ROS QoS for replayable visual-game lifecycle events."""

from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


def visual_game_event_qos(depth: int = 256, *, lifespan_sec: float | None = None) -> QoSProfile:
    """Keep recent lifecycle events available to late-joining side consumers."""
    profile = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=max(1, int(depth)),
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    if lifespan_sec is not None:
        profile.lifespan = Duration(seconds=float(lifespan_sec))
    return profile


def visual_game_event_consumer_qos(depth: int = 256) -> QoSProfile:
    """Consume only events published while the side-effecting consumer is alive."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=max(1, int(depth)),
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
