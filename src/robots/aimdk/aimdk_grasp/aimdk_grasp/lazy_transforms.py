"""TF that is parsed only when a lookup needs it.

``tf2_ros.TransformListener`` converts every /tf message to Python as it
arrives. On the X2 that is ~1300 transforms a second (three publishers, ~30
transforms each), and it was half the idle grasp pipeline's GIL on the Orin
(live 2026-10-21), load that delays the head camera driver's frames for every
subscriber there. The pipeline looks transforms up only at image times: per
request and during an execution's watch. So /tf and /tf_static wait in reader
queues (:mod:`robot_runtime.on_demand`) and a lookup
first parses what arrived since the previous one.
"""

from __future__ import annotations

#: /tf messages kept between lookups: ~1 s of the X2's ~45 a second. A capture
#: looks up its image's time, at most ~0.5 s back (the image's age plus the
#: older sets it may fall back on). Every message kept is deserialized at the
#: next lookup, and that is what a capture after an idle spell waited for: a
#: robot /tf message carries 31 transforms, ~2 ms to convert on the Orin, so a
#: full queue of 200 held a capture 0.5-0.85 s (live 2026-10-23).
TF_QUEUE = 45
STATIC_QUEUE = 100
AUTHORITY = "lazy_transforms"


class LazyTransforms:
    """``lookup_transform`` as ``tf2_ros.Buffer`` has it, over /tf and /tf_static taken on demand."""

    def __init__(self, subscriptions, *, queue: int = TF_QUEUE):
        """``subscriptions`` is an ``OnDemandSubscriptions``; a lookup drains all of it."""
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from tf2_msgs.msg import TFMessage
        from tf2_ros import Buffer

        self.buffer = Buffer()
        self._subscriptions = subscriptions
        # The durability tf2_ros.TransformListener subscribes with.
        dynamic = QoSProfile(depth=queue, durability=DurabilityPolicy.VOLATILE)
        static = QoSProfile(depth=STATIC_QUEUE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        subscriptions.subscribe(TFMessage, "/tf", dynamic, queue, self._take)
        subscriptions.subscribe(TFMessage, "/tf_static", static, STATIC_QUEUE, self._take_static)

    def _take(self, message, _received: float) -> None:
        for transform in message.transforms:
            self.buffer.set_transform(transform, AUTHORITY)

    def _take_static(self, message, _received: float) -> None:
        for transform in message.transforms:
            self.buffer.set_transform_static(transform, AUTHORITY)

    def lookup_transform(self, target_frame: str, source_frame: str, time_point, timeout=None):
        self._subscriptions.drain()
        if timeout is None:
            return self.buffer.lookup_transform(target_frame, source_frame, time_point)
        return self.buffer.lookup_transform(target_frame, source_frame, time_point, timeout)
