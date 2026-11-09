"""Camera readers that can be released while the robot moves.

The Gateway pauses ``motion_imitation_stream`` around robot motion and the stream
republishes its state on ``PAUSED_STATE_TOPIC``. A reader built here drops its DDS
subscription while paused and creates it again afterwards, so the platform stops
sending that process the frames at all (each subscribing process receives its own
copy over the network). Without a stream nothing is ever published, and a follower
stays subscribed as before.

Each reader spins its subscription on a thread of its own, and only that thread
creates or destroys it, between callbacks. Destroying a subscription under a
``MultiThreadedExecutor`` races the executor's take and ends ``spin`` with
``InvalidHandle``, which stopped the stream node on the robot on 2026-11-08.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, ShutdownException, SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

from embodied_common.motion_imitation_contract import PAUSED_STATE_TOPIC

# Latched: a reader that starts, or restarts, while a skill runs learns it at once.
PAUSED_STATE_QOS = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

# How long a pause can wait for the reader thread: one spin period plus one callback.
_SPIN_PERIOD_SEC = 0.05
_APPLY_TIMEOUT_SEC = 1.0


class PausableSubscription:
    """One subscription that exists only while it is not paused.

    ``callback`` runs on the reader's own thread. ``state_topic`` is the latched state
    to follow; ``None`` leaves ``set_paused`` to the owner (the stream node itself).
    """

    def __init__(
        self,
        node,
        message_type,
        topic: str,
        callback: Callable,
        qos,
        *,
        on_change: Callable[[bool], None] | None = None,
        state_topic: str | None = PAUSED_STATE_TOPIC,
    ) -> None:
        self._node = node
        self._args = (message_type, topic, callback, qos)
        self._on_change = on_change
        self._lock = threading.Lock()  # one transition at a time
        self.paused = False
        self.transitions = 0
        self._condition = threading.Condition()
        self._wanted = (False, 0)  # (paused, generation)
        self._applied = -1
        self._subscribed = False
        # No global arguments: launch's ``__node:=`` remap would give this node the owner's name.
        self._reader = rclpy.create_node(
            f"{node.get_name()}_camera_reader",
            context=node.context,
            use_global_arguments=False,
            enable_rosout=False,
            start_parameter_services=False,
        )
        self._executor = SingleThreadedExecutor(context=node.context)
        self._executor.add_node(self._reader)
        self._thread = threading.Thread(target=self._run, name="camera-reader", daemon=True)
        self._thread.start()
        self._await(0)
        if state_topic is not None:
            node.create_subscription(
                Bool, state_topic, self._on_state, PAUSED_STATE_QOS, callback_group=MutuallyExclusiveCallbackGroup()
            )

    @property
    def topic(self) -> str:
        return self._args[1]

    @property
    def subscribed(self) -> bool:
        return self._subscribed

    def _on_state(self, message: Bool) -> None:
        self.set_paused(bool(message.data))

    def set_paused(self, paused: bool) -> None:
        """Return once the reader thread has applied it. Do not hold a lock ``callback`` needs."""
        with self._lock:
            if paused == self.paused:
                return
            self.paused = paused
            self.transitions += 1
            with self._condition:
                generation = self._wanted[1] + 1
                self._wanted = (paused, generation)
            self._await(generation)
        self._node.get_logger().info(
            f"camera {self.topic} {'released: mirror perception paused' if paused else 'subscribed again'}"
        )
        if self._on_change is not None:
            self._on_change(paused)

    def _await(self, generation: int) -> None:
        with self._condition:
            if not self._condition.wait_for(lambda: self._applied >= generation, timeout=_APPLY_TIMEOUT_SEC):
                self._node.get_logger().warning(f"camera reader of {self.topic} did not apply the change in time")

    def _run(self) -> None:
        subscription = None
        try:
            while rclpy.ok(context=self._node.context):
                with self._condition:
                    paused, generation = self._wanted
                if paused and subscription is not None:
                    self._reader.destroy_subscription(subscription)
                    subscription = None
                elif not paused and subscription is None:
                    message_type, topic, callback, qos = self._args
                    subscription = self._reader.create_subscription(message_type, topic, callback, qos)
                self._subscribed = subscription is not None
                with self._condition:
                    self._applied = generation
                    self._condition.notify_all()
                try:
                    self._executor.spin_once(timeout_sec=_SPIN_PERIOD_SEC)
                except (ExternalShutdownException, ShutdownException):
                    return
                except Exception as exc:  # noqa: BLE001 - one bad frame must not end the reader
                    if rclpy.ok(context=self._node.context):
                        self._node.get_logger().error(f"camera callback on {self.topic} failed: {exc!r}")
        finally:
            self._subscribed = False
            try:
                self._executor.shutdown(timeout_sec=0.0)
                self._reader.destroy_node()
            except Exception:  # noqa: BLE001, S110 - the context is already going away
                pass
