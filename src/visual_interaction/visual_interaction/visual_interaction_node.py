"""ROS adapter for the visual interaction session.

Owns the plumbing and nothing else: parameters, the camera subscription, the
frame buffer, the timer, and publishing what the session decided. Every
decision belongs to ``visual_interaction.session``, which is why that package
has no ROS import and all of it is tested without one.

What this node does NOT do, by construction:

* hold a runtime mode, request control authority or publish a joint command —
  it resolves one camera interface and :func:`assert_observation_only` refuses
  anything from a motion family;
* speak. It publishes ``VisualInteractionUtterance`` and its diagnostics. The
  TODO on ``session.hooks.SpeechPort`` records what a coordinator must own
  before the robot says any of this out loud.
"""

from __future__ import annotations

import json
import threading
from collections import deque

import rclpy
from builtin_interfaces.msg import Time
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_srvs.srv import SetBool

from embodied_common.visual_interaction_contract import (
    DEFAULT_ABSENT_RESET_SEC,
    DEFAULT_CONFIRMATION_COUNT,
    DEFAULT_MAX_EVIDENCE_AGE_SEC,
    DEFAULT_MAX_SESSION_SEC,
    DEFAULT_MIN_GAP_SEC,
    NODE_NAME,
    SET_ENABLED_SERVICE,
    STATUS_TOPIC,
    SUPPRESS_MODEL_NOT_READY,
    UTTERANCE_TOPIC,
)
from ibrobot_msgs.msg import ActorState, VisualInteractionUtterance
from visual_interaction.session.core import SessionConfig, VisualInteractionSession
from visual_interaction.session.endpoints import CAMERA_COMPRESSED, resolve_camera
from visual_interaction.session.fake_vl import ScriptedVision
from visual_interaction.session.hooks import ActorSignal, RoundInputs, SessionHooks, Window

# Declared names and their types. Launch may pass no other name and no other
# type: rclpy drops an override it has no declaration for without a warning,
# and refuses one of the wrong type at startup.
PARAMETERS: dict[str, object] = {
    "runtime_description_json": "",
    "camera_interface": CAMERA_COMPRESSED,
    "actor_state_topic": "",
    "vl_implementation": "fake",
    "round_period_sec": 1.0,
    "window_sec": 2.0,
    "frames_per_window": 4,
    "buffer_sec": 6.0,
    "stale_stream_sec": 2.0,
    "confirmation_count": DEFAULT_CONFIRMATION_COUNT,
    "max_evidence_age_sec": DEFAULT_MAX_EVIDENCE_AGE_SEC,
    "min_gap_sec": DEFAULT_MIN_GAP_SEC,
    "absent_reset_sec": DEFAULT_ABSENT_RESET_SEC,
    "max_session_sec": DEFAULT_MAX_SESSION_SEC,
    "categories": [],
    "status_period_sec": 5.0,
}


class FrameBuffer:
    """The recent frames, and an evenly spaced window over them.

    Eviction and the window both use the *camera* stamp, deliberately: mixing
    an arrival clock into eviction while sampling by camera stamp (as the board
    demo does) silently shortens the window whenever the two clocks disagree.
    Liveness is judged separately, by arrival, where no agreement is needed.
    """

    def __init__(self, buffer_sec: float) -> None:
        self._buffer_sec = float(buffer_sec)
        self._items: deque[tuple[float, float, bytes]] = deque()
        self._lock = threading.Lock()
        self.total = 0

    def put(self, stamp: float, received_at: float, data: bytes) -> None:
        with self._lock:
            self._items.append((stamp, received_at, data))
            self.total += 1
            while self._items and self._items[0][0] < stamp - self._buffer_sec:
                self._items.popleft()

    def window(self, window_sec: float, count: int) -> Window | None:
        """An evenly spaced window of ``count`` frames, newest last.

        Returns None before anything has arrived. A thin stream repeats frames
        rather than failing: the model's input shape is fixed, and a repeat is
        visible in the stamps.
        """
        with self._lock:
            items = list(self._items)
        if not items:
            return None
        newest = items[-1][0]
        recent = [item for item in items if item[0] >= newest - window_sec] or items[-1:]
        if count <= 1:
            chosen = [recent[-1]]
        else:
            step = (len(recent) - 1) / (count - 1)
            chosen = [recent[min(len(recent) - 1, round(index * step))] for index in range(count)]
        return Window(
            frames=tuple(item[2] for item in chosen),
            stamps=tuple(item[0] for item in chosen),
            received_at=items[-1][1],
        )


class VisualInteractionNode(Node):
    """One resident node: subscribe, decide once per period, publish."""

    def __init__(self, *, hooks: SessionHooks | None = None, parameter_overrides=None) -> None:
        # parameter_overrides lets a test construct the node without a launch
        # file, the way visual_game_announcer_node does.
        super().__init__(NODE_NAME, parameter_overrides=parameter_overrides)
        for name, default in PARAMETERS.items():
            self.declare_parameter(name, default)

        # Config-only checks first, so a typo is reported as itself rather than
        # as whatever the runtime description happens to be missing.
        session_hooks = hooks or self._build_hooks()

        camera = resolve_camera(self._description(), str(self._value("camera_interface")))
        self._window_sec = float(self._value("window_sec"))
        self._frames = max(1, int(self._value("frames_per_window")))
        self._stale_stream_sec = float(self._value("stale_stream_sec"))
        self._buffer = FrameBuffer(float(self._value("buffer_sec")))

        self._session = VisualInteractionSession(
            hooks=session_hooks,
            config=SessionConfig(
                confirmation_count=int(self._value("confirmation_count")),
                max_evidence_age_sec=float(self._value("max_evidence_age_sec")),
                min_gap_sec=float(self._value("min_gap_sec")),
                absent_reset_sec=float(self._value("absent_reset_sec")),
                max_session_sec=float(self._value("max_session_sec")),
                **self._categories(),
            ),
        )

        self._actor = ActorSignal(present=False, known=False, bbox=None, stamp=0.0)
        self._counts: dict[str, int] = {}
        self._rounds = 0

        self._utterances = self.create_publisher(VisualInteractionUtterance, UTTERANCE_TOPIC, 10)
        self._status = self.create_publisher(String, STATUS_TOPIC, 10)
        self.create_subscription(
            CompressedImage,
            camera.endpoint,
            self._on_frame,
            QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5),
        )
        actor_topic = str(self._value("actor_state_topic"))
        if actor_topic:
            self.create_subscription(ActorState, actor_topic, self._on_actor, 10)

        # How an attendant stops the robot talking. A visitor cannot reach a
        # ROS service, so the console that calls this is part of the feature,
        # not an optional extra: without it there is no way to ask for quiet.
        self.create_service(SetBool, f"~/{SET_ENABLED_SERVICE}", self._on_set_enabled)

        self.create_timer(float(self._value("round_period_sec")), self._on_round)
        self.create_timer(float(self._value("status_period_sec")), self._publish_status)
        self.get_logger().info(
            f"visual interaction on {camera.endpoint}; actor_state={actor_topic or '(none)'}; "
            f"vl={self._value('vl_implementation')}; publishing {UTTERANCE_TOPIC} (no speech)"
        )

    # -- parameters -------------------------------------------------------

    def _value(self, name: str) -> object:
        return self.get_parameter(name).value

    def _description(self) -> dict:
        raw = str(self._value("runtime_description_json"))
        try:
            description = json.loads(raw)
        except ValueError as error:
            raise ValueError("runtime_description_json must be the bound runtime interface description") from error
        return description

    def _categories(self) -> dict:
        values = self._value("categories") or []
        return {"categories": tuple(str(value) for value in values)} if values else {}

    def _build_hooks(self) -> SessionHooks:
        implementation = str(self._value("vl_implementation"))
        if implementation != "fake":
            # Only the scripted port exists so far. Failing here keeps a
            # deployment from believing it configured a model that is not
            # wired yet.
            raise ValueError(f"vl_implementation {implementation!r} is not available; only 'fake' is wired")
        return SessionHooks(vision=ScriptedVision.of())

    def _on_set_enabled(self, request, response):
        """``data`` true resumes, false mutes. Only this clears a mute."""
        if request.data:
            self._session.resume()
        else:
            self._session.mute()
        response.success = True
        response.message = "muted" if self._session.muted else "active"
        self.get_logger().info(f"visual interaction {response.message} by request")
        return response

    # -- callbacks --------------------------------------------------------

    def _on_frame(self, message: CompressedImage) -> None:
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        self._buffer.put(stamp, self._now(), bytes(message.data))

    def _on_actor(self, message: ActorState) -> None:
        bbox = tuple(float(value) for value in message.bbox) if message.actor_present else None
        known = message.reason not in ("no_detections", "stale_detections")
        self._actor = ActorSignal(
            present=bool(message.actor_present),
            known=known,
            bbox=bbox,
            stamp=message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
        )

    def _on_round(self) -> None:
        window = self._buffer.window(self._window_sec, self._frames)
        if window is None:
            return
        now = self._now()
        if now - window.received_at > self._stale_stream_sec:
            # Judged by arrival: a dropped stream is a host-side fact and needs
            # no clock agreement with the robot.
            self._count(SUPPRESS_MODEL_NOT_READY)
            return

        result = self._session.step(RoundInputs(window=window, actor=self._actor, now=now))
        self._rounds += 1
        self._count(result.reason or "spoken")
        self._publish(result, window)

    # -- publishing -------------------------------------------------------

    def _publish(self, result, window: Window) -> None:
        message = VisualInteractionUtterance()
        message.header.stamp = self.get_clock().now().to_msg()
        message.session_id = result.session_id
        message.actor_epoch = max(0, int(result.actor_epoch))
        message.event_id = result.event_id
        message.text = result.text
        message.kind = result.observation.kind if result.observation else ""
        message.category = result.observation.category if result.observation else ""
        message.region = result.observation.region if result.observation else ""
        message.suppress_reason = result.reason
        message.evidence_start_stamp = _as_time(window.oldest_stamp)
        self._utterances.publish(message)

    def _publish_status(self) -> None:
        payload = {
            "node": NODE_NAME,
            "rounds": self._rounds,
            "frames": self._buffer.total,
            "session_id": self._session.session_id,
            "actor_epoch": self._session.actor_epoch,
            "muted": self._session.muted,
            "outcomes": dict(sorted(self._counts.items())),
            "spoken_topics": self._session.ledger().spoken_count(),
        }
        self._status.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def _count(self, key: str) -> None:
        self._counts[key] = self._counts.get(key, 0) + 1

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


def _as_time(seconds: float) -> Time:
    stamp = Time()
    stamp.sec = int(seconds)
    stamp.nanosec = int(round((seconds - int(seconds)) * 1e9))
    return stamp


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = VisualInteractionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
