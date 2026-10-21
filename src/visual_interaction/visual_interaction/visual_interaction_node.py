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
from collections.abc import Mapping

import rclpy
from builtin_interfaces.msg import Time
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
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
from visual_interaction.session.endpoints import CAMERA_COMPRESSED, Endpoint, resolve_camera
from visual_interaction.session.fake_vl import ScriptedVision
from visual_interaction.session.hooks import ActorSignal, RoundInputs, SessionHooks, Window

# Declared names and their types. Launch may pass no other name and no other
# type: rclpy drops an override it has no declaration for without a warning,
# and refuses one of the wrong type at startup.
# Frames kept at most, whatever the camera stamps say. 300 is ten seconds of
# a 30 Hz stream, comfortably more than any configured window.
DEFAULT_MAX_BUFFERED_FRAMES = 300

PARAMETERS: dict[str, object] = {
    "runtime_description_json": "",
    "camera_interface": CAMERA_COMPRESSED,
    # Escape hatch for a stream the runtime description does not carry, such
    # as the robot's half-resolution relay. Set it and camera_interface is
    # ignored; the QoS then has to be stated here too, since there is no
    # description to read it from.
    "camera_topic": "",
    "camera_topic_qos_json": '{"reliability": "best_effort", "durability": "volatile", "depth": 5}',
    "actor_state_topic": "",
    "vl_implementation": "fake",
    "round_period_sec": 1.0,
    "window_sec": 2.0,
    "frames_per_window": 4,
    "buffer_sec": 6.0,
    "stale_stream_sec": 2.0,
    "stale_actor_sec": 2.0,
    "max_buffered_frames": 300,
    "confirmation_count": DEFAULT_CONFIRMATION_COUNT,
    "max_evidence_age_sec": DEFAULT_MAX_EVIDENCE_AGE_SEC,
    "min_gap_sec": DEFAULT_MIN_GAP_SEC,
    "absent_reset_sec": DEFAULT_ABSENT_RESET_SEC,
    "max_session_sec": DEFAULT_MAX_SESSION_SEC,
    # A JSON array, not a ROS string array. An empty list gives rclpy no
    # element type to infer, so it declares BYTE_ARRAY and then refuses the
    # strings the launch builder passes — the node died at startup with
    # "expecting type 'BYTE_ARRAY'", which no unit test saw because they all
    # pass a non-empty override. Every other package here carries a list the
    # same way (``arm_joint_names_json``, ``allowed_skills_json``), and
    # ``runtime_description_json`` above is already JSON. Empty means the
    # contract's whole whitelist.
    "categories_json": "[]",
    "status_period_sec": 5.0,
}


def camera_qos(declared: Mapping[str, object]) -> QoSProfile:
    """Build the subscription QoS from what the interface description declares.

    Load-bearing, and it was wrong. The node hardcoded BEST_EFFORT while the X2
    profile declares this camera as reliable/transient_local — and that is what
    the robot really publishes. Measured on the board: subscribing
    BEST_EFFORT/VOLATILE received 0 frames in 7 seconds, while
    RELIABLE/TRANSIENT_LOCAL on the same topic received 10 (1.4 Hz, ~288 KB
    JPEG). An incompatible subscription is silent — no error, no warning, just
    a feature that never sees a frame — so the declared values are honoured
    rather than assumed.

    Unknown values fail closed: a typo in a profile must not quietly downgrade
    the subscription into one that receives nothing.
    """
    reliability = {
        "reliable": ReliabilityPolicy.RELIABLE,
        "best_effort": ReliabilityPolicy.BEST_EFFORT,
    }
    durability = {
        "volatile": DurabilityPolicy.VOLATILE,
        "transient_local": DurabilityPolicy.TRANSIENT_LOCAL,
    }
    history = {"keep_last": HistoryPolicy.KEEP_LAST, "keep_all": HistoryPolicy.KEEP_ALL}

    def pick(table: dict, key: str, default: str):
        raw = str(declared.get(key, default)).strip().lower()
        if raw not in table:
            raise ValueError(f"camera qos.{key} {raw!r} is not one of {sorted(table)}")
        return table[raw]

    depth = int(declared.get("depth", 5))
    if depth <= 0:
        raise ValueError("camera qos.depth must be positive")
    return QoSProfile(
        reliability=pick(reliability, "reliability", "reliable"),
        durability=pick(durability, "durability", "volatile"),
        history=pick(history, "history", "keep_last"),
        depth=depth,
    )


class FrameBuffer:
    """The recent frames, and an evenly spaced window over them.

    Eviction and the window both use the *camera* stamp, deliberately: mixing
    an arrival clock into eviction while sampling by camera stamp (as the board
    demo does) silently shortens the window whenever the two clocks disagree.
    Liveness is judged separately, by arrival, where no agreement is needed.
    """

    def __init__(self, buffer_sec: float, max_frames: int = DEFAULT_MAX_BUFFERED_FRAMES) -> None:
        self._buffer_sec = float(buffer_sec)
        if max_frames < 1:
            raise ValueError("max_frames must be at least 1")
        # A hard cap as well as the time window. Eviction by camera stamp alone
        # evicts nothing when the stamp stops advancing — a frozen or
        # always-zero clock then grows this without bound in a resident
        # process. The cap is what makes the memory bounded by the stream rate
        # rather than by the robot's clock behaving.
        self._items: deque[tuple[float, float, bytes]] = deque(maxlen=max_frames)
        self._lock = threading.Lock()
        self.total = 0

    def put(self, stamp: float, received_at: float, data: bytes) -> None:
        with self._lock:
            # deque(maxlen=...) drops the oldest once full, so the cap needs no
            # explicit check here.
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

        camera = self._camera()
        self._window_sec = float(self._value("window_sec"))
        self._frames = max(1, int(self._value("frames_per_window")))
        self._stale_stream_sec = float(self._value("stale_stream_sec"))
        self._stale_actor_sec = float(self._value("stale_actor_sec"))
        self._buffer = FrameBuffer(float(self._value("buffer_sec")), int(self._value("max_buffered_frames")))

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
        self._actor_received_at: float | None = None
        self._counts: dict[str, int] = {}
        self._rounds = 0

        self._utterances = self.create_publisher(VisualInteractionUtterance, UTTERANCE_TOPIC, 10)
        self._status = self.create_publisher(String, STATUS_TOPIC, 10)
        self.create_subscription(
            CompressedImage,
            camera.endpoint,
            self._on_frame,
            camera_qos(camera.qos),
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

    def _camera(self) -> Endpoint:
        """Where the frames come from: a described interface, or a named topic.

        ``camera_interface`` is the normal form and resolves by ID, so the
        endpoint and its QoS come from the bound runtime description. A named
        ``camera_topic`` is the escape hatch for a stream the description does
        not carry — the robot's half-resolution relay republishes a resized
        JPEG outside the HAL interface contract, which is how
        motion_imitation_stream reads it too. There is then no description to
        take QoS from, so it is stated alongside the topic.
        """
        topic = str(self._value("camera_topic")).strip()
        if not topic:
            return resolve_camera(self._description(), str(self._value("camera_interface")))
        if not topic.startswith("/"):
            raise ValueError("camera_topic must be an absolute ROS name")
        raw = str(self._value("camera_topic_qos_json"))
        try:
            declared = json.loads(raw)
        except ValueError as error:
            raise ValueError("camera_topic_qos_json must be a JSON object of QoS fields") from error
        if not isinstance(declared, dict):
            raise ValueError("camera_topic_qos_json must be a JSON object of QoS fields")
        return Endpoint(endpoint=topic, qos=declared)

    def _categories(self) -> dict:
        raw = str(self._value("categories_json"))
        try:
            values = json.loads(raw)
        except ValueError as error:
            raise ValueError("categories_json must be a JSON array of category names") from error
        if not isinstance(values, list):
            raise ValueError("categories_json must be a JSON array of category names")
        # Empty means the contract's whole whitelist, so the key is omitted and
        # SessionConfig keeps its default rather than being handed ().
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
            # ActorState describes one actor, so a second box is not available
            # on this wire. The ambiguity check therefore cannot fire from this
            # input, and the node says so rather than implying it is covered.
            others=(),
        )
        self._actor_received_at = self._now()

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

        # The actor state has its own liveness. Holding the last message
        # forever would keep answering a visitor who left, because the tracker
        # only ever sees the same stamp and so never times the gap out. Its age
        # is measured by arrival, for the same reason the stream's is.
        actor = self._actor
        if self._actor_received_at is None or now - self._actor_received_at > self._stale_actor_sec:
            actor = ActorSignal(present=False, known=False, bbox=None, stamp=now)

        result = self._session.step(RoundInputs(window=window, actor=actor, now=now))
        self._rounds += 1
        self._count(result.reason or "spoken")
        self._publish(result, window)

    # -- publishing -------------------------------------------------------

    def _publish(self, result, window: Window) -> None:
        message = VisualInteractionUtterance()
        # The evidence clock, as the message says: the observation time of the
        # newest frame the decision rests on. Publishing the current time here
        # would present an old picture as a fresh one to every consumer.
        message.header.stamp = _as_time(window.newest_stamp)
        message.evidence_start_stamp = _as_time(window.oldest_stamp)
        message.session_id = result.session_id
        message.actor_epoch = max(0, int(result.actor_epoch))
        message.event_id = result.event_id
        message.episode_seq = max(0, int(result.episode_seq))
        message.text = result.text
        message.kind = result.observation.kind if result.observation else ""
        message.category = result.observation.category if result.observation else ""
        message.region = result.observation.region if result.observation else ""
        message.suppress_reason = result.reason
        # Zero when there is nothing to say, which the message defines as "no
        # deadline" rather than "start immediately".
        message.latest_start_at = _as_time(result.latest_start_at) if result.spoken else _as_time(0.0)
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
