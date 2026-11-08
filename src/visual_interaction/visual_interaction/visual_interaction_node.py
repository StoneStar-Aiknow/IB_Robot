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
import os
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

from embodied_common.mirror_session_contract import STATE_MIRRORING, STATE_STALE_SEC, STATE_STANDBY, STATE_WAIT
from embodied_common.visual_interaction_contract import (
    DEFAULT_ABSENT_RESET_SEC,
    DEFAULT_CONFIRMATION_COUNT,
    DEFAULT_MAX_EVIDENCE_AGE_SEC,
    DEFAULT_MAX_SESSION_SEC,
    DEFAULT_MIN_GAP_SEC,
    KIND_CHATTER,
    NODE_NAME,
    SET_ENABLED_SERVICE,
    STATUS_TOPIC,
    SUPPRESS_MIRROR_INACTIVE,
    SUPPRESS_MODEL_NOT_READY,
    UTTERANCE_TOPIC,
)
from ibrobot_msgs.msg import ActorState, VisualInteractionUtterance
from visual_interaction.session.chatter import Chatter, ChatterConfig
from visual_interaction.session.core import SessionConfig, VisualInteractionSession
from visual_interaction.session.endpoints import CAMERA_COMPRESSED, Endpoint, resolve_camera
from visual_interaction.session.fake_vl import ScriptedVision
from visual_interaction.session.hooks import ActorSignal, RoundInputs, SessionHooks, Window
from visual_interaction.session.look_say import (
    DEFAULT_SLOW_LOOK_SEC,
    Q1,
    Q2,
    QUESTIONS,
    LookConfig,
    LookSay,
    classify,
    parse_yes,
    refine,
)

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
    # The mirror session's state topic (/mirror_session/state). When set, the
    # rounds are suspended with MIRROR_INACTIVE unless it currently reports
    # MIRRORING: in the magicmirror entry the feature is part of the mirror
    # experience and stays parked while the mirror is not following anyone.
    # Empty means no such gate (a deployment without a mirror).
    "mirror_state_topic": "",
    # How old the last state message may be, by arrival, before the state
    # counts as unknown — and unknown gates the round off, the conservative
    # side. The publisher heartbeats once a second.
    "mirror_state_stale_sec": STATE_STALE_SEC,
    "vl_implementation": "fake",
    # Where the board's InternVL deployment lives, for
    # vl_implementation "internvl". Empty with "fake".
    "vl_demo_root": "",
    "vl_max_new_tokens": 110,
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
    # "observe" is the closed-schema session; "chatter" says one host-owned
    # line every chatter_gap_min..max seconds while mirroring, keyed by what
    # the model names (visual_interaction.session.chatter).
    "mode": "observe",
    "chatter_gap_min_sec": 5.0,
    "chatter_gap_max_sec": 10.0,
    "chatter_fresh_sec": 3.0,
    "chatter_action_cooldown_sec": 15.0,
    "chatter_object_cooldown_sec": 60.0,
    "chatter_latest_start_sec": 4.0,
    # The mirror voice's ~/status: while it speaks or has a queue, the next
    # line waits. Empty or stale falls back to a length estimate.
    "voice_status_topic": "",
    "voice_status_stale_sec": 3.0,
    # internvl_voice's look (visual_interaction.session.look_say), as the
    # contract's look_json: its settings and the look pools' clips. Empty, the
    # default, is no look and the node is what it was; set, it replaces mode
    # and round_period_sec with the look's own loop.
    "look_json": "",
}

# How often the look's loop runs. A look blocks the executor for its ~1-2 s;
# between looks this short yield is what lets the subscriptions in.
LOOK_TICK_SEC = 0.05
# A failed look is logged at most this often; the status topic counts them all.
LOOK_ERROR_LOG_SEC = 30.0


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

        self._mode = str(self._value("mode"))
        if self._mode not in ("observe", "chatter"):
            raise ValueError(f"mode {self._mode!r} is not one of 'observe', 'chatter'")
        self._look_settings: dict | None = None
        # Why the look's vision could not even be built (no deployment at
        # demo_root): the look is then off, the node itself still runs.
        self._look_build_error: str | None = None
        if str(self._value("look_json")).strip():
            self._look_settings = self._parse_look()
            self._mode = "look"
        # Config-only checks first, so a typo is reported as itself rather than
        # as whatever the runtime description happens to be missing.
        session_hooks = hooks or self._build_hooks()

        camera = self._camera()
        self._window_sec = float(self._value("window_sec"))
        self._frames = max(1, int(self._value("frames_per_window")))
        self._stale_stream_sec = float(self._value("stale_stream_sec"))
        self._stale_actor_sec = float(self._value("stale_actor_sec"))
        self._buffer = FrameBuffer(float(self._value("buffer_sec")), int(self._value("max_buffered_frames")))

        # The mirror gate, when the deployment has one: the last state reported
        # on mirror_state_topic and when it arrived. Until the first message
        # lands the gate stays closed — TRANSIENT_LOCAL delivers the latched
        # current state at startup, so a live deployment opens it immediately
        # and a dead one never pretends to.
        self._mirror_topic = str(self._value("mirror_state_topic")).strip() or None
        self._mirror_state_stale_sec = float(self._value("mirror_state_stale_sec"))

        max_session_sec = float(self._value("max_session_sec"))
        if self._mirror_topic is not None:
            # A deployment with a mirror has one visit, not two: the mirror
            # session's start opens the interaction's session and its end closes
            # it (see _on_mirror_state). The duration limit is the guard for a
            # stand where nobody announces hand-overs, and applying it here
            # would expire a visit the mirror is still serving — the interaction
            # then goes quiet for the rest of that visit, exactly as observed on
            # the X2 board before this was tied together.
            max_session_sec = 0.0

        self._session = VisualInteractionSession(
            hooks=session_hooks,
            config=SessionConfig(
                confirmation_count=int(self._value("confirmation_count")),
                max_evidence_age_sec=float(self._value("max_evidence_age_sec")),
                min_gap_sec=float(self._value("min_gap_sec")),
                absent_reset_sec=float(self._value("absent_reset_sec")),
                max_session_sec=max_session_sec,
                **self._categories(),
            ),
        )

        self._actor = ActorSignal(present=False, known=False, bbox=None, stamp=0.0)
        self._actor_received_at: float | None = None
        self._mirror_state = ""
        self._mirror_state_at: float | None = None
        self._counts: dict[str, int] = {}
        self._rounds = 0
        # The last MODEL_NOT_READY cause logged, so a permanent
        # fault warns once instead of every round.
        self._last_model_error: str | None = None

        self._chatter: Chatter | None = None
        self._look: LookSay | None = None
        self._voice_busy_now = False
        self._voice_synthesizing_now = False
        self._voice_status_at: float | None = None
        self._last_phrase: str | None = None
        if self._mode == "chatter":
            self._chatter = Chatter(
                ChatterConfig(
                    gap_min_sec=float(self._value("chatter_gap_min_sec")),
                    gap_max_sec=float(self._value("chatter_gap_max_sec")),
                    fresh_sec=float(self._value("chatter_fresh_sec")),
                    action_cooldown_sec=float(self._value("chatter_action_cooldown_sec")),
                    object_cooldown_sec=float(self._value("chatter_object_cooldown_sec")),
                )
            )
        if self._look_settings is not None:
            self._init_look(self._look_settings)
        # A look blocks this single-threaded node for a second or more, and the
        # executor takes one message per subscription per wait. With a queue of
        # 10 the look read voice status seconds old as fresh, and believed a
        # clip that had ended was still playing (two 12 s silences on the
        # board). Depth 1 keeps only the newest; chatter keeps its old queue.
        latest_only = 1 if self._mode == "look" else 10
        if self._mode in ("chatter", "look"):
            voice_topic = str(self._value("voice_status_topic")).strip()
            if voice_topic:
                self.create_subscription(String, voice_topic, self._on_voice_status, latest_only)

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
            self.create_subscription(ActorState, actor_topic, self._on_actor, latest_only)
        if self._mirror_topic is not None:
            # Reliable + transient_local, depth 1, matching the publisher's
            # contract (embodied_common.mirror_session_contract): the latched
            # current state arrives without waiting for the next heartbeat.
            self.create_subscription(
                String,
                self._mirror_topic,
                self._on_mirror_state,
                QoSProfile(
                    depth=1,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                ),
            )

        # How an attendant stops the robot talking. A visitor cannot reach a
        # ROS service, so the console that calls this is part of the feature,
        # not an optional extra: without it there is no way to ask for quiet.
        self.create_service(SetBool, f"~/{SET_ENABLED_SERVICE}", self._on_set_enabled)

        if self._look is not None:
            self.create_timer(LOOK_TICK_SEC, self._look_round)
        else:
            self.create_timer(float(self._value("round_period_sec")), self._on_round)
        self.create_timer(float(self._value("status_period_sec")), self._publish_status)
        if self._mode in ("chatter", "look"):
            # Pay the ~20 s model load while the mirror still waits, not in
            # the first mirroring round, where it ate the visitor's turn.
            self._preload_timer = self.create_timer(1.0, self._preload)
        self.get_logger().info(
            f"visual interaction on {camera.endpoint}; actor_state={actor_topic or '(none)'}; "
            f"mirror_state={self._mirror_topic or '(none)'}; "
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
        if implementation == "fake":
            if self._mode == "look":
                raise ValueError("look_json needs vl_implementation 'internvl'; the scripted vision cannot look")
            return SessionHooks(vision=ScriptedVision.of())
        if implementation == "internvl":
            # Imported here, not at module scope: it loads the board's InternVL
            # deployment and a workstation has neither that directory nor an
            # NPU, so importing it eagerly would break every other entry point.
            from visual_interaction.session.internvl_vision import InternVlVision

            demo_root = str(self._value("vl_demo_root")).strip()
            if not demo_root:
                raise ValueError("vl_implementation 'internvl' needs vl.demo_root")
            if self._mode == "look":
                # One picture per look, and chatter's prompt for the base one
                # load() builds; the look's own questions are registered on top.
                try:
                    vision = InternVlVision(
                        demo_root=demo_root,
                        frames=1,
                        max_new_tokens=int(self._value("vl_max_new_tokens")),
                        prompt_mode="chatter",
                    )
                except ValueError as error:
                    self._look_build_error = str(error)
                    return SessionHooks(vision=ScriptedVision.of())
                return SessionHooks(vision=vision)
            return SessionHooks(
                vision=InternVlVision(
                    demo_root=demo_root,
                    frames=max(1, int(self._value("frames_per_window"))),
                    max_new_tokens=int(self._value("vl_max_new_tokens")),
                    prompt_mode="chatter" if self._mode == "chatter" else "observe",
                )
            )
        raise ValueError(f"vl_implementation {implementation!r} is not available; expected 'fake' or 'internvl'")

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

    def _on_mirror_state(self, message: String) -> None:
        """Record the mirror session's current state, keyed by arrival.

        The payload is the state topic's JSON (a ``state`` field plus the rest
        of the snapshot); only the state name is read here. A malformed payload
        leaves the previous state in place, and staleness — judged by arrival —
        then closes the gate on the next round, which needs no clock agreement
        with the publisher.

        With a mirror configured, this is also where the interaction's session
        begins: WAIT → STANDBY is the mirror taking over a mode (the set_session
        that opens a visit), and STANDBY → MIRRORING stays inside the same
        visit, so only the first edge starts a new range. One visitor therefore
        gets exactly the lifetime the mirror gives them, and the covered topics
        are forgotten when the mirror starts the next visit rather than when a
        timer inside this node says so.
        """
        try:
            payload = json.loads(message.data)
        except ValueError:
            return
        state = payload.get("state") if isinstance(payload, dict) else None
        previous = self._mirror_state
        self._mirror_state = state if isinstance(state, str) else ""
        self._mirror_state_at = self._now()
        if self._mirror_topic is None:
            return
        started = self._mirror_state in (STATE_STANDBY, STATE_MIRRORING) and previous in ("", STATE_WAIT)
        if started:
            self._session.start_session(now=self._mirror_state_at, reason="mirror_session")

    def _mirror_active(self, now: float) -> bool:
        """Whether the mirror gate is open: MIRRORING, and recently heard.

        Every other state — WAIT, STANDBY, unknown, silent — closes it, as
        does a subscription that was never configured (no gate) or one whose
        publisher has gone quiet past mirror_state_stale_sec.
        """
        if self._mirror_topic is None:
            return True
        return (
            self._mirror_state == STATE_MIRRORING
            and self._mirror_state_at is not None
            and now - self._mirror_state_at <= self._mirror_state_stale_sec
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

        # The actor state has its own liveness. Holding the last message
        # forever would keep answering a visitor who left, because the tracker
        # only ever sees the same stamp and so never times the gap out. Its age
        # is measured by arrival, for the same reason the stream's is.
        actor = self._actor
        if self._actor_received_at is None or now - self._actor_received_at > self._stale_actor_sec:
            actor = ActorSignal(present=False, known=False, bbox=None, stamp=now)

        if self._chatter is not None:
            self._chatter_round(window, actor, now)
            return

        result = self._session.step(
            RoundInputs(window=window, actor=actor, now=now, mirror_active=self._mirror_active(now))
        )
        self._rounds += 1
        self._count(result.reason or "spoken")
        self._publish(result, window)

    # -- chatter ----------------------------------------------------------

    def _preload(self) -> None:
        self._preload_timer.cancel()
        if self._look is not None:
            self._look_preload()
            return
        load = getattr(self._session.hooks.vision, "load", None)
        if load is None:
            return
        try:
            load()
        except Exception as error:  # noqa: BLE001 - the first round retries and reports it
            self.get_logger().warning(f"model preload failed: {error}")
            return
        self.get_logger().info("model preloaded")

    def _on_voice_status(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except ValueError:
            return
        if not isinstance(payload, dict):
            return
        self._voice_busy_now = (
            bool(payload.get("speaking")) or bool(payload.get("in_flight")) or (int(payload.get("queue") or 0) > 0)
        )
        self._voice_synthesizing_now = payload.get("in_flight") == "synthesis"
        self._voice_status_at = self._now()

    def _voice_heard(self, now: float) -> bool:
        return self._voice_status_at is not None and now - self._voice_status_at <= float(
            self._value("voice_status_stale_sec")
        )

    def _voice_busy(self, now: float) -> bool:
        return self._voice_heard(now) and self._voice_busy_now

    def _voice_synthesizing(self, now: float) -> bool:
        return self._voice_heard(now) and self._voice_synthesizing_now

    def _chatter_round(self, window: Window, actor: ActorSignal, now: float) -> None:
        """One chatter round: look, remember what was named, speak when due."""
        chatter = self._chatter
        chatter.voice(now, self._voice_busy(now))
        if not self._mirror_active(now):
            chatter.due(now, False)
            self._count(SUPPRESS_MIRROR_INACTIVE)
            return
        # Looking while the voice is being synthesized put both models on the
        # 310P's NPU at once, and the pose model in the mirror's path fell
        # behind: on the board (10-23) the arm's stalls and jumps gathered in
        # the seconds where the two overlapped, and were rare with either
        # alone. The pose model never yields; this round does, and the next
        # one looks again. A synthesis that starts during a round still
        # overlaps it, as the status arrives about once a second.
        if self._voice_synthesizing(now):
            self._count("skip_voice_synthesis")
            return

        phrase = self._session.hooks.vision.observe(
            RoundInputs(window=window, actor=actor, now=now, mirror_active=True)
        )
        self._rounds += 1
        if phrase is None:
            self._count(SUPPRESS_MODEL_NOT_READY)
            detail = getattr(self._session.hooks.vision, "last_error", "")
            if detail != self._last_model_error:
                self._last_model_error = detail
                self.get_logger().warning(f"quiet: MODEL_NOT_READY{f' ({detail})' if detail else ''}")
        else:
            self._last_model_error = None
            text = phrase if isinstance(phrase, str) else ""
            found = chatter.observe(now, text)
            if text != self._last_phrase:
                self._last_phrase = text
                self.get_logger().info(f'seen: "{text}" -> {"/".join(found) if found else "-"}')

        after = self._now()
        chatter.voice(after, self._voice_busy(after))
        # The gate was open when this round began. Asking again after the
        # ~1.2 s inference closed it whenever the 1 Hz state heartbeat had
        # not been taken yet (2 s stale), and each close re-armed the first
        # line, so on the board no line was ever due. A mirror that ended
        # meanwhile closes the next round; the voice refuses a late line.
        if not chatter.due(after, True):
            self._count("chatter_wait")
            return
        line = chatter.pick(after)
        self._count(f"said_{line.kind}")

        message = VisualInteractionUtterance()
        message.header.stamp = _as_time(window.newest_stamp)
        message.evidence_start_stamp = _as_time(window.oldest_stamp)
        message.session_id = "chatter"
        message.event_id = f"chatter-{chatter.spoken}"
        message.episode_seq = chatter.spoken
        message.text = line.text
        message.kind = line.kind
        message.category = line.subject
        message.suppress_reason = ""
        message.latest_start_at = _as_time(after + float(self._value("chatter_latest_start_sec")))
        self._utterances.publish(message)
        self.get_logger().info(
            f'say: "{line.text}"  [{line.kind}/{line.subject}] next in {chatter.next_due - after:.1f}s'
        )

    # -- look (internvl_voice) -------------------------------------------

    def _parse_look(self) -> dict:
        from embodied_common.internvl_voice_contract import parse_look_json

        settings, warnings = parse_look_json(str(self._value("look_json")))
        for warning in warnings:
            self.get_logger().warning(f"look: {warning}")
        return settings

    def _init_look(self, settings: dict) -> None:
        """The look's state: its decisions, its loop's bookkeeping and the model's questions."""
        self._look = LookSay(LookConfig.from_look(settings), settings["pools"])
        self._look_period = float(settings["round_period_sec"])
        # Looks back to back have no period to overrun; the plan's 2.5 s is the limit then.
        self._look_slow_sec = max(self._look_period, DEFAULT_SLOW_LOOK_SEC)
        self._look_crop = bool(settings["crop_actor"])
        self._look_latest_start_sec = float(settings["latest_start_sec"])
        self._look_dump_dir = str(settings["dump_dir"]).strip()
        self._look_ready = False
        # Set when the look is off for good (model or Q1 refused): no line at all.
        self._look_dead: str | None = None
        self._look_last_end = float("-inf")
        self._look_ticks = 0
        self._look_frame_stamp: float | None = None
        self._look_error_logged_at = float("-inf")
        self._look_mirroring = False
        # What one tick did, for dump_dir; reset at the start of every tick.
        self._look_record: dict = {}
        if self._look_build_error is not None:
            self._look_off(f"no model: {self._look_build_error}")
            return
        vision = self._session.hooks.vision
        missing = [
            name for name in ("register_questions", "prepare", "ask") if not callable(getattr(vision, name, None))
        ]
        if missing:
            self._look_off(f"the vision has no {', '.join(missing)}")
            return
        vision.register_questions(QUESTIONS)
        if not any(settings["pools"].values()):
            self.get_logger().warning("look: look_json carries no pools; the look has nothing to say")
        self.get_logger().info(
            f"look: period {self._look_period:g} s, window {self._look.config.window_sec:g} s, "
            f"verify {self._look.config.verify}, refine {self._look.config.refine_hand}, "
            f"crop {self._look_crop}, {sum(len(v) for v in settings['pools'].values())} clips"
        )

    def _look_off(self, why: str) -> None:
        self._look_dead = why
        self._look_ready = False
        self.get_logger().warning(f"look off: {why}")

    def _look_preload(self) -> None:
        if self._look_dead is not None:
            return
        vision = self._session.hooks.vision
        try:
            vision.load()
        except Exception as error:  # noqa: BLE001 - plan §4.6: a model that cannot load turns the look off
            self._look_off(f"model load failed: {error}")
            return
        failures = dict(getattr(vision, "question_failures", {}) or {})
        if Q1 in failures:
            self._look_off(f"{Q1} refused: {failures[Q1]}")
            return
        for key, why in sorted(failures.items()):
            self.get_logger().warning(f"look: question {key} refused, skipped: {why}")
        self._look_ready = True
        self.get_logger().info("look: model preloaded")

    def _on_look_mirror(self, now: float) -> None:
        """Follow MIRRORING by its value, not its freshness.

        A look blocks the executor for a second or two, so after one the last
        heartbeat is often older than mirror_state_stale_sec; resetting the
        round on that would start it over after every look. Staleness only
        holds the look and its lines (in _look_round).
        """
        mirroring = self._mirror_topic is None or self._mirror_state == STATE_MIRRORING
        if mirroring and not self._look_mirroring:
            self._look.enter(now)
            self._look_last_end = float("-inf")
            self.get_logger().info("look: mirroring; a new round")
        elif not mirroring and self._look_mirroring:
            self._look.leave()
        self._look_mirroring = mirroring

    def _look_round(self) -> None:
        """One tick of the look's loop: look if it is time, say a line if one is due.

        Single-threaded, never re-entered: the look runs inside this callback,
        and the next tick only comes once it has returned.
        """
        look = self._look
        now = self._now()
        self._on_look_mirror(now)
        # ~/set_enabled false mutes the look as well: no looking, no lines.
        if self._look_dead is not None or not look.active or self._session.muted:
            return
        if not self._mirror_active(now):
            self._count(SUPPRESS_MIRROR_INACTIVE)
            return
        look.voice(now, self._voice_busy_now, self._voice_heard(now), self._voice_status_at)
        self._look_record = {}
        if self._look_ready and look.wants_look(now) and now - self._look_last_end >= self._look_period:
            self._look_once(now)
            now = self._now()
            look.voice(now, self._voice_busy_now, self._voice_heard(now), self._voice_status_at)
        pick = look.decide(now, self._look_verify if self._look_ready else None)
        if pick is not None:
            self._look_say(pick, now)
        if self._look_record:
            self._look_record.setdefault("t", round(now, 3))
            self._look_dump()

    def _look_once(self, now: float) -> None:
        """Look at the newest frame: Q1, and Q2 when it named a gesture it could not name."""
        look = self._look
        window = self._buffer.window(0.0, 1)
        if window is None or now - window.received_at > self._stale_stream_sec:
            self._count("look_no_stream")
            return
        if self._actor_received_at is None or now - self._actor_received_at > self._stale_actor_sec:
            self._count("look_no_actor")
            return
        # Looking while the voice is synthesizing put two models on the NPU at
        # once and the mirror's pose model fell behind (see _chatter_round).
        # Clips are not synthesized, so in the clip mode this never holds.
        if self._voice_synthesizing(now):
            self._count("skip_voice_synthesis")
            return
        frame = _decode(window.frames[-1])
        if frame is None:
            self._count("look_bad_frame")
            return
        actor = self._actor
        bbox = actor.bbox if actor.present else None
        vision = self._session.hooks.vision
        record = self._look_record
        self._look_ticks += 1
        record.update(tick=self._look_ticks, t=round(now, 3), bbox=list(bbox) if bbox else None)
        started = self._now()
        try:
            seen = vision.prepare(frame, bbox, crop=self._look_crop)
            q1 = vision.ask(seen, Q1)
            record.update(q1=q1.text, crop_fallback=bool(seen.crop_fallback), image=seen.image)
            label = classify(q1.text)
            if look.wants_refine(label) and Q2 not in getattr(vision, "question_failures", {}):
                q2 = vision.ask(seen, Q2)
                record["q2"] = q2.text
                label = refine(label, q2.text)
        except Exception as error:  # noqa: BLE001 - plan §4.6: a failed look is a silent tick
            self._look_last_end = self._now()
            self._count("look_error")
            record["error"] = f"{type(error).__name__}: {error}"
            if now - self._look_error_logged_at >= LOOK_ERROR_LOG_SEC:
                self._look_error_logged_at = now
                self.get_logger().warning(f"look failed: {record['error']}")
            return
        after = self._now()
        self._look_last_end = after
        seconds = after - started
        record.update(seconds=round(seconds, 3), label=label.tag if label else None)
        self._rounds += 1
        if seconds > self._look_slow_sec:
            look.slow()
            record["slow"] = True
            self._count("look_slow")
            if look.stopped:
                self.get_logger().warning(f"look: {seconds:.1f} s looks; no more looking this round")
            return
        self._count("look")
        self._look_frame_stamp = window.newest_stamp
        look.looked(after, label, ref=seen)

    def _look_verify(self, key: str, seen) -> bool | None:
        """Q3 on the picture the named gesture was seen in; None when asking it failed.

        A question refused at load is skipped, not failed: it can never be
        asked, so that gesture goes on Q1 alone, as with ``verify: false``.
        """
        vision = self._session.hooks.vision
        if key in getattr(vision, "question_failures", {}):
            self._count("q3_skipped")
            self._look_record["q3_skipped"] = key
            return True
        try:
            answer = vision.ask(seen, key)
        except Exception as error:  # noqa: BLE001 - an unanswered Q3 is a "no": the line downgrades
            self._count("look_error")
            self._look_record["q3_error"] = f"{type(error).__name__}: {error}"
            return None
        yes = parse_yes(answer.text)
        self._look_record.update(q3=key, q3_answer=answer.text, q3_yes=yes)
        self._count("q3_yes" if yes else "q3_no")
        return yes

    def _look_say(self, pick, now: float) -> None:
        self._count(f"look_said_{pick.reason}")
        stamp = self._look_frame_stamp if self._look_frame_stamp is not None else now
        message = VisualInteractionUtterance()
        message.header.stamp = _as_time(stamp)
        message.evidence_start_stamp = _as_time(stamp)
        message.session_id = "look"
        message.event_id = f"look-{pick.total}-{pick.clip_id}"
        message.episode_seq = pick.total
        message.text = pick.text
        message.kind = KIND_CHATTER
        message.category = pick.pool
        message.suppress_reason = ""
        message.latest_start_at = _as_time(now + self._look_latest_start_sec)
        self._utterances.publish(message)
        self._look_record.update(pool=pick.pool, clip=pick.clip_id, reason=pick.reason, quick=pick.quick)
        self.get_logger().info(
            f'say: "{pick.text}"  [{pick.pool} {pick.clip_id}, {pick.reason}{", quick" if pick.quick else ""}] '
            f"{pick.seq}/{self._look.config.max_per_round}"
        )

    def _look_dump(self) -> None:
        """dump_dir: the crop each look saw, and one answers.jsonl line per tick that did something."""
        if not self._look_dump_dir:
            return
        record = dict(self._look_record)
        image = record.pop("image", None)
        try:
            os.makedirs(self._look_dump_dir, exist_ok=True)
            if image is not None and "tick" in record:
                import cv2

                name = f"look-{record['tick']:06d}.jpg"
                cv2.imwrite(os.path.join(self._look_dump_dir, name), image)
                record["image"] = name
            with open(os.path.join(self._look_dump_dir, "answers.jsonl"), "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as error:  # noqa: BLE001 - a dump is a debugging aid, never a reason to stop
            self._count("look_dump_error")
            if self._now() - self._look_error_logged_at >= LOOK_ERROR_LOG_SEC:
                self._look_error_logged_at = self._now()
                self.get_logger().warning(f"look: dump failed: {error}")

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
        self._log_round(result)

    def _log_round(self, result) -> None:
        """Log what the round decided, so a bring-up run is readable live.

        Without this the only trace of a decision is the utterance topic, which
        means watching the feature work requires a second terminal echoing a
        topic. A spoken line is logged at info because it is the event the
        feature exists to produce; a quiet round is logged at debug, since at
        one round a second the normal case is quiet and would drown the rest.

        MODEL_NOT_READY is the exception: it is the one quiet reason that means
        something is wrong rather than that there was nothing to say, so it
        gets a warning and carries the model's own error when there is one.
        """
        if result.spoken:
            observation = result.observation
            subject = f"{observation.category}/{observation.region}" if observation else "?"
            self.get_logger().info(f'say: "{result.text}"  [{subject}] event={result.event_id}')
            return
        if result.reason == SUPPRESS_MODEL_NOT_READY:
            detail = getattr(self._session.hooks.vision, "last_error", "")
            # Once per distinct cause. A model that cannot load fails the same
            # way every round, and sixty identical warnings bury the one line
            # that says what went wrong; the status topic still counts them all.
            if detail != self._last_model_error:
                self._last_model_error = detail
                self.get_logger().warning(f"quiet: MODEL_NOT_READY{f' ({detail})' if detail else ''}")
            else:
                self.get_logger().debug(f"quiet: MODEL_NOT_READY{f' ({detail})' if detail else ''}")
            return
        self._last_model_error = None
        # A refused or malformed answer carries why. Logged at info rather than
        # debug: unlike COOLDOWN or DUPLICATE, which are the feature working,
        # these say the model produced something unusable, and during bring-up
        # that is the single most useful line on the terminal.
        if result.detail:
            self.get_logger().info(f"quiet: {result.reason} ({result.detail})")
            return
        self.get_logger().debug(f"quiet: {result.reason}")

    def _publish_status(self) -> None:
        payload = {
            "node": NODE_NAME,
            "rounds": self._rounds,
            "frames": self._buffer.total,
            "session_id": self._session.session_id,
            "actor_epoch": self._session.actor_epoch,
            "muted": self._session.muted,
            # The mirror gate, for the same reason every other quiet cause is
            # reported: "parked while the mirror is not mirroring" must be
            # readable off the status topic, not guessed from silence.
            "mirror_state": self._mirror_state,
            "mirror_active": self._mirror_active(self._now()),
            "outcomes": dict(sorted(self._counts.items())),
            "spoken_topics": self._session.ledger().spoken_count(),
            "mode": self._mode,
        }
        if self._chatter is not None:
            payload["chatter_spoken"] = self._chatter.spoken
            payload["voice_busy"] = self._voice_busy(self._now())
        if self._look is not None:
            payload["look"] = {
                "ready": self._look_ready,
                "off": self._look_dead or "",
                "round_spoken": self._look.spoken,
                "spoken": self._look.total,
                "stopped": self._look.stopped,
                "looks": self._look_ticks,
            }
            payload["voice_busy"] = self._look.busy(self._now())
        self._status.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def _count(self, key: str) -> None:
        self._counts[key] = self._counts.get(key, 0) + 1

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


def _decode(data: bytes):
    """A JPEG as an H×W×3 BGR array, or None when it does not decode."""
    import cv2
    import numpy as np

    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return frame if frame is not None and frame.ndim == 3 else None


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
