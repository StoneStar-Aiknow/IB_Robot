"""AgiBot X2 runtime: the public runtime contract served over the vendor MC tier.

This is the wrapper layer. Generic IB-Robot packages talk only to contract
names (``/runtime_status``, ``/runtime/set_mode``, ``/runtime/stop``,
``/joint_states``, the declared command channels, ``/cmd_vel``) and to the
neutral interaction/telemetry interfaces declared in the runtime profile; the
vendor's topics, services, message types, mode names and priority scales stay
behind this node.

What the node does NOT do (design D1): instantiate a controller manager, load a
ros2_control hardware component, publish on the unprotected low-level joint
command tier, or re-implement balance, gait or whole-body control.
"""

from __future__ import annotations

import contextlib
import json
import sys
import threading
import time
import uuid
from collections import deque
from typing import Any

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float64MultiArray, String

from aimdk_robot import dataplane, projection
from aimdk_robot import vendor_gateway as vg
from aimdk_robot.projection import CommandRejected
from aimdk_robot.wake_ack import WakeAckPairer
from ibrobot_msgs.msg import PowerState, RuntimeStatus, SpeechWakeEvent
from ibrobot_msgs.srv import (
    GetRuntimeStatus,
    PlayExpression,
    SetLedPattern,
    SetRuntimeMode,
    SpeakText,
    StartRelocalization,
    StopRuntime,
)
from robot_runtime.contract import (
    GET_STATUS_SERVICE,
    IDLE_MODE,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_CONNECTING,
    LIFECYCLE_DEGRADED,
    LIFECYCLE_FAULTED,
    NAVIGATION_ACK_TOPIC,
    NAVIGATION_ENABLE_SERVICE,
    SET_MODE_SERVICE,
    STATUS_TOPIC,
    STOP_POLICIES,
    STOP_SERVICE,
)
from robot_runtime.interface_description import build_description, validate_description
from robot_runtime.interface_monitor import InterfaceMonitor
from robot_runtime.modes import ModeModel, ModeModelConfig
from robot_runtime.profile import load_profile
from robot_runtime.state import RuntimeState

# Contract-side endpoints this runtime adds for vendor capabilities. They are
# vendor-neutral: another robot with speech or lights serves the same names.
SPEAK_SERVICE = "/speech/speak"
EXPRESSION_SERVICE = "/expression/play"
LED_SERVICE = "/led/set_pattern"
RELOCALIZE_SERVICE = "/localization/relocalize"
POWER_TOPIC = "/power_state"
DIAGNOSTICS_TOPIC = "/diagnostics"

#: How old the platform's action report may be for a mode switch to rely on it
#: instead of asking (the state is published at 10 Hz).
_ACTION_REPORT_MAX_AGE_S = 0.5

#: Fallback for a vendor endpoint the profile's `vendor.qos` does not name.
#: The per-endpoint values come from the vendor's interface tables instead; see
#: `projection.vendor_qos`.
_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)
_RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

#: Public streams this runtime proves ready by observing them on the ROS graph.
#: The two audio bindings are here because a voice deployment may consume either:
#: the vendor speech chunks (external_vad) or the republished raw capture
#: (shared_audio). Without the second entry a shared_audio consumer can never be
#: admitted: its interface would stay 'unknown' and launch would time out.
#: Vendor-native sensors (cameras, lidar, IMUs) stay excluded: subscribing to
#: them inside this process would compete with motion control on a board that is
#: already the bottleneck.
OBSERVED_INTERFACES = ("audio.speech", "audio.capture")

#: ProcessedAudioOutput.audio_vad_state value that opens a vendor speech segment.
_VENDOR_SPEECH_BEGIN = 1
#: What audio.wake_event carries: the woken segment, named like SpeakerEmbedding names one.
_WAKE_EVENT_TYPE = "ibrobot_msgs/msg/SpeechWakeEvent"
#: Frame id carried by project speech chunks; consumers must not depend on it.
_SPEECH_FRAME_ID = "enhanced_audio"


#: Longest vendor message forwarded as one speech chunk. The vendor streams ~100 ms blocks, but
#: its VAD opens a segment with a backfill burst of the audio before onset (~650 ms, 20800 bytes,
#: observed on hardware 2026-10-20), which a 16000-byte cap cut short. A message past this bound
#: is not one the vendor stream produces; it is dropped, never truncated.
_SPEECH_MAX_CHUNK_MS = 2000
#: SpeechAudioChunk carries mono 16 kHz signed 16-bit PCM (see the message definition).
_SPEECH_MAX_CHUNK_BYTES = _SPEECH_MAX_CHUNK_MS * 16000 * 2 // 1000

#: How long to wait before asking for audio focus again after a refusal. A
#: refusal means another source holds the speaker, so retrying every tick would
#: only add service traffic to a queue that is already occupied.
_AUDIO_FOCUS_RETRY_S = 1.0

#: Bytes per sample of the PCM formats the platform player declares.
_PCM_SAMPLE_BYTES = {"S16LE": 2, "S24LE": 3, "S32LE": 4}

#: Window over which clock-offset samples are kept. Long enough that a burst of
#: callback latency cannot fill it, short enough to notice a clock being
#: stepped while the runtime is up.
_CLOCK_SKEW_WINDOW_S = 30.0


class AimdkRuntimeNode(Node):
    """Bridge between the public runtime contract and the AimDK MC tier."""

    def __init__(self, **node_kwargs: Any) -> None:
        super().__init__("aimdk_runtime", **node_kwargs)
        self.declare_parameter("profile", "")
        self.declare_parameter("use_sim", False)
        self.declare_parameter("sim_platform", "")
        self.declare_parameter("initial_mode", "")
        self.declare_parameter("instance_id", "")
        self.declare_parameter("startup_timeout_s", 20.0)
        # In simulated transport the launch entry supplies the description it
        # also feeds to the synthetic sensor publishers, so what the runtime
        # advertises and what it produces cannot drift apart.
        self.declare_parameter("interface_description_json", "")

        profile_path = self.get_parameter("profile").value
        if not profile_path:
            raise RuntimeError("aimdk_runtime requires the 'profile' parameter")
        self._profile = load_profile(profile_path)
        self._use_sim = bool(self.get_parameter("use_sim").value)
        self._sim_platform = str(self.get_parameter("sim_platform").value or "").strip()
        instance_id = str(self.get_parameter("instance_id").value or "")
        if instance_id:
            self._profile["runtime"]["instance_id"] = instance_id
        initial_mode = str(self.get_parameter("initial_mode").value or "")
        if initial_mode:
            if initial_mode not in self._profile["modes"] or initial_mode == "initial":
                raise RuntimeError(f"unsupported initial_mode {initial_mode!r}")
            self._profile["modes"]["initial"] = initial_mode

        self._vendor = self._profile.get("vendor") or {}
        self._source = str((self._vendor.get("input_source") or {}).get("name", "ibrobot.aimdk_robot"))
        self._source_priority = int((self._vendor.get("input_source") or {}).get("priority", 30))
        self._stamp_window_s = float((self._vendor.get("command") or {}).get("stamp_window_s", 0.2))
        # A command whose stamp falls outside the platform's window is dropped
        # by the platform without a word, so a clock that disagrees with the
        # robot's disables every stream at once. Hold the runtime out of ACTIVE
        # rather than let that look like a robot that ignores commands.
        self._clock_skew_limit_s = float((self._vendor.get("command") or {}).get("clock_skew_limit_s", 0.1))
        self._clock_samples: deque[tuple[float, float]] = deque(maxlen=256)
        # Facts the platform pushes with every state message. Reading them here
        # costs no extra traffic and removes the lag of asking over services.
        self._speed_envelope: projection.SpeedEnvelope | None = None
        self._player_state = projection.PLAYER_STATE_IDLE
        self._hand_statuses: list[int] = []
        self._hand_type = str((self._vendor.get("hand") or {}).get("type", "claw"))

        # Raw audio playback is focus-gated by the platform; these stay inert
        # until a profile declares a playback endpoint.
        self._audio_lock = threading.Lock()
        self._playback_buffer: deque[list[int]] = deque(maxlen=1)
        self._focus_held = False
        self._focus_attempt_s = 0.0
        # When everything handed to the platform player has been rendered (monotonic seconds).
        self._render_until = 0.0
        self._focus_priority = 6
        self._focus_weight = 0
        self._focus_release_idle_s = 1.0
        self._request_focus: Any = None
        self._abandon_focus: Any = None

        supplied = str(self.get_parameter("interface_description_json").value or "")
        if supplied:
            description = json.loads(supplied)
            validate_description(description)
        else:
            description = build_description(
                self._profile, use_sim=self._use_sim, sim_platform=self._sim_platform or None
            )
        self._speech_cb = MutuallyExclusiveCallbackGroup()
        self._speech_description = description
        self._speech_enabled = False
        # Readiness is reported the same way the facade and mock runtimes report
        # it: by observing the declared endpoint, not by trusting that we
        # published to it. Only the interfaces this runtime can afford to
        # observe are handed to the monitor.
        self._interface_monitor = InterfaceMonitor(
            self,
            {"interfaces": {k: v for k, v in description["interfaces"].items() if k in OBSERVED_INTERFACES}},
            callback_group=self._speech_cb,
        )
        modes = ModeModel(ModeModelConfig.from_profile(self._profile))
        self._state = RuntimeState(
            name=str(self._profile["runtime"]["name"]),
            version=str(self._profile["runtime"]["version"]),
            capabilities=self._profile["capabilities"],
            modes=modes,
            on_change=self._publish_status,
            interface_description=description,
            interface_states=self._interface_monitor.states,
        )
        self._modes = modes

        # The data plane (aimdk_dataplane, see aimdk_robot.dataplane) holds the
        # commands and the platform state; this process tells it what is
        # admitted and reads back what it reports. _lock guards both words.
        self._lock = threading.RLock()
        self._control = dataplane.Control()
        self._dataplane = dataplane.Status()
        # When the data plane generated its last status (monotonic s; the report
        # carries it, so the time it spent queued counts as age). Its group ages
        # are relative to that moment.
        self._dataplane_at = float("-inf")
        self._dataplane_seen = threading.Condition(self._lock)
        #: Token of the named motion replaying a trajectory on the arm channel;
        #: while set, the public arm stream is refused instead of interleaved.
        self._arm_owner: int | None = None
        #: Moves on anything that may have moved the arm or changed who commands
        #: it: a named motion taking the channel and every command reset (mode
        #: change, stop, lost arbitration) here, and a public arm target in the
        #: data plane (its count is added in arm_motion_epoch). A named motion
        #: cancelled part-way resumes only while it has not moved.
        self._arm_motion_epoch = 0
        self._arbitration_ok = False
        self._body_pose = projection.BODY_POSE_UNKNOWN
        self._navigation_enabled = False
        self._vendor_action = ""
        self._vendor_action_running = False
        # When that state arrived, and when this runtime last asked for an action (monotonic s).
        self._vendor_action_at = 0.0
        self._action_requested_at = 0.0
        self._system_state = ""
        self._hand_type_verified = False
        self._hand_type_reported: list[str] = []
        feedback = self._vendor.get("feedback") or {}
        self._feedback_window_s = float(feedback.get("staleness_s", 0.5))
        self._required_feedback = [str(group) for group in feedback.get("required_groups", ["arm", "hand"])]
        # Joint groups whose vendor fault codes stop commands. A profile opts in:
        # what a non-zero code means is per group (OmniHand faultcodes are set
        # while a finger presses an object, a normal grasp), so none by default.
        self._fault_groups = {str(group) for group in feedback.get("fault_groups", [])}
        # Fault sources that decide the lifecycle between them: each holds at
        # most one fault, DEGRADED ones are recoverable, FAULTED ones persist
        # until the platform clears the condition.
        self._fault_sources: dict[str, tuple[str, str]] = {}

        self._client_cb = ReentrantCallbackGroup()
        self._service_cb = MutuallyExclusiveCallbackGroup()
        # Telemetry that must be timely gets its own group, and so does the data
        # plane's status: a stop waits for it to acknowledge the closed output.
        self._telemetry_cb = MutuallyExclusiveCallbackGroup()
        self._joint_pub_cb = MutuallyExclusiveCallbackGroup()
        # The command streams get their own group for the same reason: behind
        # the joint torrent the arm stream was taken up about twice a second on
        # the robot, so the platform got a stale pose, then nothing once it
        # aged past stream_staleness_s, then a jump.
        self._command_cb = MutuallyExclusiveCallbackGroup()
        # The speaker stream too: behind the joint torrent a 1 s phrase came
        # out as 3-11 of its 20 chunks spread over up to 5 s, the depth-10
        # queue having dropped the rest. Mutually exclusive, so chunks of one
        # phrase are still published in the order they arrive.
        self._audio_cb = MutuallyExclusiveCallbackGroup()
        # The data plane node (aimdk_dataplane) reports its status on its own group too.
        self._dataplane_cb = MutuallyExclusiveCallbackGroup()
        # Stop and status must never queue behind a mode switch that is
        # waiting for the platform's confirmation: they run in their own
        # reentrant group, and the stop path serialises itself.
        self._stop_cb = ReentrantCallbackGroup()
        self._stop_lock = threading.Lock()

        self._status_pub = self.create_publisher(RuntimeStatus, STATUS_TOPIC, 10)
        self._power_pub = self.create_publisher(PowerState, POWER_TOPIC, _RELIABLE_QOS)
        self._diag_pub = self.create_publisher(DiagnosticArray, DIAGNOSTICS_TOPIC, 10)
        self._localization_pub: Any = None

        vg.require_vendor()
        self._setup_vendor_io()
        self._setup_contract_surface()

        self.create_timer(1.0, self._publish_status, callback_group=self._stop_cb)
        # The data plane closes the output once this word is older than
        # dataplane.CONTROL_TIMEOUT_S, so it is repeated well inside that.
        self.create_timer(0.1, self._push_control, callback_group=self._stop_cb)
        self.create_timer(1.0, self._poll_arbitration, callback_group=self._client_cb)
        self.create_timer(1.0, self._check_clock_skew, callback_group=self._client_cb)
        self.create_timer(0.2, self._check_feedback, callback_group=self._client_cb)
        self.create_timer(5.0, self._recheck_hand_type, callback_group=self._client_cb)
        self._startup_thread = threading.Thread(target=self._startup_checks, daemon=True)
        self._startup_thread.start()

    # --- wiring -------------------------------------------------------------

    def _vendor_qos(self, key: str, **defaults: Any) -> QoSProfile:
        """QoS for one vendor endpoint, as the vendor documents it."""
        reliability, durability, depth = projection.vendor_qos(self._vendor.get("qos"), key, **defaults)
        return QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE if reliability == "reliable" else ReliabilityPolicy.BEST_EFFORT,
            durability=(
                DurabilityPolicy.TRANSIENT_LOCAL if durability == "transient_local" else DurabilityPolicy.VOLATILE
            ),
            history=HistoryPolicy.KEEP_LAST,
            depth=depth,
        )

    def _setup_vendor_io(self) -> None:
        from aimdk_msgs.msg import (  # noqa: PLC0415 - vendor import stays local
            AlertCodeArray,
            AudioCapture,
            DiagnosticInfoArray,
            McCommonState,
            PmuState,
            SmSystemState,
        )
        from aimdk_msgs.srv import (  # noqa: PLC0415
            AbandonAudioFocus,
            GetCurrentInputSource,
            GetHandType,
            GetMcPresetMotionState,
            PlayEmoji,
            PlayTts,
            RequestAudioFocus,
            SetMcAction,
            SetMcInputSource,
            SetMcPresetMotion,
            SetPmuLed,
        )

        topics = self._vendor.get("topics") or {}
        services = self._vendor.get("services") or {}

        # The data plane (aimdk_dataplane) takes the platform's joint, hand and
        # arm command state and owns the command stream; this is the runtime's
        # side of the wiring to it (see aimdk_robot.dataplane).
        self._control_pub = self.create_publisher(
            String,
            dataplane.CONTROL_TOPIC,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )
        self._named_arm_pub = self.create_publisher(Float64MultiArray, dataplane.NAMED_ARM_TOPIC, 1)
        self.create_subscription(
            String, dataplane.STATUS_TOPIC, self._on_dataplane_status, 20, callback_group=self._dataplane_cb
        )

        def caller(service_type: Any, key: str, default: str) -> vg.ServiceCaller:
            endpoint = str(services.get(key, default))
            return vg.ServiceCaller(self, self.create_client(service_type, endpoint, callback_group=self._client_cb))

        self._set_action = caller(SetMcAction, "set_action", "/aimdk_5Fmsgs/srv/SetMcAction")
        self._set_input = caller(SetMcInputSource, "set_input_source", "/aimdk_5Fmsgs/srv/SetMcInputSource")
        self._get_input = caller(GetCurrentInputSource, "get_input_source", "/aimdk_5Fmsgs/srv/GetCurrentInputSource")
        self._get_hand_type = caller(GetHandType, "get_hand_type", "/aimdk_5Fmsgs/srv/GetHandType")
        self._preset_motion = caller(SetMcPresetMotion, "set_preset_motion", "/aimdk_5Fmsgs/srv/SetMcPresetMotion")
        self._preset_state = caller(
            GetMcPresetMotionState, "get_preset_motion_state", "/aimdk_5Fmsgs/srv/GetMcPresetMotionState"
        )
        self._play_tts = caller(PlayTts, "play_tts", "/aimdk_5Fmsgs/srv/PlayTts")
        self._play_emoji = caller(PlayEmoji, "play_emoji", "/aimdk_5Fmsgs/srv/PlayEmoji")
        self._set_led = caller(SetPmuLed, "set_led", "/aimdk_5Fmsgs/srv/SetPmuLed")

        if topics.get("mc_state"):
            self.create_subscription(
                McCommonState,
                str(topics["mc_state"]),
                self._on_mc_state,
                self._vendor_qos("mc_state"),
                callback_group=self._telemetry_cb,
            )
        if topics.get("system_state"):
            self.create_subscription(
                SmSystemState,
                str(topics["system_state"]),
                self._on_system_state,
                self._vendor_qos("system_state"),
                callback_group=self._telemetry_cb,
            )
        if self._state.has_capability("power.state") and topics.get("power_state"):
            self.create_subscription(
                PmuState, str(topics["power_state"]), self._on_power_state, self._vendor_qos("power_state")
            )
        if self._state.has_capability("diagnostics.codes"):
            if topics.get("diag_codes"):
                self.create_subscription(
                    DiagnosticInfoArray, str(topics["diag_codes"]), self._on_diagnostics, self._vendor_qos("diag_codes")
                )
            if topics.get("alert_codes"):
                self.create_subscription(
                    AlertCodeArray, str(topics["alert_codes"]), self._on_alerts, self._vendor_qos("alert_codes")
                )
        if self._state.has_capability("interaction.audio_capture") and topics.get("audio_capture"):
            self._setup_audio_capture(AudioCapture, topics)
        # Gated on its own capability: the speaker is served by a profile that
        # declares playback without declaring the microphones.
        if self._state.has_capability("interaction.audio_playback") and topics.get("audio_playback"):
            self._setup_audio_playback(topics, caller, RequestAudioFocus, AbandonAudioFocus)
        self._setup_processed_audio_if_declared(topics)
        if self._state.has_capability("localization.pose") and topics.get("localization_pose"):
            self._setup_localization(topics)

    def _setup_audio_capture(self, audio_capture_type: Any, topics: dict[str, Any]) -> None:
        """Republish vendor microphone audio on the project's audio contract."""
        from audio_common_msgs.msg import AudioDataStamped, AudioInfo  # noqa: PLC0415

        audio = self._vendor.get("audio") or {}
        # Reliable on the contract side too: the project's own ASR consumer
        # subscribes reliably, and a best-effort publisher would never reach it.
        self._audio_pub = self.create_publisher(
            AudioDataStamped, str(audio.get("capture_topic", "/audio/capture_stamped")), _RELIABLE_QOS
        )
        self._audio_info_pub = self.create_publisher(AudioInfo, str(audio.get("info_topic", "/audio/info")), 1)
        self._audio_stamped_type = AudioDataStamped
        self._audio_info_type = AudioInfo
        # BEST_EFFORT by default, as the SDK's raw-audio examples subscribe; the
        # profile's vendor.qos explains why (a RELIABLE reader lets the platform
        # retransmit from its 1000-deep capture history).
        self.create_subscription(
            audio_capture_type,
            str(topics["audio_capture"]),
            self._on_audio_capture,
            self._vendor_qos("audio_capture", reliability="best_effort"),
        )

    def _setup_audio_playback(
        self, topics: dict[str, Any], caller: Any, request_focus_type: Any, abandon_focus_type: Any
    ) -> None:
        """Play contract audio through the vendor speaker, holding its audio focus while it plays."""
        from aimdk_msgs.msg import AudioPlayback, FocusResponse  # noqa: PLC0415
        from audio_common_msgs.msg import AudioDataStamped  # noqa: PLC0415

        audio = self._vendor.get("audio") or {}
        # The platform subscribes RELIABLE here: a BEST_EFFORT publisher would
        # be silently dropped (observed on the robot, 2026-09-22).
        self._playback_pub = self.create_publisher(
            AudioPlayback, str(topics["audio_playback"]), self._vendor_qos("audio_playback", reliability="reliable")
        )
        self._playback_type = AudioPlayback
        priority, weight, release_idle_s, buffer_chunks = projection.audio_focus_request(audio)
        self._focus_priority = priority
        self._focus_weight = weight
        self._focus_release_idle_s = release_idle_s
        self._playback_buffer = deque(maxlen=buffer_chunks)
        self._request_focus = caller(request_focus_type, "request_audio_focus", "/aimdk_5Fmsgs/srv/RequestAudioFocus")
        self._abandon_focus = caller(abandon_focus_type, "abandon_audio_focus", "/aimdk_5Fmsgs/srv/AbandonAudioFocus")
        # Preemption is announced, not polled: a higher-priority source
        # taking the speaker arrives here, and publishing past it would
        # only fight whatever the platform decided should be heard.
        self.create_subscription(
            FocusResponse,
            str(topics.get("audio_focus_response", "/aima/hal/audio/focus_response")),
            self._on_audio_focus,
            self._vendor_qos("audio_focus_response", reliability="reliable", durability="transient_local"),
        )
        self.create_subscription(
            AudioDataStamped,
            str(audio.get("play_topic", "/audio/play")),
            self._on_audio_play,
            _SENSOR_QOS,
            # The speaker stream gets its own group: behind the joint torrent a 1 s
            # phrase came out as 3-11 of its 20 chunks spread over up to 5 s.
            callback_group=self._audio_cb,
        )
        self._focus_timer = self.create_timer(0.1, self._audio_focus_tick, callback_group=self._client_cb)

    def _public_qos(self, spec: dict[str, Any]) -> QoSProfile:
        """QoS for one project-facing endpoint, taken from its own declaration."""
        qos = spec.get("qos") or {}
        return QoSProfile(
            reliability=(
                ReliabilityPolicy.RELIABLE if qos.get("reliability") == "reliable" else ReliabilityPolicy.BEST_EFFORT
            ),
            durability=(
                DurabilityPolicy.TRANSIENT_LOCAL
                if qos.get("durability") == "transient_local"
                else DurabilityPolicy.VOLATILE
            ),
            history=HistoryPolicy.KEEP_LAST,
            depth=int(qos.get("depth", 10)),
        )

    def _wants_processed_audio(self, topics: dict[str, Any]) -> bool:
        """Whether this profile asks for the vendor's enhanced speech stream.

        The same gate the raw capture path uses: the capability must be
        declared, and the profile must name the vendor topic.
        """
        return bool(self._state.has_capability("interaction.audio_capture") and topics.get("processed_audio"))

    def _setup_processed_audio_if_declared(self, topics: dict[str, Any]) -> bool:
        """Wire the enhanced speech path only when the profile asked for it."""
        if not self._wants_processed_audio(topics):
            return False
        self._setup_processed_audio(topics)
        return True

    def _setup_processed_audio(self, topics: dict[str, Any]) -> None:
        """Republish the vendor's processed (AEC) speech stream as project chunks.

        The vendor message type stays behind this node, exactly like the raw
        capture and playback paths. PCM and the VAD boundary travel in one
        project message so no consumer has to reconstruct an ordering that DDS
        does not promise across two topics.
        """
        from aimdk_msgs.msg import ProcessedAudioOutput  # noqa: PLC0415 - vendor import stays local
        from audio_common_msgs.msg import AudioDataStamped  # noqa: PLC0415

        from ibrobot_msgs.msg import SpeechAudioChunk  # noqa: PLC0415

        interfaces = self._speech_description.get("interfaces") or {}
        speech = interfaces.get("audio.speech")
        if speech is None:
            raise RuntimeError(
                "vendor.topics.processed_audio is declared but the profile publishes no "
                "audio.speech interface; declare the project endpoint or remove the topic"
            )
        # The declared capability is a promise about what consumers can rely on.
        if speech.get("message_type") != "ibrobot_msgs/msg/SpeechAudioChunk":
            raise RuntimeError(
                f"audio.speech must be ibrobot_msgs/msg/SpeechAudioChunk, got {speech.get('message_type')!r}"
            )

        self._speech_chunk_type = SpeechAudioChunk
        self._speech_source_id = uuid.uuid4().hex
        self._speech_sequence = 0
        self._speech_vendor_sequence: int | None = None
        self._speech_vendor_stream_id: int | None = None
        self._speech_enabled = True
        self._speech_pub = self.create_publisher(
            SpeechAudioChunk, str(speech["endpoint"]), self._public_qos(speech), callback_group=self._speech_cb
        )
        # The diagnostic republish is a project-facing endpoint, so it is
        # declared like any other public interface instead of being smuggled
        # through the vendor section: that keeps it visible to the runtime
        # description and lets it carry its own QoS. Not declared means not
        # published.
        capture = interfaces.get("audio.enhanced_capture")
        if capture is None:
            self._speech_capture_pub = None
        else:
            if capture.get("message_type") != "audio_common_msgs/msg/AudioDataStamped":
                raise RuntimeError(
                    "audio.enhanced_capture must be audio_common_msgs/msg/AudioDataStamped, "
                    f"got {capture.get('message_type')!r}"
                )
            self._speech_capture_pub = self.create_publisher(
                AudioDataStamped, str(capture["endpoint"]), self._public_qos(capture)
            )
        self.create_subscription(
            ProcessedAudioOutput,
            str(topics["processed_audio"]),
            self._on_processed_audio,
            # BEST_EFFORT + VOLATILE can consume a TRANSIENT_LOCAL publisher and
            # deliberately does not ask for history: a latched utterance from
            # before this runtime started must not be recognized as new speech.
            self._vendor_qos("processed_audio", reliability="best_effort", durability="volatile"),
            callback_group=self._speech_cb,
        )
        # Availability heartbeat, not speech: it lets the runtime description
        # report the interface before the wake word, and carries no PCM.
        self.create_timer(0.5, self._publish_speech_heartbeat, callback_group=self._speech_cb)
        self._setup_wake_event(interfaces, topics)

    def _setup_wake_event(self, interfaces: dict[str, Any], topics: dict[str, Any]) -> None:
        """Report the platform's wake word as a project event, when the profile declares one.

        The vendor consumes its wake word before releasing speech, so a consumer
        cannot tell an utterance that named the robot from a follow-up. Its own
        assistant's acknowledgement on the playback topic can (see
        ``aimdk_robot.wake_ack``). Declared like every public endpoint; not
        declared means not published, and the vendor playback topic is then not
        subscribed either.
        """
        self._wake_pairer = None
        wake = interfaces.get("audio.wake_event")
        if wake is None:
            return
        if wake.get("message_type") != _WAKE_EVENT_TYPE:
            raise RuntimeError(f"audio.wake_event must be {_WAKE_EVENT_TYPE}, got {wake.get('message_type')!r}")
        if not topics.get("audio_playback"):
            raise RuntimeError(
                "audio.wake_event is declared but vendor.topics.audio_playback is not; "
                "the acknowledgement is read from the platform's playback topic"
            )
        from aimdk_msgs.msg import AudioPlayback  # noqa: PLC0415 - vendor import stays local

        ack = self._vendor.get("wake_ack") or {}
        if not str(ack.get("token_pattern") or "").strip():
            # No default: the platform's dialog answers share the pkg_name and
            # differ only in their token, so accepting any token re-creates the
            # false wake the pattern exists for (see aimdk_robot.wake_ack).
            raise RuntimeError(
                "audio.wake_event is declared but vendor.wake_ack.token_pattern is not; "
                "set the acknowledgement's token form (the X2's is \"^_[0-9]+_\")"
            )
        try:
            self._wake_pairer = WakeAckPairer(
                frozenset(str(name) for name in ack.get("pkg_names", ["tts"])),
                token_pattern=str(ack["token_pattern"]),
                window_s=float(ack.get("pair_window_s", 1.5)),
            )
        except ValueError as error:
            raise RuntimeError(f"vendor.wake_ack is invalid: {error}") from error
        self._wake_pub = self.create_publisher(
            SpeechWakeEvent, str(wake["endpoint"]), self._public_qos(wake), callback_group=self._speech_cb
        )
        # Same callback group as the speech path, so the pairer is only ever
        # touched from one thread. The runtime's own playback (pkg_name = its
        # source) is filtered out by the pairer. Its own QoS key: on this topic
        # the runtime is also a publisher, and vendor.qos.audio_playback must
        # stay RELIABLE for that (the platform subscribes RELIABLE), while this
        # subscription only listens and is BEST_EFFORT like audio_capture.
        self.create_subscription(
            AudioPlayback,
            str(topics["audio_playback"]),
            self._on_vendor_playback,
            self._vendor_qos("audio_playback_monitor", reliability="best_effort", durability="volatile"),
            callback_group=self._speech_cb,
        )

    def _on_vendor_playback(self, message: Any) -> None:
        pairer = getattr(self, "_wake_pairer", None)
        if pairer is None:
            return
        segment = pairer.on_playback(str(message.pkg_name), str(message.token_id), time.monotonic())
        if segment is not None:
            self._publish_wake_event(segment)

    def _publish_wake_event(self, segment: int) -> None:
        # Names the woken segment the way SpeakerEmbedding does, by its BEGIN
        # chunk. The vendor's own header carries no stamp on X2 (all zero,
        # observed 2026-10-24), so the event is stamped on arrival.
        event = SpeechWakeEvent()
        event.header.stamp = self.get_clock().now().to_msg()
        event.header.frame_id = _SPEECH_FRAME_ID
        event.source_id = self._speech_source_id
        event.segment_sequence = int(segment)
        self._wake_pub.publish(event)

    def _next_speech_sequence(self) -> int:
        """Allocate the project sequence, surfacing vendor-side loss when visible."""
        self._speech_sequence += 1
        return self._speech_sequence

    def _publish_speech_heartbeat(self) -> None:
        if not self._speech_enabled:
            return
        chunk = self._speech_chunk_type()
        chunk.header.stamp = self.get_clock().now().to_msg()
        chunk.header.frame_id = _SPEECH_FRAME_ID
        chunk.source_id = self._speech_source_id
        chunk.sequence = self._next_speech_sequence()
        chunk.state = 0
        self._speech_pub.publish(chunk)

    def _on_processed_audio(self, message: Any) -> None:
        """Map one vendor processed-audio message onto one project speech chunk."""
        data = bytes(message.audio_data)
        chunk = self._speech_chunk_type()
        chunk.header.stamp = message.header.stamp
        chunk.header.frame_id = _SPEECH_FRAME_ID
        chunk.source_id = self._speech_source_id
        # The vendor IDL documents MessageHeader.sequence as a consecutively
        # increasing ID, but the X2 firmware currently sends a constant
        # (observed on hardware, 2026-10-15), so an upstream BEST_EFFORT loss is
        # NOT observable on X2. Only a strict forward skip is treated as evidence
        # of one, because an earlier "anything but last + 1" test flagged every
        # message against that constant field and dropped all speech; constant,
        # repeated and reset values are not gaps. stream_id names the vendor
        # stream, and a change of it invalidates the previous continuation. On a
        # detected skip one project sequence number is left unused so the
        # consumer drops the affected segment instead of joining audio across
        # the hole.
        stream_id = int(getattr(message, "stream_id", 0) or 0)
        vendor_sequence = int(message.header.sequence)
        if stream_id != self._speech_vendor_stream_id:
            self._speech_vendor_stream_id = stream_id
            self._speech_vendor_sequence = None
        elif self._speech_vendor_sequence is not None and vendor_sequence > self._speech_vendor_sequence + 1:
            self._speech_sequence += 1
            self.get_logger().warning(
                f"vendor speech sequence jumped {self._speech_vendor_sequence} -> {vendor_sequence}; "
                "marking the project stream so the consumer drops the affected segment"
            )
        self._speech_vendor_sequence = vendor_sequence
        if len(data) > _SPEECH_MAX_CHUNK_BYTES:
            # One chunk carries one vendor message whole, so a boundary keeps all of its audio.
            # Truncating would remove speech from inside a segment, which the recognizer then
            # reports as a different word; leaving a sequence number unused makes the consumer
            # drop the segment instead.
            self._speech_sequence += 1
            self.get_logger().warning(
                f"vendor speech message of {len(data)} bytes exceeds {_SPEECH_MAX_CHUNK_MS} ms; dropping it "
                "and marking the project stream so the consumer drops the affected segment"
            )
            return
        chunk.sequence = self._next_speech_sequence()
        chunk.state = int(message.audio_vad_state.value)
        # Malformed PCM is forwarded, not dropped here: the consumer aborts the
        # affected segment instead of joining audio across a gap.
        chunk.audio_data = list(data)
        self._speech_pub.publish(chunk)
        pairer = getattr(self, "_wake_pairer", None)
        if pairer is not None and chunk.state == _VENDOR_SPEECH_BEGIN:
            segment = pairer.on_speech_begin(time.monotonic(), chunk.sequence)
            if segment is not None:
                self._publish_wake_event(segment)
        if self._speech_capture_pub is not None and data:
            from audio_common_msgs.msg import AudioDataStamped  # noqa: PLC0415

            stamped = AudioDataStamped()
            stamped.header.stamp = message.header.stamp
            stamped.header.frame_id = _SPEECH_FRAME_ID
            stamped.audio.data = list(data)
            self._speech_capture_pub.publish(stamped)

    def _setup_localization(self, topics: dict[str, Any]) -> None:
        from nav_msgs.msg import Odometry  # noqa: PLC0415

        params = self._state.capabilities.get("localization.pose", {})
        self._localization_frame = str(params.get("reference_frame", "map"))
        self._localization_pub = self.create_publisher(PoseWithCovarianceStamped, "/localization/pose", _RELIABLE_QOS)
        self.create_subscription(
            Odometry,
            str(topics["localization_pose"]),
            self._on_localization,
            self._vendor_qos("localization_pose", reliability="reliable"),
        )

    def _setup_contract_surface(self) -> None:
        self.create_service(SetRuntimeMode, SET_MODE_SERVICE, self._on_set_mode, callback_group=self._service_cb)
        self.create_service(GetRuntimeStatus, GET_STATUS_SERVICE, self._on_get_status, callback_group=self._stop_cb)
        self.create_service(StopRuntime, STOP_SERVICE, self._on_stop, callback_group=self._stop_cb)

        # The command channels themselves are the data plane's: it subscribes
        # them, applies the mode, admission and navigation gate this runtime
        # reports in its control word, and reports what it refused.

        if self._state.has_capability("interaction.tts"):
            self.create_service(SpeakText, SPEAK_SERVICE, self._on_speak, callback_group=self._service_cb)
        if self._state.has_capability("interaction.expression"):
            self.create_service(
                PlayExpression, EXPRESSION_SERVICE, self._on_expression, callback_group=self._service_cb
            )
        if self._state.has_capability("interaction.led"):
            self.create_service(SetLedPattern, LED_SERVICE, self._on_led, callback_group=self._service_cb)
        if self._state.has_capability("base.navigation_gate"):
            from std_srvs.srv import SetBool  # noqa: PLC0415

            self._nav_ack_pub = self.create_publisher(Bool, NAVIGATION_ACK_TOPIC, 10)
            self.create_service(
                SetBool, NAVIGATION_ENABLE_SERVICE, self._on_set_navigation, callback_group=self._service_cb
            )
            self.create_timer(0.5, self._publish_navigation_ack, callback_group=self._service_cb)
        if self._state.has_capability("localization.map"):
            self.create_service(
                StartRelocalization, RELOCALIZE_SERVICE, self._on_relocalize, callback_group=self._service_cb
            )
        if self._state.has_capability("motion.named") or self._state.has_capability("motion.posture"):
            from aimdk_robot.named_motion import NamedMotionServer  # noqa: PLC0415

            self._named_motion = NamedMotionServer(self)

    def _on_set_navigation(self, request, response):
        """Gate externally sourced velocity, with the platform's own precondition.

        The platform only accepts velocity in a stable-stand/locomotion action,
        and only from the arbitration holder; acknowledging the gate before both
        hold would tell a navigation stack it may drive when it may not.
        """
        enable = bool(request.data)
        if not enable:
            # Revoking the gate revokes the output, not just future input: a
            # cached velocity must not keep driving until it expires.
            with self._lock:
                self._navigation_enabled = False
            self.publish_safe_command()
            self._publish_navigation_ack()
            response.success, response.message = True, "navigation disabled"
            return response
        if self._state.stop_latched:
            response.success, response.message = False, "stop latched"
            return response
        if self._state.lifecycle != LIFECYCLE_ACTIVE:
            response.success = False
            response.message = f"runtime is {self._state.lifecycle}; navigation is only admitted while ACTIVE"
            return response
        spec = self._modes.spec()
        if not spec.allows_base:
            response.success = False
            response.message = f"mode {self._modes.mode!r} does not accept velocity commands"
            return response
        with self._lock:
            holding = self._arbitration_ok
        if not holding:
            # Try to take the claim rather than refusing on a stale flag; only
            # a real failure to register blocks the gate.
            holding = self._register_input_source()
        if not holding:
            response.success, response.message = False, "runtime does not hold platform command arbitration"
            return response
        with self._lock:
            self._navigation_enabled = True
        self._push_control()
        self._publish_navigation_ack()
        response.success, response.message = True, "navigation enabled"
        return response

    def _publish_navigation_ack(self) -> None:
        publisher = getattr(self, "_nav_ack_pub", None)
        if publisher is None:
            return
        with self._lock:
            enabled = self._navigation_enabled
        publisher.publish(Bool(data=enabled))

    # --- startup ------------------------------------------------------------

    def _startup_checks(self) -> None:
        """Reach ACTIVE only when the platform actually answers what we declared."""
        deadline = time.time() + float(self.get_parameter("startup_timeout_s").value)
        required = [("SetMcAction", self._set_action), ("SetMcInputSource", self._set_input)]
        missing = []
        while time.time() < deadline:
            missing = [name for name, client in required if not client.ready(0.5)]
            if not missing:
                break
        if missing:
            self._set_fault(
                "startup", f"vendor services unavailable: {', '.join(missing)}", lifecycle=LIFECYCLE_FAULTED
            )
            self._state.set_lifecycle(LIFECYCLE_FAULTED)
            return
        # The data plane is half of this runtime: without its report there is no
        # feedback, no hand type from the state stream and no command output.
        while time.time() < deadline and not self._dataplane_reporting():
            time.sleep(0.05)
        if not self._dataplane_reporting():
            self._set_fault(
                "startup",
                "the data plane (aimdk_dataplane) is not reporting; it carries the feedback and the command stream",
                lifecycle=LIFECYCLE_FAULTED,
            )
            self._state.set_lifecycle(LIFECYCLE_FAULTED)
            return
        if not self._verify_hand_type():
            return
        if not self._register_input_source():
            self._set_fault("arbitration", "vendor input source registration failed; commands would be discarded")
            self._state.set_lifecycle(LIFECYCLE_DEGRADED)
            return
        self._state.set_lifecycle(LIFECYCLE_ACTIVE)
        # Faults observed while CONNECTING (a quiet feedback source, a
        # blocking system mode) apply from the first ACTIVE moment.
        self._refresh_lifecycle()

    def _verify_hand_type(self) -> bool:
        """Check the declared end effector against the installed one (design D2).

        Returns False only when the platform positively reports a *different*
        hand family. Two answers are not mismatches and must not be treated as
        one: ``NONE`` means the platform has no hand state to report (the hands
        are not enumerated in this mode, or none are installed) and ``ERROR``
        is a platform-side hand fault. Both leave the declaration unverified,
        and ``_recheck_hand_type`` asks again — hands enumerate late.

        Two sources answer this: the ``GetHandType`` service and the hand state
        topic, which carries the same enumeration. Either is accepted, because
        a service that reports NONE while the state stream reports a real hand
        type has simply been asked too early.
        """
        declared = "claw" if self._hand_type == "claw" else "dexterous"
        reported: list[int] = []
        if self._get_hand_type.ready(2.0):
            result = self._get_hand_type.call(vg.hand_type_request(self))
            if result.ok and result.response is not None:
                reported = [
                    int(result.response.left_hands_type.value),
                    int(result.response.right_hands_type.value),
                ]
            else:
                self.get_logger().warning("GetHandType did not answer; declared hand type is unverified")
        else:
            self.get_logger().warning("GetHandType unavailable; declared hand type is unverified")
        with self._lock:
            from_state = list(self._dataplane.hand_types)
        families = {vg.hand_type_family(value) for value in reported} if reported else {None}
        if None in families and from_state:
            # The service had nothing definite; the state stream might.
            state_families = {vg.hand_type_family(value) for value in from_state}
            if None not in state_families:
                reported, families = from_state, state_families

        if families == {declared}:
            with self._lock:
                self._hand_type_verified = True
            self._set_fault("hand", None)
            return True

        names = [vg.hand_type_name(value) for value in reported]
        if None in families:
            # Nothing to compare against: report it, keep the declaration
            # unverified, and let the recheck decide later.
            if vg.HAND_TYPE_ERROR in reported:
                self._set_fault(
                    "hand",
                    f"platform reports a hand subsystem error ({names}); "
                    f"the declared {declared!r} end effector is not usable",
                )
            else:
                self._set_fault("hand", None)
                if names != self._hand_type_reported:
                    self.get_logger().warning(
                        f"platform reports hand type {names or 'nothing'}; declared {declared!r} "
                        "is unverified (hands may not be enumerated in the current platform mode)"
                    )
            self._hand_type_reported = names
            return True

        self._set_fault(
            "hand",
            f"hand type mismatch: profile declares {declared!r}, platform reports {names}",
            lifecycle=LIFECYCLE_FAULTED,
        )
        self._state.set_lifecycle(LIFECYCLE_FAULTED)
        return False

    def _recheck_hand_type(self) -> None:
        """Ask again while the declaration is unverified; hands enumerate late."""
        with self._lock:
            verified = self._hand_type_verified
        if verified or self._state.lifecycle in (LIFECYCLE_CONNECTING, LIFECYCLE_FAULTED):
            return
        self._verify_hand_type()

    def _register_input_source(self) -> bool:
        """ADD, then ENABLE — the SDK's documented registration sequence.

        Both are sent, but only ENABLE decides: the platform discards commands
        from a source that is "未注册或未启用", so a source that was added and
        never enabled is ignored exactly like one that was never added. ADD is
        allowed to fail — it does, with "already registered", every time this
        runs again to re-assert a claim — and that is not a failure to register.
        """
        source = self._vendor.get("input_source") or {}

        def request(action: int) -> Any:
            return vg.input_source_request(
                self,
                name=self._source,
                priority=self._source_priority,
                timeout_ms=int(source.get("timeout_ms", 1000)),
                action_value=action,
            )

        self._set_input.call(request(vg.INPUT_ACTION_ADD))
        if not self._set_input.call(request(vg.INPUT_ACTION_ENABLE)).ok:
            return False
        with self._lock:
            self._arbitration_ok = True
        return True

    def _poll_arbitration(self) -> None:
        """Report only a claim we have actually lost to a source that outranks us.

        The platform's arbitration is a *takeover* rule, not a reservation: a
        registered source becomes the holder by sending a non-zero command when
        there is no holder, the holder has timed out, or it outranks the
        holder. Two consequences the runtime has to respect, or it deadlocks
        itself into never being able to claim anything:

        - An empty holder means "nothing has sent a valid command yet", which
          the SDK states outright. It is not a lost claim, and closing output
          on it guarantees the holder stays empty forever.
        - A holder we outrank is not a lost claim either: streaming is how the
          takeover happens. Only a holder that outranks us can keep our
          commands out, and that is the one worth degrading for.
        """
        if self._state.lifecycle == LIFECYCLE_CONNECTING:
            return
        if not self._get_input.ready(0.0):
            return
        result = self._get_input.call(vg.current_input_source_request(self))
        if not result.ok or result.response is None:
            return
        holder = str(result.response.input_source.name or "")
        holder_priority = int(getattr(result.response.input_source, "priority", 0) or 0)
        outranked = bool(holder) and holder != self._source and holder_priority >= self._source_priority
        with self._lock:
            registered = self._arbitration_ok
        if outranked:
            if registered:
                # Whatever was cached was accepted under a claim we no longer
                # hold; recovery requires fresh commands, not a replay.
                self._revoke_output()
            with self._lock:
                self._arbitration_ok = False
            self._set_fault(
                "arbitration",
                f"command arbitration held by {holder!r} at priority {holder_priority} "
                f"(this runtime registers {self._source_priority}); streamed commands are not applied",
            )
            return
        if not registered:
            # Take the claim back rather than waiting to be handed it: the
            # registration is ours to re-assert, and nothing else will.
            self._register_input_source()
        with self._lock:
            registered = self._arbitration_ok
        self._set_fault(
            "arbitration",
            None if registered else "vendor input source registration failed; commands would be discarded",
        )

    # --- faults and lifecycle ----------------------------------------------

    def _set_fault(self, source: str, detail: str | None, *, lifecycle: str = LIFECYCLE_DEGRADED) -> None:
        """Record (or clear, with ``detail=None``) one source's fault and re-derive the lifecycle."""
        with self._lock:
            if detail is None:
                if source not in self._fault_sources:
                    return
                del self._fault_sources[source]
            else:
                if self._fault_sources.get(source) == (lifecycle, detail):
                    return
                self._fault_sources[source] = (lifecycle, detail)
        self._refresh_lifecycle()

    def _refresh_lifecycle(self) -> None:
        """Lifecycle = the worst live fault source; ACTIVE when none remain.

        Never touches CONNECTING (startup decides when that ends) or STOPPED
        (the latch owns the lifecycle until it is cleared). Lifecycle and
        faults move together, so a reader never sees a degraded runtime that
        names no reason.
        """
        with self._lock:
            sources = dict(self._fault_sources)
        faults = [detail for _lifecycle, detail in sources.values()]
        if self._state.stop_latched or self._state.lifecycle == LIFECYCLE_CONNECTING:
            self._state.set_status(None, faults)
            return
        if any(lifecycle == LIFECYCLE_FAULTED for lifecycle, _detail in sources.values()):
            target = LIFECYCLE_FAULTED
        elif sources:
            target = LIFECYCLE_DEGRADED
        else:
            target = LIFECYCLE_ACTIVE
        self._state.set_status(target, faults)

    def holds_arbitration(self) -> bool:
        with self._lock:
            return self._arbitration_ok

    def output_admitted(self) -> bool:
        """Whether anything may leave for the platform right now.

        The same rule gates command input and command output: not stopped,
        ACTIVE (so neither DEGRADED by lost arbitration nor FAULTED by a
        blocking system mode), and holding the platform's command claim.
        """
        if self._state.stop_latched or self._state.lifecycle != LIFECYCLE_ACTIVE:
            return False
        return self.holds_arbitration()

    def player_state(self) -> int:
        """The platform's preset-motion player state, as last reported.

        T6 firmware answers ``task_id 0`` for an accepted preset motion, so the
        named-motion executor has no handle to poll and watches this instead
        (see named_motion._await_player_preset). Kept as a board-local patch on
        top of upstream until the firmware behaviour is reconciled.
        """
        with self._lock:
            return int(self._player_state)

    # --- contract services --------------------------------------------------

    def _on_set_mode(self, request, response):
        target = str(request.mode)
        epoch = self._state.stop_epoch
        if self._state.stop_latched and target != IDLE_MODE:
            response.success = False
            response.message = f"stop latched ({self._state.stop_policy}); only {IDLE_MODE!r} is accepted"
            response.valid_transitions = [IDLE_MODE]
            return response
        decision = self._modes.can_switch(target)
        if not decision.allowed:
            response.success = False
            response.message = decision.reason
            response.valid_transitions = sorted(self._modes.valid_transitions())
            return response
        if decision.mode == self._modes.mode and decision.reason == "already active":
            # A stop leaves the runtime in idle with the latch engaged, so
            # requesting idle again is how the latch is cleared: "already
            # active" must not short-circuit that.
            if self._state.stop_latched and target == IDLE_MODE:
                if not self._clear_stop_latch():
                    response.success = False
                    response.message = (
                        "stop still in progress (waiting for the platform); "
                        f"request {IDLE_MODE!r} again once it has completed"
                    )
                    response.valid_transitions = [IDLE_MODE]
                    return response
                response.success = True
                response.message = "stop latch cleared"
                return response
            response.success = True
            response.message = "already active"
            return response
        mismatch = self._hand_mismatch(target)
        if mismatch:
            response.success = False
            response.message = mismatch
            response.valid_transitions = sorted(self._modes.valid_transitions())
            return response
        ok, detail = self._request_vendor_mode(target, epoch)
        if not ok:
            response.success = False
            response.message = detail
            response.valid_transitions = (
                [IDLE_MODE] if self._state.stop_latched else sorted(self._modes.valid_transitions())
            )
            return response
        if self._state.stop_epoch != epoch:
            # A stop engaged between the platform's confirmation and this
            # commit. The stop handler already put the runtime in idle and
            # latched it; the late success must not override that.
            response.success = False
            response.message = f"stop engaged while switching to {target!r}; request {IDLE_MODE!r} to clear the latch"
            response.valid_transitions = [IDLE_MODE]
            return response
        self._clear_commands()
        self._modes.commit(target)
        if self._state.stop_latched and target == IDLE_MODE and not self._clear_stop_latch():
            response.success = False
            response.message = f"stop still in progress; request {IDLE_MODE!r} again once it has completed"
            response.valid_transitions = [IDLE_MODE]
            return response
        self._publish_status()
        response.success = True
        response.message = detail
        return response

    def _clear_stop_latch(self) -> bool:
        """Clear the latch, unless a stop is still running. Returns whether it cleared.

        The stop handler holds ``_stop_lock`` from latching until the platform
        has answered, so the latch cannot be released underneath a stop that
        is still in progress: an idle request arriving meanwhile is refused
        and must be repeated once the stop has completed.
        """
        if not self._stop_lock.acquire(blocking=False):
            return False
        try:
            self._set_fault("stop", None)
            self._state.clear_stop()
            # A stop released the platform arbitration claim; re-arm it, or the
            # next command stream would be silently discarded.
            self._register_input_source()
            # clear_stop reports ACTIVE; a fault that is still live outranks that.
            self._refresh_lifecycle()
        finally:
            self._stop_lock.release()
        return True

    def _note_action_request(self) -> None:
        with self._lock:
            self._action_requested_at = time.monotonic()

    def _request_vendor_mode(self, mode: str, epoch: int) -> tuple[bool, str]:
        """Switch the vendor MC action and wait for its own state to confirm.

        The wait is interruptible: a stop engaged meanwhile (``epoch`` moves)
        ends it at once, so a stop never queues behind a confirmation.
        """
        action = projection.mode_action(self._profile, mode)
        with self._lock:
            reported_at = self._vendor_action_at
            running = (
                self._vendor_action == action
                and self._vendor_action_running
                # Only a report taken after the last switch this runtime asked
                # for, and recent: a stop or another mode may have moved the
                # platform since the state that is cached.
                and reported_at > self._action_requested_at
                and time.monotonic() - reported_at <= _ACTION_REPORT_MAX_AGE_S
            )
        if running:
            # The platform refuses a switch to the action it is already running
            # (MOTION_BUSY, "动作正在播放", firmware v1.1.1): a robot left in
            # UPPERBODY_REMOTE_SPLIT refused policy_stream that way (live
            # 2026-10-22). Nothing to switch.
            return True, f"platform already in {action}"
        self._note_action_request()
        result = self._set_action.call(vg.set_action_request(self, source=self._source, action_name=action))
        if not result.ok:
            reason = projection.mode_rejection_reason(result.code)
            return False, f"platform rejected {action}: {reason} ({result.message})".strip()
        deadline = time.time() + float((self._vendor.get("command") or {}).get("mode_confirm_timeout_s", 5.0))
        while time.time() < deadline:
            if self._state.stop_epoch != epoch:
                return False, f"stop engaged while waiting for the platform to confirm {action}"
            with self._lock:
                confirmed = self._vendor_action == action
            if confirmed:
                return True, f"platform confirmed {action}"
            time.sleep(0.05)
        return False, f"platform did not confirm {action} within the timeout"

    def _on_get_status(self, _request, response):
        response.status = self._state.to_msg(self.get_clock().now().to_msg())
        return response

    def _on_stop(self, request, response):
        policy = str(request.policy or STOP_POLICIES[0])
        response.cancel_latency_s = -1.0
        response.idle_latency_s = -1.0
        response.torque_off_latency_s = -1.0
        if policy not in STOP_POLICIES:
            response.success = False
            response.message = f"unknown stop policy {policy!r}; supported: {list(STOP_POLICIES)}"
            return response
        with self._lock:
            body_pose = self._body_pose
        try:
            plan = projection.resolve_stop_plan(policy, body_pose=body_pose, stop_config=self._vendor.get("stop"))
        except CommandRejected as rejected:
            # A humanoid cannot always honour TORQUE_OFF. Refusing is the honest
            # answer; silently holding instead and reporting success is not.
            response.success = False
            response.message = f"{rejected.reason}: {rejected.detail}"
            return response

        with self._stop_lock:
            started = time.monotonic()
            # (1) Close admission first, atomically: the latch and the idle
            #     mode stop both input callbacks and the output timer before
            #     anything else happens, and outlive a vendor failure below.
            self._state.engage_stop(policy)
            self._modes.commit(IDLE_MODE if IDLE_MODE in self._modes.declared_modes() else self._modes.mode)
            self._clear_commands()
            safe_confirmed = self.publish_safe_command()
            response.cancel_latency_s = time.monotonic() - started

            # (2) Ask the platform to hold (or release), then confirm. A plan
            #     with no vendor action is not an omission: this platform
            #     registers no mode that holds the robot from every posture, so
            #     closing admission is the hold and switching modes would be a
            #     change of behaviour rather than a stop.
            if plan.release_arbitration:
                self._release_input_source()
            if not safe_confirmed and not plan.vendor_action:
                # The latch and the closed admission stand, but the data plane
                # never answered for the word that carries the zero velocity:
                # the safe output is unproven, and success must not claim it.
                # The heartbeat re-sends the word (the safe counter moved), so
                # the zero still goes out the moment the data plane answers.
                detail = "the data plane did not confirm the safe command in time"
                self._set_fault("stop", f"{detail}; command admission stays closed (stop latched)")
                response.idle_latency_s = time.monotonic() - started
                response.success = False
                response.message = f"{detail}; stop latched locally, safe output unconfirmed"
                return response
            if not plan.vendor_action:
                response.idle_latency_s = time.monotonic() - started
                self._publish_status()
                response.success = True
                response.message = (
                    "stopped: command admission closed and an explicit safe command published; "
                    "the platform holds its current motion mode (no registered hold action)"
                )
                return response
            self._note_action_request()
            result = self._set_action.call(
                vg.set_action_request(self, source=self._source, action_name=plan.vendor_action)
            )
            elapsed = time.monotonic() - started
            if policy == "TORQUE_OFF":
                response.torque_off_latency_s = elapsed
            response.idle_latency_s = elapsed
            if not result.ok:
                # The local guarantees stand: nothing is admitted or published
                # until idle is requested. Only the platform-side guarantee is
                # unconfirmed, and the answer says exactly that.
                detail = (
                    f"platform rejected stop action {plan.vendor_action}: "
                    f"{projection.mode_rejection_reason(result.code)}"
                )
                if not safe_confirmed:
                    detail += "; the data plane did not confirm the safe command either"
                self._set_fault("stop", f"{detail}; command admission stays closed (stop latched)")
                response.success = False
                response.message = f"{detail}; stop latched locally, platform-side stop unconfirmed"
                return response
            if not safe_confirmed:
                # The platform holds, but the data plane never answered for the
                # zero velocity: one of the two guarantees is unproven, so the
                # answer is a failure that names which.
                detail = "the data plane did not confirm the safe command in time"
                self._set_fault("stop", f"{detail}; command admission stays closed (stop latched)")
                response.success = False
                response.message = (
                    f"platform stopped via {plan.vendor_action}, but {detail}; "
                    "stop latched locally, safe output unconfirmed"
                )
                return response
            self._publish_status()
            response.success = True
            response.message = f"stopped via {plan.vendor_action}" + (
                f" (requested {plan.downgraded_from}; platform-safe equivalent)" if plan.downgraded_from else ""
            )
            return response

    def _release_input_source(self) -> None:
        with self._lock:
            self._arbitration_ok = False

    def _revoke_output(self) -> None:
        """Drop every cached command and leave an explicit zero behind."""
        self._clear_commands()
        self.publish_safe_command()

    # --- interaction services ----------------------------------------------

    def _on_speak(self, request, response):
        response.utterance_id = ""
        if self._state.stop_latched:
            response.success, response.error_code = False, "STOP_LATCHED"
            response.message = "stop latched"
            return response
        try:
            vendor_request = vg.tts_request(
                self,
                text=str(request.text),
                priority=int(request.priority),
                interrupt=bool(request.interrupt),
                trace_id=str(request.trace_id),
                domain=str((self._vendor.get("tts") or {}).get("domain", "ibrobot")),
            )
        except CommandRejected as rejected:
            response.success, response.error_code, response.message = False, rejected.reason, rejected.detail
            return response
        if not str(request.text).strip():
            response.success, response.error_code = False, "INVALID_REQUEST"
            response.message = "text is empty"
            return response
        result = self._play_tts.call(vendor_request)
        response.success = result.ok
        response.error_code = "" if result.ok else "REJECTED"
        response.message = result.message
        if result.ok and result.response is not None:
            response.utterance_id = str(getattr(result.response.tts_resp, "trace_id", "") or request.trace_id)
        return response

    def _on_expression(self, request, response):
        table = {str(k): int(v) for k, v in (self._vendor.get("expressions") or {}).items()}
        try:
            emotion_id = projection.resolve_named(str(request.expression), table, reason="UNKNOWN_EXPRESSION")
        except CommandRejected as rejected:
            response.success, response.error_code, response.message = False, rejected.reason, rejected.detail
            return response
        mode = int(request.mode)
        if mode not in (PlayExpression.Request.MODE_ONCE, PlayExpression.Request.MODE_LOOP):
            response.success, response.error_code = False, "INVALID_REQUEST"
            response.message = f"mode {mode} is neither MODE_ONCE nor MODE_LOOP"
            return response
        result = self._play_emoji.call(
            vg.emoji_request(
                self,
                emotion_id=emotion_id,
                loop=mode == PlayExpression.Request.MODE_LOOP,
                priority=int(request.priority),
            )
        )
        response.success = result.ok
        response.error_code = "" if result.ok else "REJECTED"
        response.message = result.message
        return response

    def _on_led(self, request, response):
        pattern = int(request.pattern)
        # Only a call that took effect may claim an effective priority: the
        # field means "priority effective after this call", so echoing the
        # request on a refusal would tell the caller it won when it lost.
        response.active_priority = 0
        if pattern not in vg.LED_PATTERNS:
            response.success, response.error_code = False, "UNSUPPORTED_PATTERN"
            response.message = f"pattern {pattern} is not supported"
            return response
        result = self._set_led.call(
            vg.led_request(
                self,
                pattern=vg.LED_PATTERNS[pattern],
                rgb=(int(request.r), int(request.g), int(request.b)),
                priority=int(request.priority),
                preempt=bool(request.preempt),
                trace_id=str(request.trace_id),
            )
        )
        status_code = int(getattr(result.response, "status_code", 0)) if result.response is not None else -1
        # 0x1024: the platform's own light manager outranked this request.
        if status_code == 0x1024:
            response.success, response.error_code = False, "INSUFFICIENT_PRIORITY"
            response.message = (
                "platform light manager holds a higher priority; it does not report the "
                "effective value in this response (it is published on the vendor's led_state topic)"
            )
            return response
        response.success = result.ok and status_code == 0
        response.error_code = "" if response.success else "REJECTED"
        response.message = result.message
        if response.success:
            response.active_priority = int(request.priority)
        return response

    def _on_relocalize(self, request, response):
        from aimdk_robot.localization import relocalize  # noqa: PLC0415

        return relocalize(self, request, response)

    # --- command channels ---------------------------------------------------

    def _clear_commands(self) -> None:
        """Have the data plane drop every cached command; a named motion's progress counts as moved."""
        with self._lock:
            self._arm_motion_epoch += 1
            self._control.clear += 1
        self._push_control()

    def _hand_mismatch(self, mode: str) -> str:
        """Why ``mode`` cannot carry this robot's hands, or "".

        A profile whose hand family differs from one the platform reports is
        not a fault of the whole runtime — speech, presets and the other modes
        do not involve the hands — but every upper-body command in the
        hands-carrying action would be dropped by the platform, arms included.
        So only that action is refused, naming the reason.
        """
        if str((self._vendor.get("mc_actions") or {}).get(mode, "")) != projection.UPPER_BODY_HANDS_ACTION:
            return ""
        declared = "claw" if self._hand_type == "claw" else "dexterous"
        with self._lock:
            reported = list(self._dataplane.hand_types)
        if {vg.hand_type_family(value) for value in reported} - {None, declared}:
            names = [vg.hand_type_name(value) for value in reported]
            return (
                f"hand type mismatch: profile declares {declared!r}, platform reports {names}; "
                f"its upper-body commands in {mode!r} would be dropped"
            )
        return ""

    def publish_safe_command(self, *, wait_s: float = 0.2) -> bool:
        """An explicit zero velocity from the data plane (the tier has no fail-safe).

        Waits up to ``wait_s`` for the data plane to acknowledge the word that
        asks for it, which is the same word that closes the output when a stop
        or a revoke is behind the call. Returns whether it acknowledged: a stop
        or a revoke that hears nothing back has not proven the output closed,
        and must not be reported as one.
        """
        with self._lock:
            self._control.safe += 1
        return self._push_control(wait_s=wait_s)

    # --- the data plane -----------------------------------------------------

    def _control_word_locked(self) -> dataplane.Control:
        """The current control word, with the next sequence number. Caller holds _lock."""
        spec = self._modes.spec()
        envelope = self._speed_envelope
        self._control.seq += 1
        self._control.mode = self._modes.mode
        self._control.allows_stream = bool(spec.allows_stream)
        self._control.admitted = self.output_admitted()
        self._control.navigation_enabled = self._navigation_enabled
        self._control.arm_owner = self._arm_owner
        self._control.envelope = (
            (envelope.max_linear_mps, envelope.max_angular_radps) if envelope is not None and envelope.usable else None
        )
        return self._control

    def _push_control(self, *, wait_s: float = 0.0) -> bool:
        """Send the data plane what is admitted now; with ``wait_s``, wait for it to apply it.

        Built and sent under _lock, so a word computed before a stop cannot
        overtake the stop's own word. Returns whether the data plane confirmed
        applying this word (always False without ``wait_s``). The confirmation
        binds this send's own sequence number, captured before the wait: the
        word object is shared with later sends, but a status that confirms a
        newer word confirms this one too — every later word carries it.
        """
        if getattr(self, "_control_pub", None) is None:
            return False  # still wiring up; the heartbeat sends the first word
        with self._lock:
            word = self._control_word_locked()
            seq = word.seq
            self._control_pub.publish(String(data=word.encode()))
            if wait_s <= 0.0:
                return False
            return self._dataplane_seen.wait_for(lambda: self._dataplane.control_seq >= seq, timeout=wait_s)

    def _on_dataplane_status(self, message: String) -> None:
        try:
            status = dataplane.Status.decode(message.data)
        except (ValueError, KeyError, TypeError) as error:
            self.get_logger().error(f"ignoring a malformed data plane status: {error}")
            return
        with self._lock:
            self._dataplane = status
            # When the data plane generated it, not when this callback ran: a
            # report queued past the freshness window stays stale here instead
            # of being dated "now" the moment it is read.
            self._dataplane_at = status.at
            self._dataplane_seen.notify_all()
        for group, codes in status.faults.items():
            self._report_joint_faults(group, codes)
        if status.rejected:
            for note in status.rejected:
                self._modes.note_rejected(note)
            self._publish_status()

    def _dataplane_reporting(self) -> bool:
        """Whether the data plane's status is recent enough to judge feedback by."""
        with self._lock:
            return time.monotonic() - self._dataplane_at <= self._feedback_window_s

    def _feedback_ages_locked(self) -> dict[str, float]:
        """Seconds since each joint group last reported, end to end. Caller holds _lock.

        The group's age when the data plane generated its report, plus the time
        since that moment — the report's queue time included, not just the time
        since this process happened to read it.
        """
        since = time.monotonic() - self._dataplane_at
        return {group: age + since for group, age in self._dataplane.group_ages.items()}

    def _report_joint_faults(self, group: str, codes: list[tuple[str, int]]) -> None:
        """A joint the platform reports faulted degrades the runtime, for the groups
        the profile lists in ``vendor.feedback.fault_groups``.

        One fault source per joint group, kept apart from the ``hand`` source that
        owns hand-type verification: a faulted joint is a hardware condition on a
        correctly declared robot. The fault names every faulted joint with its
        vendor code and clears when the platform reports the group healthy again;
        a group that stops reporting keeps its last verdict.
        """
        if group not in self._fault_groups:
            return
        faulted = sorted((name, code) for name, code in codes if code != 0)
        detail = ", ".join(f"{name}={code}" for name, code in faulted)
        self._set_fault(f"{group}_joints", f"{group} joint fault: {detail}" if faulted else None)

    def _check_feedback(self) -> None:
        """A required feedback source that went quiet degrades the runtime.

        Judged on the data plane's report, so a data plane that stops reporting
        ages every group with it.
        """
        with self._lock:
            ages = self._feedback_ages_locked()
        missing = sorted(
            group for group in self._required_feedback if ages.get(group, float("inf")) > self._feedback_window_s
        )
        if missing and not self._dataplane_reporting():
            self._set_fault("feedback", "joint feedback stale: the data plane (aimdk_dataplane) is not reporting")
        elif missing:
            self._set_fault("feedback", f"joint feedback stale: {', '.join(missing)}")
        else:
            self._set_fault("feedback", None)

    def _on_mc_state(self, message: Any) -> None:
        """Read the whole state message, not the two fields the bridge started with.

        Every field here arrives on a topic already subscribed, so using them
        costs no extra traffic and removes the lag of asking for the same facts
        over services.
        """
        action = str(message.action_info.action_desc or "")
        if not action:
            action = vg.MC_ACTION_BY_VALUE.get(int(message.action_info.current_action.value), "")
        stamp = message.header.stamp
        remote_s = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        skew_s = self.get_clock().now().nanoseconds * 1e-9 - remote_s if remote_s > 0.0 else None
        holder = str(getattr(message.input_source, "name", "") or "")
        holder_priority = int(getattr(message.input_source, "priority", 0) or 0)
        fsm_state = int(getattr(message.fsm_state, "current_state", projection.FSM_STATE_UNKNOWN))
        envelope = vg.speed_envelope(message.speed_status)
        player_state = int(getattr(getattr(message.motion_status, "player_state", None), "value", 0))
        hand_statuses = vg.hand_statuses(message.runtime_model)
        with self._lock:
            self._vendor_action = action
            self._vendor_action_running = int(message.action_info.status.value) == vg.MC_ACTION_RUNNING
            self._vendor_action_at = time.monotonic()
            self._body_pose = int(message.body_status.value)
            self._speed_envelope = envelope
            self._player_state = player_state
            self._hand_statuses = hand_statuses
            if skew_s is not None:
                # One sample is offset + however long this callback waited, so
                # it is kept with its arrival time and judged against its peers
                # rather than on its own (see projection.clock_skew_estimate).
                self._clock_samples.append((time.monotonic(), skew_s))
        self._note_arbitration_holder(holder, holder_priority)
        self._publish_diagnostics(
            [(projection.fsm_diagnostic(fsm_state), str(fsm_state), DiagnosticStatus.OK)],
            prefix="aimdk/fsm_state",
        )

    def _note_arbitration_holder(self, holder: str, holder_priority: int) -> None:
        """Apply the holder the platform reports with every state message.

        The same fact the arbitration poll asks for arrives here ten times a
        second, so a preemption is seen at once instead of up to a poll period
        later — and every command published in that window would have been
        discarded.
        """
        if not holder or holder == self._source:
            return
        if holder_priority < self._source_priority:
            return
        with self._lock:
            already_lost = not self._arbitration_ok
            self._arbitration_ok = False
        if not already_lost:
            self._revoke_output()
        self._set_fault(
            "arbitration",
            f"command arbitration held by {holder!r} at priority {holder_priority} "
            f"(this runtime registers {self._source_priority}); streamed commands are not applied",
        )

    def _check_clock_skew(self) -> None:
        """Compare this host's clock with the platform's own state stamps."""
        horizon = time.monotonic() - _CLOCK_SKEW_WINDOW_S
        with self._lock:
            samples = [skew for received, skew in self._clock_samples if received >= horizon]
        estimate = projection.clock_skew_estimate(samples)
        if estimate is None:
            # No recent platform stamps. Silence is a feedback problem, which
            # _check_feedback reports; it is not evidence about the clock.
            return
        detail = projection.clock_skew_fault(estimate, limit_s=self._clock_skew_limit_s, window_s=self._stamp_window_s)
        self._set_fault("clock", detail, lifecycle=LIFECYCLE_FAULTED)

    def _on_system_state(self, message: Any) -> None:
        state = str(message.cur_state or "")
        with self._lock:
            previous, self._system_state = self._system_state, state
        if state == previous:
            return
        blocking = set(
            self._vendor.get("blocking_system_states") or ["Develop_MC", "EStop", "OTA", "Poweroff", "Reboot"]
        )
        if state in blocking:
            self._revoke_output()
            self._set_fault(
                "system",
                f"platform system mode {state!r} disables the interfaces this runtime depends on",
                lifecycle=LIFECYCLE_DEGRADED if state == "EStop" else LIFECYCLE_FAULTED,
            )
        else:
            self._set_fault("system", None)

    def _on_power_state(self, message: Any) -> None:
        power = PowerState()
        power.header.stamp = self.get_clock().now().to_msg()
        power.battery_present = True
        power.battery_percentage = float(message.battery_remaining_capacity_percentage)
        power.battery_voltage = float(message.battery_pack_voltage)
        power.battery_current = float(message.battery_current)
        power.battery_temperature = float(message.battery_temperature)
        power.battery_power = float(message.battery_output_power)
        power.battery_cycle_count = int(message.battery_cycle_count)
        power.charging = float(message.battery_current) > 0.0
        power.temperature = float(message.pmu_temperature)
        power.fan_speed_rpm = float(message.fan_speed)
        power.fan_percentage = float(message.fan_pecentage)
        power.rail_names = ["48v_bus", "12v_output", "orin", "rk3588"]
        power.rail_voltages = [
            float(message.bus_48v_voltage),
            float(message.output_12v_voltage),
            float(message.orin_voltage),
            float(message.rk3588_voltage),
        ]
        power.rail_currents = [
            float(message.bus_48v_current),
            float(message.output_12v_current),
            float(message.orin_current),
            float(message.rk3588_current),
        ]
        self._power_pub.publish(power)

    def _on_diagnostics(self, message: Any) -> None:
        self._publish_diagnostics(
            [(str(entry.info), f"{int(entry.diag_code)}", DiagnosticStatus.OK) for entry in message.diagnostics],
            prefix="aimdk/diagnostic",
        )

    def _on_alerts(self, message: Any) -> None:
        self._publish_diagnostics(
            [("platform alert", str(entry.code), DiagnosticStatus.ERROR) for entry in message.list],
            prefix="aimdk/alert",
        )

    def _publish_diagnostics(self, entries: list[tuple[str, str, int | bytes]], *, prefix: str) -> None:
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        for detail, code, level in entries:
            status = DiagnosticStatus()
            status.name = f"{prefix}/{code}"
            status.hardware_id = self._state.name
            status.level = level if isinstance(level, bytes) else bytes([level])
            status.message = detail
            status.values = [KeyValue(key="code", value=code)]
            array.status.append(status)
        self._diag_pub.publish(array)

    def _on_audio_capture(self, message: Any) -> None:
        stamped = self._audio_stamped_type()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.audio.data = list(message.data.data)
        self._audio_pub.publish(stamped)
        info = self._audio_info_type()
        info.channels = int(message.info.channels)
        info.sample_rate = int(message.info.sample_rate)
        info.sample_format = str(message.info.sample_format)
        info.coding_format = str(message.info.coding_format)
        self._audio_info_pub.publish(info)

    def _on_audio_play(self, message: Any) -> None:
        chunk = list(message.audio.data)
        with self._audio_lock:
            if self._focus_held:
                publish = True
            else:
                # Hold the head of the stream rather than drop it: focus takes
                # a service round trip, and an utterance that starts mid-word
                # is worse than one that starts a fraction of a second late.
                self._playback_buffer.append(chunk)
                publish = False
        if publish:
            self._publish_playback(chunk)

    def _publish_playback(self, chunk: list[int]) -> None:
        audio = self._vendor.get("audio") or {}
        channels = int(audio.get("playback_channels", 1))
        sample_rate = int(audio.get("playback_sample_rate", 16000))
        sample_format = str(audio.get("playback_sample_format", "S16LE"))
        playback = self._playback_type()
        playback.stamps = self.get_clock().now().to_msg()
        playback.info.channels = channels
        playback.info.sample_rate = sample_rate
        playback.info.sample_format = sample_format
        playback.info.coding_format = str(audio.get("playback_coding_format", "pcm"))
        playback.data.data = chunk
        # pkg_name identifies the playback source; a change flushes the
        # platform's buffer, so every chunk of one stream carries the same one.
        playback.pkg_name = self._source
        self._playback_pub.publish(playback)
        # The player renders what it is handed in real time, after whatever it already holds.
        frame_bytes = channels * _PCM_SAMPLE_BYTES.get(sample_format, 2)
        with self._audio_lock:
            self._render_until = max(time.monotonic(), self._render_until) + len(chunk) / (sample_rate * frame_bytes)

    def _audio_focus_tick(self) -> None:
        """Hold audio focus while forwarded audio is still playing, and ask for it only to play audio.

        ``hal_audio`` never requests focus on a publisher's behalf, so nothing
        published here is guaranteed to be heard until this succeeds. Releasing
        focus makes it abandon and flush whatever it still holds, so focus is
        kept until everything forwarded has been rendered, plus
        ``focus_release_idle_s``; holding it longer would keep preempting
        whatever else the platform wants to say.
        """
        now = time.monotonic()
        with self._audio_lock:
            buffered = bool(self._playback_buffer)
            playing = now < self._render_until + self._focus_release_idle_s
            held = self._focus_held
            retry_due = (now - self._focus_attempt_s) >= _AUDIO_FOCUS_RETRY_S
        # A refusal drops the buffer: asking again with nothing to play would only take the
        # speaker for silence once the higher-priority source lets go.
        if buffered and not held and retry_due:
            self._acquire_audio_focus()
        elif held and not (buffered or playing):
            self.release_audio_focus()

    def _acquire_audio_focus(self) -> None:
        self._focus_attempt_s = time.monotonic()
        result = self._request_focus.call(
            vg.audio_focus_request(
                self, pkg_name=self._source, priority=self._focus_priority, weight=self._focus_weight
            )
        )
        if not (result.ok and vg.focus_granted(result.response)):
            # The status code says only that the request was processed, so the
            # grant flag is the answer. Refused means something with a higher
            # priority owns the speaker; buffered audio is already stale.
            with self._audio_lock:
                self._playback_buffer.clear()
            self.get_logger().warning(
                f"audio focus refused (priority {self._focus_priority}): playback is not reaching the speaker",
                throttle_duration_sec=10.0,
            )
            return
        with self._audio_lock:
            self._focus_held = True
            pending = list(self._playback_buffer)
            self._playback_buffer.clear()
        for chunk in pending:
            self._publish_playback(chunk)

    def release_audio_focus(self) -> None:
        """Give the speaker back. All three requester fields must match."""
        if self._abandon_focus is None:
            return
        with self._audio_lock:
            if not self._focus_held:
                return
            self._focus_held = False
            self._playback_buffer.clear()
        self._abandon_focus.call(
            vg.audio_focus_release(
                self, pkg_name=self._source, priority=self._focus_priority, weight=self._focus_weight
            )
        )

    def _on_audio_focus(self, message: Any) -> None:
        if str(message.pkg_name) != self._source:
            return
        gained = bool(message.focus_gain)
        with self._audio_lock:
            if self._focus_held == gained:
                return
            self._focus_held = gained
            if not gained:
                self._playback_buffer.clear()
        if not gained:
            self.get_logger().warning("audio focus lost to a higher-priority source; playback stopped")

    def _on_localization(self, message: Any) -> None:
        pose = PoseWithCovarianceStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = str(message.header.frame_id or self._localization_frame)
        pose.pose.pose = message.pose.pose
        pose.pose.covariance = message.pose.covariance
        self._localization_pub.publish(pose)

    # --- status -------------------------------------------------------------

    def _publish_status(self) -> None:
        self._status_pub.publish(self._state.to_msg(self.get_clock().now().to_msg()))
        # Every lifecycle, mode and stop change comes through here: the data plane
        # learns of a closed admission now rather than at the next heartbeat.
        self._push_control()

    # --- accessors used by helper modules ----------------------------------

    @property
    def profile(self) -> dict[str, Any]:
        return self._profile

    @property
    def runtime_state(self) -> RuntimeState:
        return self._state

    @property
    def vendor_config(self) -> dict[str, Any]:
        return self._vendor

    @property
    def preset_motion_caller(self) -> vg.ServiceCaller:
        return self._preset_motion

    @property
    def preset_state_caller(self) -> vg.ServiceCaller:
        return self._preset_state

    @property
    def action_caller(self) -> vg.ServiceCaller:
        return self._set_action

    @property
    def service_callback_group(self) -> Any:
        return self._service_cb

    @property
    def action_callback_group(self) -> Any:
        """Long-running action execution must not block the mode/stop services."""
        return self._client_cb

    @property
    def input_source(self) -> str:
        return self._source

    def ensure_input_source(self) -> bool:
        """Re-assert the vendor input source claim, as the SDK's clients do.

        A registration made at startup is not permanent — a stop releases it —
        and the platform discards commands from a source that is not currently
        enabled. Taking the claim again costs one service round trip and is
        what the vendor's own preset-motion client does before every dispatch.
        """
        return self._register_input_source()

    def current_body_pose(self) -> int:
        with self._lock:
            return self._body_pose

    def platform_action(self) -> str:
        """The MC action the platform itself reports, not the mode we committed."""
        with self._lock:
            return self._vendor_action

    def current_mode(self) -> str:
        return self._modes.mode

    def measured_arm_positions(self) -> list[float] | None:
        """Fresh measured arm positions in vendor order, or None.

        Fresh by the end-to-end age: how old the measurement was when the data
        plane reported it, plus how long that report took to arrive here.
        """
        with self._lock:
            arm = self._dataplane.arm_measured
            age = self._dataplane.arm_measured_age_s
            if arm is None or age is None:
                return None
            since = time.monotonic() - self._dataplane_at
            return list(arm) if age + since <= self._feedback_window_s else None

    def claim_arm(self, owner: int) -> bool:
        """Give the arm channel to a named motion; the public arm stream is refused meanwhile.

        False while the public arm stream (teleop, grasping) is live: taking the
        arm from it mid-stream would leave its next absolute target to jump the
        arm back once the motion released it. The data plane judges that and
        applies the owner in one step — a status snapshot here may predate a
        target it has not shown yet — so the claim counts only once the data
        plane confirms the word and echoes the owner back. A motion replacing
        another one takes the channel over.
        """
        with self._lock:
            self._arm_owner = owner
            self._arm_motion_epoch += 1
        confirmed = self._push_control(wait_s=0.2)
        with self._lock:
            if confirmed and self._dataplane.arm_owner == owner:
                return True
            if self._arm_owner == owner:  # nobody took over meanwhile: take it back
                self._arm_owner = None
        self._push_control()
        return False

    def arm_motion_epoch(self) -> int:
        """A counter that moves whenever something may have moved the arm; see ``__init__``."""
        with self._lock:
            return self._arm_motion_epoch + self._dataplane.public_arm_count

    def release_arm(self, owner: int) -> None:
        with self._lock:
            if self._arm_owner != owner:
                return
            self._arm_owner = None
        self._push_control()

    def command_arm(self, positions: list[float], *, owner: int, epoch: int) -> bool:
        """Set the arm target for the fixed-rate publisher, if ``owner`` still may.

        False once the owner lost the channel, a stop engaged since ``epoch``,
        or output is no longer admitted: the caller must stop, not retry.
        """
        with self._lock:
            if self._arm_owner != owner or self._state.stop_epoch != epoch or not self.output_admitted():
                return False
        self._named_arm_pub.publish(Float64MultiArray(data=[float(owner), *(float(value) for value in positions)]))
        return True


def main(argv=None) -> int:
    rclpy.init(args=argv if argv is not None else sys.argv)
    try:
        node = AimdkRuntimeNode()
    except Exception as exc:  # noqa: BLE001 - startup failure must be legible
        print(f"aimdk_runtime failed to start: {exc}", file=sys.stderr)
        rclpy.shutdown()
        return 1
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # The vendor command tier has no fail-safe: leave an explicit zero
        # behind even when shutdown is already going wrong.
        with contextlib.suppress(Exception):
            node.publish_safe_command()
        with contextlib.suppress(Exception):
            node.release_audio_focus()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
