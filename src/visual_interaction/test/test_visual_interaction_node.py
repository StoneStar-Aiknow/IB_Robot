"""The ROS adapter: the frame buffer, and what the node publishes.

The decisions live in ``visual_interaction.session`` and are tested without
ROS. What is left here is the plumbing, so these tests cover the two things
plumbing gets wrong: which clock a window is built from, and whether a quiet
round still reports itself.
"""

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("ibrobot_msgs.msg")

from visual_interaction.session.endpoints import CAMERA_COMPRESSED  # noqa: E402
from visual_interaction.session.hooks import Window  # noqa: E402
from visual_interaction.visual_interaction_node import PARAMETERS, FrameBuffer  # noqa: E402


class TestFrameBuffer:
    def test_it_is_empty_before_anything_arrives(self):
        assert FrameBuffer(6.0).window(2.0, 4) is None

    def test_a_window_is_evenly_spaced_and_newest_last(self):
        buffer = FrameBuffer(6.0)
        for index in range(10):
            buffer.put(100.0 + index * 0.1, 200.0 + index * 0.1, f"f{index}".encode())

        window = buffer.window(1.0, 4)

        assert isinstance(window, Window)
        assert len(window.frames) == 4
        assert window.stamps == tuple(sorted(window.stamps))
        assert window.newest_stamp == pytest.approx(100.9)

    def test_a_thin_stream_repeats_frames_rather_than_failing(self):
        """The model's input shape is fixed, and a repeat shows in the stamps."""
        buffer = FrameBuffer(6.0)
        buffer.put(100.0, 200.0, b"only")

        window = buffer.window(2.0, 4)

        assert len(window.frames) == 4
        assert set(window.frames) == {b"only"}
        assert window.span == 0.0

    def test_eviction_and_sampling_use_the_same_clock(self):
        """Mixing an arrival clock into eviction silently shortens the window.

        The board demo evicts by arrival and samples by camera stamp, so the
        two disagreeing costs it frames it believes it has.
        """
        buffer = FrameBuffer(1.0)
        # Camera stamps advance by 0.5s; arrival times by 5s. If eviction used
        # arrival, everything but the newest frame would already be gone.
        for index in range(4):
            buffer.put(100.0 + index * 0.5, 200.0 + index * 5.0, f"f{index}".encode())

        window = buffer.window(2.0, 4)

        assert len({*window.frames}) > 1

    def test_an_old_frame_is_evicted_by_camera_stamp(self):
        buffer = FrameBuffer(1.0)
        buffer.put(100.0, 200.0, b"old")
        buffer.put(105.0, 205.0, b"new")

        window = buffer.window(10.0, 2)

        assert set(window.frames) == {b"new"}

    def test_received_at_is_the_newest_arrival_not_a_camera_stamp(self):
        """Liveness is judged by arrival, which needs no clock agreement."""
        buffer = FrameBuffer(6.0)
        buffer.put(100.0, 200.0, b"a")
        buffer.put(100.5, 207.0, b"b")

        assert buffer.window(2.0, 2).received_at == 207.0

    def test_one_frame_per_window_takes_the_newest(self):
        buffer = FrameBuffer(6.0)
        buffer.put(100.0, 200.0, b"old")
        buffer.put(100.5, 200.5, b"new")

        assert buffer.window(2.0, 1).frames == (b"new",)

    def test_the_total_counts_every_frame_including_evicted_ones(self):
        buffer = FrameBuffer(0.1)
        for index in range(5):
            buffer.put(100.0 + index, 200.0 + index, b"f")

        assert buffer.total == 5


class TestDeclaredParameters:
    def test_a_declared_default_accepts_the_value_launch_overrides_it_with(self):
        """The test that was missing, and it has to use parameter_overrides.

        ``categories`` was declared from an empty list, so rclpy inferred
        BYTE_ARRAY, and the first launch carrying real categories died with
        "expecting type 'BYTE_ARRAY'" before the node existed. No node test
        caught it because they all pass their own override for that name.

        It also has to go through ``parameter_overrides`` specifically:
        ``set_parameters`` with the same string array is *accepted* on a
        BYTE_ARRAY declaration, so a test written that way passes while launch
        still fails.
        """
        import json

        import rclpy
        from rclpy.parameter import Parameter

        rclpy.init()
        try:
            # What the launch builder actually sends for every parameter.
            overrides = [
                Parameter("runtime_description_json", value="{}"),
                Parameter("categories_json", value='["toy","sign","hat","badge"]'),
                Parameter("vl_implementation", value="fake"),
            ]
            node = rclpy.create_node("parameter_override_probe", parameter_overrides=overrides)
            try:
                for name, default in PARAMETERS.items():
                    node.declare_parameter(name, default)
                assert json.loads(node.get_parameter("categories_json").value) == [
                    "toy",
                    "sign",
                    "hat",
                    "badge",
                ]
            finally:
                node.destroy_node()
        finally:
            rclpy.shutdown()

    def test_every_name_the_launch_builder_passes_is_declared(self):
        """rclpy drops an override it has no declaration for without a warning."""
        for name in (
            "runtime_description_json",
            "camera_interface",
            "actor_state_topic",
            "vl_implementation",
            "confirmation_count",
            "max_evidence_age_sec",
            "min_gap_sec",
            "absent_reset_sec",
            "max_session_sec",
            "categories_json",
        ):
            assert name in PARAMETERS

    def test_the_defaults_carry_the_type_rclpy_enforces(self):
        assert isinstance(PARAMETERS["runtime_description_json"], str)
        assert isinstance(PARAMETERS["frames_per_window"], int)
        assert isinstance(PARAMETERS["window_sec"], float)
        # JSON rather than a ROS string array: an empty list has no
        # inferable element type, so rclpy declared BYTE_ARRAY and the
        # node died at startup on the launch builder's strings.
        assert isinstance(PARAMETERS["categories_json"], str)

    def test_the_default_model_is_the_scripted_one(self):
        """The real loop waits for phase zero, so the skeleton stays launchable."""
        assert PARAMETERS["vl_implementation"] == "fake"


CAMERA_ENDPOINT = "/aima/hal/sensor/stereo_head_front_left/rgb_image/compressed"

# rclpy gives every node these, and they cannot be declined.
RCLPY_SERVICES = frozenset(
    {
        "describe_parameters",
        "get_parameter_types",
        "get_parameters",
        "list_parameters",
        "set_parameters",
        "set_parameters_atomically",
    }
)
ROS_TOPICS = frozenset({"/parameter_events", "/rosout"})


@pytest.fixture
def node():
    """A live node, built without a launch file."""
    import json

    from rclpy.parameter import Parameter

    from visual_interaction.visual_interaction_node import VisualInteractionNode

    description = {
        "robot": {
            "id": "x2",
            "type": "x2_ultra_omnihand",
            "runtime_name": "aimdk",
            "runtime_version": "1",
        },
        "interfaces": {
            "camera.stereo_head_front_left.color_compressed": {
                "kind": "topic",
                "direction": "publish",
                "message_type": "sensor_msgs/msg/CompressedImage",
                "endpoint": CAMERA_ENDPOINT,
                "qos": {"reliability": "reliable", "history": "keep_last", "depth": 5},
            },
            # Declared by the runtime, and this node must still never touch it.
            "joint.arm_stream": {
                "kind": "topic",
                "direction": "subscribe",
                "message_type": "std_msgs/msg/Float64MultiArray",
                "endpoint": "/aima/hal/joint/arm_stream",
            },
        },
    }

    rclpy.init()
    live = VisualInteractionNode(
        parameter_overrides=[Parameter("runtime_description_json", value=json.dumps(description))]
    )
    try:
        yield live
    finally:
        live.destroy_node()
        rclpy.shutdown()


class TestNoMotionSideEffect:
    """What the node exposes is the evidence, not what it promises.

    The runtime publishes a control plane whatever this feature does, so "no
    executor was started" proves nothing. These assertions fail the moment
    somebody wires a motion endpoint into an observation-only entry point.
    """

    def test_it_subscribes_to_the_camera_and_nothing_else(self, node):
        subscribed = {name for name, _ in node.get_subscriber_names_and_types_by_node(node.get_name(), "/")}

        assert subscribed - ROS_TOPICS == {CAMERA_ENDPOINT}

    def test_it_publishes_only_its_own_utterance_and_diagnostics(self, node):
        published = {name for name, _ in node.get_publisher_names_and_types_by_node(node.get_name(), "/")}

        assert published - ROS_TOPICS == {"/visual_interaction/utterance", "/visual_interaction/status"}

    def test_it_serves_only_the_quiet_request(self, node):
        served = {name.rsplit("/", 1)[-1] for name, _ in node.get_service_names_and_types_by_node(node.get_name(), "/")}

        assert served - RCLPY_SERVICES == {"set_enabled"}

    def test_it_holds_no_runtime_mode_and_no_control_client(self, node):
        clients = {name for name, _ in node.get_client_names_and_types_by_node(node.get_name(), "/")}

        assert clients == set()

    def test_no_endpoint_belongs_to_a_motion_family(self, node):
        from visual_interaction.session.endpoints import FORBIDDEN_PREFIXES

        name = node.get_name()
        endpoints = (
            {topic for topic, _ in node.get_subscriber_names_and_types_by_node(name, "/")}
            | {topic for topic, _ in node.get_publisher_names_and_types_by_node(name, "/")}
            | {topic for topic, _ in node.get_service_names_and_types_by_node(name, "/")}
        )

        for endpoint in endpoints:
            for prefix in FORBIDDEN_PREFIXES:
                family = prefix.rstrip(".")
                assert f"/{family}/" not in endpoint, endpoint
                assert not endpoint.endswith(f"/{family}"), endpoint


class TestQuietRequest:
    def test_false_mutes_and_true_resumes(self, node):
        from std_srvs.srv import SetBool

        muted = node._on_set_enabled(SetBool.Request(data=False), SetBool.Response())
        assert muted.success
        assert muted.message == "muted"

        resumed = node._on_set_enabled(SetBool.Request(data=True), SetBool.Response())
        assert resumed.message == "active"

    def test_an_unknown_vl_implementation_is_refused_at_startup(self):
        """A deployment must not believe it configured a model that is not wired.

        Reported before the runtime description is read, so a typo here is
        named as itself rather than as whatever the description is missing.
        """
        import json

        from rclpy.parameter import Parameter

        from visual_interaction.visual_interaction_node import VisualInteractionNode

        rclpy.init()
        try:
            with pytest.raises(ValueError, match="expected 'fake' or 'internvl'"):
                VisualInteractionNode(
                    parameter_overrides=[
                        Parameter("runtime_description_json", value=json.dumps({"interfaces": {}})),
                        Parameter("vl_implementation", value="llava"),
                    ]
                )
        finally:
            rclpy.shutdown()


def record_step(node) -> dict:
    """Capture what the node hands the session, keeping the round working.

    The real step still runs: _on_round reads the result afterwards, so a stub
    returning something else would fail for the wrong reason.
    """
    seen: dict = {}
    original = node._session.step

    def recording(round_inputs):
        seen["actor"] = round_inputs.actor
        seen["window"] = round_inputs.window
        return original(round_inputs)

    node._session.step = recording
    return seen


class TestActorLiveness:
    """Review finding: the last actor state was held forever.

    The tracker compares the stamps it is handed, so replaying one message
    keeps the difference at zero and the gap never times out — the session
    kept answering a visitor who had left.
    """

    def test_a_stale_actor_state_is_treated_as_unknown(self, node):
        from visual_interaction.session.hooks import ActorSignal

        node._actor = ActorSignal(present=True, known=True, bbox=(100.0, 100.0, 200.0, 400.0), stamp=9.9)
        node._actor_received_at = node._now() - (node._stale_actor_sec + 5.0)

        seen = record_step(node)
        node._buffer.put(node._now(), node._now(), b"frame")
        node._on_round()

        assert seen["actor"].known is False
        assert seen["actor"].present is False

    def test_a_fresh_actor_state_is_passed_through(self, node):
        from visual_interaction.session.hooks import ActorSignal

        box = (100.0, 100.0, 200.0, 400.0)
        node._actor = ActorSignal(present=True, known=True, bbox=box, stamp=node._now())
        node._actor_received_at = node._now()

        seen = record_step(node)
        node._buffer.put(node._now(), node._now(), b"frame")
        node._on_round()

        assert seen["actor"].present is True
        assert seen["actor"].bbox == box

    def test_an_actor_state_that_never_arrived_is_unknown(self, node):
        seen = record_step(node)
        node._buffer.put(node._now(), node._now(), b"frame")
        node._on_round()

        assert seen["actor"].known is False


class TestBufferCap:
    """Review finding: eviction by camera stamp alone evicts nothing.

    A frozen or always-zero stamp then grows a resident process without bound.
    """

    def test_a_frozen_camera_stamp_cannot_grow_the_buffer(self):
        buffer = FrameBuffer(6.0, max_frames=300)

        for _ in range(1000):
            buffer.put(0.0, 0.0, b"x" * 100)

        assert len(buffer._items) == 300
        assert buffer.total == 1000

    def test_the_cap_keeps_the_newest_frames(self):
        buffer = FrameBuffer(600.0, max_frames=3)

        for index in range(10):
            buffer.put(float(index), float(index), f"f{index}".encode())

        assert [item[2] for item in buffer._items] == [b"f7", b"f8", b"f9"]

    def test_a_zero_cap_is_refused(self):
        with pytest.raises(ValueError, match="at least 1"):
            FrameBuffer(6.0, max_frames=0)


class TestPublishedFields:
    """Review finding: the message contradicted its own contract."""

    def _publish_and_capture(self, node, result):
        from visual_interaction.session.hooks import Window

        captured = {}
        node._utterances.publish = lambda message: captured.setdefault("message", message)
        node._publish(result, Window(frames=(b"a",), stamps=(10.9,), received_at=11.0))
        return captured["message"]

    def test_the_header_carries_the_evidence_time_not_the_publish_time(self, node):
        from visual_interaction.session.core import RoundResult

        message = self._publish_and_capture(node, RoundResult(spoken=False, reason="NO_NEW_TOPIC"))

        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        assert stamp == pytest.approx(10.9)

    def test_a_spoken_round_publishes_its_episode_and_deadline(self, node):
        from visual_interaction.session.core import RoundResult

        result = RoundResult(spoken=True, text="你举起的玩偶真有趣。", episode_seq=3, latest_start_at=18.9)
        message = self._publish_and_capture(node, result)

        assert message.episode_seq == 3
        deadline = message.latest_start_at.sec + message.latest_start_at.nanosec * 1e-9
        assert deadline == pytest.approx(18.9)

    def test_a_quiet_round_publishes_no_deadline(self, node):
        from visual_interaction.session.core import RoundResult

        message = self._publish_and_capture(node, RoundResult(spoken=False, reason="NO_NEW_TOPIC"))

        assert message.latest_start_at.sec == 0
        assert message.latest_start_at.nanosec == 0
        assert message.episode_seq == 0


class TestCameraQos:
    """The subscription must match what the interface description declares.

    This is the test that was missing. The node hardcoded BEST_EFFORT while the
    X2 profile declares the camera reliable/transient_local, which is what the
    robot really publishes. An incompatible subscription in ROS 2 is silent:
    no error, no warning, just zero frames forever. Measured on the board,
    BEST_EFFORT/VOLATILE received 0 frames in 7 seconds on a topic that gave
    10 (1.4 Hz, ~288 KB JPEG) over RELIABLE/TRANSIENT_LOCAL.
    """

    def test_the_declared_profile_is_honoured(self):
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy

        from visual_interaction.visual_interaction_node import camera_qos

        qos = camera_qos(
            {"reliability": "reliable", "durability": "transient_local", "history": "keep_last", "depth": 5}
        )

        assert qos.reliability == ReliabilityPolicy.RELIABLE
        assert qos.durability == DurabilityPolicy.TRANSIENT_LOCAL
        assert qos.history == HistoryPolicy.KEEP_LAST
        assert qos.depth == 5

    def test_the_x2_camera_declaration_does_not_become_best_effort(self):
        """The exact regression: this camera is reliable, and must stay so."""
        from rclpy.qos import ReliabilityPolicy

        from visual_interaction.visual_interaction_node import camera_qos

        declared = {"reliability": "reliable", "durability": "transient_local", "history": "keep_last", "depth": 5}

        assert camera_qos(declared).reliability != ReliabilityPolicy.BEST_EFFORT

    def test_a_best_effort_declaration_is_also_honoured(self):
        """Honouring the description means following it either way."""
        from rclpy.qos import ReliabilityPolicy

        from visual_interaction.visual_interaction_node import camera_qos

        qos = camera_qos({"reliability": "best_effort", "durability": "volatile"})

        assert qos.reliability == ReliabilityPolicy.BEST_EFFORT

    @pytest.mark.parametrize(
        ("field", "value"),
        [("reliability", "reliabel"), ("durability", "transient"), ("history", "keep_latest")],
    )
    def test_an_unknown_value_fails_rather_than_downgrading(self, field, value):
        """A profile typo must not quietly produce a subscription that receives nothing."""
        from visual_interaction.visual_interaction_node import camera_qos

        declared = {"reliability": "reliable", "durability": "transient_local", "history": "keep_last"}
        declared[field] = value

        with pytest.raises(ValueError, match=field):
            camera_qos(declared)

    def test_a_nonpositive_depth_is_refused(self):
        from visual_interaction.visual_interaction_node import camera_qos

        with pytest.raises(ValueError, match="depth"):
            camera_qos({"reliability": "reliable", "depth": 0})

    def test_the_live_node_subscribes_with_the_declared_profile(self):
        """End to end: what the description says is what the subscription uses."""
        import json

        import rclpy
        from rclpy.parameter import Parameter
        from rclpy.qos import ReliabilityPolicy

        from visual_interaction.visual_interaction_node import VisualInteractionNode

        description = {
            "robot": {"id": "x2", "type": "x2_ultra_omnihand", "runtime_name": "aimdk", "runtime_version": "1"},
            "interfaces": {
                CAMERA_COMPRESSED: {
                    "kind": "topic",
                    "direction": "publish",
                    "message_type": "sensor_msgs/msg/CompressedImage",
                    "endpoint": CAMERA_ENDPOINT,
                    "qos": {
                        "reliability": "reliable",
                        "durability": "transient_local",
                        "history": "keep_last",
                        "depth": 5,
                    },
                }
            },
        }
        rclpy.init()
        try:
            node = VisualInteractionNode(
                parameter_overrides=[
                    Parameter("runtime_description_json", value=json.dumps(description)),
                    Parameter("categories_json", value='["toy","sign","hat","badge"]'),
                ]
            )
            try:
                subscription = next(sub for sub in node.subscriptions if sub.topic_name == CAMERA_ENDPOINT)
                assert subscription.qos_profile.reliability == ReliabilityPolicy.RELIABLE
            finally:
                node.destroy_node()
        finally:
            rclpy.shutdown()


class TestCameraTopicEscapeHatch:
    """Reading a stream the runtime description does not carry.

    The robot's hri_halfres_relay republishes a resized JPEG at 30.1 Hz and
    ~80 KB per frame, against the described stereo stream's 10.0 Hz and
    ~296 KB. It is not part of the HAL interface contract, so it is named as a
    topic — the same way motion_imitation_stream reads it — and the QoS comes
    with it, since there is no description to take one from.
    """

    HALF_TOPIC = "/aima/hal/sensor/rgb_head_front_center/rgb_image_half/compressed"

    def node(self, **overrides):
        import json

        import rclpy
        from rclpy.parameter import Parameter

        from visual_interaction.visual_interaction_node import VisualInteractionNode

        values = {
            "runtime_description_json": json.dumps({"robot": {}, "interfaces": {}}),
            "categories_json": '["toy","sign","hat","badge"]',
            "camera_topic": self.HALF_TOPIC,
        }
        values.update(overrides)
        rclpy.init()
        try:
            node = VisualInteractionNode(parameter_overrides=[Parameter(k, value=v) for k, v in values.items()])
            return node, rclpy
        except Exception:
            rclpy.shutdown()
            raise

    def test_a_named_topic_is_subscribed_without_a_described_camera(self):
        """The description carries no camera at all here, and that is fine."""
        node, rclpy = self.node()
        try:
            assert any(sub.topic_name == self.HALF_TOPIC for sub in node.subscriptions)
        finally:
            node.destroy_node()
            rclpy.shutdown()

    def test_the_stated_qos_is_used(self):
        """The relay publishes BEST_EFFORT/VOLATILE, unlike the stereo stream."""
        from rclpy.qos import DurabilityPolicy, ReliabilityPolicy

        node, rclpy = self.node()
        try:
            sub = next(s for s in node.subscriptions if s.topic_name == self.HALF_TOPIC)

            assert sub.qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT
            assert sub.qos_profile.durability == DurabilityPolicy.VOLATILE
        finally:
            node.destroy_node()
            rclpy.shutdown()

    def test_a_reliable_relay_would_also_be_honoured(self):
        """Stating the QoS means following it, not assuming best effort."""
        from rclpy.qos import ReliabilityPolicy

        node, rclpy = self.node(
            camera_topic_qos_json='{"reliability": "reliable", "durability": "transient_local", "depth": 5}'
        )
        try:
            sub = next(s for s in node.subscriptions if s.topic_name == self.HALF_TOPIC)

            assert sub.qos_profile.reliability == ReliabilityPolicy.RELIABLE
        finally:
            node.destroy_node()
            rclpy.shutdown()

    def test_a_relative_topic_is_refused(self):
        with pytest.raises(ValueError, match="absolute ROS name"):
            node, rclpy = self.node(camera_topic="rgb_image_half/compressed")
            node.destroy_node()
            rclpy.shutdown()

    def test_malformed_qos_json_is_refused(self):
        with pytest.raises(ValueError, match="camera_topic_qos_json"):
            node, rclpy = self.node(camera_topic_qos_json="not json")
            node.destroy_node()
            rclpy.shutdown()
