"""The ROS adapter: the frame buffer, and what the node publishes.

The decisions live in ``visual_interaction.session`` and are tested without
ROS. What is left here is the plumbing, so these tests cover the two things
plumbing gets wrong: which clock a window is built from, and whether a quiet
round still reports itself.
"""

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("ibrobot_msgs.msg")

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
            "categories",
        ):
            assert name in PARAMETERS

    def test_the_defaults_carry_the_type_rclpy_enforces(self):
        assert isinstance(PARAMETERS["runtime_description_json"], str)
        assert isinstance(PARAMETERS["frames_per_window"], int)
        assert isinstance(PARAMETERS["window_sec"], float)
        assert isinstance(PARAMETERS["categories"], list)

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
            with pytest.raises(ValueError, match="only 'fake' is wired"):
                VisualInteractionNode(
                    parameter_overrides=[
                        Parameter("runtime_description_json", value=json.dumps({"interfaces": {}})),
                        Parameter("vl_implementation", value="internvl"),
                    ]
                )
        finally:
            rclpy.shutdown()
