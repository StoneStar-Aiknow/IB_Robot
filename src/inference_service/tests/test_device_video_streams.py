from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from inference_service.device_video_streams import DeviceVideoStreamManager
from inference_service.video_codec import (
    CodecCapabilities,
    CodecLifecycleState,
    CodecMetrics,
    EncodedPacket,
    VideoCodecRegistry,
    VideoEncoder,
)
from inference_service.video_rtp import StreamLifecycleState, StreamMetrics, StreamStatus
from robot_config.contract_utils import SpecView
from robot_config.observation_transport import (
    H264Spec,
    ObservationTransportSpec,
    RtpEndpointSpec,
    VideoBufferSpec,
    VideoMediaSpec,
    VideoReadinessSpec,
)


class _Encoder(VideoEncoder):
    def __init__(self, **_options):
        self.frames = []
        self.resets = 0
        self.closed = False

    @property
    def state(self):
        return CodecLifecycleState.CLOSED if self.closed else CodecLifecycleState.RUNNING

    @property
    def metrics(self):
        return CodecMetrics(
            input_frames=len(self.frames), output_frames=len(self.frames), output_packets=len(self.frames)
        )

    def encode(self, frame):
        self.frames.append(frame)
        return [
            EncodedPacket(
                b"\x00\x00\x00\x01\x65frame", frame.capture_timestamp_ns // 1000, frame.capture_timestamp_ns, True
            )
        ]

    def reset(self):
        self.resets += 1

    def close(self, timeout_s=1.0):
        self.closed = True


class _Sender:
    def __init__(self, **options):
        self.options = options
        self.packets = []
        self.resets = 0
        self.started = False
        self.closed = False

    @property
    def status(self):
        return StreamStatus(
            self.options["stream_id"],
            StreamLifecycleState.READY if self.started else StreamLifecycleState.CONFIGURED,
            self.started,
            self.options["selected_backend"],
            StreamMetrics(queued_frames=len(self.packets), reconnect_count=self.resets),
        )

    def start(self):
        self.started = True

    def enqueue(self, packet):
        self.packets.append(packet)
        callback = self.options.get("on_sent")
        if callback is not None:
            callback(packet)

    def reset(self):
        self.packets.clear()
        self.resets += 1

    def close(self, timeout_s=1.0):
        self.closed = True


def _spec(mode="rtp"):
    transport = ObservationTransportSpec()
    if mode == "rtp":
        transport = ObservationTransportSpec(
            mode="rtp",
            stream_id="top",
            endpoint=RtpEndpointSpec("127.0.0.1", 5004),
            h264=H264Spec(gop_frames=10),
            encoder_backend="software",
            decoder_backend="software",
            media=VideoMediaSpec(4, 2, 30.0),
            buffer=VideoBufferSpec(),
            readiness=VideoReadinessSpec(),
        )
    return SpecView(
        key="observation.images.top" if mode == "rtp" else "observation.state",
        topic="/camera/top" if mode == "rtp" else "/joint_states",
        ros_type="sensor_msgs/msg/Image" if mode == "rtp" else "sensor_msgs/msg/JointState",
        is_action=False,
        names=[],
        image_resize=(2, 4) if mode == "rtp" else None,
        image_encoding="rgb8",
        image_channels=3,
        resample_policy="hold",
        asof_tol_ms=0,
        max_age_ms=1000,
        stamp_src="header",
        clamp=None,
        safety_behavior=None,
        transport=transport,
    )


def _manager():
    registry = VideoCodecRegistry()
    encoders = []

    def factory(**options):
        encoder = _Encoder(**options)
        encoders.append(encoder)
        return encoder

    registry.register(
        "software",
        priority=0,
        probe=lambda _kind: CodecCapabilities(pixel_formats=("rgb24",)),
        encoder_factory=factory,
    )
    senders = []

    def sender_factory(**options):
        sender = _Sender(**options)
        senders.append(sender)
        return sender

    manager = DeviceVideoStreamManager(
        pipeline_id="policy",
        contract_fingerprint="contract",
        deployment_fingerprint="deployment",
        observation_specs=(_spec(), _spec("dds")),
        codec_registry=registry,
        sender_factory=sender_factory,
    )
    return manager, encoders[0], senders


def _image():
    pixels = np.arange(2 * 16, dtype=np.uint8).reshape(2, 16)
    return SimpleNamespace(width=4, height=2, step=16, encoding="rgb8", data=pixels.tobytes())


def test_device_stream_manager_sends_raw_uint8_frames_independent_of_requests():
    manager, encoder, senders = _manager()

    assert not manager.submit_ros_image(
        "observation.images.top", _image(), capture_timestamp_ns=1_000_000, receive_timestamp_ns=1_000_100
    )
    manager.bind_session("session-a", 1)
    assert manager.submit_ros_image(
        "observation.images.top", _image(), capture_timestamp_ns=1_000_000, receive_timestamp_ns=1_000_100
    )
    assert manager.submit_ros_image(
        "observation.images.top", _image(), capture_timestamp_ns=2_000_000, receive_timestamp_ns=2_000_100
    )

    assert len(encoder.frames) == len(senders[-1].packets) == 2
    assert encoder.frames[0].data.shape == (2, 4, 3)
    assert encoder.frames[0].data.dtype == np.uint8
    assert manager.stream_references[0].stream_id == "top"
    status = manager.statuses()[0]
    assert status.timestamp_mapping_valid
    assert status.keyframe_ready
    assert status.mapping_capture_timestamp_ns == 2_000_000


def test_device_stream_manager_descriptors_and_reset_are_session_scoped():
    manager, encoder, senders = _manager()

    assert manager.descriptors() == ()
    assert manager.bind_session("session-a", 1)
    first = manager.descriptors()[0]
    assert first.session_id == "session-a"
    assert first.contract_fingerprint == "contract"
    assert not manager.bind_session("session-a", 1)

    assert manager.bind_session("session-b", 2)
    second = manager.descriptors()[0]
    assert second.session_id == "session-b"
    assert second.session_generation == 2
    assert second.ssrc != first.ssrc
    assert encoder.resets == 2
    assert len(senders) == 3
    assert all(sender.closed for sender in senders[:-1])

    manager.close()
    assert encoder.closed and senders[-1].closed


def test_device_stream_manager_drops_non_monotonic_capture_timestamps():
    manager, encoder, _senders = _manager()
    manager.bind_session("session", 1)
    manager.submit_ros_image("observation.images.top", _image(), capture_timestamp_ns=10, receive_timestamp_ns=11)

    assert not manager.submit_ros_image(
        "observation.images.top", _image(), capture_timestamp_ns=10, receive_timestamp_ns=12
    )
    assert len(encoder.frames) == 1
    assert manager.statuses()[0].dropped_frames == 1


def test_device_stream_diagnostic_snapshot_is_immutable_deterministic_and_tracks_readiness():
    manager, _encoder, _senders = _manager()

    initial = manager.diagnostic_snapshots()
    assert initial == manager.diagnostic_snapshots()
    assert initial[0].observation_key == "observation.images.top"
    assert initial[0].endpoint == ("127.0.0.1", 5004)
    assert initial[0].configured_encoder_backend == initial[0].selected_encoder_backend == "software"
    assert initial[0].configured_decoder_backend == "software"
    assert initial[0].selected_decoder_backend == "not-local"
    assert initial[0].security == "none/trusted-network-only"
    assert not initial[0].ready
    with pytest.raises(FrozenInstanceError):
        initial[0].ready = True

    manager.bind_session("session", 1)
    manager.submit_ros_image("observation.images.top", _image(), capture_timestamp_ns=10, receive_timestamp_ns=11)
    assert manager.diagnostic_snapshots()[0].ready
