import threading
import time
from dataclasses import FrozenInstanceError, replace
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
        self.failed = False

    @property
    def state(self):
        if self.closed:
            return CodecLifecycleState.CLOSED
        if self.failed:
            return CodecLifecycleState.FAILED
        return CodecLifecycleState.RUNNING

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
        self.failed = False

    def close(self, timeout_s=1.0):
        self.closed = True


class _Sender:
    def __init__(self, **options):
        self.options = options
        self.packets = []
        self.resets = 0
        self.started = False
        self.closed = False
        self.failed = False

    @property
    def status(self):
        if self.failed:
            state = StreamLifecycleState.FAILED
        else:
            state = StreamLifecycleState.READY if self.started else StreamLifecycleState.CONFIGURED
        return StreamStatus(
            self.options["stream_id"],
            state,
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


def _spec(mode="rtp", *, key="observation.images.top", stream_id="top", port=5004):
    transport = ObservationTransportSpec()
    if mode == "rtp":
        transport = ObservationTransportSpec(
            mode="rtp",
            stream_id=stream_id,
            endpoint=RtpEndpointSpec("127.0.0.1", port),
            h264=H264Spec(gop_frames=10),
            encoder_backend="software",
            decoder_backend="software",
            media=VideoMediaSpec(4, 2, 30.0),
            buffer=VideoBufferSpec(),
            readiness=VideoReadinessSpec(),
        )
    return SpecView(
        key=key if mode == "rtp" else "observation.state",
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
    assert manager.flush()

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
    # Healthy encoders and senders survive session rollovers: only the SSRC
    # and the sender queue rotate, so the Ascend FFmpeg process is never
    # respawned just because the cloud session generation ticked.
    assert encoder.resets == 0
    assert len(senders) == 1
    assert senders[0].resets == 2
    assert not senders[0].closed

    manager.close()
    assert encoder.closed and senders[-1].closed


def test_bind_session_recovers_dead_sender_and_failed_encoder():
    manager, encoder, senders = _manager()
    manager.bind_session("session-a", 1)

    # Simulate a sender whose worker thread died and an encoder whose FFmpeg
    # process crashed between sessions.
    senders[0].failed = True
    encoder.failed = True

    assert manager.bind_session("session-b", 2)

    # The dead sender was replaced with a fresh one; the failed encoder was
    # reset back to RUNNING instead of taking the dead sender down with it.
    assert len(senders) == 2
    assert senders[0].closed
    assert senders[1].started
    assert encoder.resets == 1
    assert encoder.state is CodecLifecycleState.RUNNING

    manager.close()
    assert senders[1].closed


def test_latest_sent_capture_ns_tracks_wire_sends_and_clears_on_rollover():
    manager, encoder, senders = _manager()
    manager.bind_session("session-a", 1)

    # Nothing on the wire yet, and unknown streams never report a send.
    assert manager.latest_sent_capture_ns("observation.images.top") == 0
    assert manager.latest_sent_capture_ns("observation.images.unknown") == 0

    assert manager.submit_ros_image(
        "observation.images.top", _image(), capture_timestamp_ns=1_000_000_000, receive_timestamp_ns=1_000_000_000
    )
    assert manager.flush()
    # The fake sender invokes on_sent inline with the encoder's packet, so
    # the last-sent capture is the frame's capture timestamp, not its
    # receive timestamp or its local buffer arrival.
    assert manager.latest_sent_capture_ns("observation.images.top") == 1_000_000_000

    # A session rollover clears the record: pre-rollover frames must not be
    # mistaken for fresh ones by the decision gate.
    assert manager.bind_session("session-b", 2)
    assert manager.latest_sent_capture_ns("observation.images.top") == 0

    manager.close()


def test_device_stream_manager_drops_non_monotonic_capture_timestamps():
    manager, encoder, _senders = _manager()
    manager.bind_session("session", 1)
    manager.submit_ros_image("observation.images.top", _image(), capture_timestamp_ns=10, receive_timestamp_ns=11)

    assert not manager.submit_ros_image(
        "observation.images.top", _image(), capture_timestamp_ns=10, receive_timestamp_ns=12
    )
    assert manager.flush()
    assert len(encoder.frames) == 1
    assert manager.statuses()[0].dropped_frames == 1


def test_device_stream_manager_rejects_frames_after_session_clear():
    manager, encoder, _senders = _manager()
    manager.bind_session("session", 1)
    manager.clear_session()

    assert not manager.submit_ros_image(
        "observation.images.top", _image(), capture_timestamp_ns=10, receive_timestamp_ns=11
    )
    assert encoder.frames == []


def test_device_stream_manager_encodes_distinct_streams_concurrently():
    registry = VideoCodecRegistry()
    encode_barrier = threading.Barrier(2, timeout=2)

    class _ConcurrentEncoder(_Encoder):
        def encode(self, frame):
            encode_barrier.wait()
            return super().encode(frame)

    registry.register(
        "software",
        priority=0,
        probe=lambda _kind: CodecCapabilities(pixel_formats=("rgb24",)),
        encoder_factory=lambda **options: _ConcurrentEncoder(**options),
    )
    manager = DeviceVideoStreamManager(
        pipeline_id="policy",
        contract_fingerprint="contract",
        deployment_fingerprint="deployment",
        observation_specs=(
            _spec(),
            _spec(key="observation.images.wrist", stream_id="wrist", port=5006),
        ),
        codec_registry=registry,
        sender_factory=_Sender,
    )
    manager.bind_session("session", 1)
    results = []

    def submit(key):
        results.append(manager.submit_ros_image(key, _image(), capture_timestamp_ns=10, receive_timestamp_ns=11))

    threads = [
        threading.Thread(target=submit, args=("observation.images.top",)),
        threading.Thread(target=submit, args=("observation.images.wrist",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert not any(thread.is_alive() for thread in threads)
    assert results == [True, True]
    assert manager.flush()


def test_device_stream_manager_assigns_unique_ascend_encoder_channels():
    top_spec = _spec()
    wrist_spec = _spec(key="observation.images.wrist", stream_id="wrist", port=5006)
    top = replace(top_spec, transport=replace(top_spec.transport, encoder_backend="ascend"))
    wrist = replace(wrist_spec, transport=replace(wrist_spec.transport, encoder_backend="ascend"))
    encoder_options = []
    registry = VideoCodecRegistry()
    registry.register(
        "ascend",
        priority=0,
        probe=lambda _kind: CodecCapabilities(pixel_formats=("rgb24",)),
        encoder_factory=lambda **options: encoder_options.append(options) or _Encoder(),
    )

    manager = DeviceVideoStreamManager(
        pipeline_id="policy",
        contract_fingerprint="contract",
        deployment_fingerprint="deployment",
        observation_specs=(wrist, top),
        codec_registry=registry,
        sender_factory=_Sender,
    )

    assert sorted(options["channel_id"] for options in encoder_options) == [1, 2]
    manager.close()


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
    assert manager.flush()
    assert manager.diagnostic_snapshots()[0].ready


def test_submit_ros_image_returns_before_slow_encode_completes():
    """The executor callback must not block on the encoder's drain wait."""
    release = threading.Event()
    slow_started = threading.Event()

    class _SlowEncoder(_Encoder):
        def encode(self, frame):
            slow_started.set()
            release.wait(timeout=5)
            return super().encode(frame)

    registry = VideoCodecRegistry()
    registry.register(
        "software",
        priority=0,
        probe=lambda _kind: CodecCapabilities(pixel_formats=("rgb24",)),
        encoder_factory=lambda **options: _SlowEncoder(**options),
    )
    manager = DeviceVideoStreamManager(
        pipeline_id="policy",
        contract_fingerprint="contract",
        deployment_fingerprint="deployment",
        observation_specs=(_spec(), _spec("dds")),
        codec_registry=registry,
        sender_factory=_Sender,
    )
    manager.bind_session("session", 1)

    started = time.monotonic()
    assert manager.submit_ros_image(
        "observation.images.top", _image(), capture_timestamp_ns=10, receive_timestamp_ns=11
    )
    assert time.monotonic() - started < 0.5
    assert slow_started.wait(timeout=2)

    release.set()
    assert manager.flush()
    manager.close()


def test_encode_queue_overflow_drops_oldest_frames():
    """A stalled encoder drops the oldest queued frames instead of blocking submits."""
    release = threading.Event()
    stalled = threading.Event()

    class _StalledEncoder(_Encoder):
        def encode(self, frame):
            stalled.set()
            release.wait(timeout=5)
            return super().encode(frame)

    registry = VideoCodecRegistry()
    registry.register(
        "software",
        priority=0,
        probe=lambda _kind: CodecCapabilities(pixel_formats=("rgb24",)),
        encoder_factory=lambda **options: _StalledEncoder(**options),
    )
    manager = DeviceVideoStreamManager(
        pipeline_id="policy",
        contract_fingerprint="contract",
        deployment_fingerprint="deployment",
        observation_specs=(_spec(), _spec("dds")),
        codec_registry=registry,
        sender_factory=_Sender,
    )
    manager.bind_session("session", 1)

    assert manager.submit_ros_image("observation.images.top", _image(), capture_timestamp_ns=1, receive_timestamp_ns=1)
    assert stalled.wait(timeout=2)
    # The first frame is inside the encoder; the queue holds the next four and
    # drops the two oldest once it overflows.
    for capture_ns in range(2, 8):
        assert manager.submit_ros_image(
            "observation.images.top", _image(), capture_timestamp_ns=capture_ns, receive_timestamp_ns=capture_ns
        )

    release.set()
    assert manager.flush()
    assert manager.statuses()[0].dropped_frames == 2
    manager.close()


def test_concurrent_same_key_submits_enqueue_in_timestamp_order():
    """Two concurrent callbacks for one observation (reentrant callback group
    on a multithreaded executor) must not enqueue out of order: the timestamp
    gate and the enqueue share one critical section, so every accepted frame
    reaches the encoder in strictly increasing capture order."""
    manager, encoder, _senders = _manager()
    manager.bind_session("session", 1)
    barrier = threading.Barrier(2, timeout=2)
    accepted: list[bool] = []
    results_lock = threading.Lock()

    def submit(capture_ns):
        barrier.wait()
        ok = manager.submit_ros_image(
            "observation.images.top", _image(), capture_timestamp_ns=capture_ns, receive_timestamp_ns=capture_ns
        )
        with results_lock:
            accepted.append(ok)

    threads = [
        threading.Thread(target=submit, args=(10,)),
        threading.Thread(target=submit, args=(20,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert not any(thread.is_alive() for thread in threads)
    assert manager.flush()
    timestamps = [frame.capture_timestamp_ns for frame in encoder.frames]
    assert timestamps == sorted(timestamps)
    assert len(timestamps) == sum(accepted)
    assert len(timestamps) >= 1
    manager.close()


def test_encode_worker_drops_frames_retired_by_session_rollover():
    """A frame that passed the submit gate of the retired session but is only
    processed after the rollover must be dropped by the worker instead of
    surfacing as the first frame of the new session."""
    manager, encoder, _senders = _manager()
    manager.bind_session("session-a", 1)
    assert manager.submit_ros_image(
        "observation.images.top", _image(), capture_timestamp_ns=1_000_000, receive_timestamp_ns=1_000_100
    )
    assert manager.flush()
    assert [frame.capture_timestamp_ns for frame in encoder.frames] == [1_000_000]

    stream = manager._streams["observation.images.top"]
    manager.bind_session("session-b", 2)
    # Simulate a session-a frame that was already queued when the rollover
    # committed: retired session generation and retired lifecycle epoch.
    stale = (_image(), 2_000_000, 2_000_100, 1, stream.lifecycle_epoch - 1)
    stream.encode_queue.put(stale)
    assert manager.flush(timeout_s=1.0)

    assert [frame.capture_timestamp_ns for frame in encoder.frames] == [1_000_000]

    # The new session still accepts fresh frames.
    assert manager.submit_ros_image(
        "observation.images.top", _image(), capture_timestamp_ns=3_000_000, receive_timestamp_ns=3_000_100
    )
    assert manager.flush()
    assert [frame.capture_timestamp_ns for frame in encoder.frames] == [1_000_000, 3_000_000]
    manager.close()


def test_encode_failure_clears_readiness_and_recovers_encoder():
    """A swallowed encode exception must surface: readiness drops, the error
    is reported, and a failed encoder is reset so later frames can recover."""
    manager, _encoder, _senders = _manager()

    class _FailingOnceEncoder(_Encoder):
        def __init__(self, **options):
            super().__init__(**options)
            self.failed_once = False

        def encode(self, frame):
            if not self.failed_once:
                self.failed_once = True
                self.failed = True
                raise RuntimeError("encode boom")
            return super().encode(frame)

    # Swap the encoder for one that fails on the first frame.
    failing = _FailingOnceEncoder()
    manager._streams["observation.images.top"].encoder = failing
    manager.bind_session("session", 1)
    assert manager.submit_ros_image(
        "observation.images.top", _image(), capture_timestamp_ns=1_000_000, receive_timestamp_ns=1_000_100
    )
    assert manager.flush()

    status = manager.statuses()[0]
    assert not status.ready
    assert status.keyframe_ready is False
    assert "encode boom" in status.last_error
    assert status.dropped_frames == 1
    assert failing.resets == 1, "the failed encoder must be reset by the recovery path"

    # The failed encoder was reset by the recovery path, so the next frame
    # succeeds and readiness is re-established.
    assert manager.submit_ros_image(
        "observation.images.top", _image(), capture_timestamp_ns=2_000_000, receive_timestamp_ns=2_000_100
    )
    assert manager.flush()
    status = manager.statuses()[0]
    assert status.ready
    assert status.last_error == ""
    assert status.dropped_frames == 1
    manager.close()


def test_close_leaves_sender_open_while_worker_is_wedged():
    """close() must not close the sender while the encode worker is still
    alive on it: the wedged worker would use closed resources.  The encoder
    is closed first to try to unblock the worker."""
    release = threading.Event()
    stalled = threading.Event()

    class _StalledEncoder(_Encoder):
        def encode(self, frame):
            stalled.set()
            release.wait(timeout=5)
            return super().encode(frame)

    registry = VideoCodecRegistry()
    registry.register(
        "software",
        priority=0,
        probe=lambda _kind: CodecCapabilities(pixel_formats=("rgb24",)),
        encoder_factory=lambda **options: _StalledEncoder(**options),
    )
    manager = DeviceVideoStreamManager(
        pipeline_id="policy",
        contract_fingerprint="contract",
        deployment_fingerprint="deployment",
        observation_specs=(_spec(), _spec("dds")),
        codec_registry=registry,
        sender_factory=_Sender,
    )
    manager.bind_session("session", 1)
    assert manager.submit_ros_image("observation.images.top", _image(), capture_timestamp_ns=1, receive_timestamp_ns=1)
    assert stalled.wait(timeout=2)
    stream = manager._streams["observation.images.top"]
    sender = stream.sender

    started = time.monotonic()
    manager.close(timeout_s=0.2)
    elapsed = time.monotonic() - started

    # close() returns promptly even though the worker never joined.
    assert elapsed < 2.0
    # The encoder was closed to unblock the wedged write, but the sender was
    # left open: the worker may still touch it once it unblocks.
    assert stream.encoder.closed
    assert not sender.closed
    release.set()
