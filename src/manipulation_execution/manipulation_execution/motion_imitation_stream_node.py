"""Resident head-camera -> YOLOX -> PEAR frame stream for X2 motion imitation.

This is the streaming counterpart of ``imitate_human_motion_executor_node``'s
capture phase. That node opens a bounded window inside a skill task, records
every PEAR response into a list, and hands the list to the retargeter when the
window closes. This one is always on: it subscribes an encoded camera stream,
decodes it, refreshes a person box from YOLOX at low frequency, runs PEAR on
the newest frame, and publishes each accepted response immediately.

Three stages are replaceable, each behind its own interface so they can be
swapped independently: the camera endpoint is resolved by
interface ID at launch, the decode is a ``FrameDecoder`` chosen by name
(``manipulation_execution.frame_decoder``), and the choice of which person to
follow is an ``ActorSelector`` (``manipulation_execution.actor_selector``)
whose answer is published as ``ibrobot_msgs/ActorState``. The status topic
reports input rate, decode rate, decode time and time to first frame so a
replacement decoder can be compared against the one it replaces.

It is perception only. It commands no joints, holds no arbitration, switches no
mode and carries no dispatch binding, so starting it cannot move the robot. The
retargeting stage consumes the published stream separately.

The perception semantics -- person selection, the centred-crop fallback, the
call-start-to-call-start YOLOX rate limit, the response validation -- are the
windowed executor's, and the shared pieces live in
``manipulation_execution.hri_vision`` so the two stay identical.
There is no task capture window or deadline. A source generation invalidates
queued frames and in-flight results across source faults, restarts and geometry
changes; it does not scope the resident stream to an imitation task.
"""

from __future__ import annotations

import contextlib
import json
import math
import queue
import threading
import time
from dataclasses import dataclass

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

from ibrobot_msgs.msg import ActorState, Detection2D, DetectionArray, PearFrame
from ibrobot_msgs.srv import PearParameterPredict, YoloXDetect
from manipulation_execution.actor_selector import ActorDecision, ActorSelector
from manipulation_execution.frame_decoder import create_frame_decoder
from manipulation_execution.hri_vision import (
    PEAR_FIELD_LENGTHS,
    PEAR_OUTPUT_FIELDS,
    center_fallback_detection,
    put_latest,
    validated_person_confidence_threshold,
    validated_yolox_refresh_interval,
)

SCHEMA = "pear_parameter_v1"

# YOLOX shares one Ascend device with PEAR and is the expensive half. The
# windowed executor refreshes every 0.25 s because its whole task lasts 20 s;
# a resident stream can afford to be far lazier, since a standing person's
# bounding box does not move meaningfully within a second.
_DEFAULT_YOLOX_REFRESH_SEC = 1.0

# A resident stream must shed rather than queue: at 30 Hz input, one call stuck
# for the executor's 5 s would sit behind 150 skipped frames.
_DEFAULT_RPC_TIMEOUT_SEC = 2.0

# The two model services are released in the same launch batch as this node and
# spend seconds loading their OM bundles, so "not available" is expected at
# first and only becomes a complaint after this long.
_STARTUP_GRACE_SEC = 30.0

# Detections older than this, relative to the frame being decided, make the
# actor unknown. A few YOLOX refresh periods; launch requires it to exceed
# yolox_refresh_interval_sec, or the actor would go unknown between refreshes.
_DEFAULT_ACTOR_STALE_SEC = 3.0

_NMS_THRESHOLD = 0.65
_QUEUE_POLL_SEC = 0.1
_FUTURE_POLL_SEC = 0.01


@dataclass(frozen=True)
class _QueuedFrame:
    generation: int
    message: CompressedImage | Image


def _validated_positive(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return number


def _validated_topic(value: object, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} is required")
    if not text.startswith("/"):
        raise ValueError(f"{name} must be an absolute ROS name")
    return text


class MotionImitationStreamNode(Node):
    """Publish one PearFrame per accepted PEAR response, continuously."""

    def __init__(self, parameter_overrides=None) -> None:
        super().__init__("motion_imitation_stream", parameter_overrides=parameter_overrides)
        self.declare_parameter("rgb_compressed_topic", "")
        self.declare_parameter("frame_source_interface", "")
        self.declare_parameter("require_unique_publisher", False)
        self.declare_parameter("decoder", "mock_cpu")
        self.declare_parameter("decoder_params_json", "{}")
        self.declare_parameter("actor_state_topic", "/motion_imitation_stream/actor_state")
        self.declare_parameter("actor_stale_sec", _DEFAULT_ACTOR_STALE_SEC)
        self.declare_parameter("frame_topic", "/motion_imitation_stream/pear_frame")
        self.declare_parameter("status_topic", "/motion_imitation_stream/status")
        self.declare_parameter("yolox_detect_service", "/perception/hri/yolox_detect")
        self.declare_parameter("pear_parameters_service", "/perception/hri/pear_parameters")
        self.declare_parameter("person_confidence_threshold", 0.30)
        self.declare_parameter("yolox_refresh_interval_sec", _DEFAULT_YOLOX_REFRESH_SEC)
        self.declare_parameter("rpc_timeout_sec", _DEFAULT_RPC_TIMEOUT_SEC)
        self.declare_parameter("max_pear_hz", 0.0)
        self.declare_parameter("publish_fallback_frames", True)
        self.declare_parameter("status_period_sec", 5.0)

        self._rgb_topic = _validated_topic(self.get_parameter("rgb_compressed_topic").value, "rgb_compressed_topic")
        # Empty when launch used an explicit frame_source.topic override.
        self._frame_source_interface = str(self.get_parameter("frame_source_interface").value).strip()
        self._require_unique_publisher = self.get_parameter("require_unique_publisher").value
        if type(self._require_unique_publisher) is not bool:
            raise ValueError("require_unique_publisher must be a boolean")
        try:
            decoder_params = json.loads(str(self.get_parameter("decoder_params_json").value))
        except json.JSONDecodeError as exc:
            raise ValueError(f"decoder_params_json must be a JSON object: {exc}") from exc
        self._decoder_name = str(self.get_parameter("decoder").value)
        self._decoder = create_frame_decoder(self._decoder_name, decoder_params)
        frame_topic = _validated_topic(self.get_parameter("frame_topic").value, "frame_topic")
        status_topic = _validated_topic(self.get_parameter("status_topic").value, "status_topic")
        actor_state_topic = _validated_topic(self.get_parameter("actor_state_topic").value, "actor_state_topic")
        yolox_service = _validated_topic(self.get_parameter("yolox_detect_service").value, "yolox_detect_service")
        pear_service = _validated_topic(self.get_parameter("pear_parameters_service").value, "pear_parameters_service")
        self._person_confidence_threshold = validated_person_confidence_threshold(
            self.get_parameter("person_confidence_threshold").value
        )
        self._yolox_refresh_interval_sec = validated_yolox_refresh_interval(
            self.get_parameter("yolox_refresh_interval_sec").value
        )
        self._rpc_timeout = _validated_positive(self.get_parameter("rpc_timeout_sec").value, "rpc_timeout_sec")
        self._status_period_sec = _validated_positive(
            self.get_parameter("status_period_sec").value, "status_period_sec"
        )
        max_pear_hz = float(self.get_parameter("max_pear_hz").value)
        if not math.isfinite(max_pear_hz) or max_pear_hz < 0.0:
            raise ValueError("max_pear_hz must be finite and non-negative (0 disables throttling)")
        self._min_pear_period_sec = 1.0 / max_pear_hz if max_pear_hz > 0.0 else 0.0
        publish_fallback = self.get_parameter("publish_fallback_frames").value
        if type(publish_fallback) is not bool:
            raise ValueError("publish_fallback_frames must be a boolean")
        self._publish_fallback_frames = publish_fallback
        self._actor_selector = ActorSelector(
            self._person_confidence_threshold, self.get_parameter("actor_stale_sec").value
        )

        self._stats_lock = threading.RLock()
        self._source_generation = 0
        self._source_healthy = None
        self._received = 0
        self._compressed_dropped = 0
        self._out_of_order = 0
        self._source_rejected = 0
        self._last_capture_stamp = None
        self._last_encoded_at = 0.0
        self._source_geometry = None
        self._frames = 0
        self._frames_dropped = 0
        self._decode_failed = 0
        self._vision_ok = 0
        self._vision_detected = 0
        self._vision_fallback = 0
        self._vision_failed = 0
        self._yolox_calls = 0
        self._pear_calls = 0
        self._published = 0
        self._throttled = 0
        self._sequence = 0
        self._vision_sample = ""
        self._first_stamp = ""
        self._last_stamp = ""
        self._last_frame_id = ""
        self._last_frame_at = 0.0
        self._first_frame_after_sec: float | None = None
        # The newest successful YOLOX result, whole: the actor selector decides
        # from it per frame, so it needs every detection and the stamp of the
        # frame YOLOX saw, not just one pre-selected box.
        self._yolox_detections: DetectionArray | None = None
        self._yolox_result_ready = False
        self._actor_present_frames = 0
        self._actor_unknown_frames = 0
        self._actor_reason = ""
        self._announced_first_frame = False
        self._rejected_format = ""
        # Distinct failure reasons are logged once immediately and then at most
        # once per status period. The executor throttles by occurrence count,
        # which suits a 20 s window; a resident 30 Hz stream with a service
        # down would log tens of warnings a second.
        self._failure_logged_at: dict[str, float] = {}
        self._started_at = time.monotonic()
        self._window_started_at = self._started_at
        self._window_received = 0
        self._window_decode_calls = 0
        self._window_frames = 0
        self._window_published = 0
        self._window_decode_sec = 0.0
        self._window_decode_max_sec = 0.0

        self._stop = threading.Event()
        self._decode_queue: queue.Queue = queue.Queue(maxsize=1)
        self._pear_queue: queue.Queue = queue.Queue(maxsize=1)
        self._yolox_queue: queue.Queue = queue.Queue(maxsize=1)
        self._next_pear_at = 0.0

        # The workers spin-wait on call_async futures, which the executor
        # completes on the clients' callback group. Keeping the clients
        # reentrant and off the subscription's default group means a decode in
        # the camera callback cannot stall a service response behind it.
        self._client_group = ReentrantCallbackGroup()
        self._yolox_client = self.create_client(YoloXDetect, yolox_service, callback_group=self._client_group)
        self._pear_client = self.create_client(PearParameterPredict, pear_service, callback_group=self._client_group)
        self._frame_publisher = self.create_publisher(PearFrame, frame_topic, qos_profile_sensor_data)
        self._actor_publisher = self.create_publisher(ActorState, actor_state_topic, 10)
        self._status_publisher = self.create_publisher(String, status_topic, 10)

        self._pear_worker = threading.Thread(target=self._pear_loop, name="mi-stream-pear", daemon=True)
        self._yolox_worker = threading.Thread(target=self._yolox_loop, name="mi-stream-yolox", daemon=True)
        self._decode_worker = threading.Thread(target=self._decode_loop, name="mi-stream-decode", daemon=True)
        self._pear_worker.start()
        self._yolox_worker.start()
        self._decode_worker.start()
        self._rgb_subscription = self.create_subscription(
            CompressedImage,
            self._rgb_topic,
            self._on_compressed_frame,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self._status_timer = self.create_timer(self._status_period_sec, self._publish_status)
        # count_publishers queries the rclpy graph; doing it per frame while
        # holding the stats lock contends with the decode and vision workers.
        # A slow timer owns the probe; frame paths read the cached boolean.
        self._source_health_timer = self.create_timer(0.2, self._refresh_source_health)
        self.get_logger().info(
            f"motion imitation stream ready: in={self._rgb_topic} "
            f"interface={self._frame_source_interface or '(explicit topic)'} decoder={self._decoder_name} "
            f"publishers={self.count_publishers(self._rgb_topic)} out={frame_topic} actor={actor_state_topic}"
        )
        self.get_logger().info(
            f"motion imitation stream services: yolox={yolox_service} pear={pear_service} "
            f"yolox_refresh={self._yolox_refresh_interval_sec:.2f}s rpc_timeout={self._rpc_timeout:.2f}s"
        )

    def _refresh_source_health(self) -> None:
        """Probe source availability at a slow rate; frame paths read the cache."""
        if not self._require_unique_publisher:
            return
        available = self.count_publishers(self._rgb_topic) == 1
        with self._stats_lock:
            if not available and self._source_healthy is not False:
                self._invalidate_source_locked()
            self._source_healthy = available

    def _source_available(self) -> bool:
        if not self._require_unique_publisher:
            return True
        if self._source_healthy is None:
            # Before the health timer's first tick the cache is empty; probe
            # once here so early frames are not rejected. Callers already hold
            # the stats lock, which is reentrant for this thread.
            self._refresh_source_health()
        return self._source_healthy is True

    def _invalidate_source_locked(self) -> None:
        """Retire cache, queue entries and RPCs together while holding the state lock."""
        self._source_generation += 1
        self._yolox_detections = None
        self._yolox_result_ready = False
        self._source_geometry = None
        self._last_capture_stamp = None
        self._last_encoded_at = 0.0
        self._next_pear_at = 0.0
        for slot in (self._decode_queue, self._pear_queue, self._yolox_queue):
            with contextlib.suppress(queue.Empty):
                slot.get_nowait()

    def _frame_current_locked(self, message: Image, generation: int) -> bool:
        return generation == self._source_generation and (
            self._source_geometry is None or self._source_geometry == (message.width, message.height)
        )

    def _on_compressed_frame(self, message: CompressedImage) -> None:
        """Keep the newest compressed frame without decoding on the ROS callback."""
        stamp = (int(message.header.stamp.sec), int(message.header.stamp.nanosec))
        with self._stats_lock:
            if not self._source_available():
                self._source_rejected += 1
                self._yolox_detections = None
                self._yolox_result_ready = False
                return
            self._received += 1
            self._window_received += 1
            now = time.monotonic()
            if self._last_encoded_at and now - self._last_encoded_at > max(2.0, self._rpc_timeout):
                self._invalidate_source_locked()
            if stamp != (0, 0):
                if self._last_capture_stamp is not None and stamp <= self._last_capture_stamp:
                    self._out_of_order += 1
                    return
                self._last_capture_stamp = stamp
            self._last_encoded_at = now
            if put_latest(self._decode_queue, _QueuedFrame(self._source_generation, message)):
                self._compressed_dropped += 1

    def _decode_loop(self) -> None:
        while not self._stop.is_set():
            try:
                queued = self._decode_queue.get(timeout=_QUEUE_POLL_SEC)
            except queue.Empty:
                continue
            try:
                self._decode_frame(queued.message, queued.generation)
            except Exception as exc:
                with self._stats_lock:
                    self._decode_failed += 1
                self._record_vision_failure(f"decode: {type(exc).__name__}: {exc}")

    def _decode_frame(self, message: CompressedImage, generation: int | None = None) -> None:
        """Decode once, then hand the same frame to both workers."""
        if not self._source_available():
            return
        with self._stats_lock:
            if generation is None:
                generation = self._source_generation
            if generation != self._source_generation:
                return
        started = time.perf_counter()
        error = ""
        try:
            image = self._decoder.decode(message)
        except Exception as exc:
            # The interface says a bad frame returns None. A decoder that
            # raises anyway is counted the same way rather than allowed to
            # take the subscription callback down.
            image = None
            error = f" ({type(exc).__name__}: {exc})"
        decode_sec = time.perf_counter() - started
        with self._stats_lock:
            self._window_decode_calls += 1
            self._window_decode_sec += decode_sec
            self._window_decode_max_sec = max(self._window_decode_max_sec, decode_sec)
        if image is None:
            with self._stats_lock:
                self._decode_failed += 1
                count = self._decode_failed
                if not self._rejected_format:
                    self._rejected_format = str(message.format)
            # The platform's exact format string is not documented, so the
            # first rejected value is logged verbatim: a stream that decodes
            # zero frames has to say what it was handed.
            if count == 1:
                self.get_logger().warning(
                    f"compressed frame rejected by decoder {self._decoder_name}, format={str(message.format)!r}{error}"
                )
            return
        if not self._source_available():
            return
        stamp = image.header.stamp
        stamp_text = f"{int(stamp.sec)}.{int(stamp.nanosec):09d}"
        now = time.monotonic()
        with self._stats_lock:
            if generation != self._source_generation:
                return
            geometry = (image.width, image.height)
            if self._source_geometry is not None and self._source_geometry != geometry:
                last_capture, last_encoded = self._last_capture_stamp, self._last_encoded_at
                self._invalidate_source_locked()
                self._last_capture_stamp, self._last_encoded_at = last_capture, last_encoded
                generation = self._source_generation
            self._source_geometry = geometry
            self._frames += 1
            self._window_frames += 1
            if not self._first_stamp:
                self._first_stamp = stamp_text
            self._last_stamp = stamp_text
            self._last_frame_id = str(image.header.frame_id)
            self._last_frame_at = now
            first_frame = self._first_frame_after_sec is None
            if first_frame:
                self._first_frame_after_sec = now - self._started_at
            queued = _QueuedFrame(generation, image)
            dropped = put_latest(self._pear_queue, queued)
            dropped = put_latest(self._yolox_queue, queued) or dropped
            if dropped:
                self._frames_dropped += 1
        if first_frame:
            self.get_logger().info(
                f"motion imitation stream first camera frame after {now - self._started_at:.2f}s: "
                f"{image.width}x{image.height} decoder={self._decoder_name} decode={decode_sec * 1000.0:.1f}ms"
            )

    def _yolox_loop(self) -> None:
        next_refresh = 0.0
        pending_frame = None
        while not self._stop.is_set():
            timeout = _QUEUE_POLL_SEC
            if pending_frame is not None:
                timeout = min(timeout, max(0.0, next_refresh - time.monotonic()))
            with contextlib.suppress(queue.Empty):
                pending_frame = self._yolox_queue.get(timeout=timeout)
            if pending_frame is None:
                continue
            if time.monotonic() < next_refresh:
                # Keep the pending frame: the next iteration may replace it with
                # a newer one, so the refresh uses the freshest frame available
                # when the interval finally elapses.
                continue
            message = pending_frame
            pending_frame = None
            refresh_started = time.monotonic()
            try:
                self._process_yolox_frame(message.message, message.generation)
            except Exception as exc:
                self._record_vision_failure(f"{type(exc).__name__}: {exc}")
            # Measured call-start to call-start so a slow refresh does not
            # compound the period.
            next_refresh = refresh_started + self._yolox_refresh_interval_sec

    def _pear_loop(self) -> None:
        while not self._stop.is_set():
            try:
                queued = self._pear_queue.get(timeout=_QUEUE_POLL_SEC)
            except queue.Empty:
                continue
            with self._stats_lock:
                if not self._frame_current_locked(queued.message, queued.generation):
                    continue
                if self._min_pear_period_sec > 0.0:
                    now = time.monotonic()
                    if now < self._next_pear_at:
                        self._throttled += 1
                        continue
                    self._next_pear_at = now + self._min_pear_period_sec
            try:
                self._process_pear_frame(queued.message, queued.generation)
            except Exception as exc:
                self._record_vision_failure(f"{type(exc).__name__}: {exc}")

    def _process_yolox_frame(self, message: Image, generation: int | None = None) -> bool:
        """Refresh the person-box cache. Model preprocessing lives in the adapter."""
        if not self._source_available():
            return False
        with self._stats_lock:
            if generation is None:
                generation = self._source_generation
            if not self._frame_current_locked(message, generation):
                return False
        try:
            center_fallback_detection(message)
        except ValueError as exc:
            self._record_vision_failure(str(exc))
            return False
        request = YoloXDetect.Request()
        request.image = message
        request.confidence_threshold = self._person_confidence_threshold
        request.nms_threshold = _NMS_THRESHOLD
        response = self._call_vision(self._yolox_client, request, metric="yolox")
        if response is None or not self._source_available():
            return False
        if not response.success:
            self._record_vision_failure(f"yolox: {response.message}")
            return False
        cached = DetectionArray(header=message.header)
        for detection in response.detections.detections:
            kept = Detection2D()
            kept.header = message.header
            kept.label = detection.label
            kept.confidence = detection.confidence
            kept.bbox = [float(value) for value in detection.bbox]
            cached.detections.append(kept)
        # A refresh that failed above returned early and left the cache alone,
        # so the actor selector keeps deciding from the last good result until
        # it goes stale. Reaching here means the service answered: a result
        # with no qualifying person is the deliberate switch to the
        # centred-crop fallback.
        with self._stats_lock:
            if not self._frame_current_locked(message, generation):
                return False
            self._yolox_detections = cached
            self._yolox_result_ready = True
        return True

    def _process_pear_frame(self, message: Image, generation: int | None = None) -> None:
        """Decide the actor for the newest frame, run PEAR on its box, then publish."""
        if not self._source_available():
            return
        try:
            fallback_detection = center_fallback_detection(message)
        except ValueError as exc:
            self._record_vision_failure(str(exc))
            return
        with self._stats_lock:
            if generation is None:
                generation = self._source_generation
            if not self._frame_current_locked(message, generation):
                return
            ready = self._yolox_result_ready
            detections = self._yolox_detections
            decision = self._actor_selector.update(message, detections)
            self._record_actor(message, decision)
        if not ready:
            return
        # Unknown is never present, so a stale result falls back like an
        # empty one rather than reusing an old box.
        if decision.actor_present:
            detection = Detection2D()
            detection.header = message.header
            detection.label = "person"
            detection.confidence = decision.confidence
            detection.bbox = list(decision.bbox)
            source = "yolox"
        else:
            detection = fallback_detection
            source = "center_fallback"

        request = PearParameterPredict.Request()
        request.image = message
        # The compiled PEAR deployment is batch 1: exactly one person crop.
        request.detections = DetectionArray(header=message.header, detections=[detection])
        result = self._call_vision(self._pear_client, request, metric="pear")
        if result is None:
            return
        if not result.success:
            self._record_vision_failure(f"pear: {result.message}")
            return

        values = {name: [float(value) for value in getattr(result, name)] for name in PEAR_OUTPUT_FIELDS}
        missing = [name for name, value in values.items() if not value]
        nonfinite = [name for name, value in values.items() if any(not math.isfinite(v) for v in value)]
        mismatched = [
            f"{name}={len(value)}!={PEAR_FIELD_LENGTHS[name]}"
            for name, value in values.items()
            if value and len(value) != PEAR_FIELD_LENGTHS[name]
        ]
        if missing or nonfinite or mismatched:
            self._record_vision_failure(f"pear returned empty={missing} nonfinite={nonfinite} width={mismatched}")
            return

        # Sampled values, not just widths: a width check passes on an all-zero
        # response, which is exactly the failure this exists to catch.
        sample = "source={} bbox=[{}] conf={:.4f} pose[0:4]={} camera={} scale[0:3]={} latency={:.1f}ms".format(
            source,
            ", ".join(f"{float(value):.1f}" for value in detection.bbox),
            float(detection.confidence),
            [round(value, 4) for value in values["smplx_pose_raw"][:4]],
            [round(value, 4) for value in values["camera_raw"]],
            [round(value, 4) for value in values["smplx_scale"][:3]],
            float(result.inference_time_ms),
        )
        if not self._source_available():
            return
        with self._stats_lock:
            if not self._frame_current_locked(message, generation):
                return
            self._vision_ok += 1
            if source == "yolox":
                self._vision_detected += 1
            else:
                self._vision_fallback += 1
            self._vision_sample = sample
            announce = not self._announced_first_frame
            self._announced_first_frame = True
            if not self._publish_fallback_frames and source == "center_fallback":
                sequence = None
            else:
                self._sequence += 1
                self._published += 1
                self._window_published += 1
                sequence = self._sequence
            if sequence is not None:
                frame = PearFrame()
                frame.header = message.header
                frame.schema = SCHEMA
                frame.sequence = sequence
                frame.source = source
                frame.actor_id = decision.actor_id
                frame.bbox = [float(value) for value in detection.bbox]
                frame.confidence = float(detection.confidence)
                frame.inference_time_ms = float(result.inference_time_ms)
                for name in PEAR_OUTPUT_FIELDS:
                    setattr(frame, name, values[name])
                self._frame_publisher.publish(frame)
        if announce:
            self.get_logger().info(f"motion imitation stream first frame {sample}")

    def _record_actor(self, message: Image, decision: ActorDecision) -> None:
        """Count and publish the actor decision for one frame, before PEAR runs."""
        with self._stats_lock:
            if decision.actor_present:
                self._actor_present_frames += 1
            if not decision.known:
                self._actor_unknown_frames += 1
            self._actor_reason = decision.reason
        state = ActorState()
        state.header = message.header
        if decision.detections_stamp is not None:
            state.detections_stamp = decision.detections_stamp
        state.persons_count = decision.persons_count
        state.actor_present = decision.actor_present
        state.start_ready = decision.start_ready
        state.actor_id = decision.actor_id
        state.bbox = [float(value) for value in decision.bbox]
        state.confidence = float(decision.confidence)
        state.reason = decision.reason
        self._actor_publisher.publish(state)

    def _call_vision(self, client, request, *, metric: str = ""):
        """Call a vision service, bounded by the RPC timeout alone.

        Unlike the windowed executor, there is no task budget to separate
        from the RPC timeout, so every timeout here is an RPC timeout and is
        always counted.
        """
        if not client.service_is_ready():
            self._record_vision_failure(f"{client.srv_name} is not available")
            return None
        future = client.call_async(request)
        with self._stats_lock:
            if metric == "yolox":
                self._yolox_calls += 1
            elif metric == "pear":
                self._pear_calls += 1
        call_deadline = time.monotonic() + self._rpc_timeout
        while not future.done():
            if self._stop.is_set() or not rclpy.ok():
                future.cancel()
                return None
            if time.monotonic() >= call_deadline:
                future.cancel()
                self._record_vision_failure(f"{client.srv_name} timed out")
                return None
            time.sleep(_FUTURE_POLL_SEC)
        return future.result()

    def _record_vision_failure(self, detail: str) -> None:
        with self._stats_lock:
            self._vision_failed += 1
            count = self._vision_failed
        now = time.monotonic()
        last = self._failure_logged_at.get(detail)
        if last is not None and now - last < self._status_period_sec:
            return
        self._failure_logged_at[detail] = now
        if now - self._started_at < _STARTUP_GRACE_SEC and "is not available" in detail:
            # Expected while the model services load their OM bundles.
            self.get_logger().info(f"motion imitation stream waiting: {detail}")
            return
        self.get_logger().warning(f"motion imitation stream frame failed ({count}): {detail}")

    def _publish_status(self) -> None:
        now = time.monotonic()
        with self._stats_lock:
            elapsed = max(1e-6, now - self._window_started_at)
            status = {
                "topic": self._rgb_topic,
                "interface": self._frame_source_interface,
                "decoder": self._decoder_name,
                "publishers": self.count_publishers(self._rgb_topic),
                "require_unique_publisher": self._require_unique_publisher,
                "source_available": self._source_available(),
                "source_generation": self._source_generation,
                "source_rejected": self._source_rejected,
                "received": self._received,
                "compressed_dropped": self._compressed_dropped,
                "out_of_order": self._out_of_order,
                "cpu_frames": getattr(self._decoder, "cpu_frames", self._frames),
                "dvpp_frames": getattr(self._decoder, "dvpp_frames", 0),
                "decoder_error": getattr(self._decoder, "last_error", ""),
                "frames": self._frames,
                "frames_dropped": self._frames_dropped,
                "decode_failed": self._decode_failed,
                "rejected_format": self._rejected_format,
                "vision_ok": self._vision_ok,
                "vision_detected": self._vision_detected,
                "vision_fallback": self._vision_fallback,
                "vision_failed": self._vision_failed,
                "yolox_calls": self._yolox_calls,
                "pear_calls": self._pear_calls,
                "published": self._published,
                "throttled": self._throttled,
                "yolox_result_ready": self._yolox_result_ready,
                "actor_present_frames": self._actor_present_frames,
                "actor_unknown_frames": self._actor_unknown_frames,
                "actor_reason": self._actor_reason,
                # Messages arriving, then frames the decoder turned into
                # images, then PearFrames out; decode time is per call over
                # this status window, including calls that failed.
                "input_hz": round(self._window_received / elapsed, 2),
                "decode_hz": round(self._window_frames / elapsed, 2),
                "output_hz": round(self._window_published / elapsed, 2),
                "decode_ms": round(1000.0 * self._window_decode_sec / self._window_decode_calls, 2)
                if self._window_decode_calls
                else None,
                "decode_ms_max": round(1000.0 * self._window_decode_max_sec, 2) if self._window_decode_calls else None,
                "first_frame_after_sec": round(self._first_frame_after_sec, 2)
                if self._first_frame_after_sec is not None
                else None,
                "first_stamp": self._first_stamp,
                "last_stamp": self._last_stamp,
                "last_frame_id": self._last_frame_id,
                "last_age_sec": round(now - self._last_frame_at, 2) if self._last_frame_at else None,
                "vision_sample": self._vision_sample,
            }
            self._window_started_at = now
            self._window_received = 0
            self._window_decode_calls = 0
            self._window_frames = 0
            self._window_published = 0
            self._window_decode_sec = 0.0
            self._window_decode_max_sec = 0.0
        message = String()
        message.data = json.dumps(status, ensure_ascii=False)
        self._status_publisher.publish(message)
        if status["publishers"] == 0:
            reason = "source has no publisher"
        elif not status["source_available"]:
            reason = "source has conflicting publishers"
        elif status["frames"] == 0:
            reason = "source produced no decodable frames"
        elif self._last_frame_at and now - self._last_frame_at >= self._status_period_sec:
            reason = "source timed out: no recent decodable frame"
        elif (
            status["decoder_error"]
            and status["cpu_frames"] > 0
            and status["dvpp_frames"] == 0
            and self._decoder_name in ("auto", "dvpp")
        ):
            # auto falls back to CPU decode when hardware is unavailable; the
            # 30 FPS target only holds with DVPP, so surface the degradation.
            reason = "hardware decoder unavailable: decoding on CPU"
        else:
            reason = ""
        if reason:
            self.get_logger().warning(
                f"motion imitation stream warning: {reason} on {self._rgb_topic} "
                f"(publishers={status['publishers']}, received={status['received']}, "
                f"decode_failed={status['decode_failed']}, last_age_sec={status['last_age_sec']})"
            )
        else:
            self.get_logger().info(f"motion imitation stream status {message.data}")

    def destroy_node(self) -> bool:
        self._stop.set()
        stuck = False
        for worker in (self._decode_worker, self._pear_worker, self._yolox_worker):
            if worker.is_alive():
                worker.join(timeout=max(5.0, self._rpc_timeout + 1.0))
            if worker.is_alive():
                stuck = True
                self.get_logger().error(
                    f"worker {worker.name} did not stop; its decoder resources stay owned"
                    " and the shared decoder will not be closed"
                )
        if not stuck:
            self._decoder.close()
        # Finish teardown even when a worker is stuck: skipping super() would
        # leak every subscription, timer, client and publisher.
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    executor = MultiThreadedExecutor(num_threads=3)
    try:
        node = MotionImitationStreamNode()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
