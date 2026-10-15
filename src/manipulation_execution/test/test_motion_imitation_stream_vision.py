"""Perception loop of the resident motion-imitation stream.

The windowed executor scoped every frame to a capture epoch and a task
deadline. This node has neither, so these tests are the evidence that removing
that machinery did not take any perception behaviour with it: the YOLOX cache
policy, the centred-crop fallback, the response validation and the counter
invariant must all still hold, and a resident stream additionally has to count
what it sheds.

Methods are rebound onto a plain object, as
``test_imitate_human_motion_vision.py`` does, so none of this needs rclpy, a
live graph or a robot.
"""

import json
import math
import queue
import threading
from types import SimpleNamespace

import pytest
from sensor_msgs.msg import CompressedImage, Image

from ibrobot_msgs.msg import Detection2D, DetectionArray
from manipulation_execution.actor_selector import ActorSelector
from manipulation_execution.hri_vision import (
    PEAR_CROP_EXPANSION,
    PEAR_FIELD_LENGTHS,
    PEAR_OUTPUT_FIELDS,
    put_latest,
)
from manipulation_execution.motion_imitation_stream_node import MotionImitationStreamNode


class _Logger:
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, message):
        self.infos.append(message)

    def warning(self, message):
        self.warnings.append(message)

    def error(self, _message):
        pass


class _ImmediateFuture:
    def __init__(self, result):
        self._result = result
        self.cancelled = False

    def done(self):
        return True

    def result(self):
        return self._result

    def cancel(self):
        self.cancelled = True


class _Client:
    def __init__(self, name, responses=(), *, ready=True):
        self.srv_name = name
        self._responses = list(responses)
        self._ready = ready
        self.requests = []

    def service_is_ready(self):
        return self._ready

    def call_async(self, request):
        self.requests.append(request)
        return _ImmediateFuture(self._responses.pop(0))


class _Publisher:
    def __init__(self):
        self.published = []

    def publish(self, message):
        self.published.append(message)


class _StreamHarness:
    _invalidate_source_locked = MotionImitationStreamNode._invalidate_source_locked
    _refresh_source_health = MotionImitationStreamNode._refresh_source_health
    _frame_current_locked = MotionImitationStreamNode._frame_current_locked
    _publish_status = MotionImitationStreamNode._publish_status
    _source_available = MotionImitationStreamNode._source_available
    _on_compressed_frame = MotionImitationStreamNode._on_compressed_frame
    _decode_frame = MotionImitationStreamNode._decode_frame
    _pear_loop = MotionImitationStreamNode._pear_loop
    _yolox_loop = MotionImitationStreamNode._yolox_loop
    _process_yolox_frame = MotionImitationStreamNode._process_yolox_frame
    _process_pear_frame = MotionImitationStreamNode._process_pear_frame
    _record_actor = MotionImitationStreamNode._record_actor
    _call_vision = MotionImitationStreamNode._call_vision
    _record_vision_failure = MotionImitationStreamNode._record_vision_failure

    def get_logger(self):
        return self._logger


def _image(width=640, height=480):
    image = Image(width=width, height=height, encoding="bgr8", step=width * 3)
    image.header.frame_id = "head"
    image.header.stamp.sec = 3
    image.header.stamp.nanosec = 500000000
    return image


def _detection(label, confidence, bbox):
    return Detection2D(label=label, confidence=confidence, bbox=bbox)


def _yolox_response(*detections, success=True, message=""):
    return SimpleNamespace(
        success=success,
        message=message,
        detections=DetectionArray(detections=list(detections)),
    )


def _pear_response(*, success=True, message="", empty_field=None, nonfinite_field=None, short_field=None):
    # Correctly sized by default: the node rejects a response whose arrays do
    # not match the compiled deployment's widths.
    values = {name: [0.5] * width for name, width in PEAR_FIELD_LENGTHS.items()}
    if empty_field is not None:
        values[empty_field] = []
    if nonfinite_field is not None:
        values[nonfinite_field] = [float("nan")] * PEAR_FIELD_LENGTHS[nonfinite_field]
    if short_field is not None:
        values[short_field] = [0.5] * (PEAR_FIELD_LENGTHS[short_field] - 1)
    return SimpleNamespace(success=success, message=message, inference_time_ms=9.5, **values)


def _harness(*, yolox=(), pear=(), yolox_ready=True, pear_ready=True, publish_fallback=True, max_pear_hz=0.0):
    harness = _StreamHarness()
    harness._logger = _Logger()
    harness._stats_lock = threading.RLock()
    harness._source_generation = 0
    harness._source_healthy = None
    harness._last_capture_stamp = None
    harness._last_encoded_at = 0.0
    harness._stop = threading.Event()
    harness._rgb_topic = "/camera/head/image/compressed"
    harness._require_unique_publisher = False
    harness._person_confidence_threshold = 0.30
    harness._yolox_refresh_interval_sec = 1.0
    harness._rpc_timeout = 1.0
    harness._status_period_sec = 5.0
    harness._min_pear_period_sec = 1.0 / max_pear_hz if max_pear_hz else 0.0
    harness._next_pear_at = 0.0
    harness._publish_fallback_frames = publish_fallback
    harness._decode_queue = queue.Queue(maxsize=1)
    harness._pear_queue = queue.Queue(maxsize=1)
    harness._yolox_queue = queue.Queue(maxsize=1)
    harness._frames = 0
    harness._window_frames = 0
    harness._frames_dropped = 0
    harness._decode_failed = 0
    harness._window_decode_calls = 0
    harness._window_decode_sec = 0.0
    harness._window_decode_max_sec = 0.0
    harness._first_frame_after_sec = None
    harness._decoder_name = "mock_cpu"
    harness._rejected_format = ""
    harness._vision_ok = 0
    harness._vision_detected = 0
    harness._vision_fallback = 0
    harness._vision_failed = 0
    harness._yolox_calls = 0
    harness._pear_calls = 0
    harness._published = 0
    harness._window_published = 0
    harness._throttled = 0
    harness._sequence = 0
    harness._vision_sample = ""
    harness._first_stamp = ""
    harness._last_stamp = ""
    harness._last_frame_id = ""
    harness._last_frame_at = 0.0
    harness._yolox_detections = None
    harness._source_geometry = None
    harness._yolox_result_ready = False
    harness._actor_selector = ActorSelector(0.30, 3.0)
    harness._actor_publisher = _Publisher()
    harness._actor_present_frames = 0
    harness._actor_unknown_frames = 0
    harness._actor_reason = ""
    harness._announced_first_frame = False
    harness._failure_logged_at = {}
    harness._started_at = 0.0
    harness._yolox_client = _Client("/yolox", yolox, ready=yolox_ready)
    harness._pear_client = _Client("/pear", pear, ready=pear_ready)
    harness._frame_publisher = _Publisher()
    return harness


def _status_harness(monkeypatch, *, publishers, frames, last_frame_at):
    monkeypatch.setattr("manipulation_execution.motion_imitation_stream_node.time.monotonic", lambda: 20.0)
    harness = _harness()
    harness._frame_source_interface = ""
    harness._require_unique_publisher = True
    harness.count_publishers = lambda _topic: publishers
    harness._source_rejected = 0
    harness._received = frames
    harness._window_received = frames
    harness._compressed_dropped = 0
    harness._out_of_order = 0
    harness._decoder = SimpleNamespace()
    harness._window_started_at = 15.0
    harness._frames = frames
    harness._last_frame_at = last_frame_at
    harness._status_publisher = _Publisher()
    return harness


@pytest.mark.parametrize(
    "publishers, frames, last_frame_at, reason",
    [
        (0, 0, 0.0, "source has no publisher"),
        (0, 3, 19.0, "source has no publisher"),
        (2, 0, 0.0, "source has conflicting publishers"),
        (2, 3, 19.0, "source has conflicting publishers"),
        (1, 0, 0.0, "source produced no decodable frames"),
        (1, 3, 15.0, "source timed out"),
    ],
)
def test_source_problems_warn_at_status_intervals(monkeypatch, publishers, frames, last_frame_at, reason):
    harness = _status_harness(monkeypatch, publishers=publishers, frames=frames, last_frame_at=last_frame_at)
    harness._publish_status()
    assert len(harness._logger.warnings) == 1
    warning = harness._logger.warnings[0]
    assert reason in warning
    assert harness._rgb_topic in warning
    assert f"publishers={publishers}" in warning
    assert "last_age_sec=" in warning
    assert harness._logger.infos == []
    status = json.loads(harness._status_publisher.published[0].data)
    assert status["frames"] == frames
    assert status["source_available"] is (publishers == 1)


def test_healthy_source_keeps_informational_status(monkeypatch):
    harness = _status_harness(monkeypatch, publishers=1, frames=3, last_frame_at=19.0)
    harness._publish_status()
    assert harness._logger.warnings == []
    assert len(harness._logger.infos) == 1
    assert "motion imitation stream status" in harness._logger.infos[0]


def test_source_recovers_from_missing_publisher_without_restarting(monkeypatch):
    harness = _status_harness(monkeypatch, publishers=0, frames=0, last_frame_at=0.0)
    harness._publish_status()
    harness.count_publishers = lambda _topic: 1
    # The health timer, not the status callback, owns the graph probe; the
    # recovery between two status periods is exactly one of its ticks.
    harness._refresh_source_health()
    harness._frames = 1
    harness._last_frame_at = 19.0
    harness._publish_status()
    assert len(harness._logger.warnings) == 1
    assert len(harness._logger.infos) == 1
    assert len(harness._status_publisher.published) == 2


def test_expected_hardware_decoder_on_cpu_is_a_status_warning(monkeypatch):
    harness = _status_harness(monkeypatch, publishers=1, frames=3, last_frame_at=19.0)
    harness._decoder_name = "auto"
    harness._decoder = SimpleNamespace(cpu_frames=3, dvpp_frames=0, last_error="ImportError: ACL not loaded")
    harness._publish_status()
    assert len(harness._logger.warnings) == 1
    warning = harness._logger.warnings[0]
    assert "hardware decoder unavailable: decoding on CPU" in warning


def test_source_health_is_probed_by_the_timer_not_the_frame_path():
    harness = _harness()
    harness._require_unique_publisher = True
    probes = []

    def counting_probe(_topic):
        probes.append(1)
        return 1

    harness.count_publishers = counting_probe
    harness._source_healthy = True
    with harness._stats_lock:
        # The frame path reads the cache only: thirty frames, zero probes.
        for _ in range(30):
            assert harness._source_available()
    assert probes == []

    harness.count_publishers = lambda _topic: 2
    harness._refresh_source_health()
    assert harness._source_healthy is False
    assert not harness._source_available()
    assert harness._source_generation == 1


def test_first_source_check_probes_once_before_the_timer_ticks():
    harness = _harness()
    harness._require_unique_publisher = True
    probes = []
    harness.count_publishers = lambda _topic: probes.append(1) or 1
    assert harness._source_available() is True
    assert harness._source_available() is True
    assert probes == [1]


def test_resolution_change_invalidates_actor_detections():
    harness = _harness(yolox=[_yolox_response(_detection("person", 0.9, [10.0, 20.0, 110.0, 220.0]))])
    harness._decoder = SimpleNamespace(decode=lambda message: message)
    harness._decode_frame(_image())
    assert harness._process_yolox_frame(_image())
    harness._decode_frame(_image(width=320, height=240))
    assert harness._yolox_detections is None
    assert not harness._yolox_result_ready
    harness._process_pear_frame(_image(width=320, height=240))
    assert not harness._actor_publisher.published[-1].actor_present
    assert not harness._pear_client.requests
    assert harness._window_decode_calls == 2


def test_inflight_old_resolution_does_not_repopulate_actor_cache():
    harness = _harness(yolox=[_yolox_response(_detection("person", 0.9, [10.0, 20.0, 110.0, 220.0]))])
    harness._source_geometry = (320, 240)
    assert not harness._process_yolox_frame(_image())
    assert harness._yolox_detections is None
    assert not harness._yolox_result_ready


@pytest.mark.parametrize("metric", ["yolox", "pear"])
@pytest.mark.parametrize("publisher_count", [0, 2])
def test_source_fault_and_recovery_during_rpc_rejects_old_result(metric, publisher_count):
    harness = _harness(
        yolox=[_yolox_response(_detection("person", 0.9, [10.0, 20.0, 110.0, 220.0]))],
        pear=[_pear_response()],
    )
    image = _image()
    assert harness._process_yolox_frame(image)
    harness._require_unique_publisher = True
    publishers = [1]
    harness.count_publishers = lambda _topic: publishers[0]
    assert harness._source_available()
    harness._source_rejected = 0
    generation = harness._source_generation

    def result_after_fault(_client, _request, *, metric):
        publishers[0] = publisher_count
        # The health timer, not the frame path, notices the fault and retires
        # the generation; the frame callback then reads the cached verdict.
        harness._refresh_source_health()
        harness._on_compressed_frame(CompressedImage())
        publishers[0] = 1
        harness._refresh_source_health()
        assert harness._source_available()
        return _yolox_response() if metric == "yolox" else _pear_response()

    harness._call_vision = result_after_fault
    if metric == "yolox":
        assert not harness._process_yolox_frame(image, generation)
    else:
        harness._process_pear_frame(image, generation)
    assert harness._source_generation == generation + 1
    assert harness._yolox_detections is None
    assert not harness._yolox_result_ready
    assert harness._frame_publisher.published == []


def test_resolution_change_rejects_a_dequeued_old_pear_frame():
    harness = _harness(yolox=[_yolox_response(_detection("person", 0.9, [10.0, 20.0, 110.0, 220.0]))])
    harness._decoder = SimpleNamespace(decode=lambda message: message)
    harness._decode_frame(_image())
    held = harness._pear_queue.get_nowait()
    newest = _image(width=320, height=240)
    newest.header.stamp.sec = 4
    harness._decode_frame(newest)
    assert harness._process_yolox_frame(newest)
    harness._process_pear_frame(held.message, held.generation)
    assert harness._pear_client.requests == []
    assert harness._actor_publisher.published == []
    assert harness._frame_publisher.published == []


def test_pear_geometric_check_also_rejects_old_frame_without_queue_metadata():
    harness = _harness()
    harness._source_geometry = (320, 240)
    harness._process_pear_frame(_image())
    assert harness._pear_client.requests == []
    assert harness._actor_publisher.published == []


@pytest.mark.parametrize("metric", ["yolox", "pear"])
def test_geometry_change_and_return_to_same_size_rejects_old_rpc(metric):
    harness = _harness(yolox=[_yolox_response()])
    harness._decoder = SimpleNamespace(decode=lambda message: message)
    image = _image()
    harness._decode_frame(image)
    assert harness._process_yolox_frame(image)
    generation = harness._source_generation

    def result_after_resize(_client, _request, *, metric):
        harness._decode_frame(_image(width=320, height=240))
        harness._decode_frame(_image())
        return _yolox_response() if metric == "yolox" else _pear_response()

    harness._call_vision = result_after_resize
    if metric == "yolox":
        assert not harness._process_yolox_frame(image, generation)
    else:
        harness._process_pear_frame(image, generation)
    assert harness._source_geometry == (image.width, image.height)
    assert harness._source_generation == generation + 2
    assert not harness._yolox_result_ready
    assert harness._frame_publisher.published == []


def test_timestamp_restart_clears_old_detection_and_waits_for_fresh_yolox(monkeypatch):
    harness = _harness(yolox=[_yolox_response(_detection("person", 0.9, [10.0, 20.0, 110.0, 220.0]))])
    old = _image()
    old.header.stamp.sec = 100
    assert harness._process_yolox_frame(old)
    harness._last_capture_stamp = (100, 0)
    harness._last_encoded_at = 10.0
    harness._received = harness._window_received = harness._out_of_order = harness._compressed_dropped = 0
    generation = harness._source_generation
    monkeypatch.setattr("manipulation_execution.motion_imitation_stream_node.time.monotonic", lambda: 20.0)
    message = CompressedImage()
    message.header.stamp.sec = 1
    harness._on_compressed_frame(message)
    accepted = harness._decode_queue.get_nowait()
    assert accepted.generation == generation + 1
    assert accepted.message is message
    assert harness._yolox_detections is None
    assert not harness._yolox_result_ready
    fresh = _image()
    fresh.header.stamp = message.header.stamp
    harness._process_pear_frame(fresh, accepted.generation)
    assert not harness._actor_publisher.published[-1].actor_present
    assert harness._pear_client.requests == []


def test_fault_during_decode_discards_old_decoded_image():
    harness = _harness()
    generation = harness._source_generation

    def decode_after_restart(message):
        with harness._stats_lock:
            harness._invalidate_source_locked()
        return message

    harness._decoder = SimpleNamespace(decode=decode_after_restart)
    harness._decode_frame(_image(), generation)
    assert harness._frames == 0
    assert harness._pear_queue.empty()
    assert harness._yolox_queue.empty()


def test_generation_invalidation_flushes_all_waiting_slots_and_throttle():
    harness = _harness()
    harness._next_pear_at = 100.0
    for slot in (harness._decode_queue, harness._pear_queue, harness._yolox_queue):
        slot.put(object())
    with harness._stats_lock:
        harness._invalidate_source_locked()
    assert harness._source_generation == 1
    assert harness._next_pear_at == 0.0
    assert all(slot.empty() for slot in (harness._decode_queue, harness._pear_queue, harness._yolox_queue))


@pytest.mark.parametrize("publisher_count", [0, 2])
def test_fresh_detection_and_pear_resume_after_source_recovery(publisher_count):
    detection = _detection("person", 0.9, [10.0, 20.0, 110.0, 220.0])
    harness = _harness(
        yolox=[_yolox_response(detection), _yolox_response(detection)],
        pear=[_pear_response(), _pear_response()],
    )
    harness._require_unique_publisher = True
    publishers = [1]
    harness.count_publishers = lambda _topic: publishers[0]
    harness._decoder = SimpleNamespace(decode=lambda message: message)
    harness._decode_frame(_image())
    initial = harness._pear_queue.get_nowait()
    assert harness._process_yolox_frame(initial.message, initial.generation)
    harness._process_pear_frame(initial.message, initial.generation)
    publishers[0] = publisher_count
    # The timer probe retires the generation; the frame paths only read the
    # cached verdict, so they see the fault and the recovery through it.
    harness._refresh_source_health()
    assert not harness._source_available()
    publishers[0] = 1
    harness._refresh_source_health()
    recovered = _image()
    recovered.header.stamp.sec = 4
    harness._decode_frame(recovered)
    newest = harness._pear_queue.get_nowait()
    assert newest.generation > initial.generation
    assert harness._process_yolox_frame(newest.message, newest.generation)
    harness._process_pear_frame(newest.message, newest.generation)
    assert [frame.sequence for frame in harness._frame_publisher.published] == [1, 2]
    assert harness._frame_publisher.published[-1].header.stamp.sec == 4


def test_pear_is_not_called_before_the_first_yolox_result():
    harness = _harness(pear=[_pear_response()])

    harness._process_pear_frame(_image())

    assert harness._pear_client.requests == []
    assert harness._frame_publisher.published == []


@pytest.mark.parametrize("publishers", [0, 2])
def test_guarded_stream_stops_decode_and_inference_when_source_is_unavailable(publishers):
    harness = _harness()
    harness._require_unique_publisher = True
    harness.count_publishers = lambda _topic: publishers
    harness._decode_frame(_image())
    assert harness._process_yolox_frame(_image()) is False
    harness._process_pear_frame(_image())
    assert harness._window_decode_calls == 0
    assert harness._pear_client.requests == []
    assert harness._yolox_client.requests == []
    assert harness._actor_publisher.published == []
    assert harness._frame_publisher.published == []


def test_publisher_conflict_during_pear_call_discards_the_inflight_result():
    harness = _harness(yolox=[_yolox_response()], pear=[_pear_response()])
    assert harness._process_yolox_frame(_image())
    publishers = [1]
    harness._require_unique_publisher = True
    harness.count_publishers = lambda _topic: publishers[0]
    call_vision = harness._call_vision

    def conflicting_response(client, request, *, metric):
        result = call_vision(client, request, metric=metric)
        publishers[0] = 2
        # The timer probe records the conflict so the publish check that
        # follows the RPC reads the refreshed cache and discards the result.
        harness._refresh_source_health()
        return result

    harness._call_vision = conflicting_response
    harness._process_pear_frame(_image())
    assert len(harness._pear_client.requests) == 1
    assert harness._frame_publisher.published == []
    assert harness._published == 0


def test_detected_person_is_published_with_its_box():
    harness = _harness(
        yolox=[_yolox_response(_detection("person", 0.9, [10.0, 20.0, 110.0, 220.0]))],
        pear=[_pear_response()],
    )

    assert harness._process_yolox_frame(_image()) is True
    harness._process_pear_frame(_image())

    assert len(harness._frame_publisher.published) == 1
    frame = harness._frame_publisher.published[0]
    assert frame.source == "yolox"
    assert frame.actor_id == 0
    assert harness._actor_publisher.published[-1].actor_id == frame.actor_id
    assert harness._actor_publisher.published[-1].actor_present
    assert pytest.approx(frame.confidence, abs=1e-6) == 0.9
    assert list(frame.bbox) == [10.0, 20.0, 110.0, 220.0]
    assert len(frame.smplx_pose_raw) == PEAR_FIELD_LENGTHS["smplx_pose_raw"]
    assert frame.sequence == 1
    assert (harness._vision_ok, harness._vision_detected, harness._vision_fallback) == (1, 1, 0)


def test_successful_refresh_without_a_person_switches_to_the_centred_crop():
    harness = _harness(yolox=[_yolox_response()], pear=[_pear_response()])

    harness._process_yolox_frame(_image(width=640, height=480))
    harness._process_pear_frame(_image(width=640, height=480))

    frame = harness._frame_publisher.published[0]
    assert frame.source == "center_fallback"
    assert frame.actor_id == -1
    assert not harness._actor_publisher.published[-1].actor_present
    assert frame.confidence == 0.0
    side = min(640, 480) / PEAR_CROP_EXPANSION
    assert pytest.approx(frame.bbox[2] - frame.bbox[0], abs=1e-4) == side
    assert (harness._vision_ok, harness._vision_detected, harness._vision_fallback) == (1, 0, 1)


def test_failed_refresh_keeps_the_previous_box():
    harness = _harness(
        yolox=[
            _yolox_response(_detection("person", 0.8, [1.0, 2.0, 3.0, 4.0])),
            _yolox_response(success=False, message="model busy"),
        ],
        pear=[_pear_response()],
    )

    harness._process_yolox_frame(_image())
    assert harness._process_yolox_frame(_image()) is False
    harness._process_pear_frame(_image())

    # A refresh that failed must not degrade a working stream to the fallback.
    assert harness._frame_publisher.published[0].source == "yolox"
    assert harness._vision_failed == 1


def test_low_confidence_person_is_ignored():
    harness = _harness(
        yolox=[_yolox_response(_detection("person", 0.1, [1.0, 2.0, 3.0, 4.0]))],
        pear=[_pear_response()],
    )

    harness._process_yolox_frame(_image())
    harness._process_pear_frame(_image())

    assert harness._frame_publisher.published[0].source == "center_fallback"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"empty_field": "smplx_pose_raw"},
        {"nonfinite_field": "camera_raw"},
        {"short_field": "smplx_pose_raw"},
        {"success": False, "message": "inference failed"},
    ],
)
def test_rejected_pear_response_publishes_nothing_and_counts_a_failure(kwargs):
    harness = _harness(
        yolox=[_yolox_response(_detection("person", 0.9, [1.0, 2.0, 3.0, 4.0]))],
        pear=[_pear_response(**kwargs)],
    )

    harness._process_yolox_frame(_image())
    harness._process_pear_frame(_image())

    assert harness._frame_publisher.published == []
    assert harness._vision_ok == 0
    assert harness._vision_failed == 1


def test_counter_invariant_holds_across_a_mixed_run():
    harness = _harness(
        yolox=[
            _yolox_response(_detection("person", 0.9, [1.0, 2.0, 3.0, 4.0])),
            _yolox_response(),
        ],
        pear=[_pear_response(), _pear_response()],
    )

    harness._process_yolox_frame(_image())
    harness._process_pear_frame(_image())
    harness._process_yolox_frame(_image())
    harness._process_pear_frame(_image())

    assert harness._vision_ok == harness._vision_detected + harness._vision_fallback
    assert harness._vision_ok == 2
    assert harness._published == 2
    assert [frame.sequence for frame in harness._frame_publisher.published] == [1, 2]


def test_suppressed_fallback_still_counts_but_does_not_publish():
    harness = _harness(yolox=[_yolox_response()], pear=[_pear_response()], publish_fallback=False)

    harness._process_yolox_frame(_image())
    harness._process_pear_frame(_image())

    assert harness._frame_publisher.published == []
    assert harness._vision_fallback == 1
    assert harness._published == 0


def test_unavailable_service_is_a_counted_failure_without_a_call():
    harness = _harness(yolox_ready=False)

    assert harness._process_yolox_frame(_image()) is False
    assert harness._yolox_client.requests == []
    assert harness._vision_failed == 1


def test_zero_sized_image_is_rejected_before_any_service_call():
    harness = _harness()

    assert harness._process_yolox_frame(_image(width=0, height=0)) is False
    assert harness._yolox_client.requests == []
    assert harness._vision_failed == 1


def test_pear_request_carries_exactly_one_detection():
    harness = _harness(
        yolox=[_yolox_response(_detection("person", 0.9, [1.0, 2.0, 3.0, 4.0]))],
        pear=[_pear_response()],
    )

    harness._process_yolox_frame(_image())
    harness._process_pear_frame(_image())

    # The compiled deployment is batch 1; more than one crop per call is a
    # silent shape error rather than a service failure.
    assert len(harness._pear_client.requests[0].detections.detections) == 1


def test_published_frame_keeps_the_source_observation_time():
    harness = _harness(
        yolox=[_yolox_response(_detection("person", 0.9, [1.0, 2.0, 3.0, 4.0]))],
        pear=[_pear_response()],
    )

    harness._process_yolox_frame(_image())
    harness._process_pear_frame(_image())

    frame = harness._frame_publisher.published[0]
    assert (frame.header.stamp.sec, frame.header.stamp.nanosec) == (3, 500000000)
    assert frame.header.frame_id == "head"


def test_all_pear_fields_reach_the_published_frame():
    harness = _harness(
        yolox=[_yolox_response(_detection("person", 0.9, [1.0, 2.0, 3.0, 4.0]))],
        pear=[_pear_response()],
    )

    harness._process_yolox_frame(_image())
    harness._process_pear_frame(_image())

    frame = harness._frame_publisher.published[0]
    for name in PEAR_OUTPUT_FIELDS:
        assert len(getattr(frame, name)) == PEAR_FIELD_LENGTHS[name], name
    assert math.isfinite(frame.inference_time_ms)


def test_put_latest_keeps_the_newest_frame_and_reports_the_eviction():
    # Depth 1 and latest-wins is what keeps a slow consumer from growing
    # memory; the return value is how a resident stream counts what it shed,
    # which a bounded capture window never needed to know.
    slot: queue.Queue = queue.Queue(maxsize=1)

    assert put_latest(slot, "first") is False
    assert put_latest(slot, "second") is True
    assert slot.get_nowait() == "second"
    assert slot.empty()
