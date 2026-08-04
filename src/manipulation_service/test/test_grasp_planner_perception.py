import threading
from collections import deque
from types import SimpleNamespace

import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from ibrobot_msgs.msg import Detection2D, DetectionArray
from ibrobot_msgs.srv import GroundingDetect, SegmentDetections
from manipulation_service import grasp_planner_node
from manipulation_service.grasp_planner_node import GraspPlannerNode, RGBFrame


class _Future:
    def __init__(self, result=None, *, complete=True):
        self._result = result
        self._complete = complete

    def add_done_callback(self, callback):
        if self._complete:
            callback(self)

    def result(self):
        return self._result


class _Client:
    def __init__(self, result=None, *, ready=True, complete=True):
        self.result = result
        self.ready = ready
        self.complete = complete
        self.requests = []

    def wait_for_service(self, timeout_sec):
        assert timeout_sec > 0
        return self.ready

    def call_async(self, request):
        self.requests.append(request)
        return _Future(self.result, complete=self.complete)


class _Bridge:
    @staticmethod
    def imgmsg_to_cv2(image, desired_encoding):
        assert desired_encoding == "mono8"
        return np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width)


class _Logger:
    def error(self, _message):
        pass


def _image(value=0):
    array = np.full((2, 2), value, dtype=np.uint8)
    return Image(
        header=Header(frame_id="camera"),
        height=2,
        width=2,
        encoding="mono8",
        step=2,
        data=array.tobytes(),
    )


def _detection(mask_value=0, *, confidence=0.9):
    detection = Detection2D()
    detection.header.frame_id = "camera"
    detection.label = "banana"
    detection.confidence = confidence
    detection.bbox = [0.0, 0.0, 2.0, 2.0]
    detection.mask = _image(mask_value)
    return detection


def _planner(detect_client, segment_client=None):
    rgb = _image()
    return SimpleNamespace(
        _detect_client=detect_client,
        _segment_client=segment_client,
        _lock=threading.Lock(),
        _rgb_buffer=deque([RGBFrame(stamp_ns=0, msg=rgb)]),
        _bridge=_Bridge(),
        get_logger=lambda: _Logger(),
    )


def test_configured_segment_service_runs_even_when_detector_returns_zero_mask_bytes():
    detect_response = GroundingDetect.Response(
        success=True,
        detections=DetectionArray(detections=[_detection(0)]),
    )
    segmented = _detection(255)
    segment_response = SegmentDetections.Response(
        success=True,
        detections=DetectionArray(detections=[segmented]),
    )
    segment_client = _Client(segment_response)
    planner = _planner(_Client(detect_response), segment_client)

    result, failure = GraspPlannerNode._get_segmentation_mask(planner, "banana", 0.3)

    assert failure is None
    assert len(segment_client.requests) == 1
    assert np.all(result[0] == 255)


def test_segment_service_receives_only_highest_confidence_detection():
    detections = [_detection(0, confidence=0.1 + index * 0.01) for index in range(5)]
    best_detection = _detection(0, confidence=0.99)
    detections.append(best_detection)
    detect_response = GroundingDetect.Response(
        success=True,
        detections=DetectionArray(header=Header(frame_id="detector"), detections=detections),
    )
    segmented = _detection(255, confidence=best_detection.confidence)
    segment_client = _Client(
        SegmentDetections.Response(success=True, detections=DetectionArray(detections=[segmented]))
    )
    planner = _planner(_Client(detect_response), segment_client)

    result, failure = GraspPlannerNode._get_segmentation_mask(planner, "banana", 0.3)

    assert failure is None
    requested = segment_client.requests[0].detections
    assert requested.header.frame_id == "detector"
    assert len(requested.detections) == 1
    assert requested.detections[0].confidence == best_detection.confidence
    assert np.all(result[0] == 255)


def test_no_target_is_reported_without_calling_segmentation():
    detect_response = GroundingDetect.Response(success=True, detections=DetectionArray(detections=[]))
    segment_client = _Client()
    planner = _planner(_Client(detect_response), segment_client)

    result, failure = GraspPlannerNode._get_segmentation_mask(planner, "banana", 0.3)

    assert result is None
    assert failure.startswith("no_detections:")
    assert not segment_client.requests


def test_model_not_ready_failure_is_preserved():
    detect_response = GroundingDetect.Response(success=False, message="runtime not ready")
    planner = _planner(_Client(detect_response))

    result, failure = GraspPlannerNode._get_segmentation_mask(planner, "banana", 0.3)

    assert result is None
    assert failure == "detect_service_failed: runtime not ready"


def test_detection_timeout_is_reported(monkeypatch):
    class _ImmediateTimeoutEvent:
        def set(self):
            pass

        def wait(self, timeout):
            assert timeout == 10.0
            return False

    monkeypatch.setattr(grasp_planner_node.threading, "Event", _ImmediateTimeoutEvent)
    planner = _planner(_Client(complete=False))

    result, failure = GraspPlannerNode._get_segmentation_mask(planner, "banana", 0.3)

    assert result is None
    assert failure == "detect_service_timeout"
