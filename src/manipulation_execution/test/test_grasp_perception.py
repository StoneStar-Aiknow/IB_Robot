"""The perception service clients, against fake model servers over real ROS services."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image

from ibrobot_msgs.msg import Detection2D, GraspCandidate
from ibrobot_msgs.srv import GenerateGrasps, GroundingDetect, SegmentDetections
from manipulation_execution.pick.pipeline.contracts import GraspFailure
from manipulation_execution.pick.pipeline.perception import Detection, PerceptionServices, image_mask

H, W = 6, 8
RGB = np.arange(H * W * 3, dtype=np.uint8).reshape(H, W, 3)
FRAME, STAMP = "head_camera", 1_700_000_000_123_456_789
BOXES = {"cup": [1.0, 1.0, 3.0, 4.0], "bowl": [4.0, 0.0, 7.0, 5.0]}


def box_mask(box, header) -> Image:
    mask = np.zeros((H, W), dtype=np.uint8)
    x0, y0, x1, y1 = (int(value) for value in box)
    mask[y0:y1, x0:x1] = 255
    return Image(header=header, height=H, width=W, encoding="mono8", step=W, data=mask.tobytes())


class FakeModels(Node):
    """Grounding detector, box segmenter and grasp generator; each answers for the image it was sent."""

    def __init__(self):
        super().__init__("fake_models")
        self.with_masks = False
        self.stamp_offset_ns = 0
        self.reverse_masks = False
        self.fail = False
        self.delay_s = 0.0
        self.requests: dict[str, list] = {"detect": [], "segment": [], "grasps": []}
        group = ReentrantCallbackGroup()
        self.create_service(GroundingDetect, "/test/detect", self._detect, callback_group=group)
        self.create_service(SegmentDetections, "/test/segment", self._segment, callback_group=group)
        self.create_service(GenerateGrasps, "/test/grasps", self._grasps, callback_group=group)

    def _header(self, header):
        header.stamp.nanosec = (header.stamp.nanosec + self.stamp_offset_ns) % 1_000_000_000
        return header

    def _done(self, response, inference_ms=7.0):
        time.sleep(self.delay_s)
        response.inference_time_ms = inference_ms
        response.success = not self.fail
        response.message = "model crashed" if self.fail else ""
        return response

    def _detect(self, request, response):
        self.requests["detect"].append(request)
        header = self._header(request.image.header)
        response.detections.header = header
        for label, score in (("bowl", 0.4), ("cup", 0.9)):
            item = Detection2D(header=header, label=label, confidence=score, bbox=BOXES[label])
            if self.with_masks:
                item.mask = box_mask(BOXES[label], header)
            response.detections.detections.append(item)
        return self._done(response)

    def _segment(self, request, response):
        self.requests["segment"].append(request)
        header = self._header(request.image.header)
        response.detections.header = header
        items = list(request.detections.detections)
        for item in reversed(items) if self.reverse_masks else items:
            response.detections.detections.append(
                Detection2D(header=header, label=item.label, bbox=item.bbox, mask=box_mask(item.bbox, header))
            )
        return self._done(response)

    def _grasps(self, request, response):
        self.requests["grasps"].append(request)
        response.grasps.header = self._header(request.object_points.header)
        for index in range(min(3, request.max_grasps)):
            pose = np.eye(4)
            pose[:3, 3] = (index, 0.5, 0.25)
            response.grasps.grasps.append(
                GraspCandidate(
                    header=response.grasps.header, pose_matrix=pose.ravel().tolist(), confidence=0.9 - index / 10
                )
            )
        return self._done(response, inference_ms=12.0)


@pytest.fixture
def ros():
    rclpy.init()
    models = FakeModels()
    client = Node("perception_client")
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(models)
    executor.add_node(client)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    def services(**options):
        options = {
            "detect": "/test/detect",
            "segment": "/test/segment",
            "generate_grasps": "/test/grasps",
            "timeout_s": 5.0,
            **options,
        }
        return PerceptionServices(client, **options)

    yield models, services
    executor.shutdown()
    models.destroy_node()
    client.destroy_node()
    rclpy.shutdown()


def detect(services):
    return services.detect(RGB, FRAME, STAMP, "cup.", box_threshold=0.25, text_threshold=0.25)


def test_detections_come_best_first_and_their_own_masks_skip_segmentation(ros):
    models, make = ros
    models.with_masks = True
    services = make()
    detections = detect(services)
    assert [d.label for d in detections] == ["cup", "bowl"]
    sent = models.requests["detect"][0]
    assert (sent.text_prompt, sent.image.encoding, sent.image.header.frame_id) == ("cup.", "rgb8", FRAME)
    np.testing.assert_array_equal(np.frombuffer(bytes(sent.image.data), np.uint8).reshape(H, W, 3), RGB)
    masks = services.masks(RGB, FRAME, STAMP, detections)
    assert not models.requests["segment"]
    assert masks[0][1:4, 1:3].all() and masks[0].sum() == 6
    assert [call["service"] for call in services.calls] == ["/test/detect"]
    assert services.calls[0]["inference_s"] == pytest.approx(0.007)


def test_boxes_without_masks_are_segmented_in_order(ros):
    models, make = ros
    services = make()
    detections = detect(services)
    assert all(d.mask is None for d in detections)
    masks = services.masks(RGB, FRAME, STAMP, detections)
    boxes = [list(item.bbox) for item in models.requests["segment"][0].detections.detections]
    assert boxes == [BOXES["cup"], BOXES["bowl"]]
    assert masks[1][0:5, 4:7].all() and masks[1].sum() == 15
    models.reverse_masks = True
    with pytest.raises(GraspFailure, match="box order"):
        services.masks(RGB, FRAME, STAMP, detections)


def test_a_detector_without_masks_needs_a_segmentation_service(ros):
    _, make = ros
    services = make(segment=None)
    with pytest.raises(GraspFailure, match="no segmentation is bound"):
        services.masks(RGB, FRAME, STAMP, detect(services))


def test_a_response_for_another_image_is_refused(ros):
    models, make = ros
    models.stamp_offset_ns = 1
    with pytest.raises(GraspFailure, match="does not belong to the requested image"):
        detect(make())


def test_grasps_return_as_poses_in_the_cloud_frame(ros):
    models, make = ros
    services = make()
    points = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    poses, confidence = services.generate_grasps(points, FRAME, STAMP, max_grasps=2, min_confidence=-1.0)
    cloud = models.requests["grasps"][0].object_points
    assert (cloud.width, cloud.point_step, [field.name for field in cloud.fields]) == (2, 12, ["x", "y", "z"])
    np.testing.assert_allclose(np.frombuffer(bytes(cloud.data), "<f4").reshape(-1, 3), points, rtol=1e-6)
    assert poses.shape == (2, 4, 4)
    np.testing.assert_allclose(poses[1, :3, 3], (1.0, 0.5, 0.25))
    np.testing.assert_allclose(confidence, (0.9, 0.8), rtol=1e-6)
    assert services.calls[0]["request_bytes"] == 24 and services.calls[0]["rpc_s"] >= services.calls[0]["transport_s"]


def test_a_model_failure_ends_the_pick_and_names_the_service(ros):
    models, make = ros
    models.fail = True
    with pytest.raises(GraspFailure, match="/test/grasps failed: model crashed") as raised:
        make().generate_grasps(np.zeros((1, 3)), FRAME, STAMP, max_grasps=1, min_confidence=0.0)
    assert raised.value.code == "RPC_FAILED" and not raised.value.retryable


def test_waiting_stops_on_cancel_and_on_a_missing_service(ros):
    models, make = ros
    models.delay_s = 1.0
    started = time.monotonic()
    with pytest.raises(GraspFailure, match="cancelled") as raised:
        make().detect(
            RGB,
            FRAME,
            STAMP,
            "cup.",
            box_threshold=0.25,
            text_threshold=0.25,
            cancelled=lambda: time.monotonic() - started > 0.1,
        )
    assert raised.value.code == "CANCELED" and time.monotonic() - started < 0.9
    with pytest.raises(GraspFailure, match="/test/none is not available") as raised:
        detect(make(detect="/test/none", timeout_s=0.2))
    assert raised.value.code == "RPC_FAILED"


def test_masks_must_be_mono8_rows_of_the_image():
    data = np.zeros((2, 4), dtype=np.uint8)
    data[1, 2] = 1
    padded = Image(height=2, width=3, encoding="mono8", step=4, data=data.tobytes())
    np.testing.assert_array_equal(image_mask(padded), [[False] * 3, [False, False, True]])
    with pytest.raises(ValueError, match="encoding"):
        image_mask(Image(height=1, width=1, encoding="rgb8", step=3, data=b"\0\0\0"))
    assert Detection("cup", 0.9, np.zeros(4)).mask is None
