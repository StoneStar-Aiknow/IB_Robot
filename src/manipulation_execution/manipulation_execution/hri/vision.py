"""Robot-agnostic HRI vision helpers shared by the imitation executor and the stream node.

They are pure functions over ROS messages: no node, no joints,
no actuation, no pixel access -- only ``width``/``height`` header fields. The
streaming node needs exactly the same person-selection and crop-fallback
semantics, so they live here to keep one source of truth rather than two
copies that can drift apart.
"""

from __future__ import annotations

import math
import queue

from sensor_msgs.msg import Image

from ibrobot_msgs.msg import Detection2D

# The PEAR adapter crops a square around the person. A frame with no usable
# detection is still worth one inference on a centred square of this fraction
# of the short side, which is what the adapter's own crop expansion assumes.
PEAR_CROP_EXPANSION = 1.25

# The eight parameter blocks a PEAR response carries, and the width each one
# must have. Both come from the compiled deployment's output table in
# ``perception_service/pear_adapter.py`` (``PearParameterAdapter._outputs``);
# ``smplx_pose_raw`` is 52 packed 6D rotations (0 root, 1-21 body, 22-36 left
# hand, 37-51 right hand), which is what ``imitation_retargeting/frames.py``
# decodes. A short or over-long array means the bundle's output shape changed,
# which is a counted failure rather than something to publish.
PEAR_OUTPUT_FIELDS = (
    "smplx_pose_raw",
    "smplx_scale",
    "smplx_shape",
    "smplx_expression",
    "flame_pose",
    "flame_shape",
    "flame_expression",
    "camera_raw",
)
PEAR_FIELD_LENGTHS = {
    "smplx_pose_raw": 312,
    "smplx_scale": 6,
    "smplx_shape": 200,
    "smplx_expression": 50,
    "flame_pose": 14,
    "flame_shape": 300,
    "flame_expression": 50,
    "camera_raw": 3,
}


def validated_person_confidence_threshold(value: object) -> float:
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("person_confidence_threshold must be finite and within [0, 1]")
    return threshold


def validated_yolox_refresh_interval(value: object) -> float:
    interval = float(value)
    if not math.isfinite(interval) or interval <= 0.0:
        raise ValueError("yolox_refresh_interval_sec must be finite and greater than zero")
    return interval


def qualifying_persons(detections: list[Detection2D], threshold: float) -> list[Detection2D]:
    return [
        detection
        for detection in detections
        if detection.label == "person"
        and math.isfinite(float(detection.confidence))
        and float(detection.confidence) >= threshold
    ]


def select_person_detection(detections: list[Detection2D], threshold: float) -> Detection2D | None:
    persons = qualifying_persons(detections, threshold)
    return max(persons, key=lambda detection: float(detection.confidence), default=None)


def center_fallback_detection(image: Image) -> Detection2D:
    width = int(image.width)
    height = int(image.height)
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    center_x = width / 2.0
    center_y = height / 2.0
    half_side = min(width, height) / PEAR_CROP_EXPANSION / 2.0
    detection = Detection2D()
    detection.header = image.header
    detection.label = "person"
    detection.confidence = 0.0
    detection.bbox = [
        center_x - half_side,
        center_y - half_side,
        center_x + half_side,
        center_y + half_side,
    ]
    return detection


def put_latest(frame_queue: queue.Queue, frame: object) -> bool:
    """Replace whatever is queued with ``frame``. Returns True if a frame was evicted.

    The queues are depth 1 so a slow consumer can never grow memory and the
    camera callback never blocks: the newest frame always wins. The return
    value is what lets a resident stream count the frames it skipped, which a
    bounded capture window never needed to know.
    """
    evicted = False
    while True:
        try:
            frame_queue.put_nowait(frame)
            return evicted
        except queue.Full:
            try:
                frame_queue.get_nowait()
                evicted = True
            except queue.Empty:
                continue


__all__ = [
    "PEAR_CROP_EXPANSION",
    "PEAR_FIELD_LENGTHS",
    "PEAR_OUTPUT_FIELDS",
    "center_fallback_detection",
    "put_latest",
    "qualifying_persons",
    "select_person_detection",
    "validated_person_confidence_threshold",
    "validated_yolox_refresh_interval",
]
