"""The grasp models, called as the typed perception services, whatever accelerator serves them.

A robot sends its own images: platforms may forbid a model host from subscribing
to their cameras, and the caller already holds the frame it plans from. A
detection that comes with a mask (text-prompted segmentation in one model) is
used as is; otherwise the segmentation service masks it. Every response must
belong to the requested image (frame and stamp); grasp poses are in the
cloud's frame. Each call's round trip and the model's own inference time are
recorded, so transport and queueing can be told apart.

Any failure of a model service (unavailable, timed out, failed, or answering
with something that does not fit the request) is ``RPC_FAILED`` and ends the
pick: another observation does not bring a model back.
"""

from __future__ import annotations

import time
from array import array as _array
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .contracts import Flag, GraspFailure

RPC_FAILED = "RPC_FAILED"


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box_xyxy: np.ndarray  # pixels
    mask: np.ndarray | None = None  # HxW bool, when the detector segments


def _header(frame_id: str, stamp_ns: int):
    from std_msgs.msg import Header

    header = Header(frame_id=frame_id)
    header.stamp.sec, header.stamp.nanosec = divmod(int(stamp_ns), 1_000_000_000)
    return header


def _stamp_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


# Image and cloud payloads go into uint8[] fields as array('B'): rclpy checks a
# bytes value element by element (152 ms for one 1280x720 RGB image).


def rgb_image(rgb: np.ndarray, header):
    from sensor_msgs.msg import Image

    array = np.ascontiguousarray(rgb, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("rgb must be HxWx3 uint8")
    return Image(
        header=header,
        height=array.shape[0],
        width=array.shape[1],
        encoding="rgb8",
        step=array.shape[1] * 3,
        data=_array("B", array.tobytes()),
    )


def image_mask(message) -> np.ndarray:
    if message.encoding.lower() not in ("mono8", "8uc1"):
        raise ValueError(f"unsupported mask encoding {message.encoding!r}")
    if message.step < message.width:
        raise ValueError("mask image step is smaller than its row")
    array = np.ndarray(
        (message.height, message.width), dtype=np.uint8, buffer=bytes(message.data), strides=(message.step, 1)
    )
    return array != 0


def points_cloud(points: np.ndarray, header):
    from sensor_msgs.msg import PointCloud2, PointField

    array = np.ascontiguousarray(points, dtype="<f4")
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points must be Nx3")
    fields = [
        PointField(name=axis, offset=4 * index, datatype=PointField.FLOAT32, count=1)
        for index, axis in enumerate("xyz")
    ]
    return PointCloud2(
        header=header,
        height=1,
        width=len(array),
        fields=fields,
        is_bigendian=False,
        point_step=12,
        row_step=12 * len(array),
        is_dense=bool(np.isfinite(array).all()),
        data=_array("B", array.tobytes()),
    )


class PerceptionServices:
    """Clients of GroundingDetect, optional SegmentDetections and GenerateGrasps."""

    def __init__(
        self,
        node,
        *,
        detect: str,
        generate_grasps: str,
        segment: str | None = None,
        timeout_s: float = 30.0,
        callback_group=None,
    ):
        from ibrobot_msgs.srv import GenerateGrasps, GroundingDetect, SegmentDetections

        self._types = (GroundingDetect, SegmentDetections, GenerateGrasps)
        self._detect = node.create_client(GroundingDetect, detect, callback_group=callback_group)
        self._segment = (
            None if segment is None else node.create_client(SegmentDetections, segment, callback_group=callback_group)
        )
        self._grasps = node.create_client(GenerateGrasps, generate_grasps, callback_group=callback_group)
        self.timeout_s = timeout_s
        self.calls: list[dict] = []

    def detect(
        self,
        rgb,
        frame_id: str,
        stamp_ns: int,
        prompt: str,
        *,
        box_threshold: float,
        text_threshold: float,
        cancelled: Flag = lambda: False,
    ) -> list[Detection]:
        """Detections best first; each carries its mask when the detector returned one."""
        request = self._types[0].Request()
        request.image = rgb_image(rgb, _header(frame_id, stamp_ns))
        request.text_prompt = prompt
        request.box_threshold, request.text_threshold = float(box_threshold), float(text_threshold)
        response = self._call(self._detect, request, len(request.image.data), cancelled)
        self._same_image(response.detections.header, frame_id, stamp_ns)
        detections = []
        for item in response.detections.detections:
            mask = self._mask(item.mask, rgb, frame_id, stamp_ns) if item.mask.width and item.mask.height else None
            detections.append(
                Detection(str(item.label), float(item.confidence), np.asarray(item.bbox, dtype=np.float32), mask)
            )
        return sorted(detections, key=lambda detection: detection.score, reverse=True)

    def masks(
        self, rgb, frame_id: str, stamp_ns: int, detections: Sequence[Detection], *, cancelled: Flag = lambda: False
    ) -> list[np.ndarray]:
        """One mask per detection: its own, or the segmentation service's for its box."""
        missing = [index for index, detection in enumerate(detections) if detection.mask is None]
        masks = [detection.mask for detection in detections]
        if not missing:
            return masks
        if self._segment is None:
            raise GraspFailure(RPC_FAILED, "the detector returned no masks and no segmentation is bound")
        from ibrobot_msgs.msg import Detection2D, DetectionArray

        header = _header(frame_id, stamp_ns)
        request = self._types[1].Request()
        request.image = rgb_image(rgb, header)
        request.detections = DetectionArray(
            header=header,
            detections=[
                Detection2D(
                    header=header,
                    label=detections[index].label,
                    confidence=float(detections[index].score),
                    bbox=np.asarray(detections[index].box_xyxy, dtype=np.float32).tolist(),
                )
                for index in missing
            ],
        )
        response = self._call(self._segment, request, len(request.image.data), cancelled)
        self._same_image(response.detections.header, frame_id, stamp_ns)
        returned = response.detections.detections
        if len(returned) != len(missing):
            raise GraspFailure(RPC_FAILED, "segmentation must return one mask per box")
        for index, item in zip(missing, returned, strict=True):
            if not np.allclose(item.bbox, np.asarray(detections[index].box_xyxy, dtype=np.float32)):
                raise GraspFailure(RPC_FAILED, "segmentation must return masks in box order")
            masks[index] = self._mask(item.mask, rgb, frame_id, stamp_ns)
        return masks

    def generate_grasps(
        self,
        points: np.ndarray,
        frame_id: str,
        stamp_ns: int,
        *,
        max_grasps: int,
        min_confidence: float,
        cancelled: Flag = lambda: False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(Kx4x4 poses in the cloud's frame, K confidences), as the model ranked them."""
        request = self._types[2].Request()
        request.object_points = points_cloud(points, _header(frame_id, stamp_ns))
        request.max_grasps, request.min_confidence = int(max_grasps), float(min_confidence)
        response = self._call(self._grasps, request, len(request.object_points.data), cancelled)
        self._same_image(response.grasps.header, frame_id, stamp_ns)
        for item in response.grasps.grasps:
            self._same_image(item.header, frame_id, stamp_ns)
        poses = np.array(
            [np.asarray(item.pose_matrix, dtype=np.float64).reshape(4, 4) for item in response.grasps.grasps]
        ).reshape(-1, 4, 4)
        confidence = np.array([float(item.confidence) for item in response.grasps.grasps])
        return poses, confidence

    # -- pieces ---------------------------------------------------------------------

    @staticmethod
    def _same_image(header, frame_id: str, stamp_ns: int) -> None:
        if header.frame_id != frame_id or _stamp_ns(header) != int(stamp_ns):
            raise GraspFailure(RPC_FAILED, "a model response does not belong to the requested image")

    @classmethod
    def _mask(cls, message, rgb, frame_id: str, stamp_ns: int) -> np.ndarray:
        cls._same_image(message.header, frame_id, stamp_ns)
        try:
            mask = image_mask(message)
        except ValueError as exc:
            raise GraspFailure(RPC_FAILED, f"invalid mask: {exc}") from None
        if mask.shape != np.shape(rgb)[:2]:
            raise GraspFailure(RPC_FAILED, "a mask does not match the image")
        return mask

    def _call(self, client, request, payload_bytes: int, cancelled: Flag):
        name = client.srv_name
        started = time.monotonic()
        if not client.service_is_ready() and not client.wait_for_service(timeout_sec=min(2.0, self.timeout_s)):
            raise GraspFailure(RPC_FAILED, f"{name} is not available")
        future = client.call_async(request)
        deadline = started + self.timeout_s
        while not future.done():
            if cancelled():
                future.cancel()
                raise GraspFailure("CANCELED", f"cancelled waiting for {name}")
            if time.monotonic() >= deadline:
                future.cancel()
                raise GraspFailure(RPC_FAILED, f"{name} timed out")
            time.sleep(0.005)
        response = future.result()
        rpc_s = time.monotonic() - started
        inference_s = float(getattr(response, "inference_time_ms", 0.0)) * 1e-3
        self.calls.append(
            dict(
                service=name,
                rpc_s=rpc_s,
                inference_s=inference_s,
                transport_s=rpc_s - inference_s,
                request_bytes=payload_bytes,
            )
        )
        if response is None or not response.success:
            message = "" if response is None else response.message
            raise GraspFailure(RPC_FAILED, f"{name} failed: {message}")
        return response
