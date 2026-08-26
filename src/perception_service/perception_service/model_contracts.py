"""Shared validation and identity helpers for perception model services."""

import numpy as np

MAX_MASK_BATCH = 8
MAX_TEXT_BATCH = 16
MAX_DETECTIONS = 16
MAX_POINT_CLOUD_POINTS = 1_000_000
SUPPORTED_IMAGE_ENCODINGS = {"bgr8", "rgb8"}
_POINT_FIELD_FLOAT32 = 7


def validate_image_message(image) -> None:
    if image.encoding not in SUPPORTED_IMAGE_ENCODINGS:
        raise ValueError(f"unsupported RGB image encoding: {image.encoding}")
    if image.height <= 0 or image.width <= 0:
        raise ValueError("image dimensions must be positive")
    if image.step < image.width * 3:
        raise ValueError("RGB image step is smaller than its pixel width")
    if len(image.data) < image.step * image.height:
        raise ValueError("RGB image data is truncated")


def validate_mask_batch(image, masks, max_batch_size: int = MAX_MASK_BATCH) -> None:
    validate_image_message(image)
    if not masks:
        raise ValueError("at least one mask is required")
    if len(masks) > max_batch_size:
        raise ValueError(f"mask batch exceeds limit {max_batch_size}")
    for index, mask in enumerate(masks):
        if mask.encoding != "mono8":
            raise ValueError(f"mask {index} must use mono8 encoding")
        if mask.height != image.height or mask.width != image.width:
            raise ValueError(f"mask {index} dimensions do not match the source image")
        if mask.header.stamp != image.header.stamp:
            raise ValueError(f"mask {index} timestamp does not match the source image")
        if mask.step < mask.width or len(mask.data) < mask.step * mask.height:
            raise ValueError(f"mask {index} data is truncated")


def rank_detections(records, max_detections: int = MAX_DETECTIONS) -> list:
    """Order detections by descending confidence and cap them at the service limit.

    Detection graphs such as raw Grounding-DINO emit one candidate per decoder query
    (900 for the 310P deployment), so the service contract — not the model — decides
    how many detections may leave `GroundingDetect`. Ranking is stable on the original
    query index, which keeps the emitted order reproducible for identical scores.
    """
    if max_detections <= 0:
        raise ValueError("max_detections must be positive")
    ranked = sorted(range(len(records)), key=lambda index: (-float(records[index].confidence), index))
    return [records[index] for index in ranked[:max_detections]]


def validate_detection_batch(detections, max_detections: int = MAX_DETECTIONS) -> None:
    if len(detections) > max_detections:
        raise ValueError(f"detection batch exceeds limit {max_detections}")


def validate_text_batch(texts, max_batch_size: int = MAX_TEXT_BATCH) -> None:
    if not texts:
        raise ValueError("at least one text is required")
    if len(texts) > max_batch_size:
        raise ValueError(f"text batch exceeds limit {max_batch_size}")
    for index, text in enumerate(texts):
        if not text.strip():
            raise ValueError(f"text {index} must not be empty")


def point_cloud_xyz(cloud, max_points: int = MAX_POINT_CLOUD_POINTS) -> np.ndarray:
    """Decode the XYZ channels of a PointCloud2 into a contiguous ``[N, 3]`` float32 array.

    The cloud is read through its declared field offsets rather than assumed to be a
    packed XYZ buffer, so an RGB-carrying RealSense cloud decodes correctly. Points are
    returned in message order and non-finite rows are left in place - filtering them is
    the model adapter's contract, not this decoder's.
    """
    if cloud.height <= 0 or cloud.width <= 0:
        raise ValueError("point cloud dimensions must be positive")
    count = int(cloud.height) * int(cloud.width)
    if count > max_points:
        raise ValueError(f"point cloud exceeds limit {max_points} points")
    if cloud.is_bigendian:
        raise ValueError("big-endian point clouds are not supported")
    fields = {field.name: field for field in cloud.fields}
    missing = sorted({"x", "y", "z"} - set(fields))
    if missing:
        raise ValueError(f"point cloud is missing required fields: {missing}")
    offsets = []
    for name in ("x", "y", "z"):
        field = fields[name]
        if field.datatype != _POINT_FIELD_FLOAT32 or field.count != 1:
            raise ValueError(f"point cloud field {name!r} must be a single FLOAT32")
        if field.offset + 4 > cloud.point_step:
            raise ValueError(f"point cloud field {name!r} lies outside its point stride")
        offsets.append(int(field.offset))
    if len(cloud.data) < count * cloud.point_step:
        raise ValueError("point cloud data is truncated")
    raw = np.frombuffer(bytes(cloud.data), dtype=np.uint8, count=count * int(cloud.point_step))
    raw = raw.reshape(count, int(cloud.point_step))
    points = np.stack([raw[:, offset : offset + 4].copy().view(np.float32).reshape(-1) for offset in offsets], axis=1)
    return np.ascontiguousarray(points, dtype=np.float32)


def validate_ready(ready: bool, message: str) -> None:
    if not ready:
        raise RuntimeError(message or "model backend is not ready")
