"""What the X2 grasp pipeline and its grasp head send each other, on the wire.

The X2's two-host deployment runs the pipeline beside the runtime (the robot-side
host) and the head camera's capture and cuRobo on the Orin (aimdk_grasp.grasp_head).
What crosses between them is converted here:

- a plan (:class:`GraspPlan`) to and from ``aimdk_grasp_msgs/GraspPlan``;
- a registered colour image as JPEG, its depth as millimetres in a zstd frame
  (0 for no measurement; a 1280x720 pair was 6.5 MB raw, ~0.45 MB so and
  ~15 ms to encode on the Orin, 2026-10-22);
- a target mask as PNG.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

import numpy as np

from manipulation_execution.grasp_pipeline.contracts import (
    ArmSegment,
    Effect,
    EffectorAction,
    GraspPlan,
    Hold,
    Move,
    SegmentKind,
    Step,
)

JPEG_QUALITY = 95
ZSTD_LEVEL = 1
RGB_FORMAT = "rgb8; jpeg compressed bgr8"
MASK_FORMAT = "mono8; png"
_DEPTH_FORMAT = re.compile(r"16UC1; zstd mm; (\d+)x(\d+)")


class WireError(ValueError):
    """Data that does not decode as what this module wrote."""


# -- plans ------------------------------------------------------------------------


def step_to_msg(step: Step):
    from aimdk_grasp_msgs.msg import GraspPlanStep

    message = GraspPlanStep(name=step.name)
    if isinstance(step, Move):
        segment = step.segment
        message.kind = GraspPlanStep.MOVE
        message.segment_kind = (
            GraspPlanStep.TRAJECTORY if segment.kind is SegmentKind.TRAJECTORY else GraspPlanStep.TARGET
        )
        message.joint_names = list(segment.joint_names)
        message.positions = segment.positions.ravel().tolist()
        message.times = [] if segment.times is None else segment.times.tolist()
        message.refine_after = bool(step.refine_after)
    elif isinstance(step, Effect):
        message.kind, message.action = GraspPlanStep.EFFECT, step.action.value
    elif isinstance(step, Hold):
        message.kind, message.seconds = GraspPlanStep.HOLD, float(step.seconds)
    else:
        raise WireError(f"not a plan step: {step!r}")
    return message


def step_from_msg(message) -> Step:
    from aimdk_grasp_msgs.msg import GraspPlanStep

    if message.kind == GraspPlanStep.MOVE:
        names = tuple(message.joint_names)
        positions = np.asarray(message.positions, dtype=np.float64)
        if not names or positions.size % len(names):
            raise WireError(f"{message.name}: positions do not fill whole rows of {len(names)} joints")
        trajectory = message.segment_kind == GraspPlanStep.TRAJECTORY
        segment = ArmSegment(
            message.name,
            SegmentKind.TRAJECTORY if trajectory else SegmentKind.TARGET,
            names,
            positions.reshape(-1, len(names)),
            np.asarray(message.times, dtype=np.float64) if trajectory else None,
        )
        return Move(segment, refine_after=bool(message.refine_after))
    if message.kind == GraspPlanStep.EFFECT:
        return Effect(EffectorAction(message.action))
    if message.kind == GraspPlanStep.HOLD:
        return Hold(float(message.seconds), name=message.name)
    raise WireError(f"{message.name}: unknown step kind {message.kind}")


def plan_to_msg(plan: GraspPlan):
    from aimdk_grasp_msgs.msg import GraspPlan as GraspPlanMsg

    return GraspPlanMsg(
        steps=[step_to_msg(step) for step in plan.steps],
        has_release=plan.release is not None,
        release=[step_to_msg(step) for step in plan.release or ()],
        candidate_index=int(plan.candidate_index),
        report_json=to_json(plan.report),
    )


def plan_from_msg(message) -> GraspPlan:
    return GraspPlan(
        tuple(step_from_msg(step) for step in message.steps),
        int(message.candidate_index),
        tuple(step_from_msg(step) for step in message.release) if message.has_release else None,
        from_json(message.report_json),
    )


# -- images -----------------------------------------------------------------------


def encode_rgb(rgb: np.ndarray):
    import cv2
    from sensor_msgs.msg import CompressedImage

    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise WireError("colour must be HxWx3 uint8")
    ok, data = cv2.imencode(".jpg", np.ascontiguousarray(rgb[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise WireError("JPEG encoding failed")
    return CompressedImage(format=RGB_FORMAT, data=data.tobytes())


def decode_rgb(message) -> np.ndarray:
    import cv2

    if message.format != RGB_FORMAT:
        raise WireError(f"colour image format {message.format!r} is not {RGB_FORMAT!r}")
    bgr = cv2.imdecode(np.frombuffer(bytes(message.data), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise WireError("colour image does not decode")
    return np.ascontiguousarray(bgr[:, :, ::-1])


def encode_depth(depth_m: np.ndarray):
    """Metres to millimetres; non-finite, non-positive and out-of-range values become 0 (no measurement)."""
    import zstandard
    from sensor_msgs.msg import CompressedImage

    if depth_m.ndim != 2:
        raise WireError("depth must be HxW")
    millimetres = np.asarray(depth_m, dtype=np.float64) * 1000.0
    valid = np.isfinite(millimetres) & (millimetres > 0) & (millimetres < 65535.5)
    packed = np.where(valid, np.rint(np.where(valid, millimetres, 0.0)), 0).astype("<u2")
    height, width = packed.shape
    data = zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(packed.tobytes())
    return CompressedImage(format=f"16UC1; zstd mm; {height}x{width}", data=data)


def decode_depth(message) -> np.ndarray:
    """Float32 metres; 0 where nothing was measured."""
    import zstandard

    match = _DEPTH_FORMAT.fullmatch(message.format)
    if match is None:
        raise WireError(f"depth image format {message.format!r} is not '16UC1; zstd mm; HxW'")
    height, width = int(match.group(1)), int(match.group(2))
    try:
        raw = zstandard.ZstdDecompressor().decompress(bytes(message.data), max_output_size=height * width * 2)
    except zstandard.ZstdError as exc:
        raise WireError(f"depth image does not decompress: {exc}") from exc
    if len(raw) != height * width * 2:
        raise WireError("depth image is not the size its format names")
    return (np.frombuffer(raw, dtype="<u2").reshape(height, width) / 1000.0).astype(np.float32)


def encode_mask(mask: np.ndarray):
    import cv2
    from sensor_msgs.msg import CompressedImage

    if mask.ndim != 2:
        raise WireError("a mask is HxW")
    ok, data = cv2.imencode(".png", np.where(mask, 255, 0).astype(np.uint8))
    if not ok:
        raise WireError("PNG encoding failed")
    return CompressedImage(format=MASK_FORMAT, data=data.tobytes())


def decode_mask(message) -> np.ndarray:
    import cv2

    if message.format != MASK_FORMAT:
        raise WireError(f"mask format {message.format!r} is not {MASK_FORMAT!r}")
    image = cv2.imdecode(np.frombuffer(bytes(message.data), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise WireError("mask does not decode")
    return image > 0


# -- small values -----------------------------------------------------------------


def matrix_to_list(matrix: np.ndarray, shape: tuple[int, int]) -> list[float]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != shape or not np.isfinite(values).all():
        raise WireError(f"expected a finite {shape[0]}x{shape[1]} matrix")
    return values.ravel().tolist()


def matrix_from_list(values, shape: tuple[int, int]) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.size != shape[0] * shape[1]:
        raise WireError(f"expected {shape[0] * shape[1]} values for a {shape[0]}x{shape[1]} matrix")
    return matrix.reshape(shape)


def poses_to_list(poses: np.ndarray) -> list[float]:
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 4):
        raise WireError("poses must be Kx4x4")
    return values.ravel().tolist()


def poses_from_list(values) -> np.ndarray:
    poses = np.asarray(values, dtype=np.float64)
    if poses.size % 16:
        raise WireError("poses must be a whole number of 4x4 matrices")
    return poses.reshape(-1, 4, 4)


def to_json(value: Mapping) -> str:
    def plain(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        return str(item)

    return json.dumps(dict(value), sort_keys=True, default=plain)


def from_json(text: str) -> dict:
    if not text:
        return {}
    value = json.loads(text)
    if not isinstance(value, dict):
        raise WireError("a report is a JSON object")
    return value
