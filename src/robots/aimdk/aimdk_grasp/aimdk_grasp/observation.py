"""Registered observations and the current feedback a grasp request is planned from.

Geometry is tied to the image time. Current feedback is a separate stream; neither
reading the buffer nor doing computation advances a source receipt timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from threading import RLock

import numpy as np

from .feedback import Feedback
from .frames import ImageTransform, rigid
from .hand import HandMapping
from .joints import BODY_JOINTS
from .types import RGBDFrame

#: Image-time feedback further than this from the observation does not describe it.
MAX_IMAGE_FEEDBACK_OFFSET_S = 0.1


@lru_cache(maxsize=2)
def observation_hand_mapping(side: str = "left") -> HandMapping:
    """Generate ``side``'s planning model once per process, outside streaming callbacks."""
    from .robot_model import config_directory, planning_model

    return HandMapping(planning_model(config_directory(side)), side=side)


def copy_feedback(feedback: Feedback) -> Feedback:
    """An owned copy, with the body joints and each hand it carries checked against its planning model."""
    positions = dict(zip(BODY_JOINTS, feedback.vector(BODY_JOINTS), strict=True))
    hand = observation_hand_mapping().check(feedback.left_hand).copy()
    right = feedback.right_hand
    if right is not None:
        right = observation_hand_mapping("right").check(right).copy()
    if not np.isfinite(feedback.received_at):
        raise ValueError("Finite feedback source time required")
    return replace(feedback, positions=positions, left_hand=hand, right_hand=right)


@dataclass(frozen=True)
class RegisteredObservation:
    """A registered RGB-D frame, its torso-frame camera pose and the feedback at its image time."""

    frame: RGBDFrame
    transform: ImageTransform
    received_at: float
    image_feedback: Feedback
    calibration_id: str = ""

    def snapshot(self) -> RegisteredObservation:
        """A validated copy that owns its arrays."""
        frame, tf = self.frame, self.transform
        if (
            tf.stamp_ns <= 0
            or tf.camera_frame != frame.frame_id
            or tf.base_frame != "torso_link"
            or frame.depth_scale != 1.0
            or not np.isfinite(self.received_at)
        ):
            raise ValueError("Stamped torso transform and registered metre-valued depth required")
        rgb = np.array(frame.rgb, copy=True)
        depth = np.array(frame.depth, dtype=np.float32, copy=True)
        k = frame.intrinsics
        if (
            depth.ndim != 2
            or not depth.size
            or rgb.shape != (*depth.shape, 3)
            or rgb.dtype != np.uint8
            or not np.isfinite([k.fx, k.fy, k.cx, k.cy]).all()
            or min(k.fx, k.fy) <= 0
        ):
            raise ValueError("Invalid registered RGB-D observation")
        feedback = copy_feedback(self.image_feedback)
        if not feedback.healthy or abs(feedback.received_at - self.received_at) > MAX_IMAGE_FEEDBACK_OFFSET_S + 1e-9:
            raise ValueError("Healthy image-time feedback within 100 ms required")
        return replace(
            self,
            frame=replace(frame, rgb=rgb, depth=depth),
            image_feedback=feedback,
            transform=replace(tf, base_from_camera=rigid(tf.base_from_camera).copy()),
        )


@dataclass(frozen=True)
class LiveState:
    observation: RegisteredObservation
    feedback: Feedback


@dataclass(frozen=True)
class AdmittedState:
    """The ``Observation.robot_state`` of an X2 capture: what one request is planned from.

    ``live`` was fresh at ``admitted_at`` (source clock, s): the planner's
    snapshot time. ``started`` (monotonic) is when the request's budget began.
    """

    live: LiveState
    admitted_at: float
    started: float


class ObservationBuffer:
    """CPU-only input owner; callbacks continue updating while the GPU is busy."""

    def __init__(self) -> None:
        observation_hand_mapping()
        self._lock = RLock()
        self._observation: RegisteredObservation | None = None
        self._feedback: Feedback | None = None

    def update_observation(self, observation: RegisteredObservation) -> bool:
        """Keep a newer observation; returns False for one not newer than the held one."""
        value = observation.snapshot()
        with self._lock:
            previous = self._observation
            if previous is not None and (
                value.transform.stamp_ns <= previous.transform.stamp_ns or value.received_at < previous.received_at
            ):
                return False
            self._observation = value
            return True

    def update_feedback(self, feedback: Feedback) -> bool:
        """Keep newer current feedback; returns False for feedback not newer than the held one."""
        value = copy_feedback(feedback)
        with self._lock:
            if self._feedback is not None and value.received_at <= self._feedback.received_at:
                return False
            self._feedback = value
            return True

    def __call__(self) -> LiveState:
        with self._lock:
            if self._observation is None or self._feedback is None:
                raise RuntimeError("Waiting for registered observation and current feedback")
            return LiveState(self._observation.snapshot(), copy_feedback(self._feedback))
