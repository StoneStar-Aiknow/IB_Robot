"""Container detection through the existing typed perception services."""

import time
from dataclasses import dataclass, replace

import numpy as np

from manipulation_execution.grasp_pipeline.contracts import GraspFailure

from .container_perception import estimate_container
from .proposer import ImageTurn, PerceptionSettings, X2Proposer, observed_frame, upright_turns


@dataclass(frozen=True)
class DetectedContainer:
    mask: np.ndarray
    query: str


def detect_masks(services, observation, query, *, cancelled, limit=3, settings=PerceptionSettings()):
    query = query.strip()
    if not query:
        raise GraspFailure("INVALID_TARGET", "an object or container description is required")
    frame = observed_frame(observation)
    turn = ImageTurn(upright_turns(observation.base_from_camera), *frame.rgb.shape[:2])
    helper = X2Proposer(services, None, settings=settings, remote_scene=lambda *_: None)
    prompt = query if query.endswith(".") else f"{query} ."
    detections = helper._detect(frame, observation.stamp_ns, prompt, limit, turn, cancelled)
    for detection in detections:
        for mask in helper._segment(frame, observation.stamp_ns, detection, turn, cancelled):
            mask = np.asarray(mask, bool)
            if mask.shape == observation.depth_m.shape and mask.sum() >= 30:
                yield mask


class ContainerProposer:
    def __init__(self, services, *, models=None, settings=PerceptionSettings()):
        self.services, self.models, self.settings = services, models, settings

    def propose(self, observation, query, held, *, deadline, cancelled):
        failure = GraspFailure("CONTAINER_NOT_FOUND", "no container mask was detected", retryable=True)
        for mask in detect_masks(self.services, observation, query, cancelled=cancelled, settings=self.settings):
            if cancelled() or time.monotonic() >= deadline:
                raise GraspFailure("CANCELED" if cancelled() else "SKILL_TIMEOUT", "container detection interrupted")
            if self.models is None:
                return DetectedContainer(mask, query)
            try:
                settings = replace(self.settings, scene=replace(self.settings.scene, arm_side=held.arm_side))
                return estimate_container(observation, mask, held, self.models[held.arm_side], settings)
            except GraspFailure as exc:
                failure = exc
        raise failure
