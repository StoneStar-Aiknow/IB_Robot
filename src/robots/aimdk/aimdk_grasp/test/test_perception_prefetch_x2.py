"""X2 parts of the perception prefetch: posture, the container on a pick capture, a prefetched mask at the place."""

from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.container_proposer import ContainerProposer, DetectedContainer
from aimdk_grasp.joints import HAND_PARTS, HEAD, arm_joints
from aimdk_grasp.omnihand import HAND_OFFSETS
from aimdk_grasp.proposer import PerceptionSettings
from aimdk_grasp.remote_head import container_detector, x2_posture

from manipulation_execution.grasp_pipeline.contracts import Observation
from manipulation_execution.grasp_pipeline.perception import Detection

ARMS = (*arm_joints("left"), *arm_joints("right"), *HEAD)
CHANNEL = tuple(f"{side}_{part}" for side in ("left", "right") for part in HAND_PARTS)


class Arm:
    def __init__(self):
        self.asked = []

    def measured(self, names):
        self.asked.append(tuple(names))
        return np.arange(len(names), dtype=float)


def hand(healthy=True):
    positions = {name: 0.1 * index for index, name in enumerate(CHANNEL)}
    sample = SimpleNamespace(healthy=healthy, positions=positions)
    return SimpleNamespace(io=SimpleNamespace(feedback=lambda: sample, channel_joints=CHANNEL))


def test_posture_is_both_arms_the_head_and_only_the_installed_hands():
    arm = Arm()
    posture = x2_posture(arm, hand(), ("left",))()
    assert arm.asked == [ARMS]
    left = [0.1 * (HAND_OFFSETS["left"] + i) for i in range(len(HAND_PARTS))]
    assert np.allclose(posture, [*range(len(ARMS)), *left])
    both = x2_posture(arm, hand(), ("left", "right"))()
    assert len(both) == len(ARMS) + 2 * len(HAND_PARTS)


def test_an_automatic_arm_measures_through_the_left_part_and_an_unhealthy_hand_refuses():
    arm = Arm()
    selected = SimpleNamespace(parts={"left": arm, "right": Arm()})
    hands = SimpleNamespace(parts={"left": hand(healthy=False), "right": hand()})
    with pytest.raises(RuntimeError, match="unhealthy"):
        x2_posture(selected, hands, ("left",))()
    assert arm.asked == [ARMS]


def observation():
    return Observation(np.zeros((10, 10, 3), np.uint8), np.ones((10, 10)), np.eye(3), "camera", 1, "base", np.eye(4))


def test_the_container_is_detected_with_the_placement_thresholds():
    seen = []

    def detect(rgb, frame, stamp, prompt, *, box_threshold, text_threshold, **kwargs):
        seen.append((prompt, box_threshold))
        return [Detection("bowl", 0.9, np.array([0, 0, 10, 10]), np.ones((10, 10), bool))]

    services = SimpleNamespace(detect=detect)
    find = container_detector(services, {"container_perception": {"box_threshold": 0.5}}, PerceptionSettings())
    mask = find(observation(), "a black bowl", deadline=float("inf"), cancelled=lambda: False)
    assert mask.all() and seen == [("a black bowl .", 0.5)]
    nothing = container_detector(SimpleNamespace(detect=lambda *a, **k: []), {}, PerceptionSettings())
    assert nothing(observation(), "a black bowl", deadline=float("inf"), cancelled=lambda: False) is None


def test_a_prefetched_mask_becomes_the_remote_container_without_detection():
    services = SimpleNamespace(detect=lambda *a, **k: pytest.fail("a prefetched mask needs no detection"))
    mask = np.zeros((10, 10), bool)
    mask[2:5, 2:5] = True
    found = ContainerProposer(services).from_mask(
        observation(), mask, "a black bowl", None, deadline=float("inf"), cancelled=lambda: False
    )
    assert isinstance(found, DetectedContainer) and found.query == "a black bowl" and found.mask.sum() == 9
