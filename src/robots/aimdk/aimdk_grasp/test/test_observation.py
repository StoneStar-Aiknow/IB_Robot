"""Registered observations, their buffer, and the checks before a plan is authorized."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from aimdk_grasp.feedback import Feedback
from aimdk_grasp.frames import ImageTransform
from aimdk_grasp.hand import HandMapping
from aimdk_grasp.joints import BODY_JOINTS, HAND_ALIASES, LEFT_ARM
from aimdk_grasp.observation import ObservationBuffer, RegisteredObservation
from aimdk_grasp.revalidation import (
    ReobserveRequired,
    RevalidationConfig,
    require_fresh,
    require_posture,
    require_stable_patch,
)
from aimdk_grasp.robot_model import planning_model
from aimdk_grasp.types import CameraIntrinsics, RGBDFrame

#: The OmniHand gripper description's open preset, as aliases.
OPEN = {
    "R_thumb_roll_joint": 0.129,
    "R_thumb_abad_joint": -1.54,
    "R_thumb_mcp_joint": 0.255,
    **{name: 0.0 for name in HAND_ALIASES[3:]},
}


@pytest.fixture(scope="module")
def open_hand() -> np.ndarray:
    return HandMapping(planning_model(), {"open": OPEN}).preset("open")


@pytest.fixture
def observation(open_hand):
    def make(at: float = 0.0, stamp: int = 123) -> RegisteredObservation:
        feedback = Feedback(at, dict.fromkeys(BODY_JOINTS, 0.0), open_hand.copy())
        frame = RGBDFrame(
            np.zeros((20, 20, 3), dtype=np.uint8),
            np.ones((20, 20), dtype=np.float32),
            CameraIntrinsics(100.0, 100.0, 10.0, 10.0),
        )
        return RegisteredObservation(
            frame, ImageTransform(stamp, frame.frame_id, "torso_link", np.eye(4)), at, feedback, "calibration-1"
        )

    return make


def _mask(region=np.s_[8:12, 8:12]) -> np.ndarray:
    mask = np.zeros((20, 20), dtype=bool)
    mask[region] = True
    return mask


def test_buffer_owns_data_and_never_refreshes_source_age(observation):
    original = observation()
    buffer = ObservationBuffer()
    buffer.update_observation(original)
    buffer.update_feedback(original.image_feedback)
    original.frame.depth[:] = 9.0
    original.image_feedback.positions[LEFT_ARM[0]] = 1.0
    state = buffer()
    assert np.all(state.observation.frame.depth == 1.0)
    assert state.feedback.positions[LEFT_ARM[0]] == 0.0
    assert state.observation.received_at == state.feedback.received_at == 0.0
    with pytest.raises(ReobserveRequired, match="stale"):
        require_fresh(buffer(), 0.11, RevalidationConfig())
    assert not buffer.update_observation(observation(1.0, 123))
    assert not buffer.update_feedback(replace(original.image_feedback, received_at=-1.0))


@pytest.mark.parametrize("side", ["left", "right", None])
def test_snapshot_can_own_one_hand_or_no_hands_without_fabricating_the_other(observation, side):
    from aimdk_grasp.observation import copy_feedback, observation_hand_mapping

    hands = {f"{s}_hand": None for s in ("left", "right")}
    if side is not None:
        hands[f"{side}_hand"] = observation_hand_mapping(side).to_native(OPEN)
    source = replace(observation().image_feedback, **hands)
    copied = copy_feedback(source)
    if side is not None:
        np.testing.assert_array_equal(copied.hand(side), source.hand(side))
        source.hand(side)[0] += 0.01
        assert copied.hand(side)[0] != source.hand(side)[0]
    for s in ("left", "right"):
        if s != side:
            with pytest.raises(ValueError, match=f"lacks {s}"):
                copied.hand(s)


def test_buffer_waits_for_both_streams(observation):
    buffer = ObservationBuffer()
    buffer.update_observation(observation())
    with pytest.raises(RuntimeError, match="Waiting"):
        buffer()


def test_unhealthy_feedback_is_never_fresh(observation):
    buffer = ObservationBuffer()
    value = observation()
    buffer.update_observation(value)
    buffer.update_feedback(replace(value.image_feedback, healthy=False))
    with pytest.raises(ReobserveRequired, match="unhealthy"):
        require_fresh(buffer(), 0.01, RevalidationConfig())


def test_target_and_camera_changes_require_new_perception(observation):
    config = RevalidationConfig()
    reference, mask = observation(), _mask()
    newer = observation(0.02, 124)
    require_stable_patch(reference, newer, mask, config)
    newer.frame.depth[8:12, 8:12] -= 0.05
    with pytest.raises(ReobserveRequired, match="changed"):
        require_stable_patch(reference, newer, mask, config)
    newer = observation(0.02, 124)
    newer.frame.depth[0, 0] -= 0.05
    require_stable_patch(reference, newer, mask, config)
    newer = observation(0.02, 124)
    newer.frame.rgb[8:12, 8:12] = 200
    with pytest.raises(ReobserveRequired, match="changed"):
        require_stable_patch(reference, newer, mask, config)


def test_target_depth_validity_flicker_is_not_a_change(observation):
    """Transparent target: valid depth pixels come and go between still frames."""
    config = RevalidationConfig()
    reference, newer = observation(), observation(0.02, 124)
    mask = _mask(np.s_[6:14, 6:14])
    # 10 of 100 ROI pixels lose depth, 10 others regain it; valid depth agrees.
    reference.frame.depth[6, 6:16] = 0.0
    newer.frame.depth[13, 4:14] = np.nan
    assert require_stable_patch(reference, newer, mask, config)["depth_changed_fraction"] == 0.0
    # Depth-only flicker of a few % of the ROI with unchanged RGB is noise
    # (live 2026-10-09: 2.2-2.6 %); valid depth moving on > 5 % still rejects.
    newer.frame.depth[8:10, 8:12] += 0.05
    fraction = require_stable_patch(reference, newer, mask, config)["depth_changed_fraction"]
    assert config.changed_fraction < fraction < config.depth_changed_fraction
    newer.frame.depth[10:12, 8:12] += 0.05
    with pytest.raises(ReobserveRequired, match="changed"):
        require_stable_patch(reference, newer, mask, config)


def test_image_geometry_must_match_but_extrinsic_and_calibration_are_not_compared(observation):
    config, mask, reference = RevalidationConfig(), _mask(), observation()
    for change in (
        lambda o: replace(o, frame=replace(o.frame, intrinsics=CameraIntrinsics(101.0, 100.0, 10.0, 10.0))),
        lambda o: replace(o, transform=replace(o.transform, camera_frame="other_optical_frame")),
    ):
        with pytest.raises(ReobserveRequired, match="Camera"):
            require_stable_patch(reference, change(observation(0.02, 124)), mask, config)
    # Head motion is require_posture's; calibration identity is the execution scene's.
    newer = replace(observation(0.02, 124), calibration_id="other")
    newer.transform.base_from_camera[0, 3] = 0.01
    require_stable_patch(reference, newer, mask, config)


def test_posture_change_beyond_the_start_tolerance_requires_a_replan(observation):
    config = RevalidationConfig()
    reference = observation().image_feedback
    moved = replace(reference, positions={**reference.positions, "head_pitch_joint": 0.02})
    assert require_posture(reference, moved, config) == pytest.approx(0.02)
    moved = replace(reference, positions={**reference.positions, "head_pitch_joint": 0.03})
    with pytest.raises(ReobserveRequired, match="moved"):
        require_posture(reference, moved, config)
    # The left arm may be excluded once it is the one moving.
    moved = replace(reference, positions={**reference.positions, LEFT_ARM[0]: 0.5})
    assert require_posture(reference, moved, config, include_left=False) == 0.0


def test_image_feedback_must_belong_to_the_observation(observation):
    original = observation()
    with pytest.raises(ValueError, match="image-time"):
        replace(original, image_feedback=replace(original.image_feedback, received_at=0.2)).snapshot()
    with pytest.raises(ValueError, match="torso"):
        replace(original, transform=replace(original.transform, base_frame="base_link")).snapshot()
