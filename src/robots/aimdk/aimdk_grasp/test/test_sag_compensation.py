"""Task-pose commands raised by the predicted gravity sag land the measured arm on target."""

import numpy as np
import pytest
from aimdk_grasp.arm_gravity import ArmGravity
from aimdk_grasp.joints import LEFT_ARM, RIGHT_ARM
from aimdk_grasp.sag_compensation import SagCompensation, deployed_hands

JOINTS = LEFT_ARM + RIGHT_ARM
# grasp_ready (2026-11-07 task start), per arm as configured.
GRASP_READY = np.array(
    [-0.0486, 0.4486, 0.3506, -1.45, 0.0038, 0.0678, -0.0002]
    + [-0.0554, -0.4486, -0.3478, -1.45, -0.0029, 0.0678, 0.0001]
)
GRAVITY = {side: ArmGravity.load(side=side) for side in ("left", "right")}


def settle(command, gravity=GRAVITY):
    """Where an arm holding ``command`` at kp 40 comes to rest: q = command - sag(q)."""
    sag = SagCompensation(JOINTS, gravity=gravity)
    q = np.array(command, dtype=float)
    for _ in range(100):
        q = np.asarray(command, dtype=float) - sag.offsets(q)[0]
    return q


def test_uncompensated_grasp_ready_lands_outside_the_old_tolerance():
    # The 2026-11-07 failure: right shoulder roll 0.111 rad short, over the 0.1 rad check.
    error = np.abs(settle(GRASP_READY) - GRASP_READY)
    assert error.max() > 0.1
    assert JOINTS[int(np.argmax(error))].endswith("shoulder_roll_joint")


def test_compensated_commands_land_the_arms_on_grasp_ready():
    command = SagCompensation(JOINTS, gravity=GRAVITY)(GRASP_READY)[0]
    assert np.abs(command - GRASP_READY).max() > 0.1  # the commands really move
    assert np.abs(settle(command) - GRASP_READY).max() < 0.01


def test_only_the_commands_are_raised_and_they_stay_inside_the_limits():
    path = np.linspace(GRASP_READY - 0.2, GRASP_READY, 5)
    sag = SagCompensation(JOINTS, gravity=GRAVITY)
    np.testing.assert_allclose(sag(path), path + sag.offsets(path))
    # A limit just above the target keeps the raised command 5 mrad inside it.
    roll = JOINTS.index("left_shoulder_roll_joint")
    limits = {JOINTS[roll]: (-0.061, GRASP_READY[roll] + 0.02)}
    clipped = SagCompensation(JOINTS, limits, gravity=GRAVITY)(GRASP_READY)[0]
    assert clipped[roll] == pytest.approx(GRASP_READY[roll] + 0.015)


def test_the_default_models_are_the_configured_arm_models():
    np.testing.assert_allclose(
        SagCompensation(JOINTS).offsets(GRASP_READY), SagCompensation(JOINTS, gravity=GRAVITY).offsets(GRASP_READY)
    )


def test_a_trajectory_without_both_arms_or_in_another_width_is_refused():
    with pytest.raises(ValueError, match="right arm joints"):
        SagCompensation(LEFT_ARM, gravity=GRAVITY)
    with pytest.raises(ValueError, match="width"):
        SagCompensation(JOINTS, gravity=GRAVITY).offsets(np.zeros((2, 7)))


def test_both_hands_installed_or_undeclared_keep_the_fitted_models():
    reference = SagCompensation(JOINTS, gravity=GRAVITY).offsets(GRASP_READY)
    both = SagCompensation(JOINTS, gravity=GRAVITY, installed_hands=("left", "right"))
    np.testing.assert_allclose(both.offsets(GRASP_READY), reference)
    assert both.describe() == "left with hand, right with hand"


def test_an_arm_without_a_hand_is_raised_by_its_bare_wrist_model():
    """2026-11-07: the hand-less left arm overshot grasp_ready by 0.03-0.045 rad on the hand model."""
    left = slice(0, 7)
    bare_physics = {"left": GRAVITY["left"].without_hand(), "right": GRAVITY["right"]}
    hand_model = SagCompensation(JOINTS, gravity=GRAVITY)(GRASP_READY)[0]
    overshoot = np.abs(settle(hand_model, bare_physics) - GRASP_READY)[left].max()
    assert overshoot > 0.02
    right_only = SagCompensation(JOINTS, gravity=GRAVITY, installed_hands=["right"])
    command = right_only(GRASP_READY)[0]
    assert np.abs(settle(command, bare_physics) - GRASP_READY).max() < 0.01
    np.testing.assert_allclose(command[7:], hand_model[7:])  # the arm with its hand is unchanged
    assert right_only.describe() == "left bare wrist, right with hand"


def test_removing_the_hand_derives_a_new_model_and_leaves_the_fitted_one():
    fitted = GRAVITY["left"]
    mass, payload = fitted.chain[-1]["mass"], fitted.payload
    bare = fitted.without_hand()
    assert bare.chain[-1]["mass"] == pytest.approx(0.081) and bare.payload == 0.0
    assert (fitted.chain[-1]["mass"], fitted.payload) == (mass, payload)
    assert np.all(np.abs(bare.sag(GRASP_READY[:7])) <= np.abs(fitted.sag(GRASP_READY[:7])) + 1e-9)


def test_the_installed_hands_come_from_the_deployed_grasp_execution():
    assert deployed_hands({"backend_options": {"installed_hands": ["right"]}}) == ("right",)
    assert deployed_hands({"backend_options": {"installed_hands": ["right", "left"]}}) == ("left", "right")
    assert deployed_hands({"backend_options": {}}) == ("left", "right")  # omitted: both, as the grasp backend
    assert deployed_hands(None) is None
    with pytest.raises(ValueError):
        deployed_hands({"backend_options": {"installed_hands": "right"}})
