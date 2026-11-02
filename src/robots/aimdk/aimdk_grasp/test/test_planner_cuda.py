"""The migrated planner plans a grasp on the GPU with this package's planning model.

The other planner tests replace cuRobo with stubs. This one builds the real
LeftArmPlanner from config/x2_ultra on a synthetic scene (a table below the
hand, a box far off, the standing leg in the fine scene), warms it and plans a
grasp to a palm goal the arm can reach, so the generated planning model, its
locks, covers and limits are exercised end to end. It needs CUDA and cuRobo.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from aimdk_grasp.feedback import Feedback
from aimdk_grasp.joints import HEAD, LEFT_ARM, RIGHT_ARM
from aimdk_grasp.leg_obstacle import leg_obstacle
from aimdk_grasp.scene_estimation import LEG_OBSTACLE, SceneEstimate, SupportPlane
from test_planner_contracts import GRIPPER

torch = pytest.importorskip("torch")
pytest.importorskip("curobo")
if not torch.cuda.is_available():
    pytest.skip("the planner needs CUDA", allow_module_level=True)

from aimdk_grasp.planner import GraspPlan, LeftArmPlanner, grasp_to_palm_transform  # noqa: E402

#: A recorded standing pose (live 2026-10-09): both arms hanging, the left hand open (native order).
STANDING = {
    **dict(zip(LEFT_ARM, (0.3485, -0.0225, -0.0156, -1.1655, -0.0106, -0.0021, -0.0067), strict=True)),
    **dict(zip(RIGHT_ARM, (0.3631, 0.0252, 0.02, -1.1765, -0.003, 0.0086, 0.0013), strict=True)),
    **dict(zip(HEAD, (0.0001, 0.0), strict=True)),
}
OPEN_HAND = np.array([-0.1316, 1.5488, -0.2606, 0.0009, 0.0, 0.0, -0.0016, 0.0, -0.0039, 0.0])
#: A left-arm state with the palm forward of the body, where the grasp goal is taken.
REACHED = np.array([-0.9, 0.25, 0.0, -1.2, 0.0, 0.0, 0.0])


def box(center, size):
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float) / 2
    faces = [0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3]
    return {"vertices": corners * size + center, "faces": np.array(faces), "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]}


def build_planner():
    """A warmed planner on the synthetic scene, with the feedback and time it was built at."""
    now = time.monotonic()
    feedback = Feedback(now, dict(STANDING), OPEN_HAND.copy())
    table = {"dims": [0.6, 1.0, 0.04], "pose": [0.55, 0.15, -0.62, 1.0, 0.0, 0.0, 0.0]}
    far = box(np.array([1.5, -1.0, 0.0]), 0.05)
    scene = {"cuboid": {"table": table}, "mesh": {"environment": far}}
    fine = {"cuboid": {"table": table}, "mesh": {"environment": far, LEG_OBSTACLE: leg_obstacle()}}
    support = SupportPlane(np.array([0.55, 0.15, -0.60]), np.array([0.0, 0.0, 1.0]), np.empty((0, 3)))
    estimate = SceneEstimate(scene, now, 1, "torso_link", False, support, {}, fine)
    value = LeftArmPlanner(feedback, estimate, now, gripper=GRIPPER, max_goalset=4, planning_profile="latency")
    value.warmup()
    return value, feedback, now


@pytest.fixture(scope="module")
def planner():
    value, feedback, now = build_planner()
    yield value, feedback, now
    value.close()


def test_a_reachable_palm_goal_plans_three_fine_checked_segments(planner):
    planner, feedback, now = planner
    goal = planner._palm_poses(REACHED[None])[0]
    camera_from_grasp = goal @ np.linalg.inv(grasp_to_palm_transform(GRIPPER))
    assert planner.screen_grasps(feedback, camera_from_grasp[None], np.eye(4), now).tolist() == [True]
    plan = planner.plan_grasp(feedback, camera_from_grasp, np.eye(4), now, approach_offset=-0.10)
    assert isinstance(plan, GraspPlan) and plan.goalset_index == 0
    np.testing.assert_allclose(plan.approach.positions[0], feedback.vector(LEFT_ARM), atol=1e-5)
    # From the hanging arm to a goal 0.4 m up and forward: the approach really moves.
    assert np.abs(plan.approach.positions[-1] - plan.approach.positions[0]).max() > 0.3
    np.testing.assert_array_equal(plan.approach.positions[-1], plan.grasp.positions[0])
    np.testing.assert_array_equal(plan.grasp.positions[-1], plan.lift.positions[0])
    for segment in (plan.approach, plan.grasp, plan.lift):
        assert planner.valid_positions(segment.positions).all()
    # The grasp ends with the palm on its goal; the lift raises it or holds.
    palm = planner._palm_poses(plan.grasp.positions[-1:])[0]
    assert np.linalg.norm(palm[:3, 3] - goal[:3, 3]) < 0.01
    assert plan.lift_method in ("vertical", "hold")


def test_the_standing_leg_and_a_hand_behind_the_torso_are_invalid(planner):
    planner, feedback, _ = planner
    start = feedback.vector(LEFT_ARM)
    diagnostics = planner.constraint_diagnostics(start[None])
    assert diagnostics["valid"].all() and not diagnostics["hand_behind_torso"].any()
    # Shoulder pitched back: the hand swings behind the torso's frontal plane.
    behind = start.copy()
    behind[0] = 1.2
    assert planner.constraint_diagnostics(behind[None])["hand_behind_torso"].all()
    # Forearm swung in across the pelvis, the hand still in front of the torso:
    # the palm enters the standing left hip (the 10-15 palm hit). Only the leg
    # obstacle covers it.
    across = np.array([0.25, 0.06, -0.69, -0.95, 0.63, -0.07, 0.0])
    diagnostics = planner.constraint_diagnostics(across[None])
    assert not diagnostics["valid"].any() and not diagnostics["hand_behind_torso"].any()
    assert diagnostics["world_penetration_m"].max() > 0
    # A rejected path says where it hits: here a hand link, low beside the torso, not the table far ahead.
    said = planner.violations(np.stack([start, across]))
    assert said.startswith("samples 1-1 of 2; world ") and "links=[" in said and "behind" not in said
    x, y, z = (float(value) for value in said.split("at torso (")[1].split(")")[0].split(", "))
    assert abs(x) < 0.4 and z < 0.0
    assert planner.violations(start[None]) == "no violation"
