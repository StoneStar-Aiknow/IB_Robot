"""The X2 grasp planner adapter: prefilter, checks after planning, and the plan's steps."""

from __future__ import annotations

import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.arm_gravity import ArmGravity
from aimdk_grasp.feedback import Feedback
from aimdk_grasp.frames import ImageTransform
from aimdk_grasp.grasp_request import (
    NO_EXECUTABLE_CANDIDATE,
    REOBSERVE_REQUIRED,
    GraspRequestConfig,
    X2GraspPlanner,
    left_palm_reach,
)
from aimdk_grasp.hand import HandMapping
from aimdk_grasp.joints import BODY_JOINTS, LEFT_ARM
from aimdk_grasp.observation import AdmittedState, LiveState, RegisteredObservation
from aimdk_grasp.robot_model import arm_limits, planning_model
from aimdk_grasp.scene_estimation import SceneEstimate
from aimdk_grasp.trajectory import TimedTrajectory
from aimdk_grasp.types import CameraIntrinsics, RGBDFrame
from test_arm_stream import APPROACH, GRASP, START
from test_observation import OPEN

from manipulation_execution.pick.pipeline.contracts import (
    Effect,
    EffectorAction,
    GraspFailure,
    Hold,
    Move,
    Observation,
    Proposals,
)

STAMP, FRAME, SIZE = 5_000_000_000, "optical", 20
LIFT = GRASP + np.array([0.1, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0])
BASE_FROM_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])


@pytest.fixture(scope="module")
def hand():
    return HandMapping(planning_model(), {"open": OPEN})


@pytest.fixture(scope="module")
def gravity():
    return ArmGravity.load()


def feedback(hand_pose, at, arm=START):
    positions = dict.fromkeys(BODY_JOINTS, 0.0)
    positions.update(zip(LEFT_ARM, arm, strict=True))
    return Feedback(at, positions, np.asarray(hand_pose, dtype=float))


def registered(hand_pose, stamp, *, rgb=0):
    frame = RGBDFrame(
        np.full((SIZE, SIZE, 3), rgb, np.uint8),
        np.ones((SIZE, SIZE), np.float32),
        CameraIntrinsics(100.0, 100.0, 10.0, 10.0),
        FRAME,
    )
    at = stamp * 1e-9
    transform = ImageTransform(stamp, FRAME, "torso_link", BASE_FROM_CAMERA)
    return RegisteredObservation(frame, transform, at, feedback(hand_pose, at), "calibration")


def candidates():
    """Three candidates: in reach (0.9), beyond reach (0.95), in reach (0.5)."""
    center, radius = left_palm_reach()
    camera_from_base = np.linalg.inv(BASE_FROM_CAMERA)
    poses = []
    for offset in (0.0, radius + 0.5, 0.01):
        goal = np.eye(4)
        goal[:3, 3] = center + [offset, 0.0, 0.0]
        poses.append(camera_from_base @ goal)
    return np.stack(poses), np.array([0.9, 0.95, 0.5])


def scene(stamp=STAMP):
    return SceneEstimate({}, stamp * 1e-9, stamp, FRAME, True, None, {}, {})


def trajectory(start, end):
    return TimedTrajectory(np.linspace(0.0, 2.0, 10), np.linspace(start, end, 10))


class FakeService:
    """GraspPlanningService.plan: picks the second kept candidate and plans START->APPROACH->GRASP->LIFT."""

    def __init__(self, lift=LIFT):
        self.calls, self.lift = [], lift

    def plan(self, feedback, estimate, grasps, confidence, base_from_camera, **options):
        self.calls.append(dict(feedback=feedback, grasps=grasps, confidence=confidence, **options))
        start = feedback.vector(LEFT_ARM)
        plan = SimpleNamespace(
            approach=trajectory(start, APPROACH), grasp=trajectory(APPROACH, GRASP), lift=trajectory(GRASP, self.lift)
        )
        return plan, dict(candidate_index=1)


@pytest.fixture
def setup(hand, gravity):
    planned = registered(hand.preset("open"), STAMP)
    newer = registered(hand.preset("open"), STAMP + 33_000_000)
    current = feedback(hand.preset("open"), STAMP * 1e-9 + 0.01)
    state = {"next": LiveState(newer, current), "now": STAMP * 1e-9 + 0.05}

    def make(service=None, **config):
        return X2GraspPlanner(
            service or FakeService(),
            grasp_to_palm=np.eye(4),
            hand=hand,
            gravity=gravity,
            limits=arm_limits(),
            latest_state=lambda: state["next"],
            clock=lambda: state["now"],
            config=GraspRequestConfig(**config),
        )

    admitted = AdmittedState(LiveState(planned, current), STAMP * 1e-9 + 0.02, time.monotonic())
    return make, admitted, state


def plan(planner, admitted, *, grasps=None, mask=None, estimate=None):
    poses, confidence = candidates() if grasps is None else grasps
    region = np.zeros((SIZE, SIZE), bool)
    region[8:12, 8:12] = True
    registered_ = admitted.live.observation
    observation = Observation(
        registered_.frame.rgb,
        registered_.frame.depth,
        np.array([[100.0, 0, 10], [0, 100.0, 10], [0, 0, 1]]),
        FRAME,
        STAMP,
        "torso_link",
        BASE_FROM_CAMERA,
        robot_state=admitted,
    )
    proposals = Proposals(
        region if mask is None else mask, np.zeros((16, 3)), poses, confidence, {}, scene=estimate or scene()
    )
    return planner.plan(observation, proposals, deadline=time.monotonic() + 30.0, cancelled=lambda: False)


def test_a_reachable_candidate_becomes_the_x2_grasp_sequence(setup):
    make, admitted, _ = setup
    service = FakeService()
    result = plan(make(service), admitted)
    call = service.calls[0]
    # The candidate beyond reach is dropped; the rest go most confident first.
    np.testing.assert_allclose(call["confidence"], [0.9, 0.5])
    assert call["snapshot_now"] == admitted.admitted_at and call["screen_check"]
    assert result.candidate_index == 2 and result.report["candidate_prefilter"]["kept_indices"] == [0, 2]
    assert [step.name for step in result.steps] == [
        "preshape",
        "approach",
        "grasp",
        "settle_grasp",
        "close",
        "lift",
        "hold_lifted",
    ]
    assert isinstance(result.steps[0], Effect) and result.steps[0].action is EffectorAction.PRESHAPE
    assert isinstance(result.steps[3], Hold) and result.steps[3].seconds == 0.3 and result.steps[-1].seconds == 3.0
    approach = result.steps[1]
    assert isinstance(approach, Move) and approach.segment.joint_names == LEFT_ARM
    np.testing.assert_allclose(approach.segment.start, START)
    assert result.release is None  # a put-back retraces the plan
    report = result.report
    assert report["target_recheck"]["checked_stamp_ns"] > report["target_recheck"]["planned_stamp_ns"]
    assert report["max_sag_rad"] > 0 and report["continuity_max_error_rad"] == 0.0


def test_a_lift_that_only_holds_the_grasp_pose_is_refused(setup):
    """Closing and holding in place would report a pick that never left the table (legacy lift_not_planned)."""
    make, admitted, _ = setup
    with pytest.raises(GraspFailure, match="did not plan a lift") as raised:
        plan(make(FakeService(lift=GRASP)), admitted)
    assert raised.value.code == NO_EXECUTABLE_CANDIDATE
    steps = [step.name for step in plan(make(FakeService()), admitted).steps]
    assert steps[-2:] == ["lift", "hold_lifted"]


def test_a_closed_hand_is_refused_before_planning(setup, hand):
    make, admitted, _ = setup
    closed = hand.preset("open").copy()
    closed[4] += 0.5
    admitted = replace(
        admitted, live=replace(admitted.live, feedback=replace(admitted.live.feedback, left_hand=closed))
    )
    service = FakeService()
    with pytest.raises(GraspFailure, match="Current feedback: Initial left hand is not open") as raised:
        plan(make(service), admitted)
    assert raised.value.code == NO_EXECUTABLE_CANDIDATE and not service.calls


def test_no_candidate_in_reach_is_no_executable_candidate(setup):
    make, admitted, _ = setup
    poses, confidence = candidates()
    with pytest.raises(GraspFailure, match="within left-palm reach") as raised:
        plan(make(), admitted, grasps=(poses[1:2], confidence[1:2]))
    assert raised.value.code == NO_EXECUTABLE_CANDIDATE and not raised.value.retryable


@pytest.mark.parametrize(
    "change, reason",
    [
        ("same_image", "not newer than the planned image"),
        ("target_changed", "Target/observed region changed"),
        ("arm_moved", "Robot moved beyond the planning snapshot"),
    ],
)
def test_a_recheck_that_does_not_confirm_the_target_asks_for_a_new_observation(setup, hand, change, reason):
    make, admitted, state = setup
    newer = state["next"]
    if change == "same_image":
        newer = LiveState(admitted.live.observation, newer.feedback)
    elif change == "target_changed":
        newer = replace(newer, observation=registered(hand.preset("open"), STAMP + 33_000_000, rgb=200))
    else:
        arm = START + np.array([0.1, 0, 0, 0, 0, 0, 0])
        newer = replace(newer, feedback=feedback(hand.preset("open"), newer.feedback.received_at, arm))
    state["next"] = newer
    with pytest.raises(GraspFailure, match=reason) as raised:
        plan(make(), admitted)
    assert raised.value.code == REOBSERVE_REQUIRED and raised.value.retryable


def test_a_plan_beyond_the_controller_limits_is_refused(setup):
    make, admitted, _ = setup
    beyond = LIFT.copy()
    beyond[6] = arm_limits()[LEFT_ARM[6]][1] + 0.05
    with pytest.raises(GraspFailure, match="gravity compensation refused the plan") as raised:
        plan(make(FakeService(lift=beyond)), admitted)
    assert raised.value.code == NO_EXECUTABLE_CANDIDATE


def test_a_scene_from_another_image_is_refused(setup):
    make, admitted, _ = setup
    with pytest.raises(GraspFailure, match="planned image"):
        plan(make(), admitted, estimate=scene(STAMP + 1))


def test_a_cancel_inside_planning_ends_the_goal(setup):
    make, admitted, _ = setup

    class Cancelling(FakeService):
        def plan(self, *args, checkpoint, **options):
            raise GraspFailure("CANCELED", "goal cancelled")

    with pytest.raises(GraspFailure) as raised:
        plan(make(Cancelling()), admitted)
    assert raised.value.code == "CANCELED"
