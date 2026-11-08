"""Carrying reuses the arm's warmed planner: reserved palm spheres, a bounded hand cover, no rebuild."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.feedback import Feedback
from aimdk_grasp.hand import HandMapping
from aimdk_grasp.held_object import held_center
from aimdk_grasp.joints import HEAD, LEFT_ARM, RIGHT_ARM, arm_joints
from aimdk_grasp.place_request import opening_cover, placement_cover
from aimdk_grasp.planner import ATTACH_LINK, ATTACH_SLOTS, DISABLED_RADIUS, robot_config, screen_robot_config
from aimdk_grasp.planner_service import GraspPlanningService
from aimdk_grasp.robot_model import config_directory, planning_model
from aimdk_grasp.robot_spheres import RobotSphereModel
from test_hand import PRESETS
from test_planner_contracts import GRIPPER

#: A hanging arm (live 2026-10-09, native order).
ARM = (0.3485, -0.0225, -0.0156, -1.1655, -0.0106, -0.0021, -0.0067)


def mapping(side):
    return HandMapping(planning_model(config_directory(side)), PRESETS, side=side)


def feedback(side, hand):
    positions = {**dict.fromkeys(LEFT_ARM + RIGHT_ARM + HEAD, 0.0), **dict(zip(arm_joints(side), ARM, strict=True))}
    left = hand if side == "left" else mapping("left").preset("open")
    right = hand if side == "right" else mapping("right").preset("open")
    return Feedback(0.0, positions, np.asarray(left, float), right_hand=np.asarray(right, float))


@pytest.mark.parametrize("side", ["left", "right"])
def test_every_planner_reserves_disabled_palm_spheres_that_never_hit_the_hand(side):
    kin = robot_config(feedback(side, mapping(side).preset("open")), side=side)["robot_cfg"]["kinematics"]
    assert kin["extra_links"][ATTACH_LINK]["parent_link_name"] == "R_palm"
    assert ATTACH_LINK in kin["collision_link_names"]
    slots = kin["collision_spheres"][ATTACH_LINK]
    assert len(slots) == ATTACH_SLOTS and all(sphere["radius"] == DISABLED_RADIUS for sphere in slots)
    ignored = set(kin["self_collision_ignore"][ATTACH_LINK])
    assert {"R_palm", "R_thumb_roll", f"{side}_wrist_roll_link"} <= ignored
    # The contact screen plans the empty arm and keeps none of it.
    screen = screen_robot_config({"robot_cfg": {"kinematics": kin}}, [], side=side)["robot_cfg"]["kinematics"]
    assert ATTACH_LINK not in screen["collision_link_names"]


def uncovered_m(model, state, hands, cover):
    """Worst distance (m) by which a finger sphere of ``hands`` leaves the open-hand model plus ``cover``."""
    q = state.vector(model.arm)[None]
    poses = np.vstack([np.atleast_2d(hands), model.mapping.preset("open")[None]])
    spheres, _ = model.spheres(state, np.repeat(q, len(poses), axis=0), hand=poses)
    palm = model.last_link_frames[:, model.link_indices["R_palm"]]
    moving = np.array([name.startswith("R_") for name in model.sphere_links])
    local = np.einsum("nsj,njk->nsk", spheres[:, moving, :3] - palm[:, None, :3, 3], palm[:, :3, :3])
    radii = spheres[0, moving, 3]
    holders = np.vstack([np.column_stack([local[-1], radii]), cover])
    swept = local[:-1].reshape(-1, 3)
    gap = np.linalg.norm(swept[:, None] - holders[None, :, :3], axis=2) + np.tile(radii, len(poses) - 1)[:, None]
    return float((gap - holders[None, :, 3]).min(axis=1).max())


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_carry_cover_fits_the_slots_and_covers_the_closed_hand_and_the_object(side):
    hand_map = mapping(side)
    model = RobotSphereModel(mapping=hand_map, directory=config_directory(side))
    closed = hand_map.preset("close")
    state = feedback(side, closed)
    cloud = np.random.default_rng(0).normal(size=(64, 3))
    cloud = 0.03 * cloud / np.linalg.norm(cloud, axis=1, keepdims=True) + [0.0, 0.04, 0.10]
    held = SimpleNamespace(arm_side=side, points_in_palm=tuple(cloud.ravel()), radius_m=0.04)

    carry = placement_cover(model, state, held)
    opening = opening_cover(model, state, held)

    assert carry.shape[1] == 4 and 2 <= len(carry) <= ATTACH_SLOTS and (carry[:, 3] > 0).all()
    np.testing.assert_allclose(carry[-1], [*held_center(cloud), 0.04])
    assert np.max(np.linalg.norm(cloud - carry[-1, :3], axis=1)) <= carry[-1, 3]
    assert 1 <= len(opening) <= ATTACH_SLOTS and (opening[:, 3] > 0).all()
    # Closed fingers leave the open-hand spheres by centimetres; the filler covers them.
    assert uncovered_m(model, state, closed, np.empty((0, 4))) > 0.02
    assert uncovered_m(model, state, closed, carry[:-1]) <= 1e-6
    # Every pose while the hand opens at the release is covered by the opening cover.
    sweep = np.linspace(closed, hand_map.preset("open"), 9)[:-1]
    assert uncovered_m(model, state, sweep, opening) <= 1e-6
    # An open hand needs nothing beyond the warmed model: only the object is attached.
    assert len(placement_cover(model, feedback(side, hand_map.preset("open")), held)) == 1


class Planner:
    """Locks a hand: a closed hand mismatches unless the session carries."""

    def __init__(self, feedback, estimate, received_at, **options):
        self.hand, self.max_goalset, self.events = feedback["hand"], options["max_goalset"], []
        self.attached = None

    def locked_joints_match(self, feedback, *, carrying=False):
        return carrying or feedback["hand"] == self.hand

    def warmup(self):
        pass

    def update_scene(self, feedback, estimate, *, carrying=False):
        assert self.locked_joints_match(feedback, carrying=carrying)
        self.events.append("update")

    def attach_object(self, spheres):
        self.attached = spheres
        self.events.append("attach")

    def detach_object(self):
        self.attached = None
        self.events.append("detach")

    def close(self):
        self.events.append("close")


def test_a_carry_session_attaches_without_rebuilding_and_always_detaches():
    with GraspPlanningService(gripper=GRIPPER, planner_factory=Planner) as service:
        service.prepare({"hand": "open"}, SimpleNamespace(received_at=1.0))
        warmed = service._planner
        with service.session({"hand": "closed"}, SimpleNamespace(received_at=2.0), attached=[[0, 0, 0.1, 0.04]]) as p:
            assert p is warmed and p.attached == [[0, 0, 0.1, 0.04]]
        with (
            pytest.raises(RuntimeError, match="no path"),
            service.session({"hand": "closed"}, SimpleNamespace(received_at=3.0), attached=[[0, 0, 0.1, 0.04]]),
        ):
            raise RuntimeError("no path")
        assert service._planner is warmed and warmed.attached is None
        assert warmed.events == ["update", "attach", "detach", "update", "attach", "detach"]
        # Without a carry a closed hand is still a locked-model change.
        service.prepare({"hand": "closed"}, SimpleNamespace(received_at=4.0))
        assert service._planner is not warmed and warmed.events[-1] == "close"


def joint_path_planner(valid=lambda q: np.ones(len(q), bool)):
    from aimdk_grasp.planner import LeftArmPlanner

    planner = object.__new__(LeftArmPlanner)
    planner.arm = tuple(LEFT_ARM)
    planner.motion = SimpleNamespace(trajopt_solver=SimpleNamespace(config=SimpleNamespace(interpolation_dt=0.02)))
    planner.valid_positions = valid
    return planner


def test_the_way_home_is_a_checked_joint_path_that_starts_and_stops_at_rest():
    planner = joint_path_planner()
    # From the arm over the container (about 0.6 rad away) to HOME.
    start, home = np.array([-0.2, 0.3, 0.1, -1.6, 0.2, 0.0, 0.0]), np.array([0.4, 0.0, 0.0, -1.2, 0.0, 0.0, 0.0])
    path = planner._checked_joint_path(start, home)
    np.testing.assert_allclose(path.positions[[0, -1]], [start, home])
    speed = np.abs(np.diff(path.positions, axis=0)) / np.diff(path.times)[:, None]
    assert speed.max() <= planner.HOME_JOINT_SPEED + 1e-6
    assert speed[[0, -1]].max() < 0.1 * planner.HOME_JOINT_SPEED
    # One colliding sample, anywhere, refuses the straight path.
    blocked = joint_path_planner(lambda q: np.arange(len(q)) != len(q) // 2)
    assert blocked._checked_joint_path(start, home) is None


def test_a_blocked_straight_way_home_goes_around_by_pose_then_to_the_exact_joints():
    from aimdk_grasp.trajectory import TimedTrajectory

    start, home = np.array(ARM), np.array([0.4, 0.0, 0.0, -1.2, 0.0, 0.0, 0.0])
    # Only the straight start->home line is blocked.
    planner = joint_path_planner(lambda q: np.full(len(q), not np.allclose(q[0], start) or len(q) < 3))
    planner._palm_poses = lambda q: np.repeat(np.eye(4)[None], len(q), axis=0)
    asked = []

    def plan_pose(feedback, goal, now, *, deadline=None):
        asked.append(feedback.vector(LEFT_ARM))
        return TimedTrajectory(np.array([0.0, 0.5]), np.stack([start, home + 0.01]), tuple(LEFT_ARM))

    planner.plan_pose = plan_pose
    planner._checked_joint_path = lambda a, b: (
        None if np.allclose(a, start) else TimedTrajectory(np.array([0.0, 0.1]), np.stack([a, b]), tuple(LEFT_ARM))
    )
    state = feedback("left", mapping("left").preset("open"))
    path = planner._path_home(start, home, 0.0, state, deadline=None)
    np.testing.assert_allclose(asked[0], start)
    np.testing.assert_allclose(path.positions[-1], home)
    assert np.all(np.diff(path.times) > 0)


def test_a_blocked_way_home_drives_the_transfer_back_before_going_straight_home():
    from aimdk_grasp.trajectory import TimedTrajectory

    home = np.array([0.4, 0.0, 0.0, -1.2, 0.0, 0.0, 0.0])
    handover = np.array([-0.5, 0.2, 0.0, -1.4, 0.0, 0.0, 0.0])
    above = np.array([-0.7, 0.4, 0.1, -1.0, 0.1, 0.0, 0.0])
    retreat_end = above + 0.005  # the retreat reaches the pose, not the exact joints
    transfer = TimedTrajectory(
        np.array([0.0, 0.4, 0.8]), np.stack([handover, (handover + above) / 2, above]), tuple(LEFT_ARM)
    )
    # Straight lines from over the basket clip the table; from the handover they do not.
    planner = joint_path_planner(lambda q: np.full(len(q), not (np.allclose(q[0], retreat_end) and len(q) > 3)))
    planner.plan_pose = lambda *a, **k: pytest.fail("the way back needs no pose plan")
    path = planner._path_home(
        retreat_end, home, 0.0, feedback("left", mapping("left").preset("open")), deadline=None, back=transfer
    )
    np.testing.assert_allclose(path.positions[0], retreat_end)
    np.testing.assert_allclose(path.positions[-1], home)
    assert any(np.allclose(q, handover) for q in path.positions)
    assert np.all(np.diff(path.times) > 0)


def test_the_pick_moves_reversed_lead_from_the_handover_back_home():
    from aimdk_grasp.held_object import return_path

    from manipulation_execution.grasp_pipeline.contracts import ArmSegment, SegmentKind

    home, contact, top = np.zeros(7), np.full(7, 0.3), np.full(7, 0.5)
    approach = ArmSegment(
        "approach",
        SegmentKind.TRAJECTORY,
        tuple(LEFT_ARM),
        np.stack([home, contact / 2, contact]),
        np.array([0.0, 0.5, 1.0]),
    )
    lift = ArmSegment(
        "lift_carry", SegmentKind.TRAJECTORY, tuple(LEFT_ARM), np.stack([contact, top]), np.array([0.0, 0.4])
    )
    positions, times = return_path([approach, lift])
    path = np.asarray(positions).reshape(len(times), 7)
    np.testing.assert_allclose(path, np.stack([top, contact, contact / 2, home]))
    np.testing.assert_allclose(times, [0.0, 0.4, 0.9, 1.4])
    # A target segment carries no timing: no recorded way back.
    target = ArmSegment("approach", SegmentKind.TARGET, tuple(LEFT_ARM), np.stack([home, contact]))
    assert return_path([target, lift]) == ((), ())


def test_when_no_straight_line_works_the_way_home_follows_picks_own_path_back():
    from aimdk_grasp.trajectory import TimedTrajectory

    home = np.array([0.4, 0.0, 0.0, -1.2, 0.0, 0.0, 0.0])
    handover = np.array([-0.5, 0.2, 0.0, -1.4, 0.0, 0.0, 0.0])
    above = np.array([-0.7, 0.4, 0.1, -1.0, 0.1, 0.0, 0.0])
    transfer = TimedTrajectory(np.array([0.0, 0.8]), np.stack([handover, above]), tuple(LEFT_ARM))
    via = np.array([0.0, 0.1, 0.0, -1.3, 0.0, 0.0, 0.0])
    picked = TimedTrajectory(np.array([0.0, 0.5, 1.0]), np.stack([handover, via, home]), tuple(LEFT_ARM))

    def valid(q):
        # Long straight lines (more than 3 samples) clip the table; recorded paths do not.
        return np.full(len(q), len(q) <= 3)

    planner = joint_path_planner(valid)
    planner.plan_pose = lambda *a, **k: pytest.fail("a recorded way home needs no pose plan")
    state = feedback("left", mapping("left").preset("open"))
    path = planner._path_home(above, home, 0.0, state, deadline=None, back=transfer, return_path=picked)
    np.testing.assert_allclose(path.positions[[0, -1]], [above, home])
    assert any(np.allclose(q, via) for q in path.positions)
    # After a drop the arm is back at the handover: the recorded path alone suffices.
    path = planner._path_home(handover, home, 0.0, state, deadline=None, return_path=picked)
    np.testing.assert_allclose(path.positions, picked.positions)


def test_a_wrist_settled_a_hair_past_its_model_bound_still_gets_the_carry_cover():
    hand_map = mapping("right")
    model = RobotSphereModel(mapping=hand_map, directory=config_directory("right"))
    upper = {joint["name"]: joint["bounds"] for joint in model.joints}["right_wrist_pitch_joint"][1]
    cloud = 0.03 * np.random.default_rng(0).normal(size=(64, 3)) + [0.0, 0.04, 0.10]
    held = SimpleNamespace(arm_side="right", points_in_palm=tuple(cloud.ravel()), radius_m=0.04)

    def state(wrist_pitch):
        base = feedback("right", hand_map.preset("close"))
        positions = {name: float(base.vector([name])[0]) for name in LEFT_ARM + RIGHT_ARM + HEAD}
        positions["right_wrist_pitch_joint"] = wrist_pitch
        return Feedback(0.0, positions, base.hand("left"), right_hand=base.hand("right"))

    # Live 2026-11-06: 0.526 measured while carrying, model bound 0.5236. The
    # cover is palm-frame, so it equals the one at the bound.
    past, at_bound = placement_cover(model, state(upper + 0.0035), held), placement_cover(model, state(upper), held)
    np.testing.assert_allclose(past, at_bound, atol=1e-9)


def test_a_start_a_hair_past_a_model_bound_plans_from_just_inside_it():
    from aimdk_grasp.planner import START_BOUND_INSET_RAD, START_BOUND_SLACK_RAD, LeftArmPlanner

    lower, upper = np.full(7, -1.0), np.full(7, 0.5236)
    planner = object.__new__(LeftArmPlanner)
    planner.arm = tuple(RIGHT_ARM)
    bounds = np.stack([lower, upper]).astype(np.float32)
    planner._arm_bounds = (bounds[0].astype(float), bounds[1].astype(float))

    def start(wrist_pitch):
        positions = {**dict.fromkeys(LEFT_ARM + RIGHT_ARM + HEAD, 0.0), "right_wrist_pitch_joint": wrist_pitch}
        return planner._measured_start(Feedback(0.0, positions, np.zeros(10)))

    index = RIGHT_ARM.index("right_wrist_pitch_joint")
    taken = start(0.527)
    assert taken[index] == pytest.approx(np.float32(0.5236) - START_BOUND_INSET_RAD)
    assert np.all(np.delete(taken, index) == 0.0)
    # Inside the bound, or further past it than the slack: left exactly as measured.
    assert start(0.52)[index] == 0.52
    far = float(np.float32(0.5236)) + START_BOUND_SLACK_RAD + 0.005
    assert start(far)[index] == far
