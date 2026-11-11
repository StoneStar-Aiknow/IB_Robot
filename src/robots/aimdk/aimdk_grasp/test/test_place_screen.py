"""Placement screens both ends of every candidate and pairs connected IK branches."""

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.joints import LEFT_ARM
from aimdk_grasp.planner import LeftArmPlanner, PlacementSearchExhausted

# Candidate k: above at x=k, z=1; release at x=k, z=0 (the stub tells them apart).
ABOVE, RELEASE = 1.0, 0.0


def _targets(count):
    targets = []
    for k in range(count):
        above, release = np.eye(4), np.eye(4)
        above[:3, 3] = (k, 0.0, ABOVE)
        release[:3, 3] = (k, 0.0, RELEASE)
        targets.append((above, release))
    return targets


def _state(value):
    return np.full(7, value, dtype=np.float32)


def _planner(states, *, invalid=(), descent=None, opening_invalid=(), straight=False):
    """``states[(k, z)]``: the screen's successful IK states for that goal (none: unreachable).

    ``opening_invalid`` states collide only while the opening hand is attached.
    ``straight`` lets the transfer try the checked straight joint path (else TrajOpt only).
    """
    calls = SimpleNamespace(screen=[], valid=[], pose=[], segment=[], goalset=0, opening_valid=[])
    attached = []
    seeds = 3

    def batch(transforms, start):
        calls.screen.append(len(transforms))
        success = np.zeros((len(transforms), seeds), dtype=bool)
        positions = np.zeros((len(transforms), seeds, 7), dtype=np.float32)
        for row, transform in enumerate(transforms):
            found = states.get((int(round(transform[0, 3])), transform[2, 3]), [])
            if found:
                success[row, : len(found)] = True
                positions[row, : len(found)] = found
        return success, positions

    def valid(positions):
        opening = bool(attached)
        (calls.opening_valid if opening else calls.valid).append(len(positions))
        bad = list(invalid) + (list(opening_invalid) if opening else [])
        return np.array([not any(np.allclose(q, b) for b in bad) for q in positions])

    @contextmanager
    def attachment(spheres):
        attached.append(spheres)
        try:
            yield
        finally:
            attached.pop()

    def goalset(*args, **kwargs):
        calls.goalset += 1
        raise AssertionError("the screened search must not run the goalset IK")

    def plan_pose(feedback, goal, now, *, deadline, seed_config):
        calls.pose.append((int(round(goal[0, 3])), np.asarray(seed_config).copy()))
        return SimpleNamespace(positions=np.zeros((2, 7)))

    def segment(start, goal, check, extra_seeds=None):
        calls.segment.append((goal[2, 3], None if extra_seeds is None else np.asarray(extra_seeds).copy()))
        if goal[2, 3] == RELEASE and descent is not None:
            result = descent(len(calls.segment))
            if result is not None:
                return None, result
        # The descent ends on its first seed (the stub's release branch).
        end = np.zeros((1, 7)) if extra_seeds is None or goal[2, 3] != RELEASE else np.asarray(extra_seeds)[:1]
        return SimpleNamespace(positions=np.concatenate([np.zeros((1, 7)), end])), None

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    planner = SimpleNamespace(
        snapshot_matches=lambda *args: True,
        carrying=True,
        screen_ik=object(),
        _screen_batch_size=2,
        _measured_start=lambda feedback: np.zeros(7),
        _require_valid_start=lambda q: None,
        _batch_contact_ik=batch,
        valid_positions=valid,
        _state=lambda q: SimpleNamespace(position=np.asarray(q, dtype=np.float32)[None]),
        motion=SimpleNamespace(
            ik_solver=SimpleNamespace(solve_pose=goalset),
            trajopt_solver=SimpleNamespace(config=SimpleNamespace(interpolation_dt=0.02)),
        ),
        plan_pose=plan_pose,
        arm=LEFT_ARM,
        PLACE_STRAIGHT_TRANSFER=straight,
        HOME_CHECK_STEP=LeftArmPlanner.HOME_CHECK_STEP,
        HOME_JOINT_SPEED=LeftArmPlanner.HOME_JOINT_SPEED,
        _linear_grasp_segment=segment,
        _without_placed_object=Context,
        _attachment=attachment,
        violations=lambda positions: "opening hand collides",
        _path_home=lambda *args, **kwargs: SimpleNamespace(positions=np.zeros((2, 7))),
    )
    planner._place_screen = lambda *args: LeftArmPlanner._place_screen(planner, *args)
    planner._place_opening_screen = lambda *args: LeftArmPlanner._place_opening_screen(planner, *args)
    planner._place_pair = LeftArmPlanner._place_pair
    planner._checked_joint_path = lambda *args: LeftArmPlanner._checked_joint_path(planner, *args)
    planner._place_transfer = lambda *args, **kwargs: LeftArmPlanner._place_transfer(planner, *args, **kwargs)
    return planner, calls


def _plan(planner, count, *, opening=None):
    return LeftArmPlanner.plan_place(
        planner,
        "feedback",
        _targets(count),
        np.zeros(7),
        0.0,
        check_deadline=lambda: None,
        deadline=float("inf"),
        opening=opening,
    )


OPENING = np.zeros((2, 4))


def test_screen_tries_the_cheapest_joint_route_among_both_ended_candidates():
    states = {
        (0, ABOVE): [_state(0.0)],  # candidate 0: release unreachable, never tried
        (1, ABOVE): [_state(1.0)],  # cost 1.0 + 0.0
        (1, RELEASE): [_state(1.0)],
        # Candidate 2 (later in preference) costs 0.3 + 0.3: its above state 0.3, not 2.0
        # (2.0 is 0.05 from a release state but 2.0 away from the start at 0).
        (2, ABOVE): [_state(2.0), _state(0.3)],
        (2, RELEASE): [_state(2.05), _state(0.6), _state(3.0)],
    }
    planner, calls = _planner(states)
    _plan(planner, 3)
    assert calls.goalset == 0
    assert calls.screen == [2, 1, 2, 1]  # two batched screens, chunked by _screen_batch_size
    candidate, seed = calls.pose[0]
    assert candidate == 2 and seed.shape == (1, 1, 7)
    np.testing.assert_allclose(seed[0, 0], _state(0.3))
    # Descent gets that candidate's release states, nearest to the chosen above state first.
    goal_z, extra = calls.segment[0]
    assert goal_z == RELEASE
    np.testing.assert_allclose(extra, np.stack([_state(0.6), _state(2.05), _state(3.0)]))
    assert planner.place_report["candidate"] == 2 and planner.place_report["rounds"] == 1
    # The retreat stays the native call without extra seeds.
    assert calls.segment[1] == (ABOVE, None)


def test_screen_feasibility_is_fine_checked_incrementally():
    states = {
        (0, ABOVE): [_state(1.0), _state(0.0)],
        (0, RELEASE): [_state(0.0)],
    }
    planner, calls = _planner(states, invalid=[_state(1.0)])
    _plan(planner, 1)
    # Above: first state collides, the second is checked and valid; release: one check.
    assert calls.valid == [1, 1, 1]
    assert calls.pose[0][0] == 0
    # Where the screen's time went, per end: IK batches, fine rounds and the states they checked.
    parts = planner.place_report["timings_s"]["screen_parts"]
    counts = {
        end: {k: parts[end][k] for k in ("ik_batches", "fine_rounds", "fine_states", "feasible")} for end in parts
    }
    assert counts == {
        "above": {"ik_batches": 1, "fine_rounds": 2, "fine_states": 2, "feasible": 1},
        "release": {"ik_batches": 1, "fine_rounds": 1, "fine_states": 1, "feasible": 1},
    }
    assert all(parts[end]["ik_s"] >= 0 and parts[end]["fine_s"] >= 0 for end in parts)


def test_screen_drops_a_failed_candidate_and_tries_the_next():
    states = {(k, z): [_state(0.0)] for k in range(2) for z in (ABOVE, RELEASE)}
    planner, calls = _planner(states, descent=lambda call: "ik" if call == 1 else None)
    _plan(planner, 2)
    assert [candidate for candidate, _ in calls.pose] == [0, 1]
    report = planner.place_report
    assert {key: report[key] for key in ("candidate", "rounds", "candidates", "failed_stages")} == {
        "candidate": 1,
        "rounds": 2,
        "candidates": 2,
        "failed_stages": {"descend": 1},
    }
    assert "opening_screen" not in report  # no opening hand: nothing pre-checked
    # Why the first candidate was refused survives the later success.
    assert report["failures"] == ["candidate 0 at descend: descent: ik"]


def test_screen_without_a_both_ended_candidate_is_a_recoverable_exhausted_search():
    states = {(0, ABOVE): [_state(0.0)], (1, RELEASE): [_state(0.0)]}
    planner, calls = _planner(states)
    with pytest.raises(PlacementSearchExhausted, match="reachable above and release pose"):
        _plan(planner, 2)
    assert calls.pose == [] and calls.goalset == 0


def test_an_opening_failure_is_tried_last_never_dropped():
    # Candidate 0 is the cheaper route, but its release state collides once the hand opens.
    states = {
        (0, ABOVE): [_state(0.1)],
        (0, RELEASE): [_state(0.1)],
        (1, ABOVE): [_state(0.5)],
        (1, RELEASE): [_state(0.5)],
    }
    planner, calls = _planner(states, opening_invalid=[_state(0.1)], descent=lambda call: "ik" if call == 1 else None)
    with pytest.raises(PlacementSearchExhausted):
        _plan(planner, 2, opening=OPENING)
    # Pre-checked once per candidate under the opening hand; candidate 1 goes first,
    # fails its descent, and candidate 0 is still tried (its own round check stays the authority).
    assert calls.opening_valid[:1] == [2]
    assert [candidate for candidate, _ in calls.pose] == [1, 0]
    report = planner.place_report
    assert report["failed_stages"] == {"descend": 1, "opening": 1}
    assert report["opening_screen"] == {"checked_states": 2, "deprioritized": 1}


def test_release_seeds_that_pass_the_opening_come_first():
    states = {
        (0, ABOVE): [_state(0.0)],
        # Nearest release state to the above state collides once the hand opens.
        (0, RELEASE): [_state(0.05), _state(0.4)],
    }
    planner, calls = _planner(states, opening_invalid=[_state(0.05)])
    _plan(planner, 1, opening=OPENING)
    goal_z, extra = calls.segment[0]
    assert goal_z == RELEASE
    np.testing.assert_allclose(extra, np.stack([_state(0.4), _state(0.05)]))
    assert planner.place_report["candidate"] == 0 and planner.place_report["failed_stages"] == {}


def test_place_report_times_the_screens_and_every_round_stage():
    states = {(k, z): [_state(0.0)] for k in range(2) for z in (ABOVE, RELEASE)}
    planner, _ = _planner(states, descent=lambda call: "ik" if call == 1 else None)
    _plan(planner, 2, opening=OPENING)
    timings = planner.place_report["timings_s"]
    assert timings["screen"] >= 0 and timings["opening_screen"] >= 0 and timings["total"] >= 0
    failed, done = timings["rounds"]
    assert failed["candidate"] == 0 and failed["failed"] == "descend" and set(failed) >= {"transfer", "descend"}
    assert done["candidate"] == 1 and "failed" not in done
    assert set(done) == {"candidate", "transfer", "transfer_path", "descend", "opening", "retreat", "home"}
    assert done["transfer_path"] == "trajopt"  # the straight transfer is off in these stubs


def test_an_exhausted_search_still_reports_its_timings():
    states = {(0, ABOVE): [_state(0.0)], (0, RELEASE): [_state(0.0)]}
    planner, _ = _planner(states, descent=lambda call: "ik")
    with pytest.raises(PlacementSearchExhausted):
        _plan(planner, 1)
    report = planner.place_report
    assert report["rounds"] == 1 and report["failed_stages"] == {"descend": 1}
    assert report["timings_s"]["rounds"][0]["failed"] == "descend" and report["timings_s"]["total"] >= 0


def test_without_a_home_the_search_stops_at_the_retreat():
    """BK6 (2026-11-09): home TrajOpt refusals discarded good places and cost 4.3 s; the workflow returns the arm."""
    states = {(0, ABOVE): [_state(0.0)], (0, RELEASE): [_state(0.0)]}
    planner, calls = _planner(states)

    def refuse(*args, **kwargs):
        raise AssertionError("no way home is planned")

    planner._path_home = refuse
    paths = LeftArmPlanner.plan_place(
        planner, "feedback", _targets(1), None, 0.0, check_deadline=lambda: None, deadline=float("inf")
    )
    assert len(paths) == 3 and planner.place_report["candidate"] == 0 and not planner.place_report["failures"]
    assert not [round_ for round_ in planner.place_report["timings_s"]["rounds"] if "home" in round_]


def test_a_free_straight_joint_path_carries_to_the_screened_above_state_without_trajopt():
    states = {(0, ABOVE): [_state(0.3)], (0, RELEASE): [_state(0.3)]}
    planner, calls = _planner(states, straight=True)
    transfer = _plan(planner, 1)[0]
    assert calls.pose == []  # no TrajOpt solve
    np.testing.assert_allclose(transfer.positions[0], np.zeros(7))
    np.testing.assert_allclose(transfer.positions[-1], _state(0.3), atol=1e-6)
    # Starts and stops at rest, no joint faster than the way home's speed.
    speed = np.abs(np.diff(transfer.positions, axis=0)) / np.diff(transfer.times)[:, None]
    assert speed.max() <= LeftArmPlanner.HOME_JOINT_SPEED + 1e-6 and speed[0].max() < 0.05 and speed[-1].max() < 0.05
    assert planner.place_report["timings_s"]["rounds"][0]["transfer_path"] == "straight"


def test_a_colliding_straight_joint_path_falls_back_to_trajopt_from_the_same_seed():
    states = {(0, ABOVE): [_state(0.3)], (0, RELEASE): [_state(0.3)]}
    goal = _state(0.3)
    count = int(np.ceil(float(np.max(goal)) / LeftArmPlanner.HOME_CHECK_STEP)) + 1
    midway = np.linspace(np.zeros(7), goal, count)[count // 2]  # one of the straight path's checked samples
    planner, calls = _planner(states, invalid=[midway], straight=True)
    _plan(planner, 1)
    candidate, seed = calls.pose[0]
    assert candidate == 0
    np.testing.assert_allclose(seed[0, 0], _state(0.3))
    assert planner.place_report["timings_s"]["rounds"][0]["transfer_path"] == "trajopt"
