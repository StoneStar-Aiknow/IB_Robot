"""The X2 planner's goalset search over the grasp candidates."""

from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.grasp_session import STRATEGIES, select_plan
from aimdk_grasp.planner import GraspPlanningError


@pytest.mark.parametrize("profile", ["default", "latency"])
@pytest.mark.parametrize("side_first", [False, True])
def test_configured_ten_cm_first_keeps_every_candidate_and_strategy(profile, side_first):
    calls = []

    def refuse(feedback, grasps, matrix, now, **offsets):
        calls.append((int(grasps[0, 0, 3]), offsets["approach_offset"], offsets["lift_offset"]))
        raise RuntimeError("no solution")

    grasps = np.repeat(np.eye(4)[None], 3, axis=0)
    grasps[:, 0, 3] = np.arange(3)
    report = {}
    with pytest.raises(RuntimeError, match="No candidate"):
        select_plan(
            SimpleNamespace(max_goalset=1, planning_profile=profile, plan_grasp=refuse),
            None,
            SimpleNamespace(grasps=grasps, confidence=np.ones(3)),
            np.eye(4),
            None,
            report,
            strategy_quota=1,
            first_tier_longest_first=side_first,
            approach_order_m=[0.10, 0.15, 0.07, 0.05],
        )
    assert [call[1:] for call in calls[:4]] == [(-0.10, -0.06), (-0.15, -0.06), (-0.07, -0.06), (-0.05, -0.05)]
    expected = {(i, approach, lift) for i in range(3) for approach, lift in STRATEGIES}
    assert set(calls) == expected and len(calls) == len(expected)
    assert set(map(tuple, report["strategies"])) == set(STRATEGIES)
    assert report["first_tier_strategies"] == report["strategies"]


@pytest.mark.parametrize("order", [[0.10], [0.10, 0.15, 0.07, 0.07], [0.10, 0.15, 0.07, float("nan")], "0.10"])
def test_approach_order_cannot_silently_drop_or_invent_distances(order):
    from aimdk_grasp.grasp_request import GraspRequestConfig

    with pytest.raises(ValueError, match="approach_order_m"):
        GraspRequestConfig(approach_order_m=order)


def _grasps_with_approach(*approaches):
    grasps = np.repeat(np.eye(4)[None], len(approaches), axis=0)
    for grasp, approach in zip(grasps, approaches, strict=False):
        z = np.asarray(approach, dtype=float) / np.linalg.norm(approach)
        x = np.cross([0.0, 1.0, 0.0] if abs(z[1]) < 0.9 else [1.0, 0.0, 0.0], z)
        x /= np.linalg.norm(x)
        grasp[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    return grasps


def test_failed_goalset_winner_does_not_hide_other_candidates():
    calls = []

    def plan(feedback, grasps, matrix, now, **offsets):
        ids = grasps[:, 0, 3].astype(int).tolist()
        calls.append((ids, offsets["lift_offset"], now))
        if len(ids) == 3:
            raise GraspPlanningError("selected candidate failed fine validation", 1)
        return SimpleNamespace(goalset_index=1)

    observation = SimpleNamespace(grasps=np.repeat(np.eye(4)[None], 3, axis=0), confidence=np.array([0.2, 0.9, 0.6]))
    observation.grasps[:, 0, 3] = np.arange(3)
    report = {}
    _, index, lift = select_plan(
        SimpleNamespace(max_goalset=3, plan_grasp=plan),
        None,
        observation,
        np.eye(4),
        [True] * 3,
        report,
        now=lambda: 7.0,
    )
    assert calls == [([1, 2, 0], STRATEGIES[0][1], 7.0), ([1, 0], STRATEGIES[0][1], 7.0)]
    assert index == 0 and lift == STRATEGIES[0][1]
    assert report["planning_attempts"][0]["candidate"] == 2


@pytest.mark.parametrize("lifting", [{2}, set()])
def test_a_plan_that_cannot_lift_is_kept_while_another_candidate_may_lift(lifting):
    # Candidate 1 (most confident) plans with a hold lift; the search goes on.
    calls = []

    def plan(feedback, grasps, matrix, now, **offsets):
        ids = grasps[:, 0, 3].astype(int).tolist()
        calls.append(ids)
        chosen = ids[0]
        return SimpleNamespace(goalset_index=0, lift_method="vertical" if chosen in lifting else "hold")

    observation = SimpleNamespace(grasps=np.repeat(np.eye(4)[None], 3, axis=0), confidence=np.array([0.2, 0.9, 0.6]))
    observation.grasps[:, 0, 3] = np.arange(3)
    planner = SimpleNamespace(max_goalset=1, vertical_lift_m=0.05, plan_grasp=plan)
    report = {}
    result, index, _ = select_plan(
        planner, None, observation, np.eye(4), [True] * 3, report, strategies=[(-0.1, -0.06)]
    )
    if lifting:
        # The next candidate lifts: it wins, the hold plan of candidate 1 stays unused.
        assert index == 2 and result.lift_method == "vertical" and calls == [[1], [2]]
        assert report["lift_fallback"] == {"returned": "vertical", "held_candidate": 1}
    else:
        # None lifts: still a plan, the first one found (never fewer plans than before).
        assert index == 1 and result.lift_method == "hold" and calls == [[1], [2], [0]]
        assert report["lift_fallback"]["returned"] == "hold"
    assert report["winning_candidate"]["candidate"] == index
    # Without a vertical lift wanted, the first plan is returned as before.
    calls.clear()
    _, index, _ = select_plan(
        SimpleNamespace(max_goalset=1, vertical_lift_m=0.0, plan_grasp=plan),
        None,
        observation,
        np.eye(4),
        [True] * 3,
        {},
        strategies=[(-0.1, -0.06)],
    )
    assert index == 1 and calls == [[1]]


def test_recovered_plan_reports_actual_clearance_and_original_request():
    observation = SimpleNamespace(grasps=np.eye(4)[None], confidence=[0.9])
    recovered = SimpleNamespace(goalset_index=0, planning_method="contact_clearance", approach_offset_m=-0.2)
    planner = SimpleNamespace(max_goalset=1, plan_grasp=lambda *args, **options: recovered)
    report = {}
    plan, index, lift = select_plan(planner, None, observation, np.eye(4), None, report, strategies=[(-0.1, -0.12)])
    assert plan is recovered and index == 0 and lift == -0.12
    for row in (report["winning_candidate"], report["planning_attempts"][0]):
        assert row["approach_m"] == -0.2 and row["requested_approach_m"] == -0.1
        assert row["lift_m"] == -0.12 and row["planning_method"] == "contact_clearance"


def test_goalset_capacity_uses_strategies_outside_chunks():
    calls = []

    def fail(feedback, grasps, matrix, now, **offsets):
        calls.append((grasps[:, 0, 3].astype(int).tolist(), offsets["approach_offset"], offsets["lift_offset"]))
        raise RuntimeError("goalset unreachable")

    observation = SimpleNamespace(grasps=np.repeat(np.eye(4)[None], 3, axis=0), confidence=np.array([0.2, 0.9, 0.6]))
    observation.grasps[:, 0, 3] = np.arange(3)
    with pytest.raises(RuntimeError, match="No candidate"):
        select_plan(SimpleNamespace(max_goalset=2, plan_grasp=fail), None, observation, np.eye(4), [True] * 3, {})
    assert calls == [(ids, approach, lift) for approach, lift in STRATEGIES for ids in ([1, 2], [0])]


def test_strategy_quota_visits_alternatives_then_exhausts_deferred_candidates():
    calls = []

    def fail(feedback, grasps, matrix, now, **offsets):
        ids = grasps[:, 0, 3].astype(int).tolist()
        calls.append((offsets["approach_offset"], offsets["lift_offset"], ids))
        raise GraspPlanningError("selected candidate failed", 0)

    observation = SimpleNamespace(grasps=np.repeat(np.eye(4)[None], 3, axis=0), confidence=np.array([0.9, 0.6, 0.2]))
    observation.grasps[:, 0, 3] = np.arange(3)
    with pytest.raises(RuntimeError, match="No candidate"):
        select_plan(
            SimpleNamespace(max_goalset=3, plan_grasp=fail), None, observation, np.eye(4), None, {}, strategy_quota=1
        )
    assert calls[: len(STRATEGIES)] == [(a, lift, [0, 1, 2]) for a, lift in STRATEGIES]
    assert calls[len(STRATEGIES) :] == [(a, lift, ids) for a, lift in STRATEGIES for ids in ([1, 2], [2])]


@pytest.mark.parametrize("outcome", ["success", "failure", "publish"])
def test_deadline_does_not_retry_or_publish_expired_results(monkeypatch, outcome):
    clock = [1.0]
    monkeypatch.setattr("aimdk_grasp.grasp_session.time.perf_counter", lambda: clock[0])
    calls, published = [], []

    def plan(*args, deadline, on_approach, **kwargs):
        calls.append(kwargs)
        assert deadline == 2.0
        clock[0] = 3.0
        if outcome == "failure":
            raise GraspPlanningError("late failure", 0)
        if outcome == "publish":
            on_approach(None, 0)
        return SimpleNamespace(goalset_index=0)

    observation = SimpleNamespace(grasps=np.repeat(np.eye(4)[None], 2, axis=0), confidence=[0.9, 0.6])
    report = {}
    with pytest.raises(TimeoutError):
        select_plan(
            SimpleNamespace(max_goalset=2, plan_grasp=plan),
            None,
            observation,
            np.eye(4),
            None,
            report,
            deadline=2.0,
            on_approach=published.append,
        )
    assert len(calls) == 1 and published == []
    assert "winning_candidate" not in report


def test_latency_profile_visits_each_approach_before_repeating_lift_options():
    calls = []

    def fail(*args, approach_offset, lift_offset, **kwargs):
        calls.append((approach_offset, lift_offset))
        raise RuntimeError("unreachable")

    observation = SimpleNamespace(grasps=np.eye(4)[None], confidence=[1.0])
    with pytest.raises(RuntimeError, match="No candidate"):
        select_plan(
            SimpleNamespace(max_goalset=1, planning_profile="latency", plan_grasp=fail),
            None,
            observation,
            np.eye(4),
            None,
            {},
        )
    assert calls == [
        (-0.05, -0.05),
        (-0.10, -0.06),
        (-0.15, -0.06),
        (-0.07, -0.06),
        (-0.10, -0.12),
        (-0.15, -0.12),
        (-0.07, -0.12),
        (-0.10, -0.20),
        (-0.15, -0.20),
        (-0.07, -0.20),
    ]
    assert sorted(calls) == sorted(STRATEGIES)


def test_priority_orders_before_confidence_and_group_caps_goalset():
    calls = []

    def plan(feedback, grasps, matrix, now, **offsets):
        calls.append(grasps[:, 0, 3].astype(int).tolist())
        if len(calls) == 1:
            raise GraspPlanningError("first choice failed fine validation", 0)
        return SimpleNamespace(goalset_index=0)

    observation = SimpleNamespace(
        grasps=np.repeat(np.eye(4)[None], 5, axis=0), confidence=np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    )
    observation.grasps[:, 0, 3] = np.arange(5)
    report = {}
    _, index, _ = select_plan(
        SimpleNamespace(max_goalset=100, plan_grasp=plan),
        None,
        observation,
        np.eye(4),
        None,
        report,
        priority=np.array([1, 1, 0, 0, 1]),
        goalset_group=2,
    )
    assert report["planning_order"] == [2, 3, 0, 1, 4]
    assert report["goalset_group"] == 2 and report["priority_tiers"] == [2, 3]
    # The second goalset stays inside the preferred tier; it never mixes in 0.
    assert calls == [[2, 3], [3]] and index == 3
    with pytest.raises(ValueError, match="goalset_group"):
        select_plan(
            SimpleNamespace(max_goalset=1, plan_grasp=plan), None, observation, np.eye(4), None, {}, goalset_group=0
        )


def test_later_candidates_go_last_in_their_tier_and_never_share_a_goalset_with_the_others():
    calls = []

    def plan(feedback, grasps, matrix, now, **offsets):
        ids = grasps[:, 0, 3].astype(int).tolist()
        calls.append((ids, offsets["approach_offset"]))
        raise GraspPlanningError("approach rejected")

    observation = SimpleNamespace(
        grasps=np.repeat(np.eye(4)[None], 6, axis=0), confidence=np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    )
    observation.grasps[:, 0, 3] = np.arange(6)
    report = {}
    with pytest.raises(RuntimeError, match="No candidate"):
        select_plan(
            SimpleNamespace(max_goalset=4, plan_grasp=plan),
            None,
            observation,
            np.eye(4),
            None,
            report,
            priority=np.array([0, 0, 0, 0, 0, 1]),
            later=np.array([True, False, True, False, False, False]),
            strategies=[(-0.05, -0.05), (-0.10, -0.06)],
        )
    assert report["planning_order"] == [1, 3, 4, 0, 2, 5] and report["priority_tiers"] == [5, 1]
    # Same tier, same strategies: each strategy tries the others, then the later ones, before the next tier.
    assert calls == [
        ([1, 3, 4], -0.05),
        ([0, 2], -0.05),
        ([1, 3, 4], -0.10),
        ([0, 2], -0.10),
        ([5], -0.05),
        ([5], -0.10),
    ]

    # With the mixed retry, a tier that failed so is tried once more in mixed goalsets, in the
    # same order, before the next tier; a tier without later candidates is not repeated.
    calls.clear()
    with pytest.raises(RuntimeError, match="No candidate"):
        select_plan(
            SimpleNamespace(max_goalset=4, plan_grasp=plan),
            None,
            observation,
            np.eye(4),
            None,
            {},
            priority=np.array([0, 0, 0, 0, 0, 1]),
            later=np.array([True, False, True, False, False, False]),
            strategies=[(-0.05, -0.05), (-0.10, -0.06)],
            mixed_retry=True,
        )
    assert calls == [
        ([1, 3, 4], -0.05),
        ([0, 2], -0.05),
        ([1, 3, 4], -0.10),
        ([0, 2], -0.10),
        ([1, 3, 4, 0], -0.05),
        ([2], -0.05),
        ([1, 3, 4, 0], -0.10),
        ([2], -0.10),
        ([5], -0.05),
        ([5], -0.10),
    ]

    # Integer levels: each level after the lower ones of its tier, in goalsets of its own.
    calls.clear()
    with pytest.raises(RuntimeError, match="No candidate"):
        select_plan(
            SimpleNamespace(max_goalset=4, plan_grasp=plan),
            None,
            observation,
            np.eye(4),
            None,
            {},
            priority=np.array([0, 0, 0, 0, 0, 1]),
            later=np.array([1, 0, 2, 0, 1, 0]),
            strategies=[(-0.05, -0.05)],
        )
    assert calls == [([1, 3], -0.05), ([0, 4], -0.05), ([2], -0.05), ([5], -0.05)]


def test_lower_tier_waits_until_every_strategy_failed_on_the_preferred_tier():
    calls = []

    def plan(feedback, grasps, matrix, now, **offsets):
        ids = grasps[:, 0, 3].astype(int).tolist()
        calls.append((ids, offsets["approach_offset"]))
        # The preferred candidate plans only at the longer approach.
        if ids == [0] and offsets["approach_offset"] == -0.10:
            return SimpleNamespace(goalset_index=0)
        if ids == [1]:
            return SimpleNamespace(goalset_index=0)
        raise GraspPlanningError("approach rejected", 0)

    observation = SimpleNamespace(grasps=np.repeat(np.eye(4)[None], 2, axis=0), confidence=np.array([0.5, 0.9]))
    observation.grasps[:, 0, 3] = np.arange(2)
    report = {}
    _, index, _ = select_plan(
        SimpleNamespace(max_goalset=5, plan_grasp=plan),
        None,
        observation,
        np.eye(4),
        None,
        report,
        priority=np.array([0, 1]),
        strategies=[(-0.05, -0.05), (-0.10, -0.06)],
    )
    assert index == 0
    assert calls == [([0], -0.05), ([0], -0.10)]


def test_first_tier_tries_the_longest_approach_first():
    calls = []

    def fail(feedback, grasps, matrix, now, **offsets):
        calls.append((grasps[:, 0, 3].astype(int).tolist(), offsets["approach_offset"]))
        raise GraspPlanningError("approach rejected", 0)

    observation = SimpleNamespace(grasps=np.repeat(np.eye(4)[None], 2, axis=0), confidence=np.array([0.9, 0.5]))
    observation.grasps[:, 0, 3] = np.arange(2)
    report = {}
    with pytest.raises(RuntimeError, match="No candidate"):
        select_plan(
            SimpleNamespace(max_goalset=5, plan_grasp=fail),
            None,
            observation,
            np.eye(4),
            None,
            report,
            priority=np.array([0, 1]),
            strategies=[(-0.05, -0.05), (-0.10, -0.06), (-0.15, -0.06)],
            first_tier_longest_first=True,
        )
    assert [c for c in calls if c[0] == [0]] == [([0], -0.15), ([0], -0.10), ([0], -0.05)]
    # Lower tiers keep the configured order.
    assert [c for c in calls if c[0] == [1]] == [([1], -0.05), ([1], -0.10), ([1], -0.15)]


def test_without_a_planned_lift_each_approach_distance_is_tried_once():
    calls = []

    def fail(feedback, grasps, matrix, now, **offsets):
        calls.append((offsets["approach_offset"], offsets["lift_offset"]))
        raise GraspPlanningError("approach rejected", 0)

    observation = SimpleNamespace(grasps=np.eye(4)[None], confidence=np.array([0.9]))
    with pytest.raises(RuntimeError, match="No candidate"):
        select_plan(
            SimpleNamespace(max_goalset=5, plan_grasp=fail, _plan_lift=False),
            None,
            observation,
            np.eye(4),
            None,
            {},
            strategies=[(-0.10, -0.06), (-0.10, -0.12), (-0.15, -0.06), (-0.15, -0.20)],
        )
    assert calls == [(-0.10, -0.06), (-0.15, -0.06)]
