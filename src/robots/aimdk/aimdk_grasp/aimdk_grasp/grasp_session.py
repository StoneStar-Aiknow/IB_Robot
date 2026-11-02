"""The X2 planner's goalset search over the grasp candidates (select_plan).

Planner-independent ordering (approach_priority, enclosed_points) lives in
manipulation_execution.grasp_pipeline.candidates.
"""

import json
import sys
import time
from collections.abc import Callable

import numpy as np

# Side grasps need room to bring the hand around the object: TrajOpt paths to
# a 5 cm pre-grasp swept the fingertips through the bottle. With the same side
# candidates searched at one distance each (six 2026-09-30 captures), 5/7/10/15
# cm planned 2/3/3/5 of 6, and every 15 cm success came on the first attempt.
# The preferred tier of a side_first request tries the longest approach first.
SIDE_FIRST_LONGEST_APPROACH = True


# (approach, lift) offsets along the palm axis, m, in the order select_plan tries them.
STRATEGIES = [
    (-0.15, -0.20),
    (-0.15, -0.12),
    (-0.15, -0.06),
    (-0.10, -0.20),
    (-0.10, -0.12),
    (-0.10, -0.06),
    (-0.07, -0.20),
    (-0.07, -0.12),
    (-0.07, -0.06),
    (-0.05, -0.05),
]


def select_plan(
    planner,
    feedback,
    observation,
    matrix,
    feasible,
    report,
    checkpoint=lambda: None,
    *,
    now: float | Callable[[], float] = 0.0,
    on_approach=None,
    strategy_quota=5,
    strategies=None,
    deadline=None,
    priority=None,
    goalset_group=None,
    first_tier_longest_first=False,
    later=None,
    later_alone=True,
    mixed_retry=False,
):
    """Try strategy settings around one or more candidate goalsets.

    Give each strategy a bounded first turn, then exhaust its deferred candidates.
    This avoids spending the whole request on one difficult offset. No candidate
    or strategy is discarded by the quota. Each call gives cuRobo all candidates
    in a goalset, so its initial IK solve is shared instead of repeated once per
    candidate. A planner may cap the goalset size; groups preserve confidence
    order and the returned local goalset index is mapped back to the artifact.

    priority (lower first, e.g. approach_priority) orders ahead of confidence.
    goalset_group caps each goalset below the planner capacity so cuRobo picks
    among the best few candidates, not whichever of all feasible ones it
    reaches first (live 2026-09-30: it chose confidence rank 42 of 100 while
    five better feasible candidates were never tried).
    """
    if deadline is not None and not np.isfinite(deadline):
        raise ValueError("Finite monotonic planning deadline required")
    if strategies is None:
        # Try the short complete motion first for response latency. Preserve the
        # entire original search space, but visit each approach distance before
        # repeating it with a different lift. A failed approach/contact cannot
        # benefit from changing only the last segment's offset.
        strategies = STRATEGIES
        if getattr(planner, "planning_profile", "default") == "latency":
            by_approach = {}
            # After the short reach, try medium clearance before the longest
            # approach. Keep the remaining clearance and lift alternatives.
            medium = [pair for pair in STRATEGIES[:-1] if pair[0] == -0.10]
            other = [pair for pair in STRATEGIES[:-1] if pair[0] != -0.10]
            for pair in [STRATEGIES[-1], *medium, *other]:
                by_approach.setdefault(pair[0], []).append(pair)
            # A longer lift can make a reachable contact branch fail. Give each
            # approach its shortest existing lift before spending time on more.
            for pairs in by_approach.values():
                pairs.sort(key=lambda pair: abs(pair[1]))
            strategies = [
                pairs[turn]
                for turn in range(max(map(len, by_approach.values())))
                for pairs in by_approach.values()
                if turn < len(pairs)
            ]
    if not getattr(planner, "_plan_lift", True):
        # Without a planned lift the lift offset changes nothing: keep one
        # pair per approach distance, first occurrence, so no identical
        # attempt is repeated.
        seen, unique = set(), []
        for approach, lift in strategies:
            if approach not in seen:
                seen.add(approach)
                unique.append((approach, lift))
        strategies = unique
    strategies = np.asarray(strategies, dtype=float)
    if strategies.ndim != 2 or strategies.shape[1:] != (2,) or not len(strategies) or not np.isfinite(strategies).all():
        raise ValueError("Nonempty finite approach/lift strategy pairs required")
    candidates = range(len(observation.grasps)) if feasible is None else np.flatnonzero(feasible).tolist()
    rank = np.zeros(len(observation.grasps), dtype=int) if priority is None else np.asarray(priority)
    # later: candidates that go after the others inside their tier (same tier, same strategies);
    # an integer level orders them further (higher later), a bool is level 0 or 1.
    behind = np.zeros(len(observation.grasps), dtype=int) if later is None else np.asarray(later).astype(int)
    ordered = sorted(candidates, key=lambda i: (rank[i], behind[i], -observation.confidence[i]))
    report["planning_order"] = ordered
    report["planning_attempts"] = []
    if not ordered:
        raise RuntimeError("No candidate produced a complete three-segment plan")
    capacity = int(getattr(planner, "max_goalset", len(ordered)))
    if capacity < 1:
        raise ValueError("Planner max_goalset must be positive")
    if goalset_group is not None:
        if not isinstance(goalset_group, int) or goalset_group < 1:
            raise ValueError("goalset_group must be a positive integer")
        capacity = min(capacity, goalset_group)
        report["goalset_group"] = goalset_group
    if not isinstance(strategy_quota, int) or strategy_quota < 1:
        raise ValueError("strategy_quota must be a positive integer")
    report["strategy_quota"] = strategy_quota
    report["strategies"] = strategies.tolist()
    # Exhaust every strategy on one priority tier before the next: within one
    # strategy a lower tier used to succeed before the preferred tier reached
    # the approach distance it needed (plan6: side grasps all failed at 5 cm and
    # all planned at 10 cm, but a top-down one had already planned at 5 cm).
    # Goalsets never mix tiers, or cuRobo may pick the lower one inside a group.
    tiers = [[i for i in ordered if rank[i] == level] for level in sorted({int(rank[i]) for i in ordered})]
    report["priority_tiers"] = [len(tier) for tier in tiers]
    # A plan whose vertical lift fell back to a hold (the executor refuses to
    # run it until lift; live 2026-10-14: hold_ik on the first candidate with
    # 11 more screened) is kept while the search goes on for one that lifts.
    # It is returned when none does or the deadline comes first: never fewer
    # plans than before, only fewer that cannot lift.
    wants_lift = getattr(planner, "vertical_lift_m", 0) > 0
    fallback = None

    def held(exc=None):
        plan, index, lift, row = fallback
        report["winning_candidate"] = dict(row)
        report["lift_fallback"] = {
            "returned": "hold",
            "reason": "deadline" if exc is not None else "no candidate lifts",
        }
        return plan, index, lift

    for number, tier in enumerate(tiers):
        order = list(strategies)
        if number == 0 and first_tier_longest_first:
            # Stable: longest approach first, the original order within one distance.
            order.sort(key=lambda pair: -abs(pair[0]))
            report["first_tier_strategies"] = np.asarray(order).tolist()
        queues = [(approach, lift, list(tier)) for approach, lift in order]
        # A second, exhaustive pass preserves the original search space. Retain each
        # queue rather than selecting its failed candidates again on the second pass.
        turns = [(turn, queue) for turn in ("breadth", "deferred") for queue in queues]
        if mixed_retry and later_alone and len({int(behind[i]) for i in tier}) > 1:
            # Last, the whole tier again in mixed goalsets: a candidate that failed in
            # a split one can plan in another company (live 2026-10-14 clr_1 on the
            # Orin: 48 failed in [29, 48], planned in [29, 48, 49]).
            turns += [("mixed", (approach, lift, list(tier))) for approach, lift in order]
        for turn, (approach, lift, remaining) in turns:
            attempts = 0
            while remaining and (turn == "deferred" or attempts < strategy_quota):
                if deadline is not None and time.perf_counter() >= deadline:
                    if fallback is not None:
                        return held(TimeoutError())
                    raise TimeoutError("Planning deadline exceeded")
                attempts += 1
                group = remaining[:capacity]
                # Goalsets never mix later candidates with the others either: cuRobo may pick one.
                if later_alone and turn != "mixed":
                    group = [i for i in group if behind[i] == behind[group[0]]]
                row = {"candidates": group, "approach_m": approach, "lift_m": lift, "search_turn": turn}
                report["planning_attempts"].append(row)
                start = time.perf_counter()

                # Called by plan_grasp within this iteration; the defaults bind its group and row.
                def publish(trajectory, local_index, group=group, row=row, start=start):
                    assert on_approach is not None
                    if not 0 <= local_index < len(group):
                        raise RuntimeError("Planner returned an invalid approach goalset index")
                    if deadline is not None and time.perf_counter() >= deadline:
                        raise TimeoutError("Planning deadline exceeded")
                    row.update(
                        approach_ready=True, candidate=group[local_index], approach_ready_s=time.perf_counter() - start
                    )
                    on_approach(trajectory)

                try:
                    options: dict = {} if on_approach is None else {"on_approach": publish}
                    if deadline is not None:
                        options["deadline"] = deadline
                    plan = planner.plan_grasp(
                        feedback,
                        observation.grasps[group],
                        matrix,
                        now() if callable(now) else now,
                        approach_offset=approach,
                        lift_offset=lift,
                        **options,
                    )
                    if deadline is not None and time.perf_counter() >= deadline:
                        raise TimeoutError("Planning deadline exceeded")
                    local_index = plan.goalset_index
                    if not 0 <= local_index < len(group):
                        raise RuntimeError("Planner returned an invalid goalset index")
                    index = group[local_index]
                    row["success"] = True
                    row["candidate"] = index
                    row["confidence"] = float(observation.confidence[index])
                    row["planning_method"] = getattr(plan, "planning_method", "approach_first")
                    row["lift_method"] = getattr(plan, "lift_method", "optimized")
                    actual_offset = getattr(plan, "approach_offset_m", None)
                    if actual_offset is not None:
                        row["requested_approach_m"] = approach
                        row["approach_m"] = actual_offset
                    row["wall_s"] = time.perf_counter() - start
                    # Not once its approach was published: the arm may be moving on it.
                    if wants_lift and row["lift_method"] == "hold" and not row.get("approach_ready"):
                        if fallback is None:
                            fallback = (plan, index, lift, dict(row))
                        row["kept_as_fallback"] = True
                        remaining.remove(index)
                        continue
                    report["winning_candidate"] = dict(row)
                    if fallback is not None:
                        report["lift_fallback"] = {"returned": "vertical", "held_candidate": fallback[1]}
                    return plan, index, lift
                except TimeoutError as exc:
                    row.update(success=False, timed_out=True, error=str(exc))
                    if fallback is not None:
                        return held(exc)
                    raise
                except RuntimeError as exc:
                    row.update(success=False, error=str(exc))
                    if row.get("approach_ready"):
                        # The arm may already be moving. Retrying from the original
                        # feedback or switching candidates would invalidate its start.
                        raise
                    rejected = getattr(exc, "goalset_index", None)
                    if rejected is None:
                        del remaining[: len(group)]
                    elif 0 <= rejected < len(group):
                        row["candidate"] = group[rejected]
                        remaining.remove(group[rejected])
                    else:
                        raise RuntimeError("Planner returned an invalid failed goalset index") from exc
                finally:
                    row["wall_s"] = time.perf_counter() - start
                    print(json.dumps(row), flush=True)
                    # Not over an error on its way out: a deadline or cancel raised
                    # here would replace it, and a PlannerFault so hidden leaves the
                    # owner reusing a planner whose CUDA context is poisoned.
                    if sys.exc_info()[0] is None:
                        checkpoint()
    if fallback is not None:
        return held()
    raise RuntimeError("No candidate produced a complete three-segment plan")
