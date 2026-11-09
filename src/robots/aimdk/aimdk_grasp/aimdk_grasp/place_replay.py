"""Replay recorded X2 place plans offline, as the grasp head made them.

The grasp head records each X2PlacePlanner.plan call when
IBROBOT_X2_PLAN_INPUTS_DIR names a directory (place_request.save_place_inputs):
feedback, the container estimate and its scene, the held object, the place
targets with both hand covers, the admitted snapshot time, the budget and the
live outcome with its timings. This replays the planner's plan_place on a
resident service per arm, built from the robot configuration as the head
builds it, on this host's GPU. No robot, no ROS.

Every call's clocks (scene, feedback, admitted snapshot) are shifted by one
offset, so the planner's age checks see what they saw live while the calls can
be replayed in any order and repeated. A planner build (first call of an arm,
or another locked posture) is not timed, as it is not part of a live place.

Usage:
    replay_x2_place_inputs DIR_OR_PKL... --output result.json [--repeats 2] [--regenerate-targets]
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import time
from dataclasses import replace
from pathlib import Path

DEFAULT_CONFIG = "aimdk_x2_voice_mirror_grasp"


def recorded_places(paths: list[Path]) -> list[Path]:
    files = []
    for path in paths:
        files += sorted(path.glob("place_*.pkl")) if path.is_dir() else [path]
    if not files:
        raise SystemExit("no recorded place plans")
    return files


def shifted(recorded: dict, offset_s: float):
    """Feedback, container (with its scene) and admitted time with every clock moved by offset_s."""
    feedback = replace(recorded["feedback"], received_at=recorded["feedback"].received_at + offset_s)
    container = recorded["container"]
    scene = replace(container.scene, received_at=container.scene.received_at + offset_s)
    return feedback, replace(container, scene=scene), recorded["admitted_at"] + offset_s


def x2_place_services(config_name: str):
    """A function giving each arm's (planning service, robot) as the grasp head builds them (backend.head_parts)."""
    from aimdk_grasp.backend import robot_for_side, x2_robot
    from aimdk_grasp.planner_service import GraspPlanningService
    from robot_config.config_path import resolve_robot_config_path
    from robot_config.loader import load_robot_config_dict

    config = load_robot_config_dict(resolve_robot_config_path(config_name=config_name), defer_interface_binding=True)
    base = x2_robot(config.get("grasp_execution") or {})
    built: dict = {}

    def for_side(side):
        if side not in built:
            robot = robot_for_side(base, side)
            planning = robot.planning
            service = GraspPlanningService(
                gripper=robot.gripper,
                side=side,
                planning_profile=planning.planning_profile,
                approach_seed_policy=planning.approach_seed_policy,
                approach_over_height_m=planning.approach_over_height_m,
                approach_order_m=planning.approach_order_m,
                max_goalset=planning.max_goalset,
            )
            built[side] = (service, robot)
        return built[side]

    for_side.built = built
    return for_side


def _duration(path):
    times = getattr(path, "times", None)
    return None if times is None or not len(times) else round(float(times[-1]), 4)


def replay(files, services_for, *, budget_s=None, repeats=1, regenerate_targets=False):
    """One row per call and repeat: the live outcome beside the replayed one."""
    from aimdk_grasp.place_request import held_return_path
    from aimdk_grasp.place_targets import place_targets

    rows, clock = [], 1_000.0
    for repeat in range(repeats):
        for path in files:
            recorded = pickle.loads(path.read_bytes())
            offset = clock - recorded["admitted_at"]
            clock += 100.0
            feedback, container, now = shifted(recorded, offset)
            held = recorded["held"]
            service, _ = services_for(recorded["arm_side"])
            live = recorded.get("outcome") or {}
            budget = budget_s if budget_s is not None else recorded["budget_s"]
            row = dict(
                call=path.name,
                repeat=repeat,
                arm_side=recorded["arm_side"],
                live_outcome=live.get("outcome"),
                live_candidate=(live.get("planner") or {}).get("candidate"),
                live_plan_place_s=(live.get("timings_s") or {}).get("plan_place"),
                budget_s=budget,
            )
            targets = recorded.get("targets")
            if targets is None and not regenerate_targets:
                # Refused before planning (place_targets): nothing to plan unless the targets are regenerated.
                row.update(skipped=True, success=False, error=f"live refusal: {live.get('outcome')}", wall_s=0.0)
                rows.append(row)
                print(json.dumps(row, default=str), flush=True)
                continue
            if regenerate_targets:
                began = time.perf_counter()
                targets = place_targets(container, held, recorded["palm"], recorded["place_config"])
                row.update(targets_s=round(time.perf_counter() - began, 4), targets_changed=len(targets))
            planner = service._planner
            row["rebuilt"] = planner is None or not planner.locked_joints_match(feedback, carrying=True)
            if row["rebuilt"]:
                service.prepare(feedback, container.scene, carrying=True)  # not part of a live place
            began = time.perf_counter()
            deadline = float("inf") if budget is None else began + budget

            def check(deadline=deadline):
                if time.perf_counter() >= deadline:
                    raise TimeoutError("replayed place budget exceeded")

            report: dict = {}
            try:
                with service.session(feedback, container.scene, attached=recorded["attached"]) as planner:
                    try:
                        paths = planner.plan_place(
                            feedback,
                            targets,
                            held.home_positions
                            if getattr(recorded["place_config"], "plan_return_home", True)
                            else None,
                            now,
                            check_deadline=check,
                            deadline=deadline,
                            opening=recorded["opening"],
                            return_path=held_return_path(held),
                        )
                    finally:
                        report = dict(getattr(planner, "place_report", {}))
                row.update(
                    success=True,
                    candidate=report.get("candidate"),
                    rounds=report.get("rounds"),
                    motion_s=[_duration(part) for part in paths],
                )
            except Exception as exc:  # noqa: BLE001 - a failed call is a result
                row.update(success=False, error=f"{type(exc).__name__}: {exc}"[:600], rounds=report.get("rounds"))
            row["wall_s"] = round(time.perf_counter() - began, 4)
            row["timings_s"] = report.get("timings_s")
            row["failures"] = report.get("failures")
            rows.append(row)
            print(json.dumps(row, default=str), flush=True)
    return rows


def summary(rows) -> dict:
    skipped = sum(bool(row.get("skipped")) for row in rows)
    rows = [row for row in rows if not row.get("skipped")]
    planned = [row for row in rows if row["success"]]
    times = sorted(row["wall_s"] for row in rows)
    screens = sorted(row["timings_s"]["screen"] for row in rows if (row.get("timings_s") or {}).get("screen"))
    return dict(
        skipped_refusals=skipped,
        calls=len(rows),
        planned=len(planned),
        live_planned=sum(row["live_outcome"] == "ok" for row in rows),
        same_candidate=sum(row.get("candidate") == row["live_candidate"] for row in planned),
        median_wall_s=statistics.median(times) if times else None,
        max_wall_s=max(times) if times else None,
        median_screen_s=statistics.median(screens) if screens else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="place_*.pkl files or directories of them")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot-config", default=DEFAULT_CONFIG)
    parser.add_argument("--budget-s", type=float, default=None, help="planning budget (default: each call's live one)")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--regenerate-targets", action="store_true", help="recompute place_targets with this code (covers stay)"
    )
    args = parser.parse_args()
    files = recorded_places(args.inputs)
    services_for = x2_place_services(args.robot_config)
    try:
        rows = replay(
            files,
            services_for,
            budget_s=args.budget_s,
            repeats=args.repeats,
            regenerate_targets=args.regenerate_targets,
        )
    finally:
        for service, _ in services_for.built.values():
            service.close()
    result = dict(
        robot_config=args.robot_config,
        budget_s=args.budget_s,
        regenerate_targets=args.regenerate_targets,
        summary=summary(rows),
        rows=rows,
    )
    args.output.write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
