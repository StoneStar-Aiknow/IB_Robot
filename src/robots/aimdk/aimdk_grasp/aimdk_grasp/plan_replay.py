"""Replay recorded X2 plan calls offline, as the grasp head made them, optionally with the leg as coarse boxes.

The grasp head records each GraspPlanningService.plan call when
IBROBOT_X2_PLAN_INPUTS_DIR names a directory (planner_service.save_plan_inputs):
feedback, scene estimate (coarse and fine), candidates, camera pose, options and
the live outcome. This replays them on a resident service built from the robot
configuration, as the head builds it, on this host's GPU. No robot, no ROS.

Every call's clocks (scene, feedback, admitted snapshot) are shifted by one
offset, so the planner's age checks see what they saw live while the calls can
be replayed in any order and repeated.

Usage:
    replay_x2_plan_inputs DIR_OR_PKL... --output result.json [--coarse-leg-boxes 0.025] [--repeats 2]
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import time
from dataclasses import replace
from pathlib import Path

DEFAULT_CONFIG = "aimdk_x2_grasp_cuda"


def recorded_calls(paths: list[Path]) -> list[Path]:
    files = []
    for path in paths:
        files += sorted(path.glob("plan_*.pkl")) if path.is_dir() else [path]
    if not files:
        raise SystemExit("no recorded plan calls")
    return files


def with_leg_boxes(estimate, shrink_m: float):
    """The estimate with the pelvis and left leg as boxes in its coarse scene only (scene_estimation.leg_boxes)."""
    from aimdk_grasp.scene_estimation import leg_boxes

    scene = dict(estimate.scene, cuboid={**estimate.scene["cuboid"], **leg_boxes(shrink_m)})
    return replace(estimate, scene=scene)


def shifted(recorded: dict, offset_s: float):
    """Feedback, estimate and options with every clock moved by offset_s."""
    feedback = replace(recorded["feedback"], received_at=recorded["feedback"].received_at + offset_s)
    estimate = replace(recorded["estimate"], received_at=recorded["estimate"].received_at + offset_s)
    options = dict(recorded["options"])
    if options.get("snapshot_now") is not None:
        options["snapshot_now"] = options["snapshot_now"] + offset_s
    return feedback, estimate, options


def x2_service(config_name: str):
    """The grasp head's planning service for a robot configuration (backend.head_parts)."""
    from aimdk_grasp.backend import x2_robot
    from aimdk_grasp.planner_service import GraspPlanningService
    from robot_config.config_path import resolve_robot_config_path
    from robot_config.loader import load_robot_config_dict

    config = load_robot_config_dict(resolve_robot_config_path(config_name=config_name), defer_interface_binding=True)
    robot = x2_robot(config.get("grasp_execution") or {})
    planning = robot.planning
    service = GraspPlanningService(
        gripper=robot.gripper,
        planning_profile=planning.planning_profile,
        approach_seed_policy=planning.approach_seed_policy,
        max_goalset=planning.max_goalset,
    )
    return service, dict(
        planning_profile=planning.planning_profile,
        approach_seed_policy=planning.approach_seed_policy,
        max_goalset=planning.max_goalset,
    )


def replay(files, service, *, leg_shrink_m=None, budget_s=None, repeats=1):
    """One row per call and repeat: live outcome beside the replayed one."""
    rows, clock = [], 1_000.0
    for repeat in range(repeats):
        for path in files:
            recorded = pickle.loads(path.read_bytes())
            offset = clock - recorded["estimate"].received_at
            clock += 100.0
            feedback, estimate, options = shifted(recorded, offset)
            if leg_shrink_m is not None:
                estimate = with_leg_boxes(estimate, leg_shrink_m)
            budget = budget_s if budget_s is not None else recorded["budget_s"]
            rebuilt = service._planner is None or not service._planner.locked_joints_match(feedback)
            row = dict(call=path.name, repeat=repeat, live=recorded["outcome"], rebuilt=rebuilt, budget_s=budget)
            began = time.perf_counter()
            try:
                if rebuilt:
                    service.prepare(feedback, estimate)  # a build is not part of a live plan call
                    began = time.perf_counter()
                _, report = service.plan(
                    feedback,
                    estimate,
                    recorded["camera_from_grasps"],
                    recorded["confidence"],
                    recorded["base_from_camera"],
                    deadline=None if budget is None else time.perf_counter() + budget,
                    **options,
                )
                attempts = report.get("planning_attempts") or []
                row.update(success=True, candidate=report.get("candidate_index"), attempts=len(attempts))
            except Exception as exc:  # noqa: BLE001 - a failed call is a result
                row.update(success=False, error=f"{type(exc).__name__}: {exc}"[:600])
            row["wall_s"] = time.perf_counter() - began
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "live"}), flush=True)
    return rows


def summary(rows) -> dict:
    planned = [row for row in rows if row["success"]]
    times = sorted(row["wall_s"] for row in rows)
    return dict(
        calls=len(rows),
        planned=len(planned),
        live_planned=sum(bool(row["live"].get("success")) for row in rows),
        median_wall_s=statistics.median(times) if times else None,
        max_wall_s=max(times) if times else None,
        median_planned_wall_s=statistics.median(row["wall_s"] for row in planned) if planned else None,
        fine_failures=sum("fine collision" in row.get("error", "") for row in rows),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="plan_*.pkl files or directories of them")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot-config", default=DEFAULT_CONFIG)
    parser.add_argument("--coarse-leg-boxes", type=float, default=None, metavar="SHRINK_M")
    parser.add_argument("--budget-s", type=float, default=None, help="planning budget (default: each call's live one)")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    files = recorded_calls(args.inputs)
    service, options = x2_service(args.robot_config)
    try:
        rows = replay(files, service, leg_shrink_m=args.coarse_leg_boxes, budget_s=args.budget_s, repeats=args.repeats)
    finally:
        service.close()
    result = dict(
        robot_config=args.robot_config,
        service=options,
        coarse_leg_boxes=args.coarse_leg_boxes,
        budget_s=args.budget_s,
        summary=summary(rows),
        rows=rows,
    )
    args.output.write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
