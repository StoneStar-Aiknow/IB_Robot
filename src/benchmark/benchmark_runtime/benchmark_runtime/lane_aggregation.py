"""Static lane fan-in over durable canonical facts, with fail-closed validation."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from benchmark_runtime._strict_json import dumps_strict, loads_strict, require_json_object
from benchmark_runtime.canonical_models import ArtifactStatus, CanonicalEpisodeRecord, CanonicalRunSummary
from benchmark_runtime.canonical_report import (
    CanonicalReportError,
    CanonicalReportWriter,
    load_episode_records,
    reduce_episode_records,
)
from benchmark_runtime.plan import BenchmarkPlan, benchmark_plan_from_json, benchmark_plan_to_json, partition_plan


def read_object(path: Path) -> dict:
    return require_json_object(loads_strict(path.read_text(encoding="utf-8")), str(path))


def global_plan_from_manifest(manifest: dict) -> BenchmarkPlan:
    plan = benchmark_plan_from_json(dumps_strict(manifest["resolved_plan"]))
    return replace(plan, lane_index=None, planned_episodes=plan.global_planned_episodes)


def _global_artifact(artifact: ArtifactStatus, lane_index: int) -> ArtifactStatus:
    ref = artifact.ref
    # URI schemes need not contain //; never pass URI refs through pathlib.
    if not re.match(r"[A-Za-z][A-Za-z0-9+.-]*:", ref) and not Path(ref).is_absolute():
        ref = str(Path("lanes") / f"lane_{lane_index:03d}" / ref)
    return replace(artifact, ref=ref, metadata={**artifact.metadata, "lane_index": lane_index})


def _aggregate_lanes(
    run_root: str | Path, *, run_id: str, plan: BenchmarkPlan, lane_count: int
) -> tuple[CanonicalRunSummary, list[CanonicalEpisodeRecord]]:
    """Snapshot terminal lanes once; missing/invalid lanes and duplicate work are partial.

    The caller must wait for terminal manifests, or explicitly call this on a
    deadline/fail-fast shutdown. Nonterminal lane records are never consumed.
    Every lane owns the complete configured episode plan. Global episodes retain
    their lane-local episode_id; artifact references are global run-relative
    paths or unchanged absolute paths/URIs with lane-qualified identity.
    """
    if plan.lane_index is not None:
        raise ValueError("aggregator requires the global plan")
    partition_plan(plan, 0, lane_count)
    lane_plans = [partition_plan(plan, index, lane_count) for index in range(lane_count)]
    root = Path(run_root)
    errors: list[str] = []
    records = {}
    scope_failures: list[ArtifactStatus] = []
    lane_statuses = {}
    lanes_root = root / "lanes"
    if lanes_root.is_dir():
        expected_names = {f"lane_{index:03d}" for index in range(lane_count)}
        unexpected = sorted(item.name for item in lanes_root.iterdir() if item.name not in expected_names)
        if unexpected:
            errors.append(f"unexpected lane entries: {unexpected}")
    for index, expected in enumerate(lane_plans):
        lane_name = f"lane_{index:03d}"
        lane_root = root / "lanes" / lane_name
        try:
            manifest = read_object(lane_root / "run_manifest.json")
            summary = read_object(lane_root / "summary.json")
            status = manifest.get("status")
            if status not in {"complete", "partial", "failed"}:
                raise ValueError("lane has no durable terminal manifest")
            if summary.get("run_status") != status:
                raise ValueError("lane manifest/summary terminal status mismatch")
            if manifest.get("run_id") != run_id or summary.get("run_id") != run_id:
                raise ValueError("lane run_id mismatch")
            if summary.get("suite") != plan.suite:
                raise ValueError("lane suite mismatch")
            actual_plan = benchmark_plan_from_json(dumps_strict(manifest["resolved_plan"]))
            if actual_plan != expected:
                raise ValueError("lane plan differs from expected complete lane plan")
            if manifest.get("lane_index") != index or manifest.get("lane_count") != lane_count:
                raise ValueError("lane manifest identity mismatch")
            lane_statuses[lane_name] = status
            if status != "complete":
                errors.append(f"{lane_name}: terminal status {status}")
            failures = [
                _global_artifact(ArtifactStatus(**item), index) for item in summary.get("artifact_failures", [])
            ]
            scope_failures.extend(failures)
            if failures:
                errors.append(f"{lane_name}: artifact failures")
            if not (lane_root / "episodes.jsonl").is_file():
                raise ValueError("lane episodes.jsonl is missing")
            lane_records = load_episode_records(lane_root / "episodes.jsonl")
            expected_keys = set(expected.episode_work)
            accepted_keys = set()
            for record in lane_records:
                key = (record.task_id, record.episode_index)
                lane_key = (index, *key)
                if key in accepted_keys:
                    errors.append(f"{lane_name}: duplicate episode {key}")
                    continue
                if key not in expected_keys:
                    errors.append(f"{lane_name}: unassigned episode {key}")
                    continue
                task = plan.tasks[record.task_id]
                if (
                    record.run_id != run_id
                    or record.suite != plan.suite
                    or record.seed != plan.seed
                    or record.task_name != task.name
                ):
                    errors.append(f"{lane_name}: episode identity mismatch {key}")
                    continue
                if plan.init_state_policy.use_init_state_id and (
                    record.requested_init_state_id != record.episode_index
                    or (
                        plan.init_state_policy.selection == "episode_index_modulo_available_native_init_state_count"
                        and record.actual_init_state_id != record.episode_index % task.init_state_count
                    )
                ):
                    errors.append(f"{lane_name}: init-state identity mismatch {key}")
                    continue
                work_index = plan.episode_work.index(key)
                if record.metadata.get("lane_index") != index:
                    errors.append(f"{lane_name}: episode lane identity mismatch {key}")
                    continue
                if record.metadata.get("global_work_index", work_index) != work_index:
                    errors.append(f"{lane_name}: global work identity mismatch {key}")
                    continue
                record = replace(
                    record,
                    artifacts=tuple(_global_artifact(artifact, index) for artifact in record.artifacts),
                    metadata={**record.metadata, "lane_index": index, "global_work_index": work_index},
                )
                accepted_keys.add(key)
                records[lane_key] = record
            if accepted_keys != expected_keys:
                errors.append(f"{lane_name}: missing assigned episodes {sorted(expected_keys - accepted_keys)}")
            reduced = reduce_episode_records(
                lane_records, run_id=run_id, suite=plan.suite, planned_episodes=expected.planned_episodes
            )
            # Scope-level artifact failures can increase only the artifact counter.
            for key, value in reduced.counters.to_dict().items():
                if key == "artifact_failure_episodes":
                    continue
                if require_json_object(summary.get("counters"), "lane counters").get(key) != value:
                    raise ValueError(f"lane summary counter mismatch: {key}")
        except (OSError, ValueError, TypeError, KeyError, CanonicalReportError) as exc:
            errors.append(f"{lane_name}: {exc}")
    ordered = [
        records[(index, *key)] for index in range(lane_count) for key in plan.episode_work if (index, *key) in records
    ]
    if any(record.infrastructure_error or record.artifact_failures for record in ordered):
        errors.append("episode infrastructure/artifact failures")
    summary = reduce_episode_records(
        ordered,
        run_id=run_id,
        suite=plan.suite,
        planned_episodes=lane_count * plan.global_planned_episodes,
        planned_tasks={
            task_id: {
                "planned_episodes": lane_count * plan.episodes_per_task,
                "task_name": plan.tasks[task_id].name,
            }
            for task_id in plan.selected_task_ids
        },
        run_status="partial" if errors else "complete",
        metadata={"lane_count": lane_count, "lane_statuses": lane_statuses, "aggregation_errors": errors},
    )
    failures = tuple(
        {
            (item.metadata["lane_index"], item.ref, item.kind, item.scope, item.status, item.error): item
            for item in (*summary.artifact_failures, *scope_failures)
        }.values()
    )
    failed_refs = {(item.metadata["lane_index"], item.ref) for item in failures}
    affected = {
        (item.metadata.get("lane_index"), item.task_id, item.episode_index)
        for item in ordered
        if any((artifact.metadata["lane_index"], artifact.ref) in failed_refs for artifact in item.artifacts)
    }
    counters = replace(summary.counters, artifact_failure_episodes=len(affected))
    tasks = tuple(
        replace(
            task,
            artifact_failure_episodes=sum(task_id == task.task_id for _, task_id, _ in affected),
            status="partial" if any(task_id == task.task_id for _, task_id, _ in affected) else task.status,
        )
        for task in summary.tasks
    )
    return replace(summary, counters=counters, tasks=tasks, artifact_failures=failures), ordered


def aggregate_lanes(run_root: str | Path, *, run_id: str, plan: BenchmarkPlan, lane_count: int) -> CanonicalRunSummary:
    """Validate terminal lane facts and return a global summary without writing."""
    return _aggregate_lanes(run_root, run_id=run_id, plan=plan, lane_count=lane_count)[0]


def write_global_report(
    run_root: str | Path, *, run_id: str, plan: BenchmarkPlan, lane_count: int
) -> CanonicalRunSummary:
    """Write canonical fan-in once; never overwrite an earlier global run."""
    root = Path(run_root)
    summary, records = _aggregate_lanes(root, run_id=run_id, plan=plan, lane_count=lane_count)
    writer = CanonicalReportWriter(root)
    # start_run permits only a native directory; fan-in additionally owns lanes.
    writer.start_run({"run_id": run_id, "status": "running"}, allow_lanes=True)
    for record in records:
        writer.append_episode(record)
    writer.finalize(
        summary,
        manifest={
            "run_id": run_id,
            "status": summary.run_status,
            "lane_count": lane_count,
            "resolved_plan": loads_strict(benchmark_plan_to_json(plan)),
        },
    )
    return summary
