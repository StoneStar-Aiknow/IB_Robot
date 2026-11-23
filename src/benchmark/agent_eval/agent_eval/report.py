"""Run report: what ran (identity and hashes), what happened per episode, the aggregate.

Success is the platform's (``native_result.native_success``) and nothing else: an Agent DONE
or a skill that SUCCEEDED never counts. An episode without a platform verdict (interrupted:
the platform call failed, the connection dropped, the client or the runner aborted it; or no
result arrived) is neither a success nor a failure: it is listed as an infrastructure error
and left out of the judged denominator.

Every episode is also written as a ``benchmark_runtime`` canonical record
(``canonical/{run_manifest.json,episodes.jsonl,summary.json}``), whose reducer gives the
counters and the two rates the report shows side by side, with no single headline rate:
``native_model_success_rate`` (successes / judged episodes) and ``planned_success_rate``
(successes / planned episodes). A run with any unjudged or missing episode is ``partial``.

Token usage is reported only when the Agent's events carry it; otherwise the field is
``null`` with the reason. Secrets are never read: the planner identity is provider, model and
the base URL's host only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from benchmark_runtime.canonical_models import ArtifactStatus, CanonicalEpisodeRecord, CanonicalRunSummary
from benchmark_runtime.canonical_report import CanonicalReportWriter, reduce_episode_records

REPORT_SCHEMA = "agent_eval.report/2"
NO_USAGE_REASON = "planner usage not exposed on /agent/event"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str | os.PathLike | None) -> str | None:
    if not path or not Path(path).is_file():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompt_hash(guide: str, task_spec_source: Mapping[str, Any]) -> str:
    """SHA-256 over the recipe guide and the canonical task spec: the prompt's fixed part."""
    spec = json.dumps(task_spec_source, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_text(guide + "\n" + spec)


def git_identity(repo_dir: str | os.PathLike) -> dict[str, Any]:
    """HEAD commit and whether the work tree has uncommitted changes (``None`` outside git)."""

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", str(repo_dir), *args], capture_output=True, text=True, timeout=20, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def planner_identity(robot: Mapping[str, Any]) -> dict[str, Any]:
    """Provider / model / base URL host of the robot config's Agent planner (never keys)."""
    planner = (((robot.get("embodied") or {}).get("agent") or {}).get("planner")) or {}
    base_url = planner.get("base_url")
    return {
        "mode": planner.get("mode"),
        "provider": planner.get("provider"),
        "model": planner.get("model"),
        "base_url_host": urlparse(base_url).hostname if isinstance(base_url, str) else None,
    }


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float] | None:
    """Wilson score interval (95 % by default); ``None`` when there are no trials."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def planner_usage(record: Mapping[str, Any]) -> tuple[dict[str, int] | None, str | None]:
    """Sum prompt/completion tokens if any Agent event detail carries them."""
    totals = {"prompt_tokens": 0, "completion_tokens": 0}
    seen = False
    for decision in record.get("decisions", []):
        for event in [decision.get("terminal") or {}, *decision.get("events", [])]:
            detail = event.get("detail") if isinstance(event, Mapping) else None
            if not isinstance(detail, Mapping):
                continue
            usage = detail.get("usage") if isinstance(detail.get("usage"), Mapping) else detail
            for key in totals:
                value = usage.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    totals[key] += value
                    seen = True
    return (totals, None) if seen else (None, NO_USAGE_REASON)


def _skill_outcomes(record: Mapping[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for decision in record.get("decisions", []):
        terminal = decision.get("terminal") or {}
        counts[str(terminal.get("state") or terminal.get("event_type") or "UNKNOWN")] += 1
    return dict(sorted(counts.items()))


def _verdict(native: Mapping[str, Any], end: Mapping[str, Any], finish_reason: Any) -> tuple[bool | None, str | None]:
    """(native success or None, infrastructure error or None): a verdict only from an uninterrupted result."""
    success = native.get("native_success")
    if native.get("interrupted") or end.get("interrupted"):
        reason = end.get("reason") or native.get("failure_reason") or "interrupted"
        return None, f"interrupted: {reason}"
    if not native:
        return None, f"no native result received (finish: {finish_reason})"
    if not isinstance(success, bool):
        return None, "the platform gave no verdict"
    return success, None


def episode_entry(
    record: Mapping[str, Any], *, layout: str | None = None, seed: int | None = None, video_dir: str | None = None
) -> dict[str, Any]:
    """The report line of one episode record (written by the runner)."""
    native = record.get("native_result") or {}
    end = record.get("episode_end") or {}
    tokens, tokens_reason = planner_usage(record)
    native_success, infrastructure_error = _verdict(native, end, record.get("finish_reason"))
    return {
        "session_id": record.get("session_id"),
        "generation": record.get("generation"),
        "layout": layout,
        # The platform's own seed wins: a client "seed" may only be the base of a pass. An
        # episode the runner aborted has none (the result carries no platform extras).
        "seed": native.get("seed") if native.get("seed") is not None else seed,
        "client_seed": seed,
        "native_result_available": native_success is not None,
        "native_success": native_success,
        "infrastructure_error": infrastructure_error,
        "native_score": native.get("native_score"),
        "failure_reason": native.get("failure_reason") if native else "no native result received",
        "terminated": bool(native.get("terminated")),
        "truncated": bool(native.get("truncated")),
        "take_action_cnt": native.get("take_action_cnt"),
        "sim_time_s": end.get("sim_time_s"),
        "wall_s": record.get("wall_s"),
        "decisions": len(record.get("decisions", [])),
        "skill_outcomes": _skill_outcomes(record),
        "finish_reason": record.get("finish_reason"),
        "agent_session_id": record.get("agent_session_id"),
        "agent_request_ids": list(record.get("agent_request_ids", [])),
        "request_text_sha256": [d.get("request_text_sha256") for d in record.get("decisions", [])],
        "platform_ticks": end.get("platform_ticks"),
        "interrupted": bool(native.get("interrupted") or end.get("interrupted")),
        "artifacts": {
            "record": record.get("record_path"),
            "platform": native.get("artifact"),
            "video_dir": video_dir,
        },
        "tokens": tokens,
        "tokens_reason": tokens_reason,
        "cost": None,
        "cost_reason": "no price table configured",
    }


def _artifact(ref: Any, kind: str, *, required: bool) -> ArtifactStatus | None:
    if not ref:
        return None
    exists = Path(str(ref)).exists()
    return ArtifactStatus(
        ref=str(ref),
        kind=kind,
        scope="episode",
        required=required,
        status="available" if exists else "missing",
        error=None if exists else "not found when the report was written",
    )


def canonical_record(
    entry: Mapping[str, Any], *, run_id: str, suite: str, task_id: int, task_name: str, episode_index: int
) -> CanonicalEpisodeRecord:
    """The ``benchmark_runtime`` canonical record of one report entry (task = one suite layout entry)."""
    available = bool(entry.get("native_result_available"))
    steps = next(
        (v for v in (entry.get("take_action_cnt"), entry.get("platform_ticks")) if isinstance(v, int) and v >= 0), 0
    )
    seed = entry.get("seed") if isinstance(entry.get("seed"), int) else entry.get("client_seed")
    artifacts = [
        _artifact(entry["artifacts"].get("record"), "agent_eval_episode_record", required=True),
        _artifact(entry["artifacts"].get("platform"), "platform_artifact", required=False),
        _artifact(entry["artifacts"].get("video_dir"), "platform_video_dir", required=False),
    ]
    return CanonicalEpisodeRecord(
        run_id=run_id,
        suite=suite,
        task_id=int(task_id),
        task_name=str(task_name),
        episode_index=int(episode_index),
        seed=int(seed),
        native_result_available=available,
        is_success=entry.get("native_success") if available else None,
        executed_steps=int(steps),
        duration_s=float(entry.get("wall_s") or 0.0),
        terminated=bool(entry.get("terminated")),
        truncated=bool(entry.get("truncated")),
        termination_reason=str(entry.get("failure_reason") or "") if available else "interrupted",
        infrastructure_error=entry.get("infrastructure_error"),
        artifacts=tuple(a for a in artifacts if a is not None),
        metadata={
            "layout": entry.get("layout"),
            "pass": entry.get("pass"),
            "session_id": entry.get("session_id"),
            "generation": entry.get("generation"),
            "client_seed": entry.get("client_seed"),
            "seed_source": "platform" if entry.get("seed") != entry.get("client_seed") else "client_pass",
            "finish_reason": entry.get("finish_reason"),
            "decisions": entry.get("decisions"),
            "native_score": entry.get("native_score"),
            "agent_session_id": entry.get("agent_session_id"),
        },
    )


class CanonicalRun:
    """The run's canonical files: records appended (fsynced) as episodes end, the summary at the end."""

    def __init__(
        self, run_root: str | os.PathLike, *, run_id: str, suite: str, planned: Mapping[int, Mapping[str, Any]]
    ):
        self.run_id, self.suite = run_id, suite
        self.planned = {int(k): dict(v) for k, v in planned.items()}
        self.writer = CanonicalReportWriter(run_root)
        self.manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "running",
            "suite": suite,
            "producer": "agent_eval",
            "planned_episodes": self.planned_episodes,
            "tasks": {str(k): v for k, v in self.planned.items()},
        }
        self.writer.start_run(self.manifest)
        self._next_index: dict[int, int] = {}

    @property
    def planned_episodes(self) -> int:
        return sum(int(v["planned_episodes"]) for v in self.planned.values())

    def append(self, entry: Mapping[str, Any], task_id: int) -> CanonicalEpisodeRecord:
        index = self._next_index.get(task_id, 0)
        self._next_index[task_id] = index + 1
        record = canonical_record(
            entry,
            run_id=self.run_id,
            suite=self.suite,
            task_id=task_id,
            task_name=str(self.planned.get(task_id, {}).get("task_name", entry.get("layout"))),
            episode_index=index,
        )
        self.writer.append_episode(record)
        return record

    def finalize(self, *, stopped: bool) -> CanonicalRunSummary:
        """Reduce the records; a run that stopped early is never ``complete``."""
        records = self.writer.records
        summary = reduce_episode_records(
            records,
            run_id=self.run_id,
            suite=self.suite,
            planned_episodes=self.planned_episodes,
            planned_tasks=self.planned,
            run_status=("partial" if records else "failed") if stopped else None,
        )
        self.writer.finalize(summary, manifest={**self.manifest, "status": summary.run_status})
        return summary


def aggregate(
    episodes: Sequence[Mapping[str, Any]],
    summary: CanonicalRunSummary,
) -> dict:
    """Counters and both rates from the canonical summary; no single headline rate."""
    counters = summary.counters.to_dict()
    judged = counters["completed_native_episodes"]
    successes = counters["successful_native_episodes"]
    interval = wilson_interval(successes, judged)
    return {
        "run_status": summary.run_status,
        # Interrupted or unjudged episodes, or fewer episodes than planned: not a complete run.
        "complete": summary.run_status == "complete",
        "counters": counters,
        "rates": {
            "native_model_success_rate": counters["native_model_success_rate"],
            "planned_success_rate": counters["planned_success_rate"],
        },
        "interrupted": sum(1 for ep in episodes if ep.get("interrupted")),
        "unjudged": counters["started_episodes"] - judged,
        "rate_definitions": {
            "native_model_success_rate": "native successes / episodes with a platform verdict (judged)",
            "planned_success_rate": "native successes / planned episodes",
            "unjudged": "episodes without a platform verdict (interrupted, aborted, no result): "
            "neither success nor failure, counted as infrastructure errors",
            "wilson_95": "over judged episodes",
        },
        "wilson_95": None
        if interval is None
        else {
            "low": round(interval[0], 4),
            "high": round(interval[1], 4),
            "note": "coarse alarm only: overlapping intervals do not show equivalence",
        },
        "finish_reasons": dict(sorted(Counter(str(ep.get("finish_reason")) for ep in episodes).items())),
    }


def build_report(
    *,
    run_id: str,
    repo_dir: str | os.PathLike,
    body,
    robot: Mapping[str, Any],
    task_spec,
    recipe,
    layouts: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    summary: CanonicalRunSummary,
    suite_path: str | None = None,
    platform_identity: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "run_id": run_id,
        "git": git_identity(repo_dir),
        "robot_config": body.robot_config,
        "runtime_profile": {"provider": body.runtime_provider, "profile": body.runtime_profile},
        "planner": planner_identity(robot),
        "task": task_spec.task,
        "recipe": recipe.name,
        # Scope: the fixed recipe guide + task card only. The planner's system prompt, skill
        # catalog and per-step text are not in it; per-request text hashes are in the episodes.
        "recipe_task_hash": prompt_hash(recipe.guide, task_spec.source),
        "full_prompt_hash": None,
        "full_prompt_hash_reason": "the planner assembles system prompt + catalog + images inside the Agent; "
        "only per-request text hashes (episodes[].request_text_sha256) are recorded",
        "platform_identity": dict(platform_identity or {}),
        "config_hashes": {
            "suite": file_sha256(suite_path),
            "task_spec": task_spec.sha256,
            "recipe": recipe.sha256,
            "robot_config": file_sha256(body.robot_config_path),
            "robot_config_sources": {str(p): file_sha256(p) for p in body.robot_config_sources},
            "runtime_profile": file_sha256(body.runtime_profile_path),
        },
        "layouts": [dict(layout) for layout in layouts],
        "episodes": list(episodes),
        "aggregate": aggregate(episodes, summary),
        **dict(extra or {}),
    }


def write_report(report: Mapping[str, Any], out_dir: str | os.PathLike) -> Path:
    """``report.json`` (the canonical records are written as episodes end, see :class:`CanonicalRun`)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return report_path
