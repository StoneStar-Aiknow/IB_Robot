# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
# Licensed under the Mulan PSL v2.
"""Tune and install a strict schedule for one manifest-driven PI0.5 deployment."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from inference_manifest import CompiledDeployment, load_inference_manifest
from inference_manifest.json_utils import load_json_strict
from inference_service.pi05_schedule import PI05DenoisingSchedule, uniform_pi05_schedule, write_pi05_schedule
from model_utils import loss_compare_cli
from model_utils.pi05_export.convert_om import replace_pi05_ascend_schedule
from model_utils.pi05_export.curvature_schedule import build_curvature_schedule, load_curvature_log

DEFAULT_DENSE_STEPS = 20
DEFAULT_CANDIDATE_STEPS = (3, 4, 5)
DEFAULT_ETA = (0.5, 1.0)
DEFAULT_BASE = (1e-3,)
LOWER_IS_BETTER = frozenset({"raw_l1", "unnorm_l1", "inference_time", "average_latency_ms", "normalized_mean_w1_std"})
HIGHER_IS_BETTER = frozenset({"raw_cos", "unnorm_cos", "normalized_first_frame_cos"})


@dataclass(frozen=True)
class Candidate:
    schedule: PI05DenoisingSchedule
    source: dict[str, object]

    @property
    def name(self) -> str:
        return self.schedule.name

    @property
    def num_steps(self) -> int:
        return self.schedule.step_count


@dataclass(frozen=True)
class CandidateResult:
    candidate: Candidate
    metrics: dict[str, float | None]
    log_path: Path
    metrics_path: Path
    status: int


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    candidates: Path
    best: Path
    logs: Path
    metrics: Path
    curvature: Path


def _eta_name(value: float) -> str:
    text = f"{value:g}".replace(".", "p").replace("-", "m")
    return f"eta{text}"


def _base_name(value: float) -> str:
    text = f"{value:g}".replace(".", "p").replace("-", "m")
    return f"base{text}"


def _metric_value(result: CandidateResult, metric: str) -> float | None:
    value = result.metrics.get(metric)
    if type(value) not in (int, float) or not math.isfinite(value):
        return None
    return float(value)


def _candidate_key(result: CandidateResult, metric: str) -> tuple[float, int, str]:
    value = _metric_value(result, metric)
    if value is None:
        raise ValueError(f"candidate {result.candidate.name!r} has no finite metric {metric!r}")
    score = value if metric in LOWER_IS_BETTER else -value
    return score, result.candidate.num_steps, result.candidate.name


def _successful_results(results: list[CandidateResult], metric: str) -> list[CandidateResult]:
    return [result for result in results if result.status == 0 and _metric_value(result, metric) is not None]


def _select_result(
    results: list[CandidateResult],
    metric: str,
    max_threshold: float | None,
) -> CandidateResult:
    successful = _successful_results(results, metric)
    if not successful:
        raise RuntimeError(f"No successful candidate produced metric {metric!r}")
    if max_threshold is None:
        return min(successful, key=lambda result: _candidate_key(result, metric))

    if metric in LOWER_IS_BETTER:
        eligible = [result for result in successful if float(_metric_value(result, metric)) <= max_threshold]
    else:
        eligible = [result for result in successful if float(_metric_value(result, metric)) >= max_threshold]
    if not eligible:
        comparator = "at most" if metric in LOWER_IS_BETTER else "at least"
        raise RuntimeError(f"No candidate has {metric} {comparator} {max_threshold}")
    fewest_steps = min(result.candidate.num_steps for result in eligible)
    return min(
        (result for result in eligible if result.candidate.num_steps == fewest_steps),
        key=lambda result: _candidate_key(result, metric),
    )


def _select_per_step_results(results: list[CandidateResult], metric: str) -> list[CandidateResult]:
    successful = _successful_results(results, metric)
    return [
        min(
            (result for result in successful if result.candidate.num_steps == steps),
            key=lambda result: _candidate_key(result, metric),
        )
        for steps in sorted({result.candidate.num_steps for result in successful})
    ]


def _read_metrics(path: Path) -> dict[str, float | None]:
    value = load_json_strict(path)
    if not isinstance(value, dict):
        raise ValueError(f"loss_compare metrics must be an object: {path}")
    aggregates = value.get("aggregates", value)
    if not isinstance(aggregates, dict):
        raise ValueError(f"loss_compare metrics aggregates must be an object: {path}")
    metrics = {}
    for key, metric in aggregates.items():
        if metric is None:
            metrics[str(key)] = None
        elif type(metric) in (int, float) and math.isfinite(metric):
            metrics[str(key)] = float(metric)
    return metrics


def _run_loss_compare(
    *,
    profile: str,
    config_path: str,
    policy_path: Path,
    deployment: str,
    candidate: Candidate,
    schedule_path: Path,
    metrics_path: Path,
    log_path: Path,
    curvature_log_path: Path | None = None,
) -> CandidateResult:
    write_pi05_schedule(candidate.schedule, schedule_path)
    metrics_path.unlink(missing_ok=True)
    if curvature_log_path is not None:
        curvature_log_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        "-m",
        "model_utils.loss_compare",
        "--config",
        config_path,
        "--profile",
        profile,
        "--policy_path",
        str(policy_path),
        "--deployment",
        deployment,
        "--schedule-override-path",
        str(schedule_path),
        "--metrics-json",
        str(metrics_path),
    ]
    if curvature_log_path is not None:
        command.extend(["--curvature-log-path", str(curvature_log_path)])
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    log_path.write_text(process.stdout + process.stderr, encoding="utf-8")
    metrics = _read_metrics(metrics_path) if process.returncode == 0 and metrics_path.is_file() else {}
    return CandidateResult(
        candidate=candidate,
        metrics=metrics,
        log_path=log_path,
        metrics_path=metrics_path,
        status=process.returncode,
    )


def _build_candidates(
    dense_log: Path,
    *,
    candidate_steps: list[int],
    etas: list[float],
    bases: list[float],
) -> list[Candidate]:
    dense_schedule, scores = load_curvature_log(dense_log)
    candidates = []
    for steps in sorted(set(candidate_steps)):
        candidates.append(
            Candidate(
                schedule=uniform_pi05_schedule(steps, name=f"uniform_{steps}"),
                source={"method": "uniform"},
            )
        )
        for eta in etas:
            for base in bases:
                name = f"curvature_{_eta_name(eta)}_{_base_name(base)}_{steps}"
                candidates.append(
                    Candidate(
                        schedule=build_curvature_schedule(
                            dense_schedule,
                            scores,
                            num_steps=steps,
                            base=base,
                            eta=eta,
                            name=name,
                        ),
                        source={
                            "method": "curvature",
                            "dense_schedule": dense_schedule.name,
                            "eta": eta,
                            "base": base,
                        },
                    )
                )
    return candidates


def _candidate_report(result: CandidateResult, artifacts: ArtifactPaths) -> dict[str, object]:
    return {
        "name": result.candidate.name,
        "step_count": result.candidate.num_steps,
        "timesteps": list(result.candidate.schedule.timesteps),
        "source": result.candidate.source,
        "metrics": result.metrics,
        "status": result.status,
        "log": result.log_path.relative_to(artifacts.root).as_posix(),
        "metrics_json": result.metrics_path.relative_to(artifacts.root).as_posix(),
    }


def _write_report(
    paths: ArtifactPaths,
    *,
    selected: CandidateResult,
    per_step: list[CandidateResult],
    results: list[CandidateResult],
    args: argparse.Namespace,
    resolved: loss_compare_cli.ResolvedConfig,
) -> None:
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    report = {
        "format": "pi05-schedule-tuning-report-v1",
        "generated_at": generated_at,
        "profile": args.profile,
        "config": resolved.config_path,
        "policy_path": str(args.policy_path),
        "deployment": args.deployment,
        "selection": {"metric": args.metric, "max_threshold": args.max_threshold},
        "selected": _candidate_report(selected, paths),
        "best_per_step": [_candidate_report(result, paths) for result in per_step],
        "candidates": [_candidate_report(result, paths) for result in results],
    }
    (paths.root / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# PI0.5 Schedule Tuning Report",
        "",
        f"- Generated: `{generated_at}`",
        f"- Profile: `{args.profile}`",
        f"- Deployment: `{args.deployment}`",
        f"- Metric: `{args.metric}`",
        f"- Selected: `{selected.candidate.name}` ({selected.candidate.num_steps} steps)",
        "",
        "| Candidate | Steps | Metric | Status |",
        "|---|---:|---:|---:|",
    ]
    for result in sorted(results, key=lambda item: (item.candidate.num_steps, item.candidate.name)):
        value = _metric_value(result, args.metric)
        rendered = "NA" if value is None else f"{value:.8g}"
        lines.append(f"| `{result.candidate.name}` | {result.candidate.num_steps} | {rendered} | {result.status} |")
    (paths.root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_inputs(args: argparse.Namespace, resolved: loss_compare_cli.ResolvedConfig) -> Path:
    if resolved.args.generate_target:
        raise ValueError("The selected loss_compare profile is in generate-target mode; tuning is compute-only")
    for name in ("batch_path", "target_path", "raw_target_path"):
        value = getattr(resolved.args, name)
        if not value or not Path(value).expanduser().is_file():
            raise ValueError(f"loss_compare profile requires an existing {name}: {value!r}")
    noise_dir = resolved.args.noise_dir
    if not noise_dir or not Path(noise_dir).expanduser().is_dir():
        raise ValueError(f"loss_compare profile requires an existing noise_dir: {noise_dir!r}")

    policy_path = args.policy_path.expanduser().resolve(strict=True)
    selected = load_inference_manifest(policy_path, args.deployment)
    deployment = selected.deployment
    if selected.policy.policy_type != "pi05":
        raise ValueError("schedule tuning requires a PI0.5 policy bundle")
    if not isinstance(deployment, CompiledDeployment) or deployment.backend != "ascend":
        raise ValueError("schedule tuning requires a compiled Ascend deployment")
    if deployment.execution != ("vlm", "action_expert"):
        raise ValueError("schedule tuning requires PI0.5 vlm/action_expert execution")
    outputs = [binding for binding in deployment.bindings["action_expert"].outputs if binding.semantic == "action"]
    runtime_name = outputs[0].runtime_name if len(outputs) == 1 else None
    output_name = next(
        (part for part in reversed((runtime_name or "").split(":")) if part in {"action", "velocity", "v_t"}),
        runtime_name,
    )
    if output_name not in {"velocity", "v_t"}:
        raise ValueError("schedule tuning requires a velocity/v_t Action Expert output")
    return policy_path


def _artifact_paths(root: Path) -> ArtifactPaths:
    return ArtifactPaths(
        root=root,
        candidates=root / "schedules" / "candidates",
        best=root / "schedules" / "best",
        logs=root / "logs",
        metrics=root / "metrics",
        curvature=root / "curvature",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="Existing loss_compare profile")
    parser.add_argument("--config", default=None, help="Optional loss_compare YAML config")
    parser.add_argument("--policy-path", required=True, type=Path, help="PI0.5 policy bundle")
    parser.add_argument("--deployment", required=True, help="Named compiled Ascend velocity deployment")
    parser.add_argument("--dense-steps", type=int, default=DEFAULT_DENSE_STEPS)
    parser.add_argument("--candidate-steps", nargs="+", type=int, default=list(DEFAULT_CANDIDATE_STEPS))
    parser.add_argument("--eta", nargs="+", type=float, default=list(DEFAULT_ETA))
    parser.add_argument("--base", nargs="+", type=float, default=list(DEFAULT_BASE))
    parser.add_argument("--metric", choices=sorted(LOWER_IS_BETTER | HIGHER_IS_BETTER), default="raw_l1")
    parser.add_argument(
        "--max-threshold",
        type=float,
        default=None,
        help="Select the fewest steps meeting this maximum (lower metrics) or minimum (higher metrics)",
    )
    parser.add_argument("--artifacts-dir", required=True, type=Path, help="Output directory for schedules/report/logs")
    parser.add_argument("--force", action="store_true", help="Replace existing tuner outputs")
    parser.add_argument("--no-install", action="store_true", help="Do not update the deployment manifest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dense_steps < 2:
        raise SystemExit("--dense-steps must be at least 2")
    if any(steps < 1 for steps in args.candidate_steps):
        raise SystemExit("--candidate-steps must contain only positive integers")
    if any(not math.isfinite(value) or value <= 0.0 for value in args.eta):
        raise SystemExit("--eta must contain only finite positive values")
    if any(not math.isfinite(value) or value < 0.0 for value in args.base):
        raise SystemExit("--base must contain only finite non-negative values")
    if args.max_threshold is not None and not math.isfinite(args.max_threshold):
        raise SystemExit("--max-threshold must be finite")

    profile_args = ["--profile", args.profile]
    if args.config:
        profile_args.extend(["--config", args.config])
    resolved = loss_compare_cli.resolve(profile_args)
    try:
        policy_path = _validate_inputs(args, resolved)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    artifacts_root = args.artifacts_dir.expanduser().resolve()
    paths = _artifact_paths(artifacts_root)
    existing_outputs = [
        artifacts_root / "report.json",
        artifacts_root / "report.md",
        paths.root / "schedules" / "selected.json",
    ]
    if not args.force and any(path.exists() for path in existing_outputs):
        raise SystemExit(f"Refusing to replace existing tuner outputs under {artifacts_root}; pass --force")
    for directory in (paths.candidates, paths.best, paths.logs, paths.metrics, paths.curvature):
        directory.mkdir(parents=True, exist_ok=True)

    dense_candidate = Candidate(
        schedule=uniform_pi05_schedule(args.dense_steps, name=f"dense_uniform_{args.dense_steps}"),
        source={"method": "dense_uniform"},
    )
    dense_result = _run_loss_compare(
        profile=args.profile,
        config_path=resolved.config_path,
        policy_path=policy_path,
        deployment=args.deployment,
        candidate=dense_candidate,
        schedule_path=paths.candidates / f"{dense_candidate.name}.json",
        metrics_path=paths.metrics / f"{dense_candidate.name}.json",
        log_path=paths.logs / f"{dense_candidate.name}.log",
        curvature_log_path=paths.curvature / "dense.jsonl",
    )
    if dense_result.status != 0:
        raise SystemExit(f"Dense curvature run failed; see {dense_result.log_path}")

    candidates = _build_candidates(
        paths.curvature / "dense.jsonl",
        candidate_steps=args.candidate_steps,
        etas=args.eta,
        bases=args.base,
    )
    results = []
    for candidate in candidates:
        result = _run_loss_compare(
            profile=args.profile,
            config_path=resolved.config_path,
            policy_path=policy_path,
            deployment=args.deployment,
            candidate=candidate,
            schedule_path=paths.candidates / f"{candidate.name}.json",
            metrics_path=paths.metrics / f"{candidate.name}.json",
            log_path=paths.logs / f"{candidate.name}.log",
        )
        results.append(result)
        value = _metric_value(result, args.metric)
        print(f"[pi05-tune-schedule] {candidate.name}: status={result.status} {args.metric}={value}")

    selected = _select_result(results, args.metric, args.max_threshold)
    per_step = _select_per_step_results(results, args.metric)
    selected_path = paths.root / "schedules" / "selected.json"
    write_pi05_schedule(selected.candidate.schedule, selected_path)
    for result in per_step:
        write_pi05_schedule(result.candidate.schedule, paths.best / f"steps_{result.candidate.num_steps}.json")
    _write_report(paths, selected=selected, per_step=per_step, results=results, args=args, resolved=resolved)
    if not args.no_install:
        replace_pi05_ascend_schedule(policy_path, args.deployment, selected_path)
    print(f"[pi05-tune-schedule] selected {selected.candidate.name}; artifacts={artifacts_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
