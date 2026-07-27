from __future__ import annotations

import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from inference_service.pi05_schedule import PI05DenoisingSchedule, load_pi05_schedule
from model_utils.pi05_export import tune_schedule


def _result(root, name: str, steps: int, value: float, *, status: int = 0):
    candidate = tune_schedule.Candidate(
        PI05DenoisingSchedule(
            name=name,
            timesteps=tuple(1.0 - index / steps for index in range(steps)) + (0.0,),
        ),
        {"method": "test"},
    )
    return tune_schedule.CandidateResult(
        candidate=candidate,
        metrics={"raw_l1": value},
        log_path=root / "logs" / f"{name}.log",
        metrics_path=root / "metrics" / f"{name}.json",
        status=status,
    )


def test_tuner_selects_best_candidate_and_retains_best_per_step(tmp_path):
    results = [
        _result(tmp_path, "three-a", 3, 0.20),
        _result(tmp_path, "three-b", 3, 0.10),
        _result(tmp_path, "four", 4, 0.05),
        _result(tmp_path, "failed", 2, 0.01, status=1),
    ]

    selected = tune_schedule._select_result(results, "raw_l1", None)
    threshold_selected = tune_schedule._select_result(results, "raw_l1", 0.15)
    per_step = tune_schedule._select_per_step_results(results, "raw_l1")

    assert selected.candidate.name == "four"
    assert threshold_selected.candidate.name == "three-b"
    assert [result.candidate.name for result in per_step] == ["three-b", "four"]


def test_tuner_report_is_structured_and_lists_best_per_step(tmp_path):
    paths = tune_schedule._artifact_paths(tmp_path)
    for directory in (paths.logs, paths.metrics):
        directory.mkdir(parents=True)
    results = [_result(tmp_path, "three", 3, 0.10), _result(tmp_path, "four", 4, 0.05)]
    args = Namespace(
        profile="board",
        policy_path=tmp_path / "bundle",
        deployment="ascend-velocity",
        metric="raw_l1",
        max_threshold=None,
    )
    resolved = SimpleNamespace(config_path="/tmp/loss.yaml")

    tune_schedule._write_report(
        paths,
        selected=results[1],
        per_step=results,
        results=results,
        args=args,
        resolved=resolved,
    )

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["format"] == "pi05-schedule-tuning-report-v1"
    assert report["selected"]["name"] == "four"
    assert [entry["step_count"] for entry in report["best_per_step"]] == [3, 4]
    assert "three" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_tuner_runs_loss_compare_with_explicit_transient_paths(monkeypatch, tmp_path):
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        metrics_path = command[command.index("--metrics-json") + 1]
        with open(metrics_path, "w", encoding="utf-8") as stream:
            json.dump({"aggregates": {"raw_l1": 0.125}}, stream)
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(tune_schedule.subprocess, "run", run)
    candidate = tune_schedule.Candidate(
        PI05DenoisingSchedule(name="candidate", timesteps=(1.0, 0.5, 0.0)),
        {"method": "test"},
    )
    schedule_path = tmp_path / "candidate.json"
    metrics_path = tmp_path / "metrics.json"
    curvature_path = tmp_path / "curvature.jsonl"

    result = tune_schedule._run_loss_compare(
        profile="board",
        config_path="/tmp/loss.yaml",
        policy_path=tmp_path / "bundle",
        deployment="ascend-velocity",
        candidate=candidate,
        schedule_path=schedule_path,
        metrics_path=metrics_path,
        log_path=tmp_path / "candidate.log",
        curvature_log_path=curvature_path,
    )

    assert result.metrics["raw_l1"] == 0.125
    assert "--schedule-override-path" in captured["command"]
    assert "--curvature-log-path" in captured["command"]
    assert "--generate-target" not in captured["command"]
    assert "env" not in captured["kwargs"]
    assert load_pi05_schedule(schedule_path) == candidate.schedule


def test_tuner_rejects_generate_target_profile():
    with pytest.raises(ValueError, match="compute-only"):
        tune_schedule._validate_inputs(
            Namespace(),
            SimpleNamespace(args=SimpleNamespace(generate_target=True)),
        )
