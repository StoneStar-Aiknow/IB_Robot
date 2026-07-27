from __future__ import annotations

import json

import numpy as np
import pytest

from inference_service.pi05_schedule import PI05DenoisingSchedule, load_pi05_schedule
from model_utils.pi05_export.curvature_schedule import build_curvature_schedule, load_curvature_log, main


def _record(schedule: PI05DenoisingSchedule, scores: list[float]) -> str:
    return json.dumps({"schedule": schedule.to_dict(), "curvature_scores": scores})


def test_load_curvature_log_averages_records_with_one_strict_schedule(tmp_path):
    dense = PI05DenoisingSchedule(name="dense", timesteps=(1.0, 0.5, 0.0))
    log_path = tmp_path / "curvature.jsonl"
    log_path.write_text(_record(dense, [1.0, 3.0]) + "\n" + _record(dense, [3.0, 5.0]) + "\n", encoding="utf-8")

    loaded, scores = load_curvature_log(log_path)

    assert loaded == dense
    np.testing.assert_array_equal(scores, [2.0, 4.0])


def test_build_curvature_schedule_returns_strict_schedule():
    dense = PI05DenoisingSchedule(name="dense", timesteps=(1.0, 0.5, 0.0))

    schedule = build_curvature_schedule(dense, np.array([1.0, 3.0]), num_steps=2, base=0.0, name="tuned")

    assert schedule.name == "tuned"
    assert schedule.step_count == 2
    assert schedule.timesteps[0] == 1.0
    assert schedule.timesteps[-1] == 0.0
    assert schedule.timesteps[1] == pytest.approx(1.0 / 3.0)


def test_curvature_cli_writes_strict_schedule_files(tmp_path):
    dense = PI05DenoisingSchedule(name="dense", timesteps=(1.0, 0.5, 0.0))
    log_path = tmp_path / "curvature.jsonl"
    log_path.write_text(_record(dense, [1.0, 1.0]) + "\n", encoding="utf-8")
    output_dir = tmp_path / "schedules"

    assert main(["--log", str(log_path), "--num-steps", "2", "--output-dir", str(output_dir)]) == 0

    schedule = load_pi05_schedule(output_dir / "curvature_2.json")
    assert schedule.name == "curvature_2"
    assert schedule.timesteps == (1.0, 0.5, 0.0)


def test_curvature_log_rejects_legacy_timestep_record(tmp_path):
    log_path = tmp_path / "legacy.jsonl"
    log_path.write_text(json.dumps({"ts": [1.0, 0.0], "scores": [0.0]}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly"):
        load_curvature_log(log_path)
