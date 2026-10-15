"""Unit tests for benchmark_robotwin.evaluation (plan request parsing)."""

from __future__ import annotations

import pytest
from benchmark_robotwin.evaluation import RobotwinEvaluationError, resolve_plan_request


def _base_options() -> dict:
    return {
        "task_name": "beat_block_hammer",
        "task_config": "demo_clean",
        "embodiment": "aloha-agilex",
        "env_cfg_type": "aloha_agilex",
        "evaluation": {
            "enabled": True,
            "suite": "robotwin",
            "tasks": [0],
            "episodes_per_task": 2,
            "max_steps": 400,
            "seed": 10000,
            "video": {"enabled": True},
            "output": {},
            "timeouts": {"startup_timeout_sec": 300.0, "max_duration_sec": 600.0},
        },
    }


def test_parse_valid_request() -> None:
    request = resolve_plan_request(_base_options())
    assert request is not None
    assert request.task_name == "beat_block_hammer"
    assert request.tasks == (0,)
    assert request.episodes_per_task == 2
    assert request.max_steps == 400
    assert request.save_sim_states is False


def test_disabled_evaluation_returns_none() -> None:
    options = _base_options()
    options["evaluation"]["enabled"] = False
    assert resolve_plan_request(options) is None


def test_missing_evaluation_returns_none() -> None:
    options = _base_options()
    del options["evaluation"]
    assert resolve_plan_request(options) is None


def test_save_sim_states_not_supported() -> None:
    options = _base_options()
    options["evaluation"]["save_sim_states"] = True
    with pytest.raises(RobotwinEvaluationError, match="save_sim_states"):
        resolve_plan_request(options)


def test_single_video_not_supported() -> None:
    # Per-episode finalize would overwrite video.mp4 each episode; Phase 1
    # fails closed instead of keeping only the last episode's recording.
    options = _base_options()
    options["evaluation"]["video"]["single_video"] = True
    with pytest.raises(RobotwinEvaluationError, match="single_video"):
        resolve_plan_request(options)


def test_save_sim_states_default_is_false() -> None:
    request = resolve_plan_request(_base_options())
    assert request is not None
    assert request.save_sim_states is False


@pytest.mark.parametrize(
    "field, value",
    [
        ("episodes_per_task", 0),
        ("max_steps", -1),
        ("seed", -5),
    ],
)
def test_non_positive_integers_rejected(field: str, value: int) -> None:
    options = _base_options()
    options["evaluation"][field] = value
    with pytest.raises(RobotwinEvaluationError):
        resolve_plan_request(options)


def test_duplicate_task_ids_rejected() -> None:
    options = _base_options()
    options["evaluation"]["tasks"] = [0, 0]
    with pytest.raises(RobotwinEvaluationError, match="duplicates"):
        resolve_plan_request(options)


def test_non_finite_timeout_rejected() -> None:
    options = _base_options()
    options["evaluation"]["timeouts"] = {
        "startup_timeout_sec": float("inf"),
        "max_duration_sec": 600.0,
    }
    with pytest.raises(RobotwinEvaluationError, match="finite"):
        resolve_plan_request(options)


def test_malformed_video_block_rejected() -> None:
    options = _base_options()
    options["evaluation"]["video"] = "enabled"
    with pytest.raises(RobotwinEvaluationError, match="video"):
        resolve_plan_request(options)
