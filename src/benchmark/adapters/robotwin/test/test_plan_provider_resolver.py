"""Unit tests for the fixture-driven plan provider and resolver."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmark_robotwin.evaluation import RobotwinPlanRequest
from benchmark_robotwin.plan_provider import RobotwinPlanProvider, RobotwinProviderError
from benchmark_robotwin.plan_resolver import RobotwinPlanResolver, RobotwinPlanResolverError

_INSTRUCTION = "there is a hammer and a block on the table"


@pytest.fixture()
def robotwin_root(tmp_path: Path) -> Path:
    root = tmp_path / "robotwin"
    (root / "env_cfg" / "eval").mkdir(parents=True)
    (root / "env_cfg" / "eval" / "all_tasks.yml").write_text("tasks:\n  - beat_block_hammer\n  - click_bell\n")
    (root / "env_cfg" / "task_config").mkdir(parents=True)
    (root / "env_cfg" / "task_config" / "_eval_step_limit.yml").write_text("beat_block_hammer: 400\nclick_bell: 300\n")
    description = root / "description" / "task_instruction"
    description.mkdir(parents=True)
    (description / "beat_block_hammer.json").write_text(json.dumps({"full_description": _INSTRUCTION}))
    return root


def _request(task_name: str = "beat_block_hammer") -> RobotwinPlanRequest:
    return RobotwinPlanRequest(
        suite="robotwin",
        task_name=task_name,
        task_config="demo_clean",
        embodiment="aloha-agilex",
        env_cfg_type="aloha_agilex",
        task_order_index=0,
        tasks=(0,),
        episodes_per_task=2,
        max_steps=1000,
        seed=10000,
        save_sim_states=False,
        video_enabled=True,
        write_native=True,
        write_canonical=True,
        startup_timeout_sec=300.0,
        max_duration_sec=600.0,
    )


def test_resolve_single_task_reads_manifest_and_step_limit(robotwin_root: Path) -> None:
    task = RobotwinPlanProvider(robotwin_root).resolve_single_task("beat_block_hammer")
    assert task.task_name == "beat_block_hammer"
    assert task.instruction == _INSTRUCTION
    assert task.step_limit == 400


def test_instruction_falls_back_to_task_name(robotwin_root: Path) -> None:
    task = RobotwinPlanProvider(robotwin_root).resolve_single_task("click_bell")
    assert task.instruction == "click bell"
    assert task.step_limit == 300


def test_unknown_task_rejected(robotwin_root: Path) -> None:
    with pytest.raises(RobotwinProviderError, match="not in canonical"):
        RobotwinPlanProvider(robotwin_root).resolve_single_task("no_such_task")


def test_missing_manifest_rejected(tmp_path: Path) -> None:
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(RobotwinProviderError, match="task manifest not found"):
        RobotwinPlanProvider(empty_root).resolve_single_task("beat_block_hammer")


def test_resolver_builds_plan(robotwin_root: Path) -> None:
    plan = RobotwinPlanResolver(RobotwinPlanProvider(robotwin_root)).resolve(_request())
    assert plan.benchmark_type == "robotwin"
    assert plan.suite == "robotwin"
    assert [t.name for t in plan.tasks] == ["beat_block_hammer"]
    assert plan.tasks[0].prompt == _INSTRUCTION
    assert plan.max_steps == 400  # min(request.max_steps=1000, step_limit=400)
    assert plan.planned_episodes == 2


def test_resolver_rejects_nonzero_task_id(robotwin_root: Path) -> None:
    request = _request()
    object.__setattr__(request, "tasks", (1,))
    with pytest.raises(RobotwinPlanResolverError, match="task_id 0"):
        RobotwinPlanResolver(RobotwinPlanProvider(robotwin_root)).resolve(request)
