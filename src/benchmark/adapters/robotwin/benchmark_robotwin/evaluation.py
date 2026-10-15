"""RoboTwin evaluation request parser: YAML ``benchmark.evaluation`` → frozen plan request.

Mirrors ``benchmark_libero.evaluation`` but with RoboTwin-specific fields.
The evaluation block is optional; when ``enabled`` is falsy or absent the
adapter falls back to a non-evaluation single-task mode.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class RobotwinEvaluationError(ValueError):
    """Raised when the evaluation block is malformed."""


@dataclass(frozen=True, slots=True)
class RobotwinPlanRequest:
    """Validated plan request parsed from ``benchmark.evaluation``."""

    suite: str
    task_name: str
    task_config: str
    embodiment: str
    env_cfg_type: str
    task_order_index: int
    tasks: tuple[int, ...] | None
    episodes_per_task: int
    max_steps: int
    seed: int
    save_sim_states: bool
    video_enabled: bool
    write_native: bool
    write_canonical: bool
    startup_timeout_sec: float
    max_duration_sec: float
    lane_count: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.suite, str) or not self.suite.strip():
            raise RobotwinEvaluationError("suite must be a non-empty string")
        if not isinstance(self.task_name, str) or not self.task_name.strip():
            raise RobotwinEvaluationError("task_name must be a non-empty string")
        if not isinstance(self.task_config, str) or not self.task_config.strip():
            raise RobotwinEvaluationError("task_config must be a non-empty string")
        if not isinstance(self.embodiment, str) or not self.embodiment.strip():
            raise RobotwinEvaluationError("embodiment must be a non-empty string")
        if not isinstance(self.env_cfg_type, str) or not self.env_cfg_type.strip():
            raise RobotwinEvaluationError("env_cfg_type must be a non-empty string")
        if (
            not isinstance(self.task_order_index, int)
            or isinstance(self.task_order_index, bool)
            or self.task_order_index < 0
        ):
            raise RobotwinEvaluationError("task_order_index must be a non-negative int")
        if self.tasks is not None:
            if not isinstance(self.tasks, tuple):
                raise RobotwinEvaluationError("tasks must be a tuple or null")
            seen: set[int] = set()
            for idx, tid in enumerate(self.tasks):
                if not isinstance(tid, int) or isinstance(tid, bool) or tid < 0:
                    raise RobotwinEvaluationError(f"tasks[{idx}] must be a non-negative int")
                if tid in seen:
                    raise RobotwinEvaluationError(f"tasks[{idx}] duplicates task id {tid}")
                seen.add(tid)
        if (
            not isinstance(self.episodes_per_task, int)
            or isinstance(self.episodes_per_task, bool)
            or self.episodes_per_task <= 0
        ):
            raise RobotwinEvaluationError("episodes_per_task must be a positive int")
        if not isinstance(self.max_steps, int) or isinstance(self.max_steps, bool) or self.max_steps <= 0:
            raise RobotwinEvaluationError("max_steps must be a positive int")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise RobotwinEvaluationError("seed must be a non-negative int")
        if not isinstance(self.save_sim_states, bool):
            raise RobotwinEvaluationError("save_sim_states must be a bool")
        if not isinstance(self.video_enabled, bool):
            raise RobotwinEvaluationError("video_enabled must be a bool")
        if not isinstance(self.write_native, bool):
            raise RobotwinEvaluationError("write_native must be a bool")
        if not isinstance(self.write_canonical, bool):
            raise RobotwinEvaluationError("write_canonical must be a bool")
        if not isinstance(self.startup_timeout_sec, int | float) or isinstance(self.startup_timeout_sec, bool):
            raise RobotwinEvaluationError("startup_timeout_sec must be a number")
        if not isinstance(self.max_duration_sec, int | float) or isinstance(self.max_duration_sec, bool):
            raise RobotwinEvaluationError("max_duration_sec must be a number")
        if float(self.startup_timeout_sec) <= 0 or float(self.max_duration_sec) <= 0:
            raise RobotwinEvaluationError("timeouts must be positive")
        import math

        if not math.isfinite(float(self.startup_timeout_sec)) or not math.isfinite(float(self.max_duration_sec)):
            raise RobotwinEvaluationError("timeouts must be finite")
        if not isinstance(self.lane_count, int) or isinstance(self.lane_count, bool) or self.lane_count < 1:
            raise RobotwinEvaluationError("lane_count must be a positive int")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RobotwinEvaluationError(f"{name} must be a Mapping, got {type(value).__name__}")
    return value


def resolve_plan_request(options: Mapping[str, Any]) -> RobotwinPlanRequest | None:
    """Parse the ``benchmark.evaluation`` block from the adapter options.

    Returns ``None`` when evaluation is disabled (``enabled`` falsy or absent).
    """
    evaluation = options.get("evaluation")
    if evaluation is None:
        return None
    evaluation = _require_mapping(evaluation, "benchmark.evaluation")
    if not bool(evaluation.get("enabled", False)):
        return None

    task_name = str(options.get("task_name", "")).strip()
    task_config = str(options.get("task_config", "demo_clean")).strip()
    embodiment = str(options.get("embodiment", "aloha-agilex")).strip()
    env_cfg_type = str(options.get("env_cfg_type", "aloha_agilex")).strip()

    suite = str(evaluation.get("suite", "robotwin")).strip()

    raw_tasks = evaluation.get("tasks")
    if raw_tasks is None:
        tasks: tuple[int, ...] | None = None
    elif isinstance(raw_tasks, list):
        tasks = tuple(int(t) for t in raw_tasks)
    else:
        raise RobotwinEvaluationError("tasks must be a list of ints or null")

    episodes_per_task = int(evaluation.get("episodes_per_task", 2))
    max_steps = int(evaluation.get("max_steps", 1000))
    seed = int(evaluation.get("seed", 10000))

    # Phase 1 does not persist RoboTwin sim states: the SAPIEN state is not
    # trivially serializable and the native reporter only emits videos.
    # Fail closed on an explicit opt-in instead of silently producing nothing
    # while the disk preflight still budgets for the declared artifacts.
    save_sim_states = evaluation.get("save_sim_states", False)
    if save_sim_states:
        raise RobotwinEvaluationError(
            "save_sim_states=true is not supported by the RoboTwin adapter in Phase 1; "
            "remove the key or set it to false"
        )
    video_block = evaluation.get("video", {})
    video_block = _require_mapping(video_block, "benchmark.evaluation.video")
    video_enabled = bool(video_block.get("enabled", True))
    # Phase 1 writes one MP4 per episode. The single-video merge needs a
    # task-level streaming writer (see the LIBERO reporter); fail closed on
    # the unsupported flag instead of silently keeping only the last episode.
    if bool(video_block.get("single_video", False)):
        raise RobotwinEvaluationError(
            "video.single_video=true is not supported by the RoboTwin adapter in Phase 1; "
            "remove the key or set it to false to get one MP4 per episode"
        )

    output = evaluation.get("output", {})
    output = _require_mapping(output, "benchmark.evaluation.output")
    write_native = bool(output.get("write_native", True))
    write_canonical = bool(output.get("write_canonical", True))

    timeouts = evaluation.get("timeouts", {})
    timeouts = _require_mapping(timeouts, "benchmark.evaluation.timeouts")
    startup_timeout_sec = float(timeouts.get("startup_timeout_sec", 120.0))
    max_duration_sec = float(timeouts.get("max_duration_sec", 300.0))

    task_order_index = int(evaluation.get("task_order_index", 0))

    # Static parallel lanes mirror the LIBERO contract: every lane runs the
    # complete plan, fail-fast is mandatory, and dynamic scheduling is
    # rejected up front instead of failing per lane at runtime.
    parallel = evaluation.get("parallel", {})
    parallel = _require_mapping(parallel, "benchmark.evaluation.parallel")
    if parallel.get("scheduling", "static") != "static":
        raise RobotwinEvaluationError("RoboTwin evaluation supports static scheduling only; dynamic is unsupported")
    if parallel.get("enabled", False) and parallel.get("fail_fast", True) is not True:
        raise RobotwinEvaluationError("RoboTwin static parallel MVP requires fail_fast=true")
    lane_count = int(parallel.get("lane_count", 1)) if parallel.get("enabled", False) else 1

    return RobotwinPlanRequest(
        suite=suite,
        task_name=task_name,
        task_config=task_config,
        embodiment=embodiment,
        env_cfg_type=env_cfg_type,
        task_order_index=task_order_index,
        tasks=tasks,
        episodes_per_task=episodes_per_task,
        max_steps=max_steps,
        seed=seed,
        save_sim_states=save_sim_states,
        video_enabled=video_enabled,
        write_native=write_native,
        write_canonical=write_canonical,
        startup_timeout_sec=startup_timeout_sec,
        max_duration_sec=max_duration_sec,
        lane_count=lane_count,
    )
