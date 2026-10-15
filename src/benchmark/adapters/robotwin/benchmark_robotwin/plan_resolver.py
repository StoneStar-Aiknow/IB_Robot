"""RoboTwin plan resolver: ``RobotwinPlanRequest`` → ``BenchmarkPlan``.

Builds an immutable :class:`benchmark_runtime.plan.BenchmarkPlan` from a
single resolved :class:`RobotwinProviderTask`. The plan always contains
exactly one task (``task_id=0``) in Phase 1, matching the single-task
evaluation mode.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmark_robotwin.evaluation import RobotwinPlanRequest
from benchmark_robotwin.plan_provider import RobotwinPlanProvider, RobotwinProviderTask
from benchmark_runtime.plan import (
    BenchmarkArtifactFlags,
    BenchmarkPlan,
    BenchmarkPlanTask,
    BenchmarkTimeouts,
    InitStatePolicy,
)


class RobotwinPlanResolverError(ValueError):
    """Raised when the plan cannot be resolved."""


@dataclass(frozen=True, slots=True)
class _ResolvedTask:
    task_id: int
    name: str
    prompt: str
    problem_folder: str
    step_limit: int


class RobotwinPlanResolver:
    """Resolve a single-task RoboTwin plan."""

    def __init__(self, provider: RobotwinPlanProvider) -> None:
        self._provider = provider

    def resolve(self, request: RobotwinPlanRequest) -> BenchmarkPlan:
        provider_task: RobotwinProviderTask = self._provider.resolve_single_task(request.task_name)

        resolved = _ResolvedTask(
            task_id=0,
            name=provider_task.task_name,
            prompt=provider_task.instruction,
            problem_folder=provider_task.task_name,
            step_limit=provider_task.step_limit,
        )

        tasks = (
            BenchmarkPlanTask(
                task_id=resolved.task_id,
                name=resolved.name,
                prompt=resolved.prompt,
                init_state_count=max(1, request.episodes_per_task),
                metadata={
                    "problem_folder": resolved.problem_folder,
                    "bddl_file": f"robotwin:{resolved.name}",
                    "init_states_file": f"robotwin:seed:{resolved.name}",
                    "step_limit": resolved.step_limit,
                },
            ),
        )

        selected = (0,) if request.tasks is None else tuple(request.tasks)
        if not selected:
            raise RobotwinPlanResolverError("selected task ids must be non-empty")
        for tid in selected:
            if tid != 0:
                raise RobotwinPlanResolverError(f"Phase 1 supports only task_id 0; got {tid}")

        planned_episodes = len(selected) * request.episodes_per_task
        effective_max_steps = min(request.max_steps, resolved.step_limit)

        return BenchmarkPlan(
            benchmark_type="robotwin",
            adapter="robotwin",
            suite=request.suite,
            requested_task_order_index=request.task_order_index,
            effective_task_order_index=request.task_order_index,
            tasks=tasks,
            selected_task_ids=selected,
            episodes_per_task=request.episodes_per_task,
            planned_episodes=planned_episodes,
            seed=request.seed,
            init_state_policy=InitStatePolicy(
                use_init_state_id=True,
                selection="episode_index_modulo_available_native_init_state_count",
            ),
            max_steps=effective_max_steps,
            timeouts=BenchmarkTimeouts(
                startup_timeout_sec=request.startup_timeout_sec,
                max_duration_sec=request.max_duration_sec,
            ),
            artifacts=BenchmarkArtifactFlags(
                save_sim_states=request.save_sim_states,
                video_enabled=request.video_enabled,
                write_native=request.write_native,
                write_canonical=request.write_canonical,
            ),
            warnings=(),
            lane_count=request.lane_count,
        )
