"""PI0.5 stage topology derived from manifest execution roles and bindings."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from inference_manifest import TensorBinding
from inference_service.codecs import ExecutionPlan
from inference_service.pi05_schedule import PI05DenoisingSchedule
from inference_service.pipeline.executor import SequentialModelExecutor
from inference_service.pipeline.stages import (
    DirectIterationStateAdapter,
    EulerIterationStateAdapter,
    InferenceStage,
    IterationStep,
    IterativeStage,
    ModelStage,
    ResultAdapter,
)

_NOISE_SEMANTICS = frozenset({"noise", "action.noise", "_noise"})
_TIME_SEMANTICS = frozenset({"time", "timestep", "action.time", "_time"})
_VELOCITY_SEMANTICS = frozenset({"velocity", "action"})
_MODULAR_SUFFIX = ("embedding", "prefill", "action_in_proj", "time_mlp", "decode", "action_out_proj")


class PI05TopologyError(ValueError):
    """Raised when a manifest execution plan is not a supported PI0.5 topology."""


@dataclass(frozen=True)
class PI05Topology:
    pre_loop_roles: tuple[str, ...]
    loop_roles: tuple[str, ...]
    state_semantic: str
    timestep_semantic: str
    velocity_semantic: str


def derive_pi05_topology(plan: ExecutionPlan) -> PI05Topology:
    """Validate and classify PI0.5 role composition without inspecting backend identity."""

    roles = plan.role_names
    if roles == ("vlm", "action_expert"):
        pre_loop_roles = roles[:1]
        loop_roles = roles[1:]
    elif len(roles) > len(_MODULAR_SUFFIX) and roles[-len(_MODULAR_SUFFIX) :] == _MODULAR_SUFFIX:
        vision_roles = roles[: -len(_MODULAR_SUFFIX)]
        if any(role != "vision" and not role.startswith("vision_") for role in vision_roles):
            raise PI05TopologyError("PI0.5 modular topology has invalid vision roles")
        pre_loop_roles = (*vision_roles, "embedding", "prefill")
        loop_roles = ("action_in_proj", "time_mlp", "decode", "action_out_proj")
    else:
        raise PI05TopologyError(
            "PI0.5 topology must be (vlm, action_expert) or vision role(s) followed by the modular role suffix"
        )

    state_semantic = _single_input_semantic(plan, loop_roles, _NOISE_SEMANTICS, "noise")
    timestep_semantic = _single_input_semantic(plan, loop_roles, _TIME_SEMANTICS, "timestep")
    velocity_semantic = _single_output_semantic(plan, loop_roles[-1], _VELOCITY_SEMANTICS, "velocity")
    return PI05Topology(pre_loop_roles, loop_roles, state_semantic, timestep_semantic, velocity_semantic)


def create_pi05_executor(
    plan: ExecutionPlan,
    session: object,
    schedule: PI05DenoisingSchedule | None,
    result_adapter: ResultAdapter,
    *,
    num_inference_steps: int | None = None,
    initializer: Callable[[Mapping[str, object]], np.ndarray] | None = None,
    stage_for_role: Callable[[str], InferenceStage] | None = None,
    velocity_trace: list[np.ndarray] | None = None,
    embedding_stage: InferenceStage | None = None,
    time_prep_stage: InferenceStage | None = None,
    extra_components: tuple[object, ...] = (),
) -> SequentialModelExecutor:
    """Build pre-loop and iterative stages from a validated PI0.5 execution plan.

    ``schedule`` selects the update law: a validated Euler velocity schedule uses
    :class:`EulerIterationStateAdapter`; ``None`` retains the direct-state behavior
    (each model output replaces ``x_t``) using :class:`DirectIterationStateAdapter`
    with ``num_inference_steps`` uniform timesteps.

    ``embedding_stage`` replaces the default :class:`ModelStage` for the modular
    ``embedding`` host role so synthetic embedding construction runs as an
    executor-owned :class:`HostRoleStage`.

    ``time_prep_stage`` is inserted immediately before ``ModelStage(time_mlp)``
    in the modular loop body so sinusoidal timestep embedding is prepared on the
    host before the compiled time MLP consumes it.  When provided, the iteration
    state adapter emits a scalar timestep and delegates shape conversion to the
    host stage.
    """

    topology = derive_pi05_topology(plan)
    stage_factory = stage_for_role or (lambda role: ModelStage(role, session))
    pre_loop = _expand_pre_loop_stages(topology, stage_factory, embedding_stage)
    body = _expand_loop_body_stages(topology, stage_factory, time_prep_stage)
    timestep_shape = (
        None
        if time_prep_stage is not None
        else _static_timestep_shape(plan, topology.loop_roles, topology.timestep_semantic)
    )
    if schedule is not None:
        steps = schedule.iteration_steps()
        state_adapter = EulerIterationStateAdapter(
            state_semantic=topology.state_semantic,
            timestep_semantic=topology.timestep_semantic,
            velocity_semantic=topology.velocity_semantic,
            initializer=initializer,
            timestep_shape=timestep_shape,
            velocity_trace=velocity_trace,
        )
    else:
        steps = _direct_state_steps(num_inference_steps)
        state_adapter = DirectIterationStateAdapter(
            state_semantic=topology.state_semantic,
            output_semantic=topology.velocity_semantic,
            timestep_semantic=topology.timestep_semantic,
            initializer=initializer,
            timestep_shape=timestep_shape,
        )
    iterative = IterativeStage(
        steps,
        body,
        state_adapter,
        loop_roles=topology.loop_roles,
    )
    return SequentialModelExecutor(
        (*pre_loop, iterative),
        result_adapter,
        components=(session, *extra_components),
        execution_plan=plan,
    )


def _direct_state_steps(num_inference_steps: int | None) -> tuple[IterationStep, ...]:
    if type(num_inference_steps) is not int or num_inference_steps < 1:
        raise PI05TopologyError("PI0.5 direct-state mode requires a positive integer num_inference_steps")
    return tuple(
        IterationStep(index=step, timestep=1.0 - step / num_inference_steps, delta=0.0)
        for step in range(num_inference_steps)
    )


def _expand_pre_loop_stages(
    topology: PI05Topology,
    stage_factory: Callable[[str], InferenceStage],
    embedding_stage: InferenceStage | None,
) -> tuple[InferenceStage, ...]:
    stages: list[InferenceStage] = []
    for role in topology.pre_loop_roles:
        if role == "embedding" and embedding_stage is not None:
            stages.append(embedding_stage)
        else:
            stages.append(stage_factory(role))
    return tuple(stages)


def _expand_loop_body_stages(
    topology: PI05Topology,
    stage_factory: Callable[[str], InferenceStage],
    time_prep_stage: InferenceStage | None,
) -> tuple[InferenceStage, ...]:
    if time_prep_stage is None:
        return tuple(stage_factory(role) for role in topology.loop_roles)
    stages: list[InferenceStage] = []
    for role in topology.loop_roles:
        if role == "time_mlp":
            stages.append(time_prep_stage)
        stages.append(stage_factory(role))
    return tuple(stages)


def _static_timestep_shape(
    plan: ExecutionPlan,
    roles: tuple[str, ...],
    timestep_semantic: str,
) -> tuple[int, ...] | None:
    binding = _single_input_binding(plan, roles, timestep_semantic)
    if any(dimension < 1 for dimension in binding.shape):
        return None
    return binding.shape


def _single_input_semantic(
    plan: ExecutionPlan,
    roles: tuple[str, ...],
    candidates: frozenset[str],
    label: str,
) -> str:
    binding = _single_input_binding(plan, roles, candidates, label)
    return binding.semantic


def _single_input_binding(
    plan: ExecutionPlan,
    roles: tuple[str, ...],
    candidates: frozenset[str],
    label: str | None = None,
) -> TensorBinding:
    matches = [
        binding for role in roles for binding in plan.role(role).bindings.inputs if binding.semantic in candidates
    ]
    if len(matches) != 1:
        detail = label or "matching"
        raise PI05TopologyError(f"PI0.5 loop requires exactly one {detail} input semantic, found {sorted(matches)}")
    return matches[0]


def _single_output_semantic(
    plan: ExecutionPlan,
    role: str,
    candidates: frozenset[str],
    label: str,
) -> str:
    matches = {binding.semantic for binding in plan.role(role).bindings.outputs if binding.semantic in candidates}
    if len(matches) != 1:
        raise PI05TopologyError(f"PI0.5 loop requires exactly one {label} output semantic, found {sorted(matches)}")
    return matches.pop()


__all__ = ["PI05Topology", "PI05TopologyError", "create_pi05_executor", "derive_pi05_topology"]
