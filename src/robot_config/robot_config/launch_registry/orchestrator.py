"""Generic launch orchestrator over the prepared launch and frozen plan.

The registration framework lives in ``robot_config.launch_registry``
(openspec change ``split-launch-registry-package``).

Implements design decisions 3/6 of openspec change
``refactor-launch-builder-registry``. The orchestrator accepts a
:class:`~robot_config.launch_registry.preparation.PreparedLaunch` (the
preparation pipeline output: prepared context + frozen plan) and executes
the planned builders through ``execute_plan``, then assembles the final
action stream:

- **No barrier** (no controller group spawner / readiness waiter, or the
  provider is already live-bound): the merged plan-order actions are
  returned directly, preserving the legacy observable startup order
  including the tail business entries.
- **Barrier present and deferred candidates non-empty**: the
  ``runtime_ready`` candidate set is split out (``split_for_barrier``)
  and gated behind the controller readiness barrier via a single
  ``OnProcessExit`` success-gated handler registered BEFORE the barrier
  action itself; a failure/non-zero exit does not release the deferred
  set and requests Shutdown. With no deferred candidates no extra
  group/waiter failure listener is added (legacy behavior).
- **Tracing** wraps both scopes like the legacy entry: the immediate root
  actions and the true deferred actions each get
  ``scope_tracing_environment`` (readiness callbacks run after the outer
  launch environment scope has closed).
- **Live binding**: when ``live_binding_required`` is set the orchestrator
  delegates exactly once to the existing
  ``generate_bound_runtime_actions``; the callback only builds the
  already-frozen plan against the bound effective config (context swap,
  ``provider_started=True``), never re-filters registry entries. The
  provider-only non-binding combination keeps the legacy known
  limitation: ``extra_consumers`` is consumed only on the live-binding
  branch.

Business exceptions bubble up wrapped with the builder id (execute_plan);
recording/rerun tolerance policies live in their adapters, not here.
"""

from __future__ import annotations

from typing import Any

from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit

from robot_config.dds_environment import check_launch_dds
from robot_config.launch_registry.registry import LaunchBuildContext, execute_plan
from robot_config.logger_utils import get_colored_logger
from robot_config.utils import parse_bool

logger = get_colored_logger("robot_config.launch_registry.orchestrator")


def _tracing_attribute(name: str):
    """Resolve a tracing helper at call time from its source module.

    Existing tests monkeypatch ``robot_config.launch_builders.tracing``;
    binding these callables at module import (a top-level ``from``
    import) would freeze the original function and silently miss the
    patched path.
    """
    from robot_config.launch_builders import tracing

    return getattr(tracing, name)


def _readiness_attribute(name: str):
    """Resolve a readiness helper at call time (monkeypatch seam).

    Same rule as the tracing helpers: existing tests patch
    ``robot_config.launch_builders.readiness`` on the source module.
    """
    from robot_config.launch_builders import readiness

    return getattr(readiness, name)


def _tracing_enabled(prepared) -> bool:
    return parse_bool(prepared.arguments.get("enable_tracing", "false"), default=False)


def _assemble_actions(
    prepared,
    context: LaunchBuildContext,
    outputs,
    *,
    enable_tracing: bool,
) -> list[Any]:
    """Merge plan-order actions applying barrier gating and tracing scopes."""
    immediate_actions: list[Any]
    deferred_actions: list[Any]
    barrier = context.controller_readiness_barrier
    if barrier is not None and outputs.runtime_ready_actions:
        immediate_actions, deferred_actions = outputs.split_for_barrier()
        logger.info(f"Controller readiness barrier armed for {len(deferred_actions)} control-dependent action(s)")
        if enable_tracing:
            # Readiness callbacks run after the outer launch environment
            # scope has closed.
            deferred_actions = [_tracing_attribute("scope_tracing_environment")(deferred_actions)]
        # The consumer handler must be registered BEFORE the barrier
        # action itself (the barrier may appear anywhere in the merged
        # stream); the callback snapshots the action list.
        immediate_actions.insert(
            0,
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=barrier,
                    on_exit=_readiness_attribute("start_actions_on_success")(
                        deferred_actions,
                        success_message="Required controllers are active; starting control-dependent nodes.",
                        failure_reason="Controller readiness probe failed; aborting launch.",
                    ),
                )
            ),
        )
    else:
        # No barrier (or nothing deferred behind it): keep the merged
        # plan-order stream verbatim, tail entries included.
        immediate_actions = list(outputs.actions)
        deferred_actions = []

    if enable_tracing:
        requested_trace_session = prepared.arguments.get(
            "trace_session_name", _tracing_attribute("DEFAULT_TRACE_SESSION_NAME")
        )
        immediate_actions[:0] = _tracing_attribute("generate_tracing_actions")(
            enable_tracing=True, requested_session_name=requested_trace_session
        )
        return [_tracing_attribute("scope_tracing_environment")(immediate_actions)]
    return immediate_actions


def build_prepared_launch(
    prepared,
    *,
    extra_consumers=None,
) -> list[Any]:
    """Execute the frozen plan and assemble the final action stream.

    ``prepared`` is the :class:`PreparedLaunch` returned by
    ``prepare_launch``; the plan inside it was validated and frozen
    before any provider startup (design 13b). ``extra_consumers`` follows
    the legacy ``launch_setup`` convention and is consumed only on the
    live-binding branch.
    """
    context: LaunchBuildContext = prepared.context
    plan = prepared.plan
    enable_tracing = _tracing_enabled(prepared)

    if not prepared.live_binding_required:
        outputs = execute_plan(plan, context)
        if extra_consumers is not None:
            # Known legacy limitation, kept intentionally: extra_consumers
            # is only consumed by the live-binding branch. A provider-only
            # configuration with no binding requirement ignores it.
            logger.debug("extra_consumers ignored on the non-binding path (legacy known limitation)")
        logger.info(f"========== Total nodes to launch: {len(outputs.actions)} ==========")
        return _assemble_actions(prepared, context, outputs, enable_tracing=enable_tracing)

    # 13c. binding: delegate provider/live binding exactly once; the
    # callback only builds the already-frozen plan against the bound
    # effective config and never re-filters registry entries.
    import robot_config.launch_builders.runtime as runtime_module

    robot_config = context.robot_config

    def construct_consumers(effective):
        # Deferred-consumer DDS re-check (upstream robot.launch.py semantics):
        # the launch environment is validated again right before the
        # live-bound consumers are constructed, using the original launch
        # context carried by the prepared state.
        check_launch_dds(prepared.launch_context)
        context.robot_config = effective
        context.provider_started = True
        outputs = execute_plan(plan, context)
        actions = _assemble_actions(prepared, context, outputs, enable_tracing=enable_tracing)
        if extra_consumers is not None:
            actions.extend(extra_consumers(effective))
        return actions

    # The controller startup timeout resolves only on this real
    # provider/binding consumption path (never for benchmark/mock skips).
    return runtime_module.generate_bound_runtime_actions(
        robot_config,
        construct_consumers,
        use_sim=prepared.use_sim,
        display=parse_bool(prepared.arguments.get("moveit_display", "false"), default=False),
        readiness_timeout_s=_readiness_attribute("resolve_controller_startup_timeout")(robot_config, prepared.use_sim),
    )
