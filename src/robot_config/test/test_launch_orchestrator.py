"""Frozen-plan orchestration using real launch events, without spawning processes."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
from launch import LaunchContext
from launch.actions import EmitEvent, ExecuteProcess, GroupAction, LogInfo, RegisterEventHandler
from launch.events import Shutdown
from launch.events.process import ProcessExited

from robot_config.launch_builders import runtime, tracing
from robot_config.launch_builders.orchestrator import build_prepared_launch
from robot_config.launch_builders.registry import BuilderSpec, BuildResult, LaunchBuildContext, LaunchBuilderRegistry


def _prepared(monkeypatch, entries, *, barrier=None, binding=False, arguments=None, enabled=None):
    module = ModuleType("test_orchestrator_factories")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    registry = LaunchBuilderRegistry()
    for name, timing, factory in entries:
        setattr(module, name, factory)
        registry.add(BuilderSpec(id=name, factory=f"{module.__name__}:{name}", timing=timing, enabled=enabled))
    context = LaunchBuildContext(robot_config={"name": "r"}, controller_readiness_barrier=barrier)
    return SimpleNamespace(
        context=context,
        plan=registry.plan(context),
        arguments=arguments or {},
        live_binding_required=binding,
        use_sim=True,
        # The live-binding branch re-runs check_launch_dds against the
        # original launch context; give the fake prepared state a real one.
        launch_context=LaunchContext(),
    )


def _exit(action, code):
    return ProcessExited(action=action, name="test", cmd=[], cwd=None, env=None, pid=1, returncode=code)


def test_no_barrier_preserves_mixed_plan_order(monkeypatch):
    actions = [LogInfo(msg=name) for name in ("control", "perception", "recording", "mirror")]
    prepared = _prepared(
        monkeypatch,
        [
            (
                str(i),
                "runtime_ready" if i in (1, 3) else "immediate",
                lambda ctx, action=action: BuildResult(actions=[action]),
            )
            for i, action in enumerate(actions)
        ],
    )
    assert build_prepared_launch(prepared) == actions


@pytest.mark.parametrize("code", [0, 2])
def test_handler_precedes_real_barrier_and_gates_consumers(monkeypatch, code):
    barrier = ExecuteProcess(cmd=["never-executed"])
    consumer = LogInfo(msg="consumer")
    tail = LogInfo(msg="tail")
    prepared = _prepared(
        monkeypatch,
        [
            ("control", "immediate", lambda ctx: BuildResult(actions=[barrier])),
            ("perception", "runtime_ready", lambda ctx: BuildResult(actions=[consumer])),
            ("tail", "immediate", lambda ctx: BuildResult(actions=[tail])),
        ],
        barrier=barrier,
    )
    actions = build_prepared_launch(prepared)
    assert isinstance(actions[0], RegisterEventHandler)
    assert actions[1:] == [barrier, tail]
    handler = actions[0].event_handler
    event = _exit(barrier, code)
    assert handler.matches(event)
    assert not handler.matches(_exit(ExecuteProcess(cmd=["other"]), code))
    released = list(handler.handle(event, LaunchContext()))
    if code == 0:
        assert released == [consumer]
    else:
        assert len(released) == 1 and isinstance(released[0], EmitEvent)
        assert isinstance(released[0].event, Shutdown)


def test_barrier_without_consumers_adds_no_failure_listener(monkeypatch):
    barrier = ExecuteProcess(cmd=["never-executed"])
    prepared = _prepared(
        monkeypatch, [("control", "immediate", lambda ctx: BuildResult(actions=[barrier]))], barrier=barrier
    )
    assert build_prepared_launch(prepared) == [barrier]


def test_tracing_scopes_both_root_and_deferred_actions(monkeypatch):
    scopes = []

    def scope(actions):
        group = GroupAction(actions=actions)
        scopes.append((group, list(actions)))
        return group

    setup = LogInfo(msg="tracing setup")
    monkeypatch.setattr(tracing, "scope_tracing_environment", scope)
    monkeypatch.setattr(tracing, "generate_tracing_actions", lambda **kwargs: [setup])
    barrier = ExecuteProcess(cmd=["never-executed"])
    consumer = LogInfo(msg="consumer")
    prepared = _prepared(
        monkeypatch,
        [
            ("control", "immediate", lambda ctx: BuildResult(actions=[barrier])),
            ("consumer", "runtime_ready", lambda ctx: BuildResult(actions=[consumer])),
        ],
        barrier=barrier,
        arguments={"enable_tracing": "true"},
    )
    result = build_prepared_launch(prepared)
    assert len(scopes) == 2
    deferred, root = scopes
    assert deferred[1] == [consumer]
    assert result == [root[0]]
    assert root[1][0] is setup
    handler = root[1][1].event_handler
    assert list(handler.handle(_exit(barrier, 0), LaunchContext())) == [deferred[0]]
    assert root[1][-1] is barrier


def test_binding_delegates_once_and_uses_frozen_plan_and_effective_snapshot(monkeypatch):
    built, delegated, extras, predicates = [], [], [], []
    effective = {"name": "bound"}
    consumer = LogInfo(msg="bound consumer")
    extra = LogInfo(msg="extra")

    def build(context):
        built.append((context.robot_config, context.provider_started))
        return BuildResult(actions=[consumer])

    def enabled(context):
        predicates.append(context.robot_config)
        return context.robot_config["name"] == "r"

    prepared = _prepared(monkeypatch, [("consumer", "runtime_ready", build)], binding=True, enabled=enabled)

    def delegate(config, construct_consumers, **kwargs):
        delegated.append((config, kwargs))
        return construct_consumers(effective)

    monkeypatch.setattr(runtime, "generate_bound_runtime_actions", delegate)
    result = build_prepared_launch(prepared, extra_consumers=lambda config: extras.append(config) or [extra])
    assert len(delegated) == 1
    assert built == [(effective, True)]
    assert len(predicates) == 1  # no re-filtering after effective config becomes "bound"
    assert extras == [effective]
    assert result == [consumer, extra]


def test_binding_without_callback_does_not_build_consumers(monkeypatch):
    built = []
    sentinel = LogInfo(msg="binding pending or failed")
    prepared = _prepared(
        monkeypatch, [("consumer", "runtime_ready", lambda ctx: built.append(ctx) or BuildResult())], binding=True
    )
    monkeypatch.setattr(runtime, "generate_bound_runtime_actions", lambda *args, **kwargs: [sentinel])
    assert build_prepared_launch(prepared) == [sentinel]
    assert built == []


def test_provider_only_nonbinding_path_ignores_extra_consumers(monkeypatch):
    prepared = _prepared(monkeypatch, [])
    prepared.context.robot_config["runtime"] = {"provider": "so101_runtime"}
    calls = []
    assert build_prepared_launch(prepared, extra_consumers=lambda config: calls.append(config) or []) == []
    assert calls == []


def test_builder_exception_preserves_cause(monkeypatch):
    error = ValueError("boom")

    def fail(context):
        raise error

    prepared = _prepared(monkeypatch, [("consumer", "immediate", fail)])
    with pytest.raises(RuntimeError, match="'consumer'") as caught:
        build_prepared_launch(prepared)
    assert caught.value.__cause__ is error
