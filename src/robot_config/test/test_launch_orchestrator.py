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
from robot_config.launch_registry.orchestrator import build_prepared_launch
from robot_config.launch_registry.registry import BuilderSpec, BuildResult, LaunchBuildContext, LaunchBuilderRegistry


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
        sim_platform="sdk",
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


# ──────────────────────────────────────────────────────────────────────────────
# Pre-binding world bootstrap (runtime-simulation-transport D3b, task 3.2b)
# ──────────────────────────────────────────────────────────────────────────────


def _gazebo_provider_prepared(monkeypatch, tmp_path, *, binding=True):
    """A real prepared launch for a provider on the gazebo transport.

    Drives the real launch entry preparation over the builtin registry with
    the real simulation/control entries; only the provider launch file (and,
    for the consumer pass, the live-description binding) is stubbed.
    """
    from robot_config.launch_registry.builtin_registry import build_launch_registry
    from robot_config.launch_registry.preparation import prepare_launch

    provider_launch = tmp_path / "runtime.launch.py"
    provider_launch.write_text("# provider is not executed by this construction test\n", encoding="utf-8")
    monkeypatch.setattr(runtime, "resolve_runtime_launch", lambda _provider: str(provider_launch))

    loaded = {
        "name": "consumer",
        "runtime": {"provider": "test_robot", "profile": "so101_single_arm", "require_model": binding},
        "simulation": {"platform": "gazebo", "scene": "pick_banana", "robot_spawn": {"x": 0.1, "y": 0.2, "z": -0.03}},
    }
    context = LaunchContext()
    context.launch_configurations.update({"use_sim": "true", "sim_platform": ""})
    return prepare_launch(context, build_launch_registry(entry_points=[]), loaded_config=loaded)


def _sub_text(value):
    if isinstance(value, tuple | list):
        return "".join(_sub_text(item) for item in value)
    return str(getattr(value, "text", value))


def _include_summary(actions):
    """(index, source_name, arguments) for each include; the gz server include
    is identified by its ``gz_args`` argument (its launch source is a
    PathJoinSubstitution whose str() carries no readable name)."""
    from launch.actions import IncludeLaunchDescription

    summaries = []
    for index, action in enumerate(actions):
        if not isinstance(action, IncludeLaunchDescription):
            continue
        arguments = {_sub_text(key): _sub_text(value) for key, value in action.launch_arguments}
        if "gz_args" in arguments:
            name = "gz_sim.launch.py"
        elif "profile" in arguments:
            name = "provider runtime.launch.py"
        else:
            name = ""
        summaries.append((index, name, arguments))
    return summaries


def _gz_server_count(actions):
    return sum(1 for _index, name, args in _include_summary(actions) if "gz_sim.launch.py" in name)


def test_binding_branch_starts_the_world_once_before_the_provider(monkeypatch, tmp_path):
    """The real binding branch (D3b): exactly one gz server action, ordered
    before the provider include, with the resolved world handed over."""
    prepared = _gazebo_provider_prepared(monkeypatch, tmp_path, binding=True)
    assert prepared.live_binding_required is True

    actions = build_prepared_launch(prepared)
    includes = _include_summary(actions)
    gz_includes = [(index, args) for index, name, args in includes if "gz_sim.launch.py" in name]
    assert len(gz_includes) == 1, f"expected exactly one gz server action, got {len(gz_includes)}"
    provider_includes = [(index, args) for index, name, args in includes if "provider runtime.launch.py" in name]
    assert len(provider_includes) == 1
    world_index, _ = gz_includes[0]
    provider_index, provider_args = provider_includes[0]
    assert world_index < provider_index, "the world must start before the provider readiness wait"
    assert provider_args["transport"] == "gazebo"
    assert "simulated" not in provider_args
    assert provider_args["gazebo_world"] == "demo"
    assert (provider_args["spawn_x"], provider_args["spawn_y"], provider_args["spawn_z"]) == ("0.1", "0.2", "-0.03")


def test_binding_branch_consumer_pass_does_not_relaunch_the_world(monkeypatch, tmp_path):
    """Driving the real on_ready consumer callback: the second plan pass
    reuses the pre-seeded SimulationStartup and emits no second world."""
    import robot_config.launch_builders.runtime as runtime_module
    import robot_runtime.interface_description as interface_description_module

    prepared = _gazebo_provider_prepared(monkeypatch, tmp_path, binding=True)
    actions = build_prepared_launch(prepared)

    description = {
        "schema_version": 1,
        "robot": {"id": "test_robot", "type": "test", "runtime_name": "test_robot", "runtime_version": "0.1.0"},
        "execution": "simulated",
        "transport": "gazebo",
        "interfaces": {},
        "states": {},
    }
    from robot_runtime.interface_description import description_digest

    description["digest"] = description_digest(description)
    monkeypatch.setattr(interface_description_module, "load_description", lambda _path: description)
    monkeypatch.setattr(runtime_module, "materialize_interface_config", lambda effective, _d, _p: effective)

    from launch.event_handlers import OnProcessExit

    handlers = [
        action
        for action in actions
        if isinstance(action, RegisterEventHandler) and isinstance(action.event_handler, OnProcessExit)
    ]
    assert len(handlers) == 1, "the bound stream registers exactly one OnProcessExit handler (readiness)"
    on_ready = handlers[0].event_handler._OnActionEventBase__on_event
    consumer_actions = on_ready(_exit(None, 0), SimpleNamespace(is_shutdown=False))
    assert _gz_server_count(actions) + _gz_server_count(consumer_actions or []) == 1, (
        "the consumer pass must reuse the pre-seeded world, not relaunch it"
    )


def test_control_builds_after_simulation_in_the_frozen_plan(monkeypatch, tmp_path):
    """D3b non-binding branch: the provider-including control entry builds
    after the simulation entry (the world must run before the provider waits
    for its clock); controller_startup stays after both."""
    prepared = _gazebo_provider_prepared(monkeypatch, tmp_path, binding=False)
    assert prepared.live_binding_required is False
    order = [spec.id for spec in prepared.plan.specs]
    assert "simulation" in order and "control" in order
    assert order.index("simulation") < order.index("control")
    assert "controller_startup" in order
    assert order.index("control") < order.index("controller_startup")
    assert order.index("simulation") < order.index("controller_startup")

    # The non-binding stream keeps the same property: the world include is
    # emitted before the provider include.
    actions = build_prepared_launch(prepared)
    includes = _include_summary(actions)
    world = [index for index, name, _args in includes if "gz_sim.launch.py" in name]
    provider = [index for index, name, _args in includes if "provider runtime.launch.py" in name]
    assert len(world) == 1 and len(provider) == 1 and world[0] < provider[0]
