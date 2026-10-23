"""Preparation behavior, extension ownership and pre-provider validation."""

from __future__ import annotations

import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest

from robot_config.launch_builders import control, readiness
from robot_config.launch_registry.launch_options import ArgumentSpec, ArgumentSpecError
from robot_config.launch_registry.preparation import (
    PreparationError,
    _run_phase_hooks,
    compose_registry_arguments,
    prepare_launch,
)
from robot_config.launch_registry.registry import (
    BuilderSpec,
    BuildResult,
    HookSpec,
    LaunchBuilderRegistry,
    RegistryError,
    execute_plan,
)
from robot_config.runtime_target import RuntimeTargetError


def _context(**arguments):
    return SimpleNamespace(launch_configurations=arguments)


@pytest.fixture()
def business(monkeypatch):
    """Hermetic loader + resource-check observer.

    The config_overrides/preflight hook observations that used hand_sources as
    their sample entry moved to a synthetic probe entry (the hand_sources entry
    itself moved to robot_teleop, which robot_config tests must not import);
    the framework properties under test are unchanged.
    """
    calls = []
    monkeypatch.setattr(
        control, "validate_runtime_resources", lambda config, **kwargs: calls.append(("resources", config))
    )
    monkeypatch.setattr(readiness, "resolve_config_path", lambda *args: "/tmp/preparation-test.yaml")
    monkeypatch.setattr(readiness, "load_launch_robot_config", lambda *args, **kwargs: {"name": "test_robot"})
    probe = ModuleType("preparation_probe_hand")

    def probe_override_hook(config, arguments):
        calls.append(("hand", arguments["probe_profile"]))

    def probe_preflight_hook(config, arguments, target_state):
        calls.append(("confirm", config, {"use_sim": target_state.use_sim, "control_mode": target_state.control_mode}))

    probe.probe_override_hook = probe_override_hook
    probe.probe_preflight_hook = probe_preflight_hook
    probe.PROBE_BUILDER = BuilderSpec(
        id="probe_hand",
        factory=f"{probe.__name__}:probe_override_hook",
        arguments=(
            ArgumentSpec(
                name="probe_profile", declared_default="", absent_fallback="", description="Probe profile selection"
            ),
        ),
        hooks=(
            HookSpec(phase="config_overrides", factory=f"{probe.__name__}:probe_override_hook", config_keys=("probe",)),
            HookSpec(phase="preflight", factory=f"{probe.__name__}:probe_preflight_hook", config_keys=("probe",)),
        ),
    )
    monkeypatch.setitem(sys.modules, probe.__name__, probe)

    def probe_registry():
        from robot_config.launch_registry.builtin_registry import build_builtin_registry

        registry = build_builtin_registry()
        registry.add(probe.PROBE_BUILDER)
        return registry

    monkeypatch.setattr(sys.modules[__name__], "_probe_hand_registry", probe_registry, raising=False)
    return calls


def _probe_hand_registry():
    """Set by the business fixture (see above); keeps test bodies terse."""
    raise AssertionError("the business fixture installs the probe registry")


def _registry(monkeypatch, hook, *, phase="control_mode", keys=("feature",), arguments=(), enabled=None):
    module = ModuleType("test_preparation_extension")
    module.hook = hook
    module.build = lambda context: BuildResult(actions=[context.robot_config["feature"]])
    monkeypatch.setitem(sys.modules, module.__name__, module)
    registry = LaunchBuilderRegistry()
    registry.add(
        BuilderSpec(
            id="feature",
            factory=f"{module.__name__}:build",
            enabled=enabled,
            arguments=arguments,
            hooks=(HookSpec(phase=phase, factory=f"{module.__name__}:hook", config_keys=keys),),
        )
    )
    return registry


def test_normalization_copy_and_prepared_fields(business):
    """Through the builtin registry plus one synthetic hook-bearing entry
    (the former hand_sources sample moved to robot_teleop): the caller's
    loaded config is never mutated, core arguments are normalized, and the
    config_overrides hook -> resource check -> preflight hook ordering with
    an isolated preflight snapshot holds for ANY contributing entry."""
    loaded = {"name": "test_robot"}
    before = copy.deepcopy(loaded)
    prepared = prepare_launch(
        _context(use_sim="true", sim_platform=" Mock ", probe_profile="right", nav_stage=" grasp "),
        _probe_hand_registry(),
        loaded_config=loaded,
    )
    assert loaded == before
    assert prepared.arguments["sim_platform"] == "mock"
    assert prepared.arguments["nav_stage"] == "grasp"
    assert prepared.robot_config["simulation"]["platform"] == "mock"
    assert prepared.context.auto_start_controllers == "false"
    assert prepared.context.sim_backend_needs_ros2_control is False
    # overrides hook -> resource check -> preflight hook, exactly once each.
    assert [item[0] for item in business] == ["hand", "resources", "confirm"]
    # The preflight hook receives the prepared target state and an isolated
    # copy of the real snapshot (never the shared object).
    assert business[-1][2] == {"use_sim": True, "control_mode": "model_inference"}
    assert business[-1][1] is not prepared.robot_config
    assert business[-1][1] == prepared.robot_config


def test_direct_call_fallbacks(business):
    """Direct launch_setup callers observe the same absent fallbacks as
    before the business-parameter migration — through the builtin
    registry that contributes them."""
    from robot_config.launch_registry.builtin_registry import build_builtin_registry

    prepared = prepare_launch(_context(), build_builtin_registry())
    assert prepared.arguments["robot_config"] == "test_cam"
    assert prepared.arguments["moveit_display"] == "false"
    # The hand_profile fallback example moved with the hand_sources entry to
    # robot_teleop's suite (voice_asr's example moved earlier the same way).
    assert "hand_profile" not in prepared.arguments


def test_contributed_cli_hook_enables_builder_before_plan(monkeypatch, business):
    observed = []

    def hook(config, arguments):
        observed.append(config.get("default_control_mode"))
        with pytest.raises(TypeError):
            arguments["feature_enabled"] = "false"
        config["feature"] = arguments["feature_enabled"] == "true"

    registry = _registry(
        monkeypatch,
        hook,
        arguments=(
            ArgumentSpec(
                name="feature_enabled", declared_default="false", absent_fallback="false", description="Enable feature"
            ),
        ),
        enabled=lambda ctx: ctx.robot_config["feature"],
    )
    prepared = prepare_launch(_context(feature_enabled="true", control_mode="teleop"), registry)
    assert observed == [None]
    assert prepared.control_mode == "teleop"
    assert execute_plan(prepared.plan, prepared.context).actions == [True]
    disabled = prepare_launch(_context(feature_enabled="false"), registry)
    assert disabled.plan.specs == ()


@pytest.mark.parametrize("phase", ["control_mode", "preflight"])
def test_hook_failure_never_mutates_shared_snapshot(monkeypatch, phase):
    def hook(config, arguments):
        config["feature"]["enabled"] = True
        config["runtime"]["target"] = "hardware"

    registry = _registry(monkeypatch, hook, phase=phase)
    config = {"feature": {"enabled": False}, "runtime": {"target": "simulation"}}
    before = copy.deepcopy(config)
    with pytest.raises(PreparationError, match="unowned|preflight"):
        _run_phase_hooks(phase, dict(registry.hooks_by_phase())[phase], config, {})
    assert config == before


def test_nested_owned_key_can_be_created_without_allowing_siblings(monkeypatch):
    retained = []

    def hook(config, arguments):
        config.setdefault("feature", {})["enabled"] = True
        retained.append(config)

    registry = _registry(monkeypatch, hook, keys=("feature.enabled",))
    config = {}
    _run_phase_hooks("control_mode", dict(registry.hooks_by_phase())["control_mode"], config, {})
    assert config == {"feature": {"enabled": True}}
    retained[0]["feature"]["enabled"] = False
    assert config["feature"]["enabled"] is True


@pytest.mark.parametrize("key", ["runtime.target", "simulation", "contract.observations", "peripherals"])
def test_core_key_declarations_rejected_before_loading(monkeypatch, key):
    registry = _registry(monkeypatch, lambda *args: None, keys=(key,))
    with pytest.raises(PreparationError, match="core"):
        compose_registry_arguments(registry)


def test_invalid_graph_rejected_before_any_preparation(monkeypatch):
    registry = LaunchBuilderRegistry()
    registry.add(BuilderSpec(id="feature", factory="does_not_exist:build", after=("missing",)))
    with pytest.raises(RegistryError, match="missing"):
        prepare_launch(_context(), registry)


def _launch_entry_module():
    import importlib.util
    from pathlib import Path

    launch_path = Path(__file__).resolve().parents[1] / "launch" / "robot.launch.py"
    spec = importlib.util.spec_from_file_location("robot_launch_invalid_registry", launch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _provider_loaded_config():
    return {
        "name": "consumer",
        "type": "test",
        "runtime": {"provider": "test_robot", "profile": "test", "require_model": True},
    }


def test_invalid_registry_rejected_before_provider_factory(monkeypatch, tmp_path):
    """Task 4.5: patching the REAL launch entry's build_launch_registry to
    an invalid registry (all control/provider entries kept, one rogue spec
    with an unknown dependency) must fail synchronously inside launch_setup
    with RegistryError and ZERO calls to BOTH provider seams —
    generate_runtime_provider_actions and generate_bound_runtime_actions.
    (The seam migrated from build_builtin_registry to build_launch_registry
    when the entry switched to entry-point composition.)"""
    import robot_config.launch_builders.runtime as runtime_module

    provider_calls = []
    bound_calls = []
    real_provider = runtime_module.generate_runtime_provider_actions
    real_bound = runtime_module.generate_bound_runtime_actions
    monkeypatch.setattr(
        runtime_module,
        "generate_runtime_provider_actions",
        lambda *a, **k: (provider_calls.append(1), real_provider(*a, **k))[1],
    )
    monkeypatch.setattr(
        runtime_module,
        "generate_bound_runtime_actions",
        lambda *a, **k: (bound_calls.append(1), real_bound(*a, **k))[1],
    )

    module = _launch_entry_module()
    original_build = module.build_launch_registry

    def build_invalid_registry():
        registry = original_build(entry_points=[])
        registry.add(BuilderSpec(id="rogue_extension", factory="does_not_exist:build", after=("missing",)))
        return registry

    monkeypatch.setattr(module, "build_launch_registry", build_invalid_registry)

    from launch import LaunchContext

    context = LaunchContext()
    with pytest.raises(RegistryError, match="unknown dependency"):
        module.launch_setup(context, loaded_config=_provider_loaded_config())
    assert provider_calls == [] and bound_calls == []


def test_legal_registry_control_reaches_binding_delegate_once(monkeypatch, tmp_path):
    """Task 4.5 legal control: the SAME real launch entry with the unmodified
    builtin registry and a provider/require_model config reaches the
    live-binding delegate exactly once (seam observable) without starting any
    process — proving the zero-call assertions above guard a reachable seam."""
    import robot_config.launch_builders.runtime as runtime_module

    provider_calls = []
    bound_calls = []
    real_provider = runtime_module.generate_runtime_provider_actions
    real_bound = runtime_module.generate_bound_runtime_actions
    monkeypatch.setattr(
        runtime_module,
        "generate_runtime_provider_actions",
        lambda *a, **k: (provider_calls.append(1), real_provider(*a, **k))[1],
    )
    monkeypatch.setattr(
        runtime_module,
        "generate_bound_runtime_actions",
        lambda *a, **k: (bound_calls.append(1), real_bound(*a, **k))[1],
    )

    module = _launch_entry_module()
    provider_path = tmp_path / "runtime.launch.py"
    provider_path.write_text("# provider is not executed by this construction test\n", encoding="utf-8")
    monkeypatch.setattr(runtime_module, "resolve_runtime_launch", lambda _provider: str(provider_path))

    from launch import LaunchContext
    from launch.actions import IncludeLaunchDescription
    from launch_ros.actions import Node

    context = LaunchContext()
    actions = module.launch_setup(context, loaded_config=_provider_loaded_config())
    assert len(bound_calls) == 1
    assert len(provider_calls) == 1
    # No process starts: the returned stream is only constructed — one
    # provider include plus the wait_for_runtime readiness node.
    assert sum(isinstance(action, IncludeLaunchDescription) for action in actions) == 1
    assert any(isinstance(action, Node) and action.node_executable == "wait_for_runtime" for action in actions)


def test_core_argument_collision_rejected(monkeypatch):
    registry = _registry(
        monkeypatch,
        lambda *args: None,
        arguments=(
            ArgumentSpec(name="control_mode", declared_default="x", absent_fallback="x", description="collision"),
        ),
    )
    with pytest.raises(ArgumentSpecError, match="duplicate launch argument"):
        compose_registry_arguments(registry)


def test_business_argument_collision_with_builtin_contribution_rejected(monkeypatch):
    """A contributed argument reusing a builtin business name (with_moveit is
    owned by the motion_planning entry, not the core table) is rejected at
    add time against the builtin contributions. (The record arguments moved
    to dataset_tools with the recording entry; their collision behavior is
    pinned there against the injected extension.)"""
    from robot_config.launch_registry.builtin_registry import build_builtin_registry
    from robot_config.launch_registry.registry import RegistryError

    module = ModuleType("test_preparation_with_moveit_collision")
    module.build = lambda context: BuildResult()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    registry = build_builtin_registry()
    with pytest.raises(
        RegistryError, match="cannot contribute launch argument 'with_moveit'; already contributed by 'motion_planning'"
    ):
        registry.add(
            BuilderSpec(
                id="rogue",
                factory=f"{module.__name__}:build",
                arguments=(
                    ArgumentSpec(
                        name="with_moveit", declared_default="x", absent_fallback="x", description="collision"
                    ),
                ),
            )
        )


def test_generate_launch_description_rejects_cli_collision(monkeypatch):
    """Task 4.5 declaration stage: generate_launch_description composes the
    registry arguments before any OpaqueFunction runs; a business entry
    reusing a core CLI name must be rejected at that declaration stage,
    not at launch time."""
    import importlib.util
    from pathlib import Path

    launch_path = Path(__file__).resolve().parents[1] / "launch" / "robot.launch.py"
    spec = importlib.util.spec_from_file_location("robot_launch_cli_collision", launch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    registry = LaunchBuilderRegistry()
    registry.add(
        BuilderSpec(
            id="rogue",
            factory="does_not_exist:build",
            arguments=(
                ArgumentSpec(
                    name="robot_config", declared_default="rogue", absent_fallback="rogue", description="collision"
                ),
            ),
        )
    )
    # The declaration-stage composition the entry point performs via
    # compose_registry_arguments must reject the collision synchronously.
    with pytest.raises(ArgumentSpecError, match="duplicate launch argument"):
        module.compose_registry_arguments(registry)


def test_live_binding_writes_target_and_disables_sim_clock(business):
    loaded = {"name": "r", "runtime": {"provider": "so101_runtime", "require_model": True}}
    prepared = prepare_launch(
        _context(use_sim="true", sim_platform="sdk"), LaunchBuilderRegistry(), loaded_config=loaded
    )
    assert prepared.live_binding_required is True
    assert prepared.robot_config["runtime"]["target"] == "simulation"
    assert prepared.node_use_sim_time is False
    assert "target" not in loaded["runtime"]


def test_live_binding_requires_provider(business):
    with pytest.raises(PreparationError, match="requires runtime.provider"):
        prepare_launch(_context(), LaunchBuilderRegistry(), loaded_config={"runtime": {"require_model": True}})


@pytest.mark.parametrize("override,expected", [("", False), ("true", True)])
def test_teleop_inference_fallback_and_override(business, override, expected):
    prepared = prepare_launch(
        _context(control_mode="teleop", with_inference=override),
        LaunchBuilderRegistry(),
        loaded_config={"name": "r", "control_modes": {"teleop": {}}},
    )
    assert prepared.with_inference is expected


def test_execution_mode_override_requires_pipeline(business):
    with pytest.raises(PreparationError, match="requires inference_pipeline"):
        prepare_launch(_context(inference_execution_mode="monolithic"), LaunchBuilderRegistry())


@pytest.mark.parametrize(
    "target,use_sim,message",
    [
        # YAML has no runtime.target; the explicit use_sim argument and a
        # runtime_target override must agree in BOTH directions (task 4.3).
        ("hardware", "true", "requires use_sim=false but use_sim=true"),
        ("simulation", "false", "requires use_sim=true but use_sim=false"),
    ],
)
def test_use_sim_runtime_target_mismatch_rejected_in_both_directions(business, target, use_sim, message):
    """Task 4.3 launch-level rejection: an explicit runtime target that
    contradicts the explicit use_sim argument fails synchronously inside
    prepare_launch (phase 0), before any business node generation."""
    with pytest.raises(RuntimeTargetError, match=message):
        prepare_launch(
            _context(runtime_target=target, use_sim=use_sim),
            LaunchBuilderRegistry(),
        )


# ---------------------------------------------------------------------------
# migrate-builtin-builder-metadata: business overrides as registry hooks
# ---------------------------------------------------------------------------


def _builtin_registry():
    from robot_config.launch_registry.builtin_registry import build_builtin_registry

    return build_builtin_registry()


def test_embodied_overrides_match_legacy_preparation():
    """Task 3.5 equivalence anchor: the embodied config_overrides hook
    reproduces the legacy override semantics. (The voice ASR half of the
    anchor moved with the voice_asr entry to voice_asr_service's suite.)"""
    loaded = {
        "name": "r",
        "embodied": {"enabled": False, "perception": {"enabled": False}},
        "teleoperation": {"enabled": False},
    }
    prepared = prepare_launch(
        _context(
            with_embodied="true",
            with_perception="true",
        ),
        _builtin_registry(),
        loaded_config=loaded,
    )
    # Legacy embodied overrides.
    assert prepared.robot_config["embodied"]["enabled"] is True
    assert prepared.robot_config["embodied"]["perception"]["enabled"] is True


def test_builtin_hook_owned_keys_stay_exclusive(monkeypatch):
    """Spec delta: another registered entry declaring an owned key is
    rejected at add time, before any hook runs. (The voice_asr key's
    exclusivity moved with that entry's hook to voice_asr_service's suite;
    the hand-profile keys moved with the hand_sources entry to robot_teleop's
    suite.)"""
    from robot_config.launch_registry.registry import RegistryError

    for key in ("embodied",):
        registry = _builtin_registry()
        module = ModuleType("test_preparation_key_collision")
        module.build = lambda context: BuildResult()
        monkeypatch.setitem(sys.modules, module.__name__, module)
        with pytest.raises(RegistryError, match="config keys already owned"):
            registry.add(
                BuilderSpec(
                    id="rogue",
                    factory=f"{module.__name__}:build",
                    hooks=(HookSpec(phase="config_overrides", factory=f"{module.__name__}:hook", config_keys=(key,)),),
                )
            )


def _preflight_probe_registry(monkeypatch, preflight_hook):
    """Builtin table plus one synthetic preflight-only entry (the former
    hand_sources preflight sample moved to robot_teleop with its entry)."""
    module = ModuleType("preparation_preflight_probe")
    module.preflight_hook = preflight_hook
    monkeypatch.setitem(sys.modules, module.__name__, module)
    registry = _builtin_registry()
    registry.add(
        BuilderSpec(
            id="probe_hand",
            factory=f"{module.__name__}:preflight_hook",
            hooks=(HookSpec(phase="preflight", factory=f"{module.__name__}:preflight_hook", config_keys=("probe",)),),
        )
    )
    return registry


def test_preflight_hook_confirmation_runs_exactly_once(monkeypatch):
    """Spec delta: a preflight hook runs exactly once per launch, before any
    provider startup, and must not rewrite the prepared configuration."""
    confirmations = []

    def observing(config, arguments, target_state):
        confirmations.append(
            (dict(config), {"use_sim": target_state.use_sim, "control_mode": target_state.control_mode})
        )

    prepared = prepare_launch(
        _context(), _preflight_probe_registry(monkeypatch, observing), loaded_config={"name": "test_robot"}
    )
    assert len(confirmations) == 1
    assert confirmations[0][1] == {"use_sim": False, "control_mode": "model_inference"}
    assert confirmations[0][0] == dict(prepared.robot_config)


def test_preflight_hook_failure_fails_the_launch(monkeypatch):
    """A failing preflight hook fails the launch synchronously, before any
    provider factory runs."""

    def refusing(config, **kwargs):
        raise RuntimeError("Interactive startup P-pose requires a terminal.")

    registry = _preflight_probe_registry(monkeypatch, refusing)
    with pytest.raises(RuntimeError, match=r"preparation hook 'probe_hand' \(preflight\) failed"):
        prepare_launch(_context(), registry, loaded_config={"name": "test_robot"})


def test_legacy_two_parameter_hook_convention_stays_compatible(monkeypatch):
    """Design decision 4 option (b): hooks that only accept
    (robot_config, arguments) keep working when a target state is
    available; a three-parameter hook receives it."""
    observed = []

    def two_param_hook(config, arguments):
        observed.append(("two", config.get("feature")))

    def three_param_hook(config, arguments, target_state):
        observed.append(("three", target_state.use_sim, target_state.control_mode))

    module = ModuleType("test_preparation_hook_arity")
    module.hook_two = two_param_hook
    module.hook_three = three_param_hook
    module.build = lambda context: BuildResult()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    registry = LaunchBuilderRegistry()
    registry.add(
        BuilderSpec(
            id="feature_two",
            factory=f"{module.__name__}:build",
            hooks=(
                HookSpec(phase="config_overrides", factory=f"{module.__name__}:hook_two", config_keys=("feature",)),
            ),
        )
    )
    registry.add(
        BuilderSpec(
            id="feature_three",
            factory=f"{module.__name__}:build",
            hooks=(
                HookSpec(phase="config_overrides", factory=f"{module.__name__}:hook_three", config_keys=("other",)),
            ),
        )
    )
    prepare_launch(_context(control_mode="teleop"), registry, loaded_config={"name": "r"})
    assert ("two", None) in observed
    assert ("three", False, "teleop") in observed
