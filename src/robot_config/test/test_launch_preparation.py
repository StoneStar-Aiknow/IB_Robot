"""Preparation behavior, extension ownership and pre-provider validation."""

from __future__ import annotations

import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest

from robot_config.launch_builders import control, hand_sources, readiness
from robot_config.launch_builders.launch_options import ArgumentSpec, ArgumentSpecError
from robot_config.launch_builders.preparation import (
    PreparationError,
    _run_phase_hooks,
    compose_registry_arguments,
    prepare_launch,
)
from robot_config.launch_builders.registry import (
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
    calls = []
    monkeypatch.setattr(hand_sources, "apply_hand_profile", lambda config, profile: calls.append(("hand", profile)))
    monkeypatch.setattr(
        control, "validate_runtime_resources", lambda config, **kwargs: calls.append(("resources", config))
    )
    monkeypatch.setattr(
        hand_sources,
        "confirm_interactive_startup_p_pose",
        lambda config, **kwargs: calls.append(("confirm", config, kwargs)),
    )
    monkeypatch.setattr(readiness, "resolve_config_path", lambda *args: "/tmp/preparation-test.yaml")
    monkeypatch.setattr(readiness, "load_launch_robot_config", lambda *args, **kwargs: {"name": "test_robot"})
    return calls


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
    """Through the REAL builtin registry: the caller's loaded config is
    never mutated, core arguments are normalized, and the hand_sources
    config_overrides hook applies the strip().lower() normalization the
    preparation core used to perform (design decision 4: verbatim
    migration)."""
    from robot_config.launch_builders.builtin_registry import build_builtin_registry

    loaded = {"name": "test_robot"}
    before = copy.deepcopy(loaded)
    prepared = prepare_launch(
        _context(use_sim="true", sim_platform=" Mock ", hand_profile=" RIGHT ", nav_stage=" grasp "),
        build_builtin_registry(),
        loaded_config=loaded,
    )
    assert loaded == before
    assert prepared.arguments["sim_platform"] == "mock"
    assert prepared.arguments["nav_stage"] == "grasp"
    assert prepared.robot_config["simulation"]["platform"] == "mock"
    assert prepared.context.auto_start_controllers == "false"
    assert prepared.context.sim_backend_needs_ros2_control is False
    # hand hook -> resource check -> preflight P-pose hook, exactly once each.
    assert [item[0] for item in business] == ["hand", "resources", "confirm"]
    # The hook normalized the raw CLI value before applying the profile.
    assert business[0] == ("hand", "right")
    # The preflight P-pose hook receives the prepared target state and an
    # isolated copy of the real snapshot (never the shared object).
    assert business[-1][2] == {"use_sim": True, "control_mode": "model_inference"}
    assert business[-1][1] is not prepared.robot_config
    assert business[-1][1] == prepared.robot_config


def test_direct_call_fallbacks(business):
    """Direct launch_setup callers observe the same absent fallbacks as
    before the business-parameter migration — through the builtin
    registry that contributes them."""
    from robot_config.launch_builders.builtin_registry import build_builtin_registry

    prepared = prepare_launch(_context(), build_builtin_registry())
    assert prepared.arguments["robot_config"] == "test_cam"
    assert prepared.arguments["moveit_display"] == "false"
    assert prepared.arguments["voice_asr_auto_start"] == ""


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
    """Task 4.5: patching the REAL launch entry's build_builtin_registry to
    an invalid builtin registry (all control/provider entries kept, one
    rogue spec with an unknown dependency) must fail synchronously inside
    launch_setup with RegistryError and ZERO calls to BOTH provider seams —
    generate_runtime_provider_actions and generate_bound_runtime_actions."""
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
    original_build = module.build_builtin_registry

    def build_invalid_builtin_registry():
        registry = original_build()
        registry.add(BuilderSpec(id="rogue_extension", factory="does_not_exist:build", after=("missing",)))
        return registry

    monkeypatch.setattr(module, "build_builtin_registry", build_invalid_builtin_registry)

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
    """A contributed argument reusing a builtin business name (record is
    owned by the recording entry now, not the core table) is rejected at
    add time against the builtin contributions."""
    from robot_config.launch_builders.builtin_registry import build_builtin_registry
    from robot_config.launch_builders.registry import RegistryError

    module = ModuleType("test_preparation_record_collision")
    module.build = lambda context: BuildResult()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    registry = build_builtin_registry()
    with pytest.raises(
        RegistryError, match="cannot contribute launch argument 'record'; already contributed by 'recording'"
    ):
        registry.add(
            BuilderSpec(
                id="rogue",
                factory=f"{module.__name__}:build",
                arguments=(
                    ArgumentSpec(name="record", declared_default="x", absent_fallback="x", description="collision"),
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
    from robot_config.launch_builders.builtin_registry import build_builtin_registry

    return build_builtin_registry()


def _hand_profile_loaded_config() -> dict:
    """A hand-profile selection config mirroring aero_hand_teleop.yaml."""
    return {
        "name": "hand_profile_robot",
        "hand_profiles": {
            "default_profile": "right",
            "profiles": {
                "right": {
                    "sides": ["right"],
                    "hand_source": "mhandpro",
                    "active_actuators": ["aux_right"],
                    "active_devices": ["glove_right"],
                },
                "dual": {
                    "sides": ["left", "right"],
                    "hand_source": "mhandpro",
                    "active_actuators": ["aux_left", "aux_right"],
                    "active_devices": ["glove_left", "glove_right"],
                },
            },
        },
        "hand_sources": {"mhandpro": {"type": "mhandpro", "enabled": True, "mock": True, "sides": ["right"]}},
        "auxiliary_actuators": {
            "aux_left": {"profile_managed": True, "enabled": False, "joint_names": ["left_j1"]},
            "aux_right": {"profile_managed": True, "enabled": False, "joint_names": ["right_j1"]},
        },
        "teleoperation": {
            "enabled": True,
            "active_devices": ["glove_right"],
            "devices": [{"name": "glove_left"}, {"name": "glove_right"}],
        },
        "joints": {"arm": [], "hand": []},
    }


def test_hand_profile_hook_result_matches_legacy_preparation(monkeypatch):
    """Task 3.5 equivalence anchor: through the REAL builtin registry the
    hand_sources config_overrides hook produces exactly the config the
    legacy core step produced (design decision 4)."""
    resource_configs = []
    monkeypatch.setattr(control, "validate_runtime_resources", lambda config, **kwargs: resource_configs.append(config))
    loaded = _hand_profile_loaded_config()
    before = copy.deepcopy(loaded)
    prepared = prepare_launch(_context(hand_profile=" DUAL "), _builtin_registry(), loaded_config=loaded)
    # The caller's loaded config is never mutated.
    assert loaded == before
    # Verbatim hand-profile semantics: normalized selection, sides,
    # profile-managed actuator enables, active teleop devices, hand joints.
    assert prepared.robot_config["active_hand_profile"] == "dual"
    assert prepared.robot_config["hand_sources"]["mhandpro"]["sides"] == ["left", "right"]
    assert prepared.robot_config["auxiliary_actuators"]["aux_left"]["enabled"] is True
    assert prepared.robot_config["auxiliary_actuators"]["aux_right"]["enabled"] is True
    assert prepared.robot_config["teleoperation"]["active_devices"] == ["glove_left", "glove_right"]
    assert prepared.robot_config["joints"]["hand"] == ["left_j1", "right_j1"]
    # resource_check runs AFTER the hook and observes the applied profile.
    assert len(resource_configs) == 1
    assert resource_configs[0]["active_hand_profile"] == "dual"
    # The hand_profiles INPUT section is consumed, never rewritten.
    assert prepared.robot_config["hand_profiles"] == before["hand_profiles"]


def test_default_hand_profile_applies_without_cli_argument(monkeypatch):
    monkeypatch.setattr(control, "validate_runtime_resources", lambda config, **kwargs: None)
    prepared = prepare_launch(_context(), _builtin_registry(), loaded_config=_hand_profile_loaded_config())
    assert prepared.robot_config["active_hand_profile"] == "right"
    assert prepared.robot_config["auxiliary_actuators"]["aux_left"]["enabled"] is False
    assert prepared.robot_config["auxiliary_actuators"]["aux_right"]["enabled"] is True


def test_voice_and_embodied_overrides_match_legacy_preparation():
    """Task 3.5 equivalence anchor: the voice ASR and embodied
    config_overrides hooks reproduce the legacy override semantics."""
    loaded = {
        "name": "r",
        "voice_asr": {"enabled": False, "active_mode": "manual", "realtime_pre_roll_seconds": 0.5},
        "embodied": {"enabled": False, "perception": {"enabled": False}},
        "teleoperation": {"enabled": False},
    }
    prepared = prepare_launch(
        _context(
            voice_asr_auto_start="true",
            voice_asr_realtime_pre_roll_seconds="0.8",
            with_embodied="true",
            with_perception="true",
        ),
        _builtin_registry(),
        loaded_config=loaded,
    )
    # Legacy voice_start: auto-start forces enabled + continuous.
    assert prepared.robot_config["voice_asr"]["enabled"] is True
    assert prepared.robot_config["voice_asr"]["active_mode"] == "continuous"
    # Legacy voice_override: the explicit pre-roll override applies.
    assert prepared.robot_config["voice_asr"]["realtime_pre_roll_seconds"] == 0.8
    # Legacy embodied overrides.
    assert prepared.robot_config["embodied"]["enabled"] is True
    assert prepared.robot_config["embodied"]["perception"]["enabled"] is True


def test_voice_asr_explicit_false_override_disables_yaml_enabled():
    loaded = {
        "name": "r",
        "voice_asr": {"enabled": True, "active_mode": "continuous", "realtime_pre_roll_seconds": 0.5},
        "teleoperation": {"enabled": False},
    }
    prepared = prepare_launch(_context(voice_asr_auto_start="false"), _builtin_registry(), loaded_config=loaded)
    assert prepared.robot_config["voice_asr"]["enabled"] is False


def test_voice_asr_hook_runs_when_entry_is_mock_skipped_without_optional_imports(monkeypatch):
    """The config_overrides hook runs for every registered entry whether or
    not it participates: a mock-skipped voice_asr still observes the CLI
    override, and the hook never imports the optional voice_asr_service
    package (design decision 4)."""
    sys.modules.pop("voice_asr_service", None)
    loaded = {
        "name": "r",
        "voice_asr": {"enabled": False, "realtime_pre_roll_seconds": 0.5},
        "teleoperation": {"enabled": False},
    }
    prepared = prepare_launch(
        _context(
            use_sim="true",
            sim_platform="mock",
            control_mode="model_inference",
            voice_asr_realtime_pre_roll_seconds="0.8",
        ),
        _builtin_registry(),
        loaded_config=loaded,
    )
    plan_ids = [spec.id for spec in prepared.plan.specs]
    assert "voice_asr" not in plan_ids  # mock skip filtered the entry...
    assert prepared.robot_config["voice_asr"]["realtime_pre_roll_seconds"] == 0.8  # ...hook still ran
    assert "voice_asr_service" not in sys.modules


def test_builtin_hook_owned_keys_stay_exclusive(monkeypatch):
    """Spec delta: another registered entry declaring voice_asr, embodied
    or a hand-profile key is rejected at add time, before any hook runs."""
    from robot_config.launch_builders.registry import RegistryError

    for key in (
        "voice_asr",
        "embodied",
        "hand_sources",
        "auxiliary_actuators",
        "teleoperation",
        "joints",
        "active_hand_profile",
    ):
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


def test_preflight_p_pose_confirmation_runs_exactly_once(monkeypatch):
    """Spec delta: the P-pose confirmation runs exactly once per launch
    through the hand_sources preflight hook, before any provider startup."""
    confirmations = []
    monkeypatch.setattr(
        hand_sources,
        "confirm_interactive_startup_p_pose",
        lambda config, **kwargs: confirmations.append((config["active_hand_profile"], kwargs)),
    )
    prepared = prepare_launch(_context(), _builtin_registry(), loaded_config=_hand_profile_loaded_config())
    assert len(confirmations) == 1
    assert confirmations[0][0] == "right"
    assert confirmations[0][1] == {"use_sim": False, "control_mode": "model_inference"}
    # The preflight hook must not rewrite the prepared configuration.
    assert prepared.robot_config["active_hand_profile"] == "right"


def test_preflight_p_pose_failure_fails_the_launch(monkeypatch):
    """Cancel/EOF during the interactive confirmation fails the launch
    synchronously, before any provider factory runs."""

    def refusing(config, **kwargs):
        raise RuntimeError("Interactive startup P-pose requires a terminal.")

    monkeypatch.setattr(hand_sources, "confirm_interactive_startup_p_pose", refusing)
    with pytest.raises(RuntimeError, match=r"preparation hook 'hand_sources' \(preflight\) failed"):
        prepare_launch(_context(), _builtin_registry(), loaded_config=_hand_profile_loaded_config())


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
