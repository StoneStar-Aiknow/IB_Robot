"""Tests for the dual entry-point launch argument model.

The full argument set is compared parameter-by-parameter against the
legacy ``robot.launch.py`` declarations (declared_default) and the legacy
``launch_setup`` reads (absent_fallback), so any drift introduced by the
refactor is caught here instead of at board bring-up. Since openspec
change ``migrate-builtin-builder-metadata`` the core table keeps only
the 14 infrastructure parameters; the 11 business parameters come from
their owning BuilderSpec declarations, and the composed (core +
registry-contributed) set must equal the legacy table exactly.
"""

from __future__ import annotations

import pytest
from launch import LaunchContext

from robot_config.launch_builders.builtin_registry import build_builtin_registry
from robot_config.launch_builders.launch_options import (
    CORE_ARGUMENT_SPECS,
    ArgumentSpec,
    ArgumentSpecError,
    compose_argument_specs,
    declare_launch_arguments,
    read_launch_argument,
    read_launch_arguments,
)

# Frozen from robot.launch.py at 9e30e614; never read the refactored table as its own oracle.
LEGACY_DECLARED_DEFAULTS = {
    "robot_config": "so101_single_arm",
    "config_path": "",
    "use_sim": "",
    "sim_platform": "",
    "auto_start_controllers": "true",
    "control_mode": "",
    "nav_stage": "",
    "hand_profile": "",
    "voice_asr_realtime_pre_roll_seconds": "",
    "with_inference": "",
    "inference_pipeline": "",
    "inference_execution_mode": "",
    "with_moveit": "",
    "with_navigation": "",
    "navigation_mode": "",
    "moveit_display": "true",
    "record": "false",
    "record_mode": "continuous",
    "voice_asr_auto_start": "false",
    "with_embodied": "",
    "with_perception": "",
    "record_visualizer": "none",
    "runtime_target": "",
    "enable_tracing": "false",
    "trace_session_name": "ib_robot_trace",
}

# The core table after migrate-builtin-builder-metadata: infrastructure and
# preparation-core parameters only (design decision 3).
CORE_ARGUMENT_NAMES = (
    "robot_config",
    "config_path",
    "use_sim",
    "sim_platform",
    "auto_start_controllers",
    "control_mode",
    "nav_stage",
    "with_inference",
    "inference_pipeline",
    "inference_execution_mode",
    "moveit_display",
    "runtime_target",
    "enable_tracing",
    "trace_session_name",
)

# Business parameters and their owning builder entries (design decision 3),
# in builtin registration order (hand_sources -> voice_asr -> navigation ->
# motion_planning -> embodied -> recording -> rerun_visualizer).
BUSINESS_ARGUMENT_OWNERS = {
    "hand_profile": "hand_sources",
    "voice_asr_auto_start": "voice_asr",
    "voice_asr_realtime_pre_roll_seconds": "voice_asr",
    "with_navigation": "navigation",
    "navigation_mode": "navigation",
    "with_moveit": "motion_planning",
    "with_embodied": "embodied",
    "with_perception": "embodied",
    "record": "recording",
    "record_mode": "recording",
    "record_visualizer": "rerun_visualizer",
}


def _text(substitutions):
    return "".join(item.text if hasattr(item, "text") else str(item) for item in substitutions)


def _spec_table() -> dict[str, ArgumentSpec]:
    return {spec.name: spec for spec in CORE_ARGUMENT_SPECS}


def _composed_table() -> dict[str, ArgumentSpec]:
    """Core table plus the builtin registry's business contributions."""
    registry = build_builtin_registry()
    specs = compose_argument_specs(registry.contributed_arguments())
    return {spec.name: spec for spec in specs}


class TestCoreTableMatchesLegacyDeclarations:
    def test_core_table_keeps_only_infrastructure_parameters(self):
        table = _spec_table()
        assert tuple(table) == CORE_ARGUMENT_NAMES

    def test_composed_argument_set_matches_legacy_exactly(self):
        """Core + builtin business contributions must equal the legacy
        table: same name set, same declared_default, same absent_fallback
        (the declaration ORDER differs — core first, then business in
        registration order — and is the only allowed difference)."""
        table = _composed_table()
        assert set(table) == set(LEGACY_DECLARED_DEFAULTS), (
            f"missing={sorted(set(LEGACY_DECLARED_DEFAULTS) - set(table))} "
            f"extra={sorted(set(table) - set(LEGACY_DECLARED_DEFAULTS))}"
        )
        for name, expected_default in LEGACY_DECLARED_DEFAULTS.items():
            assert table[name].declared_default == expected_default, name

    @pytest.mark.parametrize(
        "name,expected_default",
        [
            ("robot_config", "so101_single_arm"),
            ("config_path", ""),
            ("use_sim", ""),
            ("sim_platform", ""),
            ("auto_start_controllers", "true"),
            ("control_mode", ""),
            ("nav_stage", ""),
            ("with_inference", ""),
            ("inference_pipeline", ""),
            ("inference_execution_mode", ""),
            ("moveit_display", "true"),
            ("runtime_target", ""),
            ("enable_tracing", "false"),
        ],
    )
    def test_declared_defaults_match_legacy(self, name, expected_default):
        spec = _spec_table()[name]
        assert spec.declared_default == expected_default

    def test_trace_session_name_declared_default_matches_legacy(self):
        spec = _spec_table()["trace_session_name"]
        assert spec.declared_default == LEGACY_DECLARED_DEFAULTS["trace_session_name"]

    @pytest.mark.parametrize("name,owner", sorted(BUSINESS_ARGUMENT_OWNERS.items()))
    def test_business_argument_is_owned_by_its_builder(self, name, owner):
        """Each business parameter is contributed by exactly one builtin
        spec — its owning entry — and is absent from the core table."""
        assert name not in _spec_table()
        owners = [spec.id for spec in build_builtin_registry().snapshot() for arg in spec.arguments if arg.name == name]
        assert owners == [owner]

    def test_business_arguments_follow_registration_order(self):
        """The composed order is the core table followed by the business
        contributions in builtin registration order (the only allowed
        observable difference to the legacy declaration order)."""
        registry = build_builtin_registry()
        specs = compose_argument_specs(registry.contributed_arguments())
        assert [spec.name for spec in specs[: len(CORE_ARGUMENT_NAMES)]] == list(CORE_ARGUMENT_NAMES)
        assert [spec.name for spec in specs[len(CORE_ARGUMENT_NAMES) :]] == list(BUSINESS_ARGUMENT_OWNERS)


class TestAbsentFallbacks:
    """absent_fallback comes from each legacy launch_setup read site."""

    def test_robot_config_falls_back_to_test_cam(self):
        # launch_setup: context.launch_configurations.get("robot_config", "test_cam")
        assert _spec_table()["robot_config"].absent_fallback == "test_cam"

    def test_moveit_display_falls_back_to_false(self):
        # launch_setup reads moveit_display with default "false" at each
        # call site (both provider display and controller paths).
        assert _spec_table()["moveit_display"].absent_fallback == "false"

    def test_voice_asr_auto_start_absent_falls_back_to_empty(self):
        # launch_setup: .get("voice_asr_auto_start", "") — an absent key
        # must NOT enable/disable the section; the empty-string rule keeps
        # YAML voice_asr.enabled for the direct embodied call path.
        assert _composed_table()["voice_asr_auto_start"].absent_fallback == ""

    @pytest.mark.parametrize(
        "name",
        [
            "auto_start_controllers",
            "record",
            "record_mode",
            "record_visualizer",
            "enable_tracing",
        ],
    )
    def test_hardcoded_read_defaults_are_absent_fallbacks(self, name):
        # launch_setup reads these with hardcoded defaults ("true",
        # "false", "continuous", "none", "false").
        table = _composed_table()
        expected = {
            "auto_start_controllers": "true",
            "record": "false",
            "record_mode": "continuous",
            "record_visualizer": "none",
            "enable_tracing": "false",
        }[name]
        assert table[name].absent_fallback == expected

    @pytest.mark.parametrize(
        "name",
        [
            "config_path",
            "use_sim",
            "sim_platform",
            "control_mode",
            "nav_stage",
            "hand_profile",
            "voice_asr_realtime_pre_roll_seconds",
            "with_inference",
            "inference_pipeline",
            "inference_execution_mode",
            "with_moveit",
            "with_navigation",
            "navigation_mode",
            "with_embodied",
            "with_perception",
            "runtime_target",
        ],
    )
    def test_empty_string_read_defaults_stay_empty(self, name):
        assert _composed_table()[name].absent_fallback == ""


class TestReadSemantics:
    def test_missing_key_yields_absent_fallback(self):
        spec = _spec_table()["robot_config"]
        assert read_launch_argument({}, spec) == "test_cam"

    def test_present_empty_string_is_returned_verbatim(self):
        spec = _composed_table()["voice_asr_auto_start"]
        assert read_launch_argument({"voice_asr_auto_start": ""}, spec) == ""

    def test_present_value_wins_over_both_defaults(self):
        spec = _spec_table()["robot_config"]
        assert read_launch_argument({"robot_config": "lekiwi"}, spec) == "lekiwi"

    def test_read_launch_arguments_covers_core_table(self):
        context = LaunchContext()
        context.launch_configurations.update({"robot_config": "lekiwi", "use_sim": "true"})
        values = read_launch_arguments(context.launch_configurations)
        assert values["robot_config"] == "lekiwi"
        assert values["use_sim"] == "true"
        assert values["moveit_display"] == "false"  # absent fallback
        assert values["config_path"] == ""

    def test_read_launch_arguments_covers_composed_table(self):
        """Reading with the builtin composed specs yields every legacy
        argument, business ones included, with the same fallback rules."""
        registry = build_builtin_registry()
        specs = compose_argument_specs(registry.contributed_arguments())
        values = read_launch_arguments({}, specs)
        assert set(values) == set(LEGACY_DECLARED_DEFAULTS)
        assert values["hand_profile"] == ""
        assert values["record"] == "false"
        assert values["record_visualizer"] == "none"


class TestComposition:
    def test_business_spec_is_appended_after_core(self):
        business = ArgumentSpec(
            name="business_knob",
            declared_default="off",
            absent_fallback="",
            description="A business knob.",
        )
        specs = compose_argument_specs([business])
        assert specs[-1] is business
        assert len(specs) == len(CORE_ARGUMENT_SPECS) + 1

    def test_business_spec_reusing_core_name_is_rejected(self):
        business = ArgumentSpec(
            name="robot_config",
            declared_default="rogue",
            absent_fallback="rogue",
            description="Collision.",
        )
        with pytest.raises(ArgumentSpecError, match="duplicate launch argument name 'robot_config'"):
            compose_argument_specs([business])

    def test_two_business_specs_with_same_name_are_rejected(self):
        first = ArgumentSpec(name="knob", declared_default="a", absent_fallback="", description="1")
        second = ArgumentSpec(name="knob", declared_default="b", absent_fallback="", description="2")
        with pytest.raises(ArgumentSpecError, match="duplicate launch argument name 'knob'"):
            compose_argument_specs([first, second])

    def test_non_spec_entry_is_rejected(self):
        with pytest.raises(ArgumentSpecError, match="ArgumentSpec instances"):
            compose_argument_specs([("name", "default")])


class TestDeclaration:
    def test_declare_launch_arguments_uses_declared_defaults(self):
        registry = build_builtin_registry()
        specs = compose_argument_specs(registry.contributed_arguments())
        declarations = declare_launch_arguments(specs)
        table = _composed_table()
        by_name = {_text(item._DeclareLaunchArgument__name): item for item in declarations}
        assert set(by_name) == set(table)
        for name, declaration in by_name.items():
            assert _text(declaration._DeclareLaunchArgument__default_value) == table[name].declared_default

    def test_declaration_does_not_load_business_run_dependencies(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name.startswith(("voice_asr_service", "lerobot")):
                raise AssertionError(f"argument declaration must not import {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        assert declare_launch_arguments()
