"""Voice ASR launch-preparation semantics (voice_asr_service's contribution).

Migrated from ``robot_config/test/test_launch_preparation.py`` and
``robot_config/test/test_launch_options.py`` (openspec change
``migrate-business-launch-builders``, task B.4): the config_overrides hook,
its business arguments and the ``voice_asr`` config-key ownership moved with
the ``voice_asr`` entry to this package, so their preparation-level tests
follow. Assertions are verbatim where the behavior is unchanged; the registry
under test now holds this package's entry alone, which is exactly the
ownership the migration establishes. The hook's import lightness (the old
"never imports the optional voice_asr_service package" assertion) is pinned
by ``test_launch_builder_import_budget.py``.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from robot_config.launch_builders.preparation import prepare_launch
from robot_config.launch_builders.registry import (
    BuilderSpec,
    BuildResult,
    HookSpec,
    LaunchBuilderRegistry,
    RegistryError,
)
from voice_asr_service.launch_builders.voice_asr import VOICE_ASR_BUILDER


def _context(**arguments):
    return SimpleNamespace(launch_configurations=arguments)


def _voice_asr_registry() -> LaunchBuilderRegistry:
    registry = LaunchBuilderRegistry()
    registry.add(VOICE_ASR_BUILDER)
    return registry


def test_voice_asr_overrides_match_legacy_preparation():
    """Task 3.5 equivalence anchor (voice ASR half): the config_overrides
    hook reproduces the legacy override semantics."""
    loaded = {
        "name": "r",
        "voice_asr": {"enabled": False, "active_mode": "manual", "realtime_pre_roll_seconds": 0.5},
        "teleoperation": {"enabled": False},
    }
    prepared = prepare_launch(
        _context(
            voice_asr_auto_start="true",
            voice_asr_realtime_pre_roll_seconds="0.8",
        ),
        _voice_asr_registry(),
        loaded_config=loaded,
    )
    # Legacy voice_start: auto-start forces enabled + continuous.
    assert prepared.robot_config["voice_asr"]["enabled"] is True
    assert prepared.robot_config["voice_asr"]["active_mode"] == "continuous"
    # Legacy voice_override: the explicit pre-roll override applies.
    assert prepared.robot_config["voice_asr"]["realtime_pre_roll_seconds"] == 0.8


def test_voice_asr_explicit_false_override_disables_yaml_enabled():
    loaded = {
        "name": "r",
        "voice_asr": {"enabled": True, "active_mode": "continuous", "realtime_pre_roll_seconds": 0.5},
        "teleoperation": {"enabled": False},
    }
    prepared = prepare_launch(_context(voice_asr_auto_start="false"), _voice_asr_registry(), loaded_config=loaded)
    assert prepared.robot_config["voice_asr"]["enabled"] is False


def test_voice_asr_hook_runs_when_entry_is_mock_skipped():
    """The config_overrides hook runs for every registered entry whether or
    not it participates: a mock-skipped voice_asr still observes the CLI
    override (the hook's import lightness is pinned by
    ``test_launch_builder_import_budget.py``)."""
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
        _voice_asr_registry(),
        loaded_config=loaded,
    )
    plan_ids = [spec.id for spec in prepared.plan.specs]
    assert "voice_asr" not in plan_ids  # mock skip filtered the entry...
    assert prepared.robot_config["voice_asr"]["realtime_pre_roll_seconds"] == 0.8  # ...hook still ran


def test_voice_asr_config_key_is_owned_exclusively(monkeypatch):
    """Spec delta (migrated from robot_config's builtin-hook coverage):
    another registered entry declaring the ``voice_asr`` config key is
    rejected at add time, before any hook runs."""
    module = ModuleType("test_voice_asr_key_collision")
    module.build = lambda context: BuildResult()
    module.hook = lambda config, arguments: None
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(RegistryError, match="config keys already owned"):
        _voice_asr_registry().add(
            BuilderSpec(
                id="rogue",
                factory=f"{module.__name__}:build",
                hooks=(
                    HookSpec(phase="config_overrides", factory=f"{module.__name__}:hook", config_keys=("voice_asr",)),
                ),
            )
        )


def test_voice_asr_business_arguments_are_owned_by_this_entry():
    """The two voice ASR business parameters are contributed by exactly
    this entry, with the legacy declared defaults and absent fallbacks
    (migrated from robot_config's launch-options suite)."""
    specs = {spec.name: spec for spec in VOICE_ASR_BUILDER.arguments}
    assert set(specs) == {"voice_asr_auto_start", "voice_asr_realtime_pre_roll_seconds"}
    # launch_setup read sites: .get("voice_asr_auto_start", "false") and
    # .get("voice_asr_realtime_pre_roll_seconds", "") — an absent key must
    # NOT enable/disable the section; the empty-string rule keeps YAML
    # voice_asr.enabled for the direct embodied call path.
    assert specs["voice_asr_auto_start"].declared_default == "false"
    assert specs["voice_asr_auto_start"].absent_fallback == ""
    assert specs["voice_asr_realtime_pre_roll_seconds"].declared_default == ""
    assert specs["voice_asr_realtime_pre_roll_seconds"].absent_fallback == ""
