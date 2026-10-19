"""Entry-point extension tests for ``build_launch_registry`` (Phase 2, task 2.3).

Openspec change ``decentralize-launch-builder-registration``, design
decision 3: installed packages contribute ``BuilderSpec`` entries through the
entry point group ``ibrobot.launch_builders``. These tests inject
:class:`importlib.metadata.EntryPoint` objects through the ``entry_points``
seam, so no package has to be installed to exercise the whole discovery
contract:

- extensions append after the builtin table, deterministically sorted by
  entry point name (then value) regardless of the order they were declared in;
- an extension may contribute launch arguments and prepare hooks, and may
  reference builtin or other extension entries through ``after``/``requires``
  with the same topological rules as builtins;
- every extension error (duplicate entry point name, import failure, missing
  attribute, non-BuilderSpec resolution, id/argument/hook-key conflicts with
  existing entries, ``provides`` declarations) is a :class:`RegistryError`
  naming the entry point and its value, raised while building the registry —
  before any build factory or hook is resolved or executed;
- extension factories stay ``module:attribute`` STRINGS resolved lazily on
  every call (the source-module monkeypatch seam), and an entry that does not
  participate never resolves its factory, so its run dependency module is
  never imported.

The real mirror_session migration (design decision 4) is covered end to end
in ``manipulation_execution`` (its own package tests); these tests pin the
generic machinery only.
"""

from __future__ import annotations

import sys
import types
from importlib.metadata import EntryPoint

import pytest

from robot_config.launch_builders.builtin_registry import (
    LAUNCH_BUILDER_ENTRY_POINT_GROUP,
    build_builtin_registry,
    build_launch_registry,
)
from robot_config.launch_builders.launch_options import ArgumentSpec, compose_argument_specs
from robot_config.launch_builders.registry import (
    BuilderSpec,
    BuildResult,
    HookSpec,
    LaunchBuildContext,
    RegistryError,
    execute_plan,
    resolve_builder_factory,
)


def _entry_point(name: str, value: str) -> EntryPoint:
    return EntryPoint(name=name, value=value, group=LAUNCH_BUILDER_ENTRY_POINT_GROUP)


def _install_module(monkeypatch, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _context(**overrides) -> LaunchBuildContext:
    defaults = {"robot_config": {"name": "test_robot"}}
    defaults.update(overrides)
    return LaunchBuildContext(**defaults)


class TestExtensionComposition:
    def test_extensions_append_after_builtins_in_entry_point_name_order(self, monkeypatch):
        """Builtin order is unchanged; extensions follow, sorted by entry
        point name (then value), no matter the order they were declared in."""
        for name in ("ext_zulu_probe", "ext_alpha_probe", "ext_mike_probe"):
            module = _install_module(monkeypatch, name)
            module.SPEC = BuilderSpec(id=name, factory=f"{name}:build")

        registry = build_launch_registry(
            entry_points=[
                _entry_point("zulu", "ext_zulu_probe:SPEC"),
                _entry_point("alpha", "ext_alpha_probe:SPEC"),
                _entry_point("mike", "ext_mike_probe:SPEC"),
            ]
        )

        ids = [spec.id for spec in registry.snapshot()]
        builtin_ids = [spec.id for spec in build_builtin_registry().snapshot()]
        assert ids == builtin_ids + ["ext_alpha_probe", "ext_mike_probe", "ext_zulu_probe"]

    def test_composition_is_deterministic_across_builds(self, monkeypatch):
        module = _install_module(monkeypatch, "ext_determinism_probe")
        module.SPEC = BuilderSpec(id="ext_determinism_probe", factory="ext_determinism_probe:build")

        entry_points = [
            _entry_point("second", "ext_determinism_probe:SPEC"),
            _entry_point("first", "ext_determinism_probe:SPEC2"),
        ]
        module.SPEC2 = BuilderSpec(id="ext_determinism_probe_alt", factory="ext_determinism_probe:build_alt")

        first_ids = [spec.id for spec in build_launch_registry(entry_points=entry_points).snapshot()]
        second_ids = [spec.id for spec in build_launch_registry(entry_points=entry_points).snapshot()]
        assert first_ids == second_ids
        assert first_ids[-2:] == ["ext_determinism_probe_alt", "ext_determinism_probe"]

    def test_entry_point_may_resolve_to_a_sequence_of_specs(self, monkeypatch):
        module = _install_module(monkeypatch, "ext_sequence_probe")
        module.SPECS = (
            BuilderSpec(id="ext_sequence_one", factory="ext_sequence_probe:build_one"),
            BuilderSpec(id="ext_sequence_two", factory="ext_sequence_probe:build_two"),
        )

        registry = build_launch_registry(entry_points=[_entry_point("seq", "ext_sequence_probe:SPECS")])
        assert [spec.id for spec in registry.snapshot()][-2:] == ["ext_sequence_one", "ext_sequence_two"]


class TestExtensionContributions:
    def test_extension_contributes_arguments_and_hooks_like_a_builtin(self, monkeypatch):
        """One entry point registration is all an extension needs to add a
        CLI argument and a prepare hook: both merge through the same
        aggregation the preparation pipeline uses, appended after the
        builtin contributions."""
        module = _install_module(monkeypatch, "ext_contrib_probe")
        module.SPEC = BuilderSpec(
            id="ext_contrib_probe",
            factory="ext_contrib_probe:build",
            arguments=(
                ArgumentSpec(
                    name="ext_contrib_scale",
                    declared_default="1.0",
                    absent_fallback="1.0",
                    description="Extension contribution probe scale",
                ),
            ),
            hooks=(
                HookSpec(
                    phase="config_overrides",
                    factory="ext_contrib_probe:override_hook",
                    config_keys=("ext_contrib",),
                ),
            ),
        )

        registry = build_launch_registry(entry_points=[_entry_point("contrib", "ext_contrib_probe:SPEC")])

        builtin_arguments = [argument.name for argument in build_builtin_registry().contributed_arguments()]
        assert [argument.name for argument in registry.contributed_arguments()] == builtin_arguments + [
            "ext_contrib_scale"
        ]
        composed = compose_argument_specs(registry.contributed_arguments())
        assert composed[-1].name == "ext_contrib_scale"

        config_overrides = dict(registry.hooks_by_phase())["config_overrides"]
        assert config_overrides[-1] == ("ext_contrib_probe", module.SPEC.hooks[0])

    def test_extension_after_and_requires_follow_builtin_graph_rules(self, monkeypatch):
        """``after``/``requires`` reference builtins and other extensions
        with unchanged semantics: ordering edges, hard participation checks
        and cycle rejection all run on the combined registry."""
        first = _install_module(monkeypatch, "ext_edge_first")
        first.SPEC = BuilderSpec(id="ext_edge_first", factory="ext_edge_first:build")
        second = _install_module(monkeypatch, "ext_edge_second")
        second.SPEC = BuilderSpec(
            id="ext_edge_second",
            factory="ext_edge_second:build",
            after=("control", "ext_edge_first"),
        )
        registry = build_launch_registry(
            entry_points=[
                _entry_point("b", "ext_edge_second:SPEC"),
                _entry_point("a", "ext_edge_first:SPEC"),
            ]
        )
        plan_ids = [spec.id for spec in registry.plan(_context()).specs]
        assert plan_ids.index("ext_edge_first") < plan_ids.index("ext_edge_second")
        assert plan_ids.index("control") < plan_ids.index("ext_edge_second")

        # requires is a hard participation dependency: a disabled target is
        # an error even when the referencing entry itself participates.
        third = _install_module(monkeypatch, "ext_edge_third")
        third.SPEC = BuilderSpec(
            id="ext_edge_third",
            factory="ext_edge_third:build",
            requires=("ext_edge_first",),
        )
        first.SPEC = BuilderSpec(
            id="ext_edge_first",
            factory="ext_edge_first:build",
            enabled=lambda _context: False,
        )
        registry = build_launch_registry(
            entry_points=[
                _entry_point("c", "ext_edge_third:SPEC"),
                _entry_point("a", "ext_edge_first:SPEC"),
            ]
        )
        with pytest.raises(RegistryError, match="requires 'ext_edge_first'"):
            registry.plan(_context())


class TestExtensionErrors:
    """Every extension error is a RegistryError naming the entry point and
    its value, raised before any build factory or hook is resolved."""

    def test_duplicate_entry_point_name_is_rejected_with_both_values(self):
        with pytest.raises(RegistryError, match="duplicate launch-builder entry point name 'clash'") as excinfo:
            build_launch_registry(
                entry_points=[
                    _entry_point("clash", "ext_a_probe:SPEC"),
                    _entry_point("clash", "ext_b_probe:SPEC"),
                ]
            )
        assert "ext_a_probe:SPEC" in str(excinfo.value)
        assert "ext_b_probe:SPEC" in str(excinfo.value)

    def test_import_failure_names_the_entry_point_and_module(self):
        with pytest.raises(RegistryError, match="no_such_extension_module_xyz") as excinfo:
            build_launch_registry(entry_points=[_entry_point("broken", "no_such_extension_module_xyz:SPEC")])
        message = str(excinfo.value)
        assert "'broken'" in message
        assert "no_such_extension_module_xyz:SPEC" in message

    def test_missing_attribute_is_rejected(self, monkeypatch):
        _install_module(monkeypatch, "ext_missing_attr_probe")
        with pytest.raises(RegistryError, match="missing attribute") as excinfo:
            build_launch_registry(entry_points=[_entry_point("broken", "ext_missing_attr_probe:NOPE")])
        assert "'broken'" in str(excinfo.value)
        assert "ext_missing_attr_probe:NOPE" in str(excinfo.value)

    @pytest.mark.parametrize(
        "resolved",
        [
            42,
            "not a spec",
            lambda _context: BuildResult(),
            [BuilderSpec(id="ok_item", factory="a:b"), "poison"],
        ],
        ids=["int", "str", "function", "sequence-with-non-spec"],
    )
    def test_non_builderspec_resolution_is_rejected(self, monkeypatch, resolved):
        module = _install_module(monkeypatch, "ext_bad_type_probe")
        module.SPEC = resolved
        with pytest.raises(RegistryError, match="must resolve to a BuilderSpec") as excinfo:
            build_launch_registry(entry_points=[_entry_point("broken", "ext_bad_type_probe:SPEC")])
        assert "'broken'" in str(excinfo.value)
        assert "ext_bad_type_probe:SPEC" in str(excinfo.value)

    def test_builtin_id_conflict_is_rejected_with_entry_point_context(self, monkeypatch):
        module = _install_module(monkeypatch, "ext_id_clash_probe")
        module.SPEC = BuilderSpec(id="control", factory="ext_id_clash_probe:build")
        with pytest.raises(RegistryError, match="duplicate builder id 'control'") as excinfo:
            build_launch_registry(entry_points=[_entry_point("clash", "ext_id_clash_probe:SPEC")])
        message = str(excinfo.value)
        assert "'clash'" in message
        assert "ext_id_clash_probe:SPEC" in message

    def test_argument_name_conflict_is_rejected_with_entry_point_context(self, monkeypatch):
        module = _install_module(monkeypatch, "ext_argument_clash_probe")
        module.SPEC = BuilderSpec(
            id="ext_argument_clash_probe",
            factory="ext_argument_clash_probe:build",
            arguments=(ArgumentSpec(name="with_moveit", declared_default="", absent_fallback="", description="clash"),),
        )
        with pytest.raises(RegistryError, match="cannot contribute launch argument 'with_moveit'") as excinfo:
            build_launch_registry(entry_points=[_entry_point("clash", "ext_argument_clash_probe:SPEC")])
        message = str(excinfo.value)
        assert "'clash'" in message
        assert "ext_argument_clash_probe:SPEC" in message

    def test_hook_config_key_conflict_is_rejected_with_entry_point_context(self, monkeypatch):
        module = _install_module(monkeypatch, "ext_hook_clash_probe")
        module.SPEC = BuilderSpec(
            id="ext_hook_clash_probe",
            factory="ext_hook_clash_probe:build",
            hooks=(HookSpec(phase="config_overrides", factory="m:h", config_keys=("embodied",)),),
        )
        with pytest.raises(RegistryError, match="config keys already owned by 'embodied'") as excinfo:
            build_launch_registry(entry_points=[_entry_point("clash", "ext_hook_clash_probe:SPEC")])
        message = str(excinfo.value)
        assert "'clash'" in message
        assert "ext_hook_clash_probe:SPEC" in message

    def test_extension_cannot_declare_typed_infrastructure_outputs(self, monkeypatch):
        module = _install_module(monkeypatch, "ext_provides_probe")
        module.SPEC = BuilderSpec(
            id="ext_provides_probe",
            factory="ext_provides_probe:build",
            provides=("control_startup",),
        )
        with pytest.raises(RegistryError, match="typed infrastructure outputs are reserved") as excinfo:
            build_launch_registry(entry_points=[_entry_point("broken", "ext_provides_probe:SPEC")])
        message = str(excinfo.value)
        assert "'broken'" in message
        assert "ext_provides_probe:SPEC" in message
        assert "control_startup" in message

    def test_invalid_entry_point_value_is_rejected(self):
        with pytest.raises(RegistryError, match="invalid value") as excinfo:
            build_launch_registry(entry_points=[_entry_point("broken", "no_separator_here")])
        assert "'broken'" in str(excinfo.value)


class TestLazyResolution:
    def test_extension_factories_resolve_lazily_from_the_source_module(self, monkeypatch):
        """The extension spec's factory is a ``module:attribute`` STRING
        resolved on every call: re-patching the source module attribute
        changes what the next resolution returns (the seam tests rely on)."""
        module = _install_module(monkeypatch, "ext_lazy_probe")
        module.SPEC = BuilderSpec(id="ext_lazy_probe", factory="ext_lazy_probe:build")

        def first(_context):
            return BuildResult()

        def second(_context):
            return BuildResult()

        module.build = first
        registry = build_launch_registry(entry_points=[_entry_point("lazy", "ext_lazy_probe:SPEC")])
        spec = registry.snapshot()[-1]
        assert spec.factory == "ext_lazy_probe:build"
        assert resolve_builder_factory(spec.factory) is first
        module.build = second
        assert resolve_builder_factory(spec.factory) is second

    def test_non_participating_extension_never_imports_its_run_dependency(self, monkeypatch):
        """Installing an extension package does not enable it: when the
        enabled predicate filters the entry out, its factory module is never
        imported (the run dependency stays untouched), while the spec module
        itself — needed to read the declaration — is."""
        module = _install_module(monkeypatch, "ext_run_dep_probe")
        module.SPEC = BuilderSpec(
            id="ext_run_dep_probe",
            # The factory module does NOT exist: resolving it while the entry
            # is disabled would raise ImportError; participating would be an
            # even louder failure. Neither happens below.
            factory="ext_run_dep_target_probe:build",
            enabled=lambda _context: False,
        )

        registry = build_launch_registry(entry_points=[_entry_point("lazy", "ext_run_dep_probe:SPEC")])
        plan = registry.plan(_context())
        assert "ext_run_dep_probe" not in [spec.id for spec in plan.specs]

        execute_plan(plan, _context())
        assert "ext_run_dep_target_probe" not in sys.modules
