"""Behavioral tests for the explicit launch-builder registry contracts.

All factories are referenced as ``module:attribute`` and resolved through
the REAL :func:`resolve_builder_factory` (importlib). Test modules are
injected into ``sys.modules`` so the resolver exercises its actual import
path; attributes are looked up on every call, so re-patching an attribute
must change behavior (monkeypatch-seam proof, not a structural stub).

Declaration semantics under test:

- ``after`` is ordering-only and optional (edges to non-participating
  targets are removed); ``requires`` is participation-only (a disabled
  target is an error) and contributes no ordering edge.
- Cycles are rejected on the combined ``after``+``requires`` declaration
  graph: a pure ``requires`` cycle or an ``after``/``requires`` mix is as
  fatal as a pure ``after`` cycle.
- The no-barrier merge preserves stable plan order (it must not
  concatenate timing buckets); :meth:`PlanOutputs.split_for_barrier` is
  the only place that separates ``runtime_ready`` candidates.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
from launch.actions import LogInfo

from robot_config.launch_builders.launch_options import ArgumentSpec, compose_argument_specs
from robot_config.launch_builders.registry import (
    BenchmarkStartup,
    BuilderSpec,
    BuildResult,
    CircularDependencyError,
    ControlStartup,
    HookSpec,
    LaunchBuildContext,
    LaunchBuilderRegistry,
    RegistryError,
    SimulationStartup,
    execute_plan,
    isolated_config,
    resolve_builder_factory,
)
from robot_config.runtime_target import RuntimeTarget


def _install_module(monkeypatch, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _context(**overrides) -> LaunchBuildContext:
    defaults = {"robot_config": {"name": "test_robot"}}
    defaults.update(overrides)
    return LaunchBuildContext(**defaults)


class TestFactoryResolution:
    def test_resolver_imports_module_and_attribute(self, monkeypatch):
        module = _install_module(monkeypatch, "registry_test_mod_a")
        marker = lambda _context: BuildResult()  # noqa: E731
        module.build_it = marker

        assert resolve_builder_factory("registry_test_mod_a:build_it") is marker

    def test_resolver_rereads_attribute_on_every_call(self, monkeypatch):
        module = _install_module(monkeypatch, "registry_test_mod_b")
        first = lambda _context: BuildResult()  # noqa: E731
        second = lambda _context: BuildResult()  # noqa: E731
        module.build_it = first

        assert resolve_builder_factory("registry_test_mod_b:build_it") is first
        module.build_it = second
        assert resolve_builder_factory("registry_test_mod_b:build_it") is second

    def test_resolver_rejects_missing_attribute_without_swallowing(self, monkeypatch):
        _install_module(monkeypatch, "registry_test_mod_missing")
        with pytest.raises(RegistryError, match="missing attribute"):
            resolve_builder_factory("registry_test_mod_missing:nowhere")

    @pytest.mark.parametrize("reference", ["no_separator", ":only_attr", "only_module:", "a:b:c"])
    def test_resolver_rejects_malformed_references(self, reference):
        with pytest.raises(RegistryError, match="factory reference"):
            resolve_builder_factory(reference)


class TestDeclarationValidation:
    def test_duplicate_id_is_rejected_at_add_time(self):
        registry = LaunchBuilderRegistry()
        registry.add(BuilderSpec(id="control", factory="a:b"))
        with pytest.raises(RegistryError, match="duplicate builder id 'control'"):
            registry.add(BuilderSpec(id="control", factory="c:d"))

    def test_unknown_after_dependency_is_rejected_before_any_factory(self, monkeypatch):
        module = _install_module(monkeypatch, "registry_test_unknown")
        calls = []
        module.build = lambda context: calls.append(context) or BuildResult()

        registry = LaunchBuilderRegistry()
        registry.add(BuilderSpec(id="first", factory="registry_test_unknown:build", after=["ghost"]))
        with pytest.raises(RegistryError, match="unknown dependency 'ghost'"):
            registry.plan(_context())
        assert calls == []

    def test_unknown_requires_dependency_is_rejected_before_any_factory(self, monkeypatch):
        module = _install_module(monkeypatch, "registry_test_unknown_req")
        calls = []
        module.build = lambda context: calls.append(context) or BuildResult()

        registry = LaunchBuilderRegistry()
        registry.add(BuilderSpec(id="first", factory="registry_test_unknown_req:build", requires=["ghost"]))
        with pytest.raises(RegistryError, match="unknown dependency 'ghost'"):
            registry.plan(_context())
        assert calls == []

    def test_after_cycle_is_rejected_before_any_factory(self, monkeypatch):
        module = _install_module(monkeypatch, "registry_test_cycle")
        calls = []
        module.build = lambda context: calls.append(context) or BuildResult()

        registry = LaunchBuilderRegistry()
        registry.add(BuilderSpec(id="a", factory="registry_test_cycle:build", after=["b"]))
        registry.add(BuilderSpec(id="b", factory="registry_test_cycle:build", after=["c"]))
        registry.add(BuilderSpec(id="c", factory="registry_test_cycle:build", after=["a"]))
        with pytest.raises(CircularDependencyError):
            registry.plan(_context())
        assert calls == []

    def test_pure_requires_cycle_is_rejected_before_any_factory(self, monkeypatch):
        """A mutual requires pair is a declaration-graph cycle even though
        requires contributes no ordering edge; the approved spec rejects
        cycles on the combined after+requires graph."""
        module = _install_module(monkeypatch, "registry_test_req_cycle")
        calls = []
        module.build = lambda context: calls.append(context) or BuildResult()

        registry = LaunchBuilderRegistry()
        registry.add(BuilderSpec(id="a", factory="registry_test_req_cycle:build", requires=["b"]))
        registry.add(BuilderSpec(id="b", factory="registry_test_req_cycle:build", requires=["a"]))
        with pytest.raises(CircularDependencyError):
            registry.plan(_context())
        assert calls == []

    def test_mixed_after_requires_cycle_is_rejected_before_any_factory(self, monkeypatch):
        """An after edge closed by a requires edge is equally fatal: the
        cycle check runs on the union of both edge kinds."""
        module = _install_module(monkeypatch, "registry_test_mixed_cycle")
        calls = []
        module.build = lambda context: calls.append(context) or BuildResult()

        registry = LaunchBuilderRegistry()
        registry.add(BuilderSpec(id="a", factory="registry_test_mixed_cycle:build", after=["b"]))
        registry.add(BuilderSpec(id="b", factory="registry_test_mixed_cycle:build", requires=["a"]))
        with pytest.raises(CircularDependencyError):
            registry.plan(_context())
        assert calls == []


class TestStableOrdering:
    def test_registration_order_is_the_tie_breaker(self, monkeypatch):
        module = _install_module(monkeypatch, "registry_test_order")
        module.build = lambda _context: BuildResult()

        registry = LaunchBuilderRegistry()
        for spec_id in ("perception", "teleop", "audio", "navigation"):
            registry.add(BuilderSpec(id=spec_id, factory="registry_test_order:build"))
        plan = registry.plan(_context())
        assert [spec.id for spec in plan.specs] == ["perception", "teleop", "audio", "navigation"]

    def test_after_reorders_only_when_target_participates(self, monkeypatch):
        """Registration order tail, middle, head with ``tail after head``:
        the stable Kahn ready set unlocks middle and head first and the
        registration-order tie-breaker picks middle (registered before
        head), so the plan is middle, head, tail."""
        module = _install_module(monkeypatch, "registry_test_after")
        module.build = lambda _context: BuildResult()

        registry = LaunchBuilderRegistry()
        registry.add(BuilderSpec(id="tail", factory="registry_test_after:build", after=["head"]))
        registry.add(BuilderSpec(id="middle", factory="registry_test_after:build"))
        registry.add(BuilderSpec(id="head", factory="registry_test_after:build"))
        plan = registry.plan(_context())
        assert [spec.id for spec in plan.specs] == ["middle", "head", "tail"]

    def test_after_edge_to_disabled_target_is_removed(self, monkeypatch):
        module = _install_module(monkeypatch, "registry_test_disabled_target")
        module.build = lambda _context: BuildResult()

        registry = LaunchBuilderRegistry()
        registry.add(
            BuilderSpec(
                id="consumer",
                factory="registry_test_disabled_target:build",
                after=["optional"],
            )
        )
        registry.add(
            BuilderSpec(
                id="optional",
                factory="registry_test_disabled_target:build",
                enabled=lambda _context: False,
            )
        )
        plan = registry.plan(_context())
        assert [spec.id for spec in plan.specs] == ["consumer"]

    def test_requires_to_disabled_target_is_an_error(self, monkeypatch):
        module = _install_module(monkeypatch, "registry_test_req_disabled")
        module.build = lambda _context: BuildResult()

        registry = LaunchBuilderRegistry()
        registry.add(BuilderSpec(id="consumer", factory="registry_test_req_disabled:build", requires=["optional"]))
        registry.add(
            BuilderSpec(
                id="optional",
                factory="registry_test_req_disabled:build",
                enabled=lambda _context: False,
            )
        )
        with pytest.raises(RegistryError, match="'optional', which is not participating"):
            registry.plan(_context())

    def test_requires_contributes_no_ordering_edge(self, monkeypatch):
        """requires guarantees participation only: with a registered before
        b and a requires b, the plan keeps registration order (if requires
        were an ordering edge, b would build first)."""
        module = _install_module(monkeypatch, "registry_test_req_order")
        module.build = lambda _context: BuildResult()

        registry = LaunchBuilderRegistry()
        registry.add(BuilderSpec(id="a", factory="registry_test_req_order:build", requires=["b"]))
        registry.add(BuilderSpec(id="b", factory="registry_test_req_order:build"))
        plan = registry.plan(_context())
        assert [spec.id for spec in plan.specs] == ["a", "b"]


class TestEnablement:
    def test_empty_result_from_participating_builder_is_not_disabling(self, monkeypatch):
        module = _install_module(monkeypatch, "registry_test_empty")
        built = []
        module.build = lambda context: built.append(context) or BuildResult(actions=[])

        registry = LaunchBuilderRegistry()
        registry.add(
            BuilderSpec(
                id="business",
                factory="registry_test_empty:build",
                # Disabled by domain config, not by the registry predicate:
                # the entry still participates and returns a legal no-op.
                enabled=lambda _context: True,
            )
        )
        outputs = execute_plan(registry.plan(_context()), _context())
        assert built and outputs.actions == []
        assert outputs.results["business"].actions == []

    def test_mock_skip_key_filters_participation(self, monkeypatch):
        module = _install_module(monkeypatch, "registry_test_mock")
        calls = []
        module.build = lambda context: calls.append(context) or BuildResult()

        registry = LaunchBuilderRegistry()
        registry.add(BuilderSpec(id="voice", factory="registry_test_mock:build", mock_skip_key="voice_asr"))

        context = _context(use_sim=True, sim_platform="mock", runtime_target=RuntimeTarget.SIMULATION)
        plan = registry.plan(context)
        assert plan.specs == ()
        execute_plan(plan, context)
        assert calls == []

        hardware_context = _context(runtime_target=RuntimeTarget.HARDWARE, use_sim=False)
        execute_plan(registry.plan(hardware_context), hardware_context)
        assert len(calls) == 1

    def test_mock_skip_is_not_benchmark_plus(self, monkeypatch):
        module = _install_module(monkeypatch, "registry_test_benchmark_mock")
        module.build = lambda _context: BuildResult()

        registry = LaunchBuilderRegistry()
        registry.add(BuilderSpec(id="voice", factory="registry_test_benchmark_mock:build"))

        context = _context(use_sim=True, sim_platform="mock", runtime_target=RuntimeTarget.BENCHMARK)
        assert context.mock_backend_active is False
        plan = registry.plan(context)
        assert [spec.id for spec in plan.specs] == ["voice"]


class TestBuiltinParticipation:
    """Plan-level participation of the REAL builtin registry: CLI-switch
    gating (with YAML fallbacks) is declared once in the registry metadata
    and the adapters never re-check it (openspec change
    ``migrate-builtin-builder-metadata``, design decision 1)."""

    @staticmethod
    def _plan_ids(**overrides) -> list[str]:
        from robot_config.launch_builders.builtin_registry import build_builtin_registry

        defaults = {
            "robot_config": {"name": "test_robot"},
            "runtime_target": RuntimeTarget.HARDWARE,
            "use_sim": False,
            "sim_platform": "",
            "control_mode": "model_inference",
            "with_inference": True,
            "arguments": {},
        }
        defaults.update(overrides)
        plan = build_builtin_registry().plan(LaunchBuildContext(**defaults))
        return [spec.id for spec in plan.specs]

    def test_mock_backend_skips_declared_subsystems_at_plan_time(self):
        plan_ids = self._plan_ids(
            use_sim=True,
            sim_platform="mock",
            runtime_target=RuntimeTarget.SIMULATION,
        )
        # Mock skip keys drop the entries from the frozen plan; the adapters
        # are never called, so their bodies no longer repeat the mock check.
        # voice_asr / speech_direction / navigation moved to their owning
        # packages (their mock-skip participation is pinned there through
        # injected entry points); perception is the retained mock-skipped
        # builtin.
        for skipped in ("perception",):
            assert skipped not in plan_ids
        # Non-skipped entries still participate (recording moved to
        # dataset_tools, voice_tts to voice_tts_service; embodied is the
        # retained always-participating sample here).
        for present in ("control", "simulation", "audio_io", "execution", "embodied"):
            assert present in plan_ids

    def test_mock_backend_skips_subsystems_via_shared_helper(self):
        """The mock skip reuses mock_mode_skips_subsystem keys, so an
        unknown subsystem key never skips."""
        from robot_config.launch_builders.hardware_mock import mock_mode_skips_subsystem

        assert mock_mode_skips_subsystem(True, "navigation") is True
        assert mock_mode_skips_subsystem(False, "navigation") is False

    def test_benchmark_and_perception_follow_the_runtime_target(self):
        plan_ids = self._plan_ids(runtime_target=RuntimeTarget.BENCHMARK, use_sim=True)
        assert "benchmark" in plan_ids
        assert "perception" not in plan_ids
        plan_ids = self._plan_ids(runtime_target=RuntimeTarget.SIMULATION, use_sim=True)
        assert "benchmark" not in plan_ids
        assert "perception" in plan_ids

    def test_execution_follows_with_inference(self):
        assert "execution" in self._plan_ids(with_inference=True)
        assert "execution" not in self._plan_ids(with_inference=False)

    def test_infrastructure_entries_always_participate(self):
        """control/simulation/controller_startup have no enable predicate
        or mock skip key: their adapters own the internal no-op branches."""
        for runtime_target in (RuntimeTarget.HARDWARE, RuntimeTarget.SIMULATION, RuntimeTarget.BENCHMARK):
            plan_ids = self._plan_ids(runtime_target=runtime_target, use_sim=True, with_inference=True)
            for infrastructure in ("control", "simulation", "controller_startup", "embodied", "motion_planning"):
                assert infrastructure in plan_ids, (runtime_target, infrastructure)


class TestSkipLogging:
    """The registry reports every non-participating entry once with the
    owning filter (mock skip key or enabled predicate) — design decision 2."""

    def _plan_with_log_spy(self, monkeypatch, context):
        from robot_config.launch_builders import registry as registry_module
        from robot_config.launch_builders.builtin_registry import build_builtin_registry

        messages = []
        monkeypatch.setattr(
            registry_module,
            "logger",
            SimpleNamespace(info=lambda msg: messages.append(msg), warning=lambda msg: None, error=lambda msg: None),
        )
        plan = build_builtin_registry().plan(context)
        return plan, messages

    def test_enabled_predicate_skips_are_logged_once_with_reason(self, monkeypatch):
        context = LaunchBuildContext(
            robot_config={"name": "r", "navigation": {"enabled": False}},
            runtime_target=RuntimeTarget.HARDWARE,
            use_sim=False,
            control_mode="model_inference",
            with_inference=True,
            arguments={"record_visualizer": "none"},
        )
        plan, messages = self._plan_with_log_spy(monkeypatch, context)
        plan_ids = [spec.id for spec in plan.specs]
        assert "benchmark" not in plan_ids
        assert "launch builder 'benchmark' skipped: enabled predicate" in messages  # hardware target
        # Each skipped entry is reported exactly once (teleop, navigation,
        # task_executor and rerun_visualizer moved out of the builtin table;
        # their skip behavior is pinned by their owning packages through
        # injected entry points; benchmark is the retained enabled-predicate
        # skip under this context).
        assert len([message for message in messages if "skipped" in message]) == 1

    def test_mock_skip_skips_are_logged_with_the_key(self, monkeypatch):
        context = LaunchBuildContext(
            robot_config={"name": "r"},
            runtime_target=RuntimeTarget.SIMULATION,
            use_sim=True,
            sim_platform="mock",
            control_mode="model_inference",
            with_inference=True,
            arguments={},
        )
        plan, messages = self._plan_with_log_spy(monkeypatch, context)
        plan_ids = [spec.id for spec in plan.specs]
        assert "perception" not in plan_ids
        assert "launch builder 'perception' skipped: mock skip key 'perception'" in messages
        # voice_asr / speech_direction / navigation moved to voice_asr_service
        # and robot_navigation; their mock-skip logging is pinned there.

    def test_participating_entries_are_not_logged(self, monkeypatch):
        context = LaunchBuildContext(
            robot_config={"name": "r"},
            runtime_target=RuntimeTarget.HARDWARE,
            use_sim=False,
            control_mode="model_inference",
            with_inference=True,
            arguments={},
        )
        _plan, messages = self._plan_with_log_spy(monkeypatch, context)
        assert not any("'control' skipped" in message for message in messages)
        assert not any("'embodied' skipped" in message for message in messages)


class TestTypedOutputOwnership:
    def test_same_output_declared_twice_is_rejected_at_add_time(self):
        registry = LaunchBuilderRegistry()
        registry.add(BuilderSpec(id="control", factory="a:b", provides=["control_startup"]))
        with pytest.raises(RegistryError, match="already owned by 'control'"):
            registry.add(BuilderSpec(id="rogue", factory="c:d", provides=["control_startup"]))

    def test_unknown_output_name_is_rejected_at_spec_construction(self):
        with pytest.raises(RegistryError, match="unknown typed output"):
            BuilderSpec(id="x", factory="a:b", provides=["hover_startup"])

    def test_business_entry_returning_control_output_is_rejected_on_first_provision(self, monkeypatch):
        module = _install_module(monkeypatch, "registry_test_rogue_output")
        # A business builder returning an output it never declared ownership
        # of must fail even though no other entry provided it before. This
        # is a registry contract violation, not a business failure: it
        # raises RegistryError directly, outside the exception wrapper.
        module.build = lambda _context: BuildResult(control_startup=ControlStartup())

        registry = LaunchBuilderRegistry()
        registry.add(BuilderSpec(id="business", factory="registry_test_rogue_output:build"))
        plan = registry.plan(_context())
        with pytest.raises(RegistryError, match="without declaring ownership"):
            execute_plan(plan, _context())

    def test_declared_owner_providing_output_is_accepted(self, monkeypatch):
        module = _install_module(monkeypatch, "registry_test_owner")
        startup = ControlStartup(robot_description={"xacro": "urdf"}, controller_names=("c1",))
        module.build = lambda _context: BuildResult(actions=[LogInfo(msg="ctrl")], control_startup=startup)

        registry = LaunchBuilderRegistry()
        registry.add(
            BuilderSpec(
                id="control",
                factory="registry_test_owner:build",
                provides=["control_startup"],
            )
        )
        outputs = execute_plan(registry.plan(_context()), _context())
        assert outputs.control_startup is startup


class TestContextPropagation:
    def test_control_and_simulation_outputs_reach_downstream_entries(self, monkeypatch):
        module = _install_module(monkeypatch, "registry_test_downstream")
        control_startup = ControlStartup(controller_names=("c1", "c2"))
        simulation_startup = SimulationStartup(entity_action=LogInfo(msg="create"))
        seen = {}

        def control_build(_context):
            return BuildResult(control_startup=control_startup)

        def simulation_build(_context):
            return BuildResult(simulation_startup=simulation_startup)

        def downstream_build(context):
            seen["control"] = context.control_startup
            seen["simulation"] = context.simulation_startup
            return BuildResult()

        module.build_control = control_build
        module.build_simulation = simulation_build
        module.build_downstream = downstream_build

        registry = LaunchBuilderRegistry()
        registry.add(
            BuilderSpec(
                id="control",
                factory="registry_test_downstream:build_control",
                provides=["control_startup"],
            )
        )
        registry.add(
            BuilderSpec(
                id="simulation",
                factory="registry_test_downstream:build_simulation",
                after=["control"],
                provides=["simulation_startup"],
            )
        )
        registry.add(
            BuilderSpec(
                id="controller_startup",
                factory="registry_test_downstream:build_downstream",
                after=["simulation"],
            )
        )
        execute_plan(registry.plan(_context()), _context())
        assert seen["control"] is control_startup
        assert seen["simulation"] is simulation_startup

    def test_benchmark_snapshot_replaces_config_for_later_entries_only(self, monkeypatch):
        module = _install_module(monkeypatch, "registry_test_benchmark_flow")
        materialized = {"name": "test_robot", "benchmark": {"run": True}, "_config_path": "/run/robot.yaml"}
        startup = BenchmarkStartup(effective_config=materialized, config_path="/run/robot.yaml", run_id="run-42")
        before, after = [], []

        def benchmark_build(_context):
            return BuildResult(benchmark_startup=startup)

        def early_build(context):
            before.append(context.robot_config)
            return BuildResult()

        def late_build(context):
            after.append((context.robot_config, context.benchmark_startup))
            return BuildResult()

        module.build_benchmark = benchmark_build
        module.build_early = early_build
        module.build_late = late_build

        registry = LaunchBuilderRegistry()
        registry.add(BuilderSpec(id="early", factory="registry_test_benchmark_flow:build_early"))
        registry.add(
            BuilderSpec(
                id="benchmark",
                factory="registry_test_benchmark_flow:build_benchmark",
                provides=["benchmark_startup"],
            )
        )
        registry.add(
            BuilderSpec(
                id="execution",
                factory="registry_test_benchmark_flow:build_late",
                after=["benchmark"],
            )
        )
        context = _context()
        original_config = context.robot_config
        outputs = execute_plan(registry.plan(context), context)
        # Entries before the benchmark entry see the pre-benchmark config...
        assert before and before[0] is original_config
        assert before[0] is not materialized
        # ...while the benchmark entry and every later entry observe the
        # materialized snapshot, including the context itself after the run.
        assert after[0][0] is materialized
        assert after[0][1] is startup
        assert context.robot_config is materialized
        assert outputs.benchmark_startup is startup
        # run identity/path flows to every later entry through one owner.
        assert after[0][0]["_config_path"] == "/run/robot.yaml"

    def test_execution_recording_rerun_receive_same_materialized_identity(self, monkeypatch):
        """Task 5.7: execution, recording and rerun — the three after=benchmark
        consumers — must all receive the SAME materialized snapshot object and
        run identity from the one benchmark owner, never re-materialized or
        re-assigned (spec: 三者接收同一物化路径/身份，调用方配置不被改写)."""
        module = _install_module(monkeypatch, "registry_test_benchmark_consumers")
        materialized = {"name": "test_robot", "_config_path": "/run/robot.yaml"}
        startup = BenchmarkStartup(effective_config=materialized, config_path="/run/robot.yaml", run_id="run-7")
        observed = {}

        def benchmark_build(_context):
            return BuildResult(benchmark_startup=startup)

        module.build_benchmark = benchmark_build
        for entry_id in ("execution", "recording", "rerun"):

            def make_consumer(name):
                def consumer(context):
                    observed.setdefault(name, []).append((context.robot_config, context.benchmark_startup))
                    return BuildResult()

                return consumer

            setattr(module, f"build_{entry_id}", make_consumer(entry_id))

        registry = LaunchBuilderRegistry()
        registry.add(
            BuilderSpec(
                id="benchmark",
                factory="registry_test_benchmark_consumers:build_benchmark",
                provides=["benchmark_startup"],
            )
        )
        for entry_id in ("execution", "recording", "rerun"):
            registry.add(
                BuilderSpec(
                    id=entry_id,
                    factory=f"registry_test_benchmark_consumers:build_{entry_id}",
                    after=["benchmark"],
                )
            )
        context = _context()
        outputs = execute_plan(registry.plan(context), context)
        assert set(observed) == {"execution", "recording", "rerun"}
        for _entry_id, entries in observed.items():
            assert len(entries) == 1
            # Same materialized object identity, same run identity owner.
            assert entries[0][0] is materialized
            assert entries[0][1] is startup
            assert entries[0][0]["_config_path"] == "/run/robot.yaml"
        assert outputs.benchmark_startup is startup
        # The caller's context config was replaced with the materialized
        # snapshot exactly once (single owner), not re-written per consumer.
        assert context.robot_config is materialized

    def test_builtin_registry_real_adapters_share_one_materialized_identity(self, monkeypatch):
        """Task 5.7 through the REAL builtin registry and REAL adapters (no
        hand-written specs): the SOURCE benchmark generator materializes the
        run config exactly once and the builtin execution entry — the
        after=benchmark consumer that stays in robot_config — receives that
        SAME materialized snapshot (new path + run_id), while the original
        loaded config is never rewritten and nothing re-materializes. The
        recording / rerun_visualizer consumers moved to dataset_tools with
        the recording entry; the full three-consumer composition (builtin +
        injected extensions) is pinned by dataset_tools' suite."""
        from robot_config.launch_builders import benchmark as benchmark_module
        from robot_config.launch_builders import execution as execution_module
        from robot_config.launch_builders.builtin_registry import build_builtin_registry

        # Builtin declaration check: the retained consumer declares after="benchmark".
        builtin_specs = {spec.id: spec for spec in build_builtin_registry().snapshot()}
        assert "execution" in builtin_specs
        assert "benchmark" in builtin_specs["execution"].after
        # The migrated consumers are extension contributions now, not builtin rows.
        assert "recording" not in builtin_specs
        assert "rerun_visualizer" not in builtin_specs

        benchmark_calls = []
        execution_configs = []

        def materializing_generate(snapshot):
            # SOURCE seam: one materialization owner — new path + run_id are
            # written here and nowhere else.
            benchmark_calls.append(snapshot)
            snapshot["benchmark"]["evaluation"]["run_id"] = "run-5.7"
            snapshot["_config_path"] = "/materialized/run-5.7.yaml"
            return []

        monkeypatch.setattr(benchmark_module, "generate_benchmark_nodes", materializing_generate)
        monkeypatch.setattr(
            execution_module, "generate_execution_nodes", lambda config, *a, **k: execution_configs.append(config) or []
        )

        loaded_config = {
            "name": "bench",
            "_config_path": "/original/robot.yaml",
            "benchmark": {"type": "libero", "adapter": "libero", "evaluation": {"enabled": True, "suite": "libero"}},
        }
        original_copy = {"_config_path": loaded_config["_config_path"], "benchmark": dict(loaded_config["benchmark"])}
        context = _context(
            robot_config=loaded_config,
            runtime_target=RuntimeTarget.BENCHMARK,
            use_sim=True,
            control_mode="benchmark",
            with_inference=True,
            arguments={},
        )
        registry = build_builtin_registry()
        outputs = execute_plan(registry.plan(context), context)

        # The SOURCE benchmark generator ran exactly once (no re-materialization).
        assert len(benchmark_calls) == 1
        materialized = benchmark_calls[0]
        # The builtin consumer generator ran exactly once and received the
        # SAME materialized snapshot object, carrying the SAME new path and
        # run identity.
        assert len(execution_configs) == 1
        assert execution_configs[0] is materialized
        assert execution_configs[0]["_config_path"] == "/materialized/run-5.7.yaml"
        assert execution_configs[0]["benchmark"]["evaluation"]["run_id"] == "run-5.7"
        # execute_plan replaced the shared context config with the materialized
        # snapshot exactly once, and the typed output owns the same identity.
        assert context.robot_config is materialized
        assert outputs.benchmark_startup.effective_config is materialized
        assert outputs.benchmark_startup.config_path == "/materialized/run-5.7.yaml"
        assert outputs.benchmark_startup.run_id == "run-5.7"
        # The caller's original config was never rewritten: no run_id write-back,
        # original _config_path intact.
        assert loaded_config["_config_path"] == original_copy["_config_path"]
        assert "run_id" not in loaded_config["benchmark"]["evaluation"]


class TestExecutePlan:
    def test_actions_merge_in_plan_order_not_timing_buckets(self, monkeypatch):
        """The no-barrier merge must preserve stable plan order: an early
        runtime_ready entry (perception) stays ahead of later immediate
        entries (voice, recording) instead of being concatenated behind
        the immediate bucket."""
        module = _install_module(monkeypatch, "registry_test_merge_order")
        module.build_perception = lambda _context: BuildResult(actions=[LogInfo(msg="perception")])
        module.build_voice = lambda _context: BuildResult(actions=[LogInfo(msg="voice")])
        module.build_recording = lambda _context: BuildResult(actions=[LogInfo(msg="recording")])

        registry = LaunchBuilderRegistry()
        registry.add(
            BuilderSpec(id="perception", factory="registry_test_merge_order:build_perception", timing="runtime_ready")
        )
        registry.add(BuilderSpec(id="voice", factory="registry_test_merge_order:build_voice"))
        registry.add(BuilderSpec(id="recording", factory="registry_test_merge_order:build_recording"))
        outputs = execute_plan(registry.plan(_context()), _context())
        assert [action.msg[0].text for action in outputs.actions] == ["perception", "voice", "recording"]
        assert [segment.spec_id for segment in outputs.segments] == ["perception", "voice", "recording"]
        assert [segment.timing for segment in outputs.segments] == ["runtime_ready", "immediate", "immediate"]

    def test_split_for_barrier_defers_only_runtime_ready_segments(self, monkeypatch):
        """split_for_barrier separates the runtime_ready candidate set in
        plan order while immediate actions keep their plan order."""
        module = _install_module(monkeypatch, "registry_test_barrier")
        module.build_perception = lambda _context: BuildResult(actions=[LogInfo(msg="perception")])
        module.build_voice = lambda _context: BuildResult(actions=[LogInfo(msg="voice")])
        module.build_speech = lambda _context: BuildResult(actions=[LogInfo(msg="speech")])
        module.build_recording = lambda _context: BuildResult(actions=[LogInfo(msg="recording")])

        registry = LaunchBuilderRegistry()
        registry.add(
            BuilderSpec(id="perception", factory="registry_test_barrier:build_perception", timing="runtime_ready")
        )
        registry.add(BuilderSpec(id="voice", factory="registry_test_barrier:build_voice"))
        registry.add(BuilderSpec(id="speech", factory="registry_test_barrier:build_speech", timing="runtime_ready"))
        registry.add(BuilderSpec(id="recording", factory="registry_test_barrier:build_recording"))
        outputs = execute_plan(registry.plan(_context()), _context())
        immediate, deferred = outputs.split_for_barrier()
        assert [action.msg[0].text for action in immediate] == ["voice", "recording"]
        assert [action.msg[0].text for action in deferred] == ["perception", "speech"]
        # The gated candidates are recorded, but the no-barrier merge
        # above is what the readiness orchestrator consumes directly.
        assert [action.msg[0].text for action in outputs.runtime_ready_actions] == ["perception", "speech"]

    def test_plan_timing_accessors_partition_specs(self, monkeypatch):
        module = _install_module(monkeypatch, "registry_test_accessor")
        module.build = lambda _context: BuildResult()
        registry = LaunchBuilderRegistry()
        registry.add(BuilderSpec(id="imm", factory="registry_test_accessor:build"))
        registry.add(BuilderSpec(id="ready", factory="registry_test_accessor:build", timing="runtime_ready"))
        plan = registry.plan(_context())
        assert [spec.id for spec in plan.immediate_specs] == ["imm"]
        assert [spec.id for spec in plan.runtime_ready_specs] == ["ready"]

    def test_factory_exception_is_wrapped_with_builder_id_and_chain(self, monkeypatch):
        module = _install_module(monkeypatch, "registry_test_failure")

        def boom(_context):
            raise KeyError("model_path")

        module.build = boom
        registry = LaunchBuilderRegistry()
        registry.add(BuilderSpec(id="execution", factory="registry_test_failure:build"))
        with pytest.raises(RuntimeError, match="'execution' failed") as wrapped:
            execute_plan(registry.plan(_context()), _context())
        assert isinstance(wrapped.value.__cause__, KeyError)

    def test_non_buildresult_return_is_rejected(self, monkeypatch):
        module = _install_module(monkeypatch, "registry_test_badresult")
        module.build = lambda _context: ["not", "a", "result"]
        registry = LaunchBuilderRegistry()
        registry.add(BuilderSpec(id="x", factory="registry_test_badresult:build"))
        plan = registry.plan(_context())
        with pytest.raises(RegistryError, match="must return a BuildResult"):
            execute_plan(plan, _context())

    def test_instances_do_not_share_state(self, monkeypatch):
        module = _install_module(monkeypatch, "registry_test_isolation")
        module.build = lambda _context: BuildResult()

        first = LaunchBuilderRegistry()
        first.add(BuilderSpec(id="only_here", factory="registry_test_isolation:build"))
        second = LaunchBuilderRegistry()
        assert "only_here" in first
        assert "only_here" not in second
        assert len(second) == 0
        # An empty registry is a legal (empty) plan, not an error; the
        # unknown-dependency rejection is covered above with a real
        # ghost target declared by a participating spec.
        assert second.plan(_context()).specs == ()

    def test_real_unknown_dependency_is_rejected_per_instance(self, monkeypatch):
        module = _install_module(monkeypatch, "registry_test_isolation_ghost")
        module.build = lambda _context: BuildResult()

        second = LaunchBuilderRegistry()
        second.add(BuilderSpec(id="lonely", factory="registry_test_isolation_ghost:build", after=["ghost"]))
        with pytest.raises(RegistryError, match="unknown dependency 'ghost'"):
            second.plan(_context())


class TestConfigIsolation:
    def test_isolated_config_deep_copies_caller_state(self):
        caller = {"name": "robot", "nested": {"list": [1, 2]}, "peripherals": [{"type": "camera"}]}
        snapshot = isolated_config(caller)
        assert snapshot == caller
        assert snapshot is not caller
        snapshot["nested"]["list"].append(3)
        snapshot["peripherals"][0]["type"] = "lidar"
        snapshot["name"] = "mutated"
        assert caller == {"name": "robot", "nested": {"list": [1, 2]}, "peripherals": [{"type": "camera"}]}

    def test_isolated_config_accepts_plain_mapping(self):
        from collections.abc import Mapping

        class ReadOnlyMap(Mapping):
            def __init__(self, data):
                self._data = data

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

            def __getitem__(self, key):
                return self._data[key]

        snapshot = isolated_config(ReadOnlyMap({"a": {"b": 1}}))
        assert isinstance(snapshot, dict)
        assert snapshot["a"] == {"b": 1}


class TestRegistrationMetadata:
    """One BuilderSpec registration contributing business arguments and
    prepare/preflight hooks (design decision 4), with collisions
    rejected at add time before any factory or hook is resolved."""

    def test_one_registration_contributes_arguments_and_hooks(self, monkeypatch):
        module = _install_module(monkeypatch, "registry_test_meta_business")
        module.build = lambda _context: BuildResult()
        module.apply_feature = lambda config: config
        module.confirm_feature = lambda config: config

        argument = ArgumentSpec(
            name="feature_enabled",
            declared_default="false",
            absent_fallback="false",
            description="Enable the test feature",
        )
        registry = LaunchBuilderRegistry()
        registry.add(
            BuilderSpec(
                id="feature",
                factory="registry_test_meta_business:build",
                arguments=[argument],
                hooks=[
                    HookSpec(
                        phase="config_overrides",
                        factory="registry_test_meta_business:apply_feature",
                        config_keys=["feature"],
                    ),
                    HookSpec(
                        phase="preflight",
                        factory="registry_test_meta_business:confirm_feature",
                        config_keys=["feature"],
                    ),
                ],
            )
        )
        # The preparation module merges the aggregated contributions with
        # the core table; the contributed argument survives composition.
        composed = compose_argument_specs(registry.contributed_arguments())
        assert argument in composed
        by_phase = dict(registry.hooks_by_phase())
        assert [spec_id for spec_id, _hook in by_phase["config_overrides"]] == ["feature"]
        assert [spec_id for spec_id, _hook in by_phase["preflight"]] == ["feature"]

    def test_hook_factory_resolves_lazily_and_rereads_attribute(self, monkeypatch):
        """Hook callables are resolved on every call through the real
        resolver; the aggregated declaration itself never resolves, so
        patching the source module changes behavior (test seam)."""
        module = _install_module(monkeypatch, "registry_test_meta_hook")
        module.build = lambda _context: BuildResult()
        first = lambda config: config  # noqa: E731
        second = lambda config: config  # noqa: E731
        module.prepare = first

        registry = LaunchBuilderRegistry()
        registry.add(
            BuilderSpec(
                id="feature",
                factory="registry_test_meta_hook:build",
                hooks=[
                    HookSpec(phase="config_overrides", factory="registry_test_meta_hook:prepare", config_keys=["voice"])
                ],
            )
        )
        # hooks_by_phase only aggregates declarations; nothing resolved yet.
        (_phase, ((_spec_id, hook),)) = [(p, e) for p, e in registry.hooks_by_phase() if e][0]
        assert resolve_builder_factory(hook.factory) is first
        module.prepare = second
        assert resolve_builder_factory(hook.factory) is second

    def test_same_spec_may_reown_its_keys_across_phases(self, monkeypatch):
        module = _install_module(monkeypatch, "registry_test_meta_reown")
        module.build = lambda _context: BuildResult()

        registry = LaunchBuilderRegistry()
        registry.add(
            BuilderSpec(
                id="feature",
                factory="registry_test_meta_reown:build",
                hooks=[
                    HookSpec(phase="config_overrides", factory="a:b", config_keys=["feature_profile"]),
                    HookSpec(phase="resource_check", factory="c:d", config_keys=["feature_profile"]),
                ],
            )
        )
        assert len(registry) == 1

    def test_cross_spec_argument_name_collision_is_rejected_at_add_time(self):
        registry = LaunchBuilderRegistry()
        registry.add(
            BuilderSpec(
                id="first",
                factory="a:b",
                arguments=[
                    ArgumentSpec(
                        name="feature_enabled", declared_default="false", absent_fallback="false", description="d"
                    )
                ],
            )
        )
        with pytest.raises(RegistryError, match="already contributed by 'first'"):
            registry.add(
                BuilderSpec(
                    id="second",
                    factory="c:d",
                    arguments=[
                        ArgumentSpec(
                            name="feature_enabled", declared_default="true", absent_fallback="true", description="d"
                        )
                    ],
                )
            )

    def test_core_argument_name_collision_is_rejected_at_composition(self, monkeypatch):
        """Collisions with the core table are caught when the preparation
        module merges contributions via compose_argument_specs (record is
        a builtin business contribution now; control_mode is core)."""
        module = _install_module(monkeypatch, "registry_test_meta_core")
        module.build = lambda _context: BuildResult()

        registry = LaunchBuilderRegistry()
        registry.add(
            BuilderSpec(
                id="feature",
                factory="registry_test_meta_core:build",
                arguments=[
                    ArgumentSpec(name="control_mode", declared_default="x", absent_fallback="x", description="d")
                ],
            )
        )
        with pytest.raises(ValueError, match="duplicate launch argument name 'control_mode'"):
            compose_argument_specs(registry.contributed_arguments())

    def test_cross_spec_config_key_collision_is_rejected_at_add_time(self):
        registry = LaunchBuilderRegistry()
        registry.add(
            BuilderSpec(
                id="first",
                factory="a:b",
                hooks=[HookSpec(phase="control_mode", factory="a:b", config_keys=["control"])],
            )
        )
        with pytest.raises(RegistryError, match="config keys already owned by 'first': control"):
            registry.add(
                BuilderSpec(
                    id="second",
                    factory="c:d",
                    hooks=[HookSpec(phase="preflight", factory="c:d", config_keys=["control"])],
                )
            )

    def test_unknown_hook_phase_is_rejected_at_declaration(self):
        with pytest.raises(RegistryError, match="is not a named preparation anchor"):
            HookSpec(phase="after_providers", factory="a:b", config_keys=["x"])

    def test_hook_without_config_keys_is_rejected(self):
        with pytest.raises(RegistryError, match="at least one config key"):
            HookSpec(phase="preflight", factory="a:b")

    def test_collisions_reject_before_any_factory_resolution(self, monkeypatch):
        """add-time rejection must not resolve factories: a colliding
        spec whose factory module does not exist still fails with the
        collision error, not an import error."""
        module = _install_module(monkeypatch, "registry_test_meta_nofactory")
        module.build = lambda _context: BuildResult()
        registry = LaunchBuilderRegistry()
        registry.add(
            BuilderSpec(
                id="first",
                factory="a:b",
                hooks=[HookSpec(phase="preflight", factory="a:b", config_keys=["arm"])],
            )
        )
        with pytest.raises(RegistryError, match="config keys already owned"):
            registry.add(
                BuilderSpec(
                    id="ghost",
                    factory="no_such_module_anywhere:build",
                    hooks=[HookSpec(phase="preflight", factory="no_such_module_anywhere:hook", config_keys=["arm"])],
                )
            )
        assert "ghost" not in registry


@pytest.mark.parametrize("keys", [("",), ("a..b",), (".a",), ("a.",), (" a",)])
def test_invalid_hook_owned_paths(keys):
    with pytest.raises(RegistryError, match="dotted path"):
        HookSpec(phase="control_mode", factory="a:b", config_keys=keys)


@pytest.mark.parametrize("first,second", [("feature", "feature.enabled"), ("feature.enabled", "feature")])
def test_hook_ancestor_ownership_collisions(first, second):
    registry = LaunchBuilderRegistry()
    registry.add(
        BuilderSpec(
            id="one", factory="a:b", hooks=(HookSpec(phase="control_mode", factory="a:b", config_keys=(first,)),)
        )
    )
    with pytest.raises(RegistryError, match="already owned"):
        registry.add(
            BuilderSpec(
                id="two", factory="a:b", hooks=(HookSpec(phase="control_mode", factory="a:b", config_keys=(second,)),)
            )
        )


def test_business_cannot_claim_infrastructure_output_even_when_first():
    registry = LaunchBuilderRegistry()
    with pytest.raises(RegistryError, match="reserved for 'control'"):
        registry.add(BuilderSpec(id="business", factory="a:b", provides=("control_startup",)))


def test_typed_output_value_is_checked(monkeypatch):
    module = _install_module(monkeypatch, "registry_test_invalid_output_type")
    module.build = lambda context: BuildResult(control_startup={})
    registry = LaunchBuilderRegistry()
    registry.add(BuilderSpec(id="control", factory=f"{module.__name__}:build", provides=("control_startup",)))
    with pytest.raises(RegistryError, match="must return ControlStartup"):
        execute_plan(registry.plan(_context()), _context())
