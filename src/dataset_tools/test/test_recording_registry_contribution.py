"""Registry-contribution tests for dataset_tools' launch-builder entries.

Migrated / reworked from robot_config's registry, options and adapter suites
(openspec change ``migrate-business-launch-builders``, task A.3/A.8):

* participation of ``rerun_visualizer`` (any ``record_visualizer`` value but
  ``none``) and the ``record`` / ``record_mode`` / ``record_visualizer``
  argument ownership moved here because the entries are now extension
  contributions of this package, not builtin table rows;
* the real-adapter benchmark-materialization test keeps its full coverage by
  composing the REAL builtin registry with this package's entry points
  injected through the same seam the registry tests use (hermetic — no
  ambient installed extensions).

robot_config keeps the framework-level equivalents restricted to retained
builtin entries.
"""

from __future__ import annotations

from importlib.metadata import EntryPoint

from dataset_tools.launch_builders.recording import RECORDING_BUILDER, RERUN_VISUALIZER_BUILDER
from robot_config.launch_builders.builtin_registry import build_launch_registry
from robot_config.launch_builders.launch_options import compose_argument_specs
from robot_config.launch_builders.registry import LaunchBuildContext, execute_plan
from robot_config.runtime_target import RuntimeTarget

_RECORDING_ENTRY_POINT = EntryPoint(
    name="recording",
    value="dataset_tools.launch_builders.recording:RECORDING_BUILDER",
    group="ibrobot.launch_builders",
)
_RERUN_ENTRY_POINT = EntryPoint(
    name="rerun_visualizer",
    value="dataset_tools.launch_builders.recording:RERUN_VISUALIZER_BUILDER",
    group="ibrobot.launch_builders",
)


def _registry():
    """The real builtin registry plus this package's two entries (injected)."""
    return build_launch_registry(entry_points=[_RECORDING_ENTRY_POINT, _RERUN_ENTRY_POINT])


def _context(**overrides) -> LaunchBuildContext:
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
    return LaunchBuildContext(**defaults)


def _plan_ids(**overrides) -> list[str]:
    return [spec.id for spec in _registry().plan(_context(**overrides)).specs]


class TestParticipation:
    def test_rerun_visualizer_participates_for_any_value_but_none(self):
        """Migrated from robot_config's registry suite: the value is
        lower()ed by the build function; unknown values still participate
        because the adapter owns the unknown-value warning dispatch."""
        assert "rerun_visualizer" not in _plan_ids(arguments={"record_visualizer": "none"})
        assert "rerun_visualizer" not in _plan_ids(arguments={})  # absent -> "none"
        assert "rerun_visualizer" in _plan_ids(arguments={"record_visualizer": "RERUN"})
        assert "rerun_visualizer" in _plan_ids(arguments={"record_visualizer": "matplotlib"})

    def test_recording_always_participates(self):
        """The recording entry declares no enabled predicate: ordinary
        launches keep it in the plan and the build function owns the
        record:=false no-op branch."""
        assert "recording" in _plan_ids(arguments={"record": "false"})
        assert "recording" in _plan_ids(arguments={"record": "true"})


class TestArgumentOwnership:
    def test_record_arguments_are_owned_by_the_extension_entries(self):
        """Migrated from robot_config's launch-options suite: record /
        record_mode / record_visualizer are contributed by exactly one spec
        each — this package's entries — with the legacy defaults."""
        registry = _registry()
        specs = {spec.id: spec for spec in registry.snapshot()}
        assert specs["recording"] is RECORDING_BUILDER
        assert specs["rerun_visualizer"] is RERUN_VISUALIZER_BUILDER
        contributed = {argument.name: spec.id for spec in registry.snapshot() for argument in spec.arguments}
        assert contributed["record"] == "recording"
        assert contributed["record_mode"] == "recording"
        assert contributed["record_visualizer"] == "rerun_visualizer"

        defaults = {
            argument.name: (argument.declared_default, argument.absent_fallback)
            for argument in (*RECORDING_BUILDER.arguments, *RERUN_VISUALIZER_BUILDER.arguments)
        }
        assert defaults["record"] == ("false", "false")
        assert defaults["record_mode"] == ("continuous", "continuous")
        assert defaults["record_visualizer"] == ("none", "none")

    def test_extension_arguments_compose_after_the_builtin_table(self):
        """Core + builtin business + this package's contributions compose
        without collisions, the extension arguments appended last."""
        registry = _registry()
        specs = compose_argument_specs(registry.contributed_arguments())
        names = [spec.name for spec in specs]
        assert names[-3:] == ["record", "record_mode", "record_visualizer"]


def test_extension_consumers_share_one_materialized_identity(monkeypatch):
    """Migrated from robot_config's registry suite (the real-adapter variant):
    through the REAL builtin registry plus this package's injected entries —
    the SOURCE benchmark generator materializes the run config exactly once;
    the builtin execution entry and this package's recording /
    rerun_visualizer entries — the three after=benchmark consumers — each
    receive that SAME materialized snapshot (new path + run_id), while the
    original loaded config is never rewritten and nothing re-materializes."""
    import dataset_tools.launch_builders.recording as recording_module
    from robot_config.launch_builders import benchmark as benchmark_module
    from robot_config.launch_builders import execution as execution_module

    # Extension declaration check: the three consumers declare after="benchmark".
    composed_specs = {spec.id: spec for spec in _registry().snapshot()}
    assert {"execution", "recording", "rerun_visualizer"} <= set(composed_specs)
    for entry_id in ("execution", "recording", "rerun_visualizer"):
        assert "benchmark" in composed_specs[entry_id].after

    benchmark_calls = []
    execution_configs = []
    recording_configs = []
    rerun_configs = []

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
    monkeypatch.setattr(
        recording_module, "generate_recording_nodes", lambda config, *a, **k: recording_configs.append(config) or []
    )
    monkeypatch.setattr(
        recording_module, "generate_rerun_viewer_node", lambda config, *a, **k: rerun_configs.append(config) or []
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
        arguments={"record": "true", "record_mode": "episodic", "record_visualizer": "rerun"},
    )
    registry = _registry()
    outputs = execute_plan(registry.plan(context), context)

    # The SOURCE benchmark generator ran exactly once (no re-materialization).
    assert len(benchmark_calls) == 1
    materialized = benchmark_calls[0]
    # Each consumer generator ran exactly once...
    assert len(execution_configs) == 1
    assert len(recording_configs) == 1
    assert len(rerun_configs) == 1
    # ...and all three received the SAME materialized snapshot object,
    # carrying the SAME new path and run identity.
    for received in (execution_configs[0], recording_configs[0], rerun_configs[0]):
        assert received is materialized
        assert received["_config_path"] == "/materialized/run-5.7.yaml"
        assert received["benchmark"]["evaluation"]["run_id"] == "run-5.7"
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
