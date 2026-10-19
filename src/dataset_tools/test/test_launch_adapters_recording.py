"""Build-function boundary tests for the recording / rerun visualizer entries.

Migrated from ``robot_config/test/test_launch_adapters.py`` (openspec change
``migrate-business-launch-builders``, task A.3/A.8): the recording and
rerun-visualizer test classes followed the module to dataset_tools; the
simulation/benchmark classes stay in robot_config with the retained modules.
Assertions are verbatim; only the import paths changed.
"""

from __future__ import annotations

from types import SimpleNamespace

from launch.actions import EmitEvent
from launch.events import Shutdown

import dataset_tools.launch_builders.recording as recording_module
from robot_config.launch_builders.registry import LaunchBuildContext
from robot_config.runtime_target import RuntimeTarget


class _FakeNode:
    def __init__(self, name: str):
        self.name = name


def _context(**overrides) -> LaunchBuildContext:
    defaults = {
        "robot_config": {"name": "test_robot"},
        "runtime_target": RuntimeTarget.HARDWARE,
        "use_sim": False,
        "sim_platform": "",
        "control_mode": "model_inference",
        "arguments": {},
        "auto_start_controllers": "true",
        "sim_backend_needs_ros2_control": True,
    }
    defaults.update(overrides)
    return LaunchBuildContext(**defaults)


def _sim_context(robot_config=None, **overrides) -> LaunchBuildContext:
    overrides.setdefault("runtime_target", RuntimeTarget.SIMULATION)
    overrides.setdefault("use_sim", True)
    overrides["robot_config"] = robot_config if robot_config is not None else {"name": "test_robot"}
    return _context(**overrides)


class TestRecordingEntry:
    def _semantic_context(self, record_mode: str) -> LaunchBuildContext:
        return _sim_context(
            robot_config={"name": "test_robot", "recording": {"semantic_dataset": True}},
            arguments={"record": "false", "record_mode": record_mode},
        )

    def test_semantic_dataset_with_episodic_mode_emits_shutdown_without_generating(self, monkeypatch):
        calls = []
        monkeypatch.setattr(recording_module, "generate_recording_nodes", lambda *a, **k: calls.append(a) or [])
        result = recording_module.build_recording(self._semantic_context("episodic"))

        shutdowns = [a for a in result.actions if isinstance(a, EmitEvent) and isinstance(a.event, Shutdown)]
        assert len(shutdowns) == 1
        assert calls == []

    def test_semantic_dataset_with_continuous_mode_forces_recording_on(self, monkeypatch):
        calls: list = []

        def fake_generate(robot_config, control_mode, mode, scheduler_enabled=False):
            calls.append((mode, scheduler_enabled))
            return [_FakeNode("recorder")]

        monkeypatch.setattr(recording_module, "generate_recording_nodes", fake_generate)
        result = recording_module.build_recording(self._semantic_context("continuous"))

        # record:=false is overridden to enabled by the semantic dataset policy.
        assert calls == [("continuous", False)]
        assert [a.name for a in result.actions] == ["recorder"]

    def test_ordinary_recording_failure_logs_and_continues(self, monkeypatch):
        def failing(*a, **k):
            raise RuntimeError("plain recording failure")

        monkeypatch.setattr(recording_module, "generate_recording_nodes", failing)
        context = _sim_context(arguments={"record": "true", "record_mode": "continuous"})
        result = recording_module.build_recording(context)

        assert result.actions == []


class TestRerunVisualizerEntry:
    def _logger_spy(self, monkeypatch) -> list[str]:
        warnings: list[str] = []

        def record_warning(msg):
            warnings.append(msg)

        monkeypatch.setattr(
            recording_module,
            "logger",
            SimpleNamespace(warning=record_warning, info=lambda msg: None, error=lambda msg: None),
        )
        return warnings

    def test_value_is_lowered_before_matching(self, monkeypatch):
        calls: list = []

        def fake_generate(robot_config):
            calls.append(robot_config)
            return [_FakeNode("viewer")]

        monkeypatch.setattr(recording_module, "generate_rerun_viewer_node", fake_generate)
        self._logger_spy(monkeypatch)
        context = _sim_context(arguments={"record_visualizer": "RERUN"})

        result = recording_module.build_rerun_visualizer(context)

        assert len(calls) == 1
        assert [a.name for a in result.actions] == ["viewer"]

    def test_unknown_value_really_warns_and_generates_nothing(self, monkeypatch):
        def unexpected(*a, **k):
            raise AssertionError("unknown visualizer values must not generate nodes")

        monkeypatch.setattr(recording_module, "generate_rerun_viewer_node", unexpected)
        warnings = self._logger_spy(monkeypatch)
        context = _sim_context(arguments={"record_visualizer": "matplotlib"})

        result = recording_module.build_rerun_visualizer(context)

        assert result.actions == []
        assert any("Unknown record_visualizer value" in msg for msg in warnings)

    def test_none_is_a_noop_without_warning(self, monkeypatch):
        def unexpected(*a, **k):
            raise AssertionError("none must not generate nodes")

        monkeypatch.setattr(recording_module, "generate_rerun_viewer_node", unexpected)
        warnings = self._logger_spy(monkeypatch)
        context = _sim_context(arguments={"record_visualizer": "none"})

        result = recording_module.build_rerun_visualizer(context)

        assert result.actions == []
        assert warnings == []
