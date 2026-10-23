"""Behavioral tests for the per-domain builtin launch-builder build functions.

Each builtin entry's ``build_<id>`` function now lives in its domain module
(openspec change ``decentralize-launch-builder-registration``; these tests
migrated from the retired ``adapters.py`` module with the same coverage).
Focused on the critical boundaries only (per the apply-phase review scope);
positive verification monkeypatches the SOURCE modules and counts real
invocations, so a patched attribute must be observed (fake-green
detection):

- simulation: hardware / benchmark / provider launches never touch the
  sim backend; the mock platform DOES enter the backend path and the
  backend receives a private config copy (local writes never leak into
  the shared context config);
- benchmark: the materialized snapshot (``_config_path`` + ``run_id``)
  is returned as ``benchmark_startup``; the shared context config is NOT
  replaced here (``execute_plan`` owns that).

The recording / rerun visualizer tolerance boundaries moved to
dataset_tools with the recording module (openspec change
``migrate-business-launch-builders``; same assertions, import paths
updated).

The controller-startup event chain (real ExecuteProcess targets) is
covered separately by the readiness/orchestrator suites; it is
deliberately not tested here with fake actions.
"""

from __future__ import annotations

import robot_config.launch_builders.benchmark as benchmark_module
import robot_config.launch_builders.sim_backend as sim_backend_module
import robot_config.launch_builders.simulation as simulation_module
from robot_config.launch_registry.registry import LaunchBuildContext
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


class TestSimulationEntry:
    def _forbid_backend(self, monkeypatch):
        def unexpected(*a, **k):
            raise AssertionError("sim backend must not be touched on this launch")

        monkeypatch.setattr(sim_backend_module, "get_sim_backend", unexpected)

    def test_hardware_launch_never_touches_sim_backend(self, monkeypatch):
        self._forbid_backend(monkeypatch)
        result = simulation_module.build_simulation(_context(use_sim=False))
        assert result.actions == []
        assert result.simulation_startup is not None
        assert result.simulation_startup.entity_action is None

    def test_benchmark_launch_never_touches_sim_backend(self, monkeypatch):
        self._forbid_backend(monkeypatch)
        result = simulation_module.build_simulation(_sim_context(runtime_target=RuntimeTarget.BENCHMARK))
        assert result.simulation_startup is not None
        assert result.simulation_startup.entity_action is None

    def test_provider_launch_never_touches_sim_backend(self, monkeypatch):
        self._forbid_backend(monkeypatch)
        config = {"name": "test_robot", "runtime": {"provider": "so101_runtime"}}
        result = simulation_module.build_simulation(_sim_context(robot_config=config))
        assert result.simulation_startup is not None
        assert result.simulation_startup.entity_action is None

    def test_mock_platform_enters_backend_with_private_config_copy(self, monkeypatch):
        received: list[dict] = []

        class FakeBackend:
            def start_backend(self, config):
                received.append(config)
                # Gazebo-style local write: must not leak to the shared config.
                config["initial_pose_x"] = 1.5
                return [_FakeNode("mock_backend")], _FakeNode("mock_entity")

            def spawn_peripheral_bridges(self, peripherals):
                return []

        monkeypatch.setattr(sim_backend_module, "get_sim_backend", lambda platform: FakeBackend())
        config = {"name": "test_robot", "simulation": {"platform": "mock"}}
        context = _sim_context(robot_config=config, sim_platform="mock")

        result = simulation_module.build_simulation(context)

        assert len(received) == 1  # the mock platform really entered the backend path
        assert received[0] is not context.robot_config  # private copy, not the shared config
        assert "initial_pose_x" not in context.robot_config  # no write-back
        assert [a.name for a in result.actions] == ["mock_backend"]
        assert result.simulation_startup.entity_action is not None


class TestBenchmarkEntry:
    def test_materialized_snapshot_returned_with_identity(self, monkeypatch):
        def fake_generate(snapshot):
            snapshot["benchmark"] = {"evaluation": {"enabled": True, "run_id": "run-42"}}
            snapshot["_config_path"] = "/tmp/run-42.yaml"
            return [_FakeNode("benchmark_bridge")]

        monkeypatch.setattr(benchmark_module, "generate_benchmark_nodes", fake_generate)
        context = _sim_context(runtime_target=RuntimeTarget.BENCHMARK)

        result = benchmark_module.build_benchmark(context)

        assert [a.name for a in result.actions] == ["benchmark_bridge"]
        startup = result.benchmark_startup
        assert startup is not None
        assert startup.run_id == "run-42"
        assert startup.config_path == "/tmp/run-42.yaml"
        assert startup.effective_config["benchmark"]["evaluation"]["run_id"] == "run-42"
        # The shared context config is NOT replaced by the build function itself;
        # execute_plan owns the cross-entry propagation.
        assert "benchmark" not in context.robot_config


def test_disabled_benchmark_evaluation_propagates_transport_without_run_identity(monkeypatch):
    def materialize(snapshot):
        snapshot["transport_marker"] = "materialized"
        return []

    monkeypatch.setattr(benchmark_module, "generate_benchmark_nodes", materialize)
    original = {"name": "r", "benchmark": {"evaluation": {"enabled": False, "run_id": "stale"}}}
    context = _sim_context(robot_config=original, runtime_target=RuntimeTarget.BENCHMARK)
    result = benchmark_module.build_benchmark(context)
    assert result.benchmark_startup.run_id == ""
    assert result.benchmark_startup.effective_config["transport_marker"] == "materialized"
    assert "transport_marker" not in original
