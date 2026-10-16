"""Startup identity transport tests using real ROS imports and test-only adapters."""

from __future__ import annotations

import json
import multiprocessing
import time
import traceback
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import rclpy
from rclpy.parameter import Parameter
from rclpy.task import Future

from benchmark_runtime import environment_node, evaluator_node
from benchmark_runtime.adapter import BenchmarkAdapter
from benchmark_runtime.io_descriptor import BenchmarkIODescriptor, FeatureDescriptor
from benchmark_runtime.models import BenchmarkCapabilities, BenchmarkEnvironmentConfig
from benchmark_runtime.plan import (
    BenchmarkArtifactFlags,
    BenchmarkPlan,
    BenchmarkPlanTask,
    BenchmarkTimeouts,
    InitStatePolicy,
    benchmark_plan_from_json,
)
from benchmark_runtime.registry import BenchmarkPlugin
from ibrobot_msgs.srv import GetBenchmarkPlan

IDENTITY = {
    "name": "test-provider",
    "version": "verified-revision",
    "provider_root": "/environment-process/provider",
    "module_path": "/environment-process/module.py",
    "assets_path": "/environment-process/assets",
    "apis_present": ["reset", "step"],
    "extension": {"verified": True},
}
DESCRIPTOR = BenchmarkIODescriptor(
    observations=(FeatureDescriptor("observation.state", "state", "float32", (1,), "C"),),
    actions=(FeatureDescriptor("action", "action", "float32", (1,), "C"),),
)


class GenericAdapter(BenchmarkAdapter):
    """Existing adapters inherit the optional identity hook unchanged."""

    def __init__(self):
        self.configured = False
        self.reset_calls = 0
        self.closed = False

    @property
    def capabilities(self):
        return BenchmarkCapabilities(False, False, False, False, False)

    def configure(self, config):
        self.configured = True

    def get_io_descriptor(self):
        return DESCRIPTOR

    def get_plan(self):
        assert self.configured
        return BenchmarkPlan(
            benchmark_type="test",
            adapter="test",
            suite="test_suite",
            requested_task_order_index=0,
            effective_task_order_index=0,
            tasks=(BenchmarkPlanTask(0, "task", "perform task"),),
            selected_task_ids=(0,),
            episodes_per_task=1,
            planned_episodes=1,
            seed=0,
            init_state_policy=InitStatePolicy(False, "provider"),
            max_steps=1,
            timeouts=BenchmarkTimeouts(30.0, 30.0),
            artifacts=BenchmarkArtifactFlags(False, False, False, True),
        )

    def list_tasks(self):
        return []

    def reset(self, request):
        self.reset_calls += 1
        raise AssertionError("startup identity must be available before reset")

    def step(self, action):
        raise AssertionError("startup must not step")

    def render(self):
        return None

    def collect_native_artifacts(self):
        return []

    def close(self):
        self.closed = True


class IdentityAdapter(GenericAdapter):
    def __init__(self):
        super().__init__()
        self.identity = json.loads(json.dumps(IDENTITY))
        self.identity_calls = 0

    def provider_identity(self):
        assert self.configured
        self.identity_calls += 1
        return self.identity


def _plugin(adapter):
    return BenchmarkPlugin(
        name="test",
        api_version=1,
        validate_environment_config=lambda section: BenchmarkEnvironmentConfig("test", "test", section),
        create_adapter=lambda: adapter,
        create_native_reporter=lambda: pytest.fail("native reporting is disabled"),
    )


def _environment_parameters(config):
    prefix = json.loads(config.read_text())["prefix"]
    return [
        Parameter("robot_config_path", value=str(config)),
        Parameter("adapter_name", value="test"),
        *[Parameter(f"{name}_service", value=f"{prefix}/{name}") for name in ("reset", "step", "plan", "finalize")],
    ]


def _evaluator(config):
    prefix = json.loads(config.read_text())["prefix"]
    node = evaluator_node.BenchmarkEvaluatorNode(
        parameter_overrides=[
            Parameter("robot_config_path", value=str(config)),
            Parameter("evaluation_enabled", value=True),
            Parameter("run_id", value="identity-test"),
            *[
                Parameter(f"benchmark_{name}_service", value=f"{prefix}/{name}")
                for name in ("plan", "reset", "finalize")
            ],
            Parameter("prepare_policy_episode_service", value=f"{prefix}/prepare"),
            Parameter("run_policy_action_server", value=f"{prefix}/run"),
            Parameter("inference_health_topic", value=f"{prefix}/health"),
        ]
    )
    try:
        node.start_production()
        # Drive readiness explicitly; no policy readiness means no reset can run.
        node._timer.cancel()
    except BaseException:
        node.destroy_node()
        raise
    return node


@pytest.fixture
def ros_context():
    rclpy.init(args=[])
    try:
        yield
    finally:
        rclpy.shutdown()


@pytest.fixture
def robot_config(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "input_features": {"observation.state": {"type": "STATE", "shape": [1]}},
                "output_features": {"action": {"type": "ACTION", "shape": [1]}},
            }
        )
    )
    prefix = f"/identity_{uuid.uuid4().hex}"
    config = tmp_path / "robot.json"
    config.write_text(
        json.dumps(
            {
                "prefix": prefix,
                "robot": {
                    "benchmark": {
                        "evaluation": {
                            "enabled": True,
                            "output": {"root": str(tmp_path / "reports")},
                            "timeouts": {"startup_timeout_sec": 30.0},
                        }
                    },
                    "contract": {
                        "observations": [
                            {
                                "key": "observation.state",
                                "type": "ibrobot_msgs/msg/StampedFloat32MultiArray",
                                "topic": f"{prefix}/state",
                                "selector": {"names": ["state"]},
                            }
                        ],
                        "actions": [
                            {
                                "key": "action",
                                "selector": {"names": ["action"]},
                                "publish": {"type": "std_msgs/msg/Float32MultiArray"},
                            }
                        ],
                    },
                    "control_modes": {
                        "model_inference": {"inference": {"pipelines": {"test": {"model_path": str(model)}}}}
                    },
                },
            }
        )
    )
    return config


@pytest.fixture
def environment(robot_config, ros_context, monkeypatch):
    adapter = IdentityAdapter()
    monkeypatch.setattr(environment_node, "load_plugin", lambda name: _plugin(adapter))
    node = environment_node.BenchmarkEnvironmentNode(parameter_overrides=_environment_parameters(robot_config))
    try:
        node.start_production()
        yield node, adapter
    finally:
        node.close()
        node.destroy_node()


def _response(environment):
    # Lightweight callback fixture works even before the generated srv is rebuilt.
    response = environment._handle_get_plan(None, SimpleNamespace())
    return SimpleNamespace(**json.loads(json.dumps(vars(response))))


def _accept_response(node, response):
    future = Future()
    future.set_result(response)
    node._plan_client = SimpleNamespace(service_is_ready=lambda: True, call_async=lambda request: future)
    node._drive_plan_and_readiness()
    node._drive_plan_and_readiness()


def test_startup_snapshots_identity_and_writes_manifest_before_reset(environment, robot_config):
    env, adapter = environment
    adapter.identity["extension"]["verified"] = False
    response = _response(env)
    assert response.success
    assert json.loads(response.provider_identity_json) == IDENTITY
    assert benchmark_plan_from_json(response.plan_json) == env._plan
    assert "provider_identity" not in json.loads(response.plan_json)
    node = _evaluator(robot_config)
    try:
        _accept_response(node, response)
        manifest_path = Path(node._canonical_manifest["run_root"]) / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["runtime_identity"]["provider"] == IDENTITY
        assert "libero_provider" not in manifest["runtime_identity"]["packages"]
        assert node._canonical_writer.records == ()
        assert not node._production_ready
        assert adapter.identity_calls == 1
        assert adapter.reset_calls == 0
        assert _response(env).provider_identity_json == response.provider_identity_json
    finally:
        node.destroy_node()


def test_generic_adapter_without_identity(robot_config, ros_context, monkeypatch):
    adapter = GenericAdapter()
    assert adapter.provider_identity() is None
    monkeypatch.setattr(environment_node, "load_plugin", lambda name: _plugin(adapter))
    env = environment_node.BenchmarkEnvironmentNode(parameter_overrides=_environment_parameters(robot_config))
    node = None
    try:
        env.start_production()
        response = _response(env)
        assert response.provider_identity_json == "null"
        node = _evaluator(robot_config)
        _accept_response(node, response)
        assert node._canonical_manifest["runtime_identity"]["provider"] is None
        assert adapter.reset_calls == 0
    finally:
        if node is not None:
            node.destroy_node()
        env.close()
        env.destroy_node()


@pytest.mark.parametrize("payload", ["", "[]", '"provider"', '{"name":NaN}', '{"name":1,"name":2}'])
def test_malformed_identity_rejected_before_manifest(environment, robot_config, payload):
    env, adapter = environment
    response = _response(env)
    response.provider_identity_json = payload
    node = _evaluator(robot_config)
    try:
        with pytest.raises(evaluator_node.EvaluatorStartupError, match="malformed provider identity"):
            _accept_response(node, response)
        assert node._canonical_manifest is None
        assert node._supervisor is None
        assert adapter.reset_calls == 0
    finally:
        node.destroy_node()


@pytest.mark.parametrize("identity", [["not an object"], {"version": float("nan")}, {1: "invalid key"}])
def test_invalid_adapter_identity_fails_startup(robot_config, ros_context, monkeypatch, identity):
    adapter = IdentityAdapter()
    adapter.identity = identity
    monkeypatch.setattr(environment_node, "load_plugin", lambda name: _plugin(adapter))
    env = environment_node.BenchmarkEnvironmentNode(parameter_overrides=_environment_parameters(robot_config))
    try:
        with pytest.raises(environment_node.EnvironmentStartupError):
            env.start_production()
        assert env._plan_srv_handle is None
        assert adapter.closed
        assert adapter.reset_calls == 0
    finally:
        env.close()
        env.destroy_node()


def test_unavailable_plan_clears_provider_identity(environment):
    env, _ = environment
    env._plan = None
    response = env._handle_get_plan(None, SimpleNamespace(provider_identity_json='{"stale":true}'))
    assert not response.success
    assert response.plan_json == ""
    assert response.provider_identity_json == "null"


def _serve_environment(config, with_identity, stop, connection):
    """Spawned ROS server: no shared adapter state with the evaluator process."""
    node = None
    rclpy.init(args=[])
    adapter = IdentityAdapter() if with_identity else GenericAdapter()
    try:
        with patch.object(environment_node, "load_plugin", return_value=_plugin(adapter)):
            node = environment_node.BenchmarkEnvironmentNode(parameter_overrides=_environment_parameters(Path(config)))
            node.start_production()
        if with_identity:
            adapter.identity["extension"]["verified"] = False
        connection.send({"ready": True})
        while not stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.05)
        connection.send({"reset_calls": adapter.reset_calls, "identity_calls": getattr(adapter, "identity_calls", 0)})
    except BaseException:
        connection.send({"error": traceback.format_exc()})
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.shutdown()
        connection.close()


@pytest.mark.parametrize("with_identity", [True, False])
def test_ros_plan_transport_from_separate_environment_process(robot_config, ros_context, with_identity):
    if "provider_identity_json" not in GetBenchmarkPlan.Response.get_fields_and_field_types():
        pytest.skip("GetBenchmarkPlan is stale: rebuild ibrobot_msgs before running ROS identity transport tests")
    context = multiprocessing.get_context("spawn")
    stop = context.Event()
    parent, child = context.Pipe()
    server = context.Process(target=_serve_environment, args=(str(robot_config), with_identity, stop, child))
    node = None
    server.start()
    child.close()
    try:
        assert parent.poll(20), "environment startup timed out"
        assert parent.recv() == {"ready": True}
        node = _evaluator(robot_config)
        deadline = time.monotonic() + 20
        while node._canonical_manifest is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            node._drive_plan_and_readiness()
        assert node._canonical_manifest is not None
        manifest_path = Path(node._canonical_manifest["run_root"]) / "run_manifest.json"
        identity = json.loads(manifest_path.read_text())["runtime_identity"]
        assert identity["provider"] == (IDENTITY if with_identity else None)
        assert "libero_provider" not in identity["packages"]
        assert not node._production_ready
        stop.set()
        assert parent.poll(5), "environment did not report startup call counts"
        assert parent.recv() == {"reset_calls": 0, "identity_calls": int(with_identity)}
    finally:
        stop.set()
        if node is not None:
            node.destroy_node()
        server.join(timeout=5)
        if server.is_alive():
            server.terminate()
            server.join(timeout=5)
        parent.close()
    assert server.exitcode == 0
