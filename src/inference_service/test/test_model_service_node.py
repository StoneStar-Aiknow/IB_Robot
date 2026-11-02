from types import SimpleNamespace

import pytest

from inference_service.model_service_node import ModelServiceNode
from inference_service.model_service_plugin import ModelServiceError, PluginRuntimeStatus


class _Response(SimpleNamespace):
    success = False
    message = ""
    error_code = ""
    inference_time_ms = 0.0
    model = None


def _host(plugin):
    return SimpleNamespace(
        plugin=plugin,
        failure_reason="",
        runtime_info=lambda: "runtime-info",
    )


def test_request_error_does_not_poison_runtime_health():
    class Plugin:
        def runtime_status(self):
            return PluginRuntimeStatus(state="ready", ready=True)

        def handle(self, _request, _response):
            raise ModelServiceError("invalid request", error_code="INVALID_TEXT")

    host = _host(Plugin())
    response = ModelServiceNode._dispatch(host, object(), _Response())

    assert not response.success
    assert response.error_code == "INVALID_TEXT"
    assert response.model == "runtime-info"
    assert host.failure_reason == ""


def test_successful_request_after_request_error_stays_available():
    class Plugin:
        calls = 0

        def runtime_status(self):
            return PluginRuntimeStatus(state="ready", ready=True)

        def handle(self, _request, _response):
            self.calls += 1
            if self.calls == 1:
                raise ModelServiceError("unsupported prompt", error_code="UNSUPPORTED_PROMPT")
            return "ok"

    host = _host(Plugin())

    first = ModelServiceNode._dispatch(host, object(), _Response())
    second = ModelServiceNode._dispatch(host, object(), _Response())

    assert not first.success
    assert first.error_code == "UNSUPPORTED_PROMPT"
    assert second.success
    assert second.message == "ok"
    assert host.failure_reason == ""


def test_destroy_node_transfers_pending_plugin_before_provider_shutdown(monkeypatch):
    import rclpy

    from inference_service.backends.errors import BackendLifecycleError
    from inference_service.runtime_composition import build_runtime_dependencies
    from inference_service.unified_runtime import RuntimeProviders

    events = []

    def close():
        events.append("plugin")
        if events.count("plugin") == 1:
            raise BackendLifecycleError("native close pending", close_pending=True)

    plugin = SimpleNamespace(close=close)
    providers = RuntimeProviders(SimpleNamespace(close=lambda: events.append("acl")), object())
    dependencies = build_runtime_dependencies(providers=providers)
    monkeypatch.setattr(ModelServiceNode, "_initialize_plugin", lambda node, _: setattr(node, "plugin", plugin))
    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            "service_type:=ibrobot_msgs/srv/RecognizeTags",
            "-p",
            "service_endpoint:=/test_cleanup",
        ]
    )
    node = None
    try:
        node = ModelServiceNode(registry_set=dependencies.registry_set, providers=providers)
        node.destroy_node()
        node = None
        assert events == ["plugin"]
        providers.close()
        assert events == ["plugin", "plugin", "acl"]
    finally:
        if node is not None:
            node.destroy_node()
        providers.close()
        rclpy.try_shutdown()


@pytest.mark.parametrize("kind", ["model_service", "policy"])
@pytest.mark.parametrize("close_pending", [False, True])
def test_constructor_rolls_back_loaded_owner_when_ros_entity_creation_fails(monkeypatch, kind, close_pending):
    import rclpy
    from rclpy.node import Node

    from inference_service import pipeline_policy_node
    from inference_service.backends.errors import BackendLifecycleError
    from inference_service.runtime_composition import build_runtime_dependencies
    from inference_service.unified_runtime import RuntimeProviders

    events = []

    def close():
        events.append("model_close")
        if close_pending and events.count("model_close") == 1:
            raise BackendLifecycleError("still owned", close_pending=True)

    owner = SimpleNamespace(close=close)
    providers = RuntimeProviders(SimpleNamespace(close=lambda: events.append("provider_close")), object())
    dependencies = build_runtime_dependencies(providers=providers)
    original_destroy = Node.destroy_node

    def destroy(node):
        events.append("ros_destroy")
        return original_destroy(node)

    monkeypatch.setattr(Node, "destroy_node", destroy)
    cls = ModelServiceNode if kind == "model_service" else pipeline_policy_node.PipelinePolicyNode
    method = "create_service" if kind == "model_service" else "create_publisher"
    original_create = getattr(cls, method)

    def create(node, *args, **kwargs):
        if "loaded" in events:
            raise RuntimeError("ROS entity creation failed")
        return original_create(node, *args, **kwargs)

    monkeypatch.setattr(cls, method, create)
    if kind == "model_service":

        def initialize(node, _):
            node.plugin = owner
            events.append("loaded")

        monkeypatch.setattr(cls, "_initialize_plugin", initialize)

        def construct():
            return cls(registry_set=dependencies.registry_set, providers=providers)
    else:
        manifest = SimpleNamespace(policy=SimpleNamespace(n_obs_steps=1))
        monkeypatch.setattr(pipeline_policy_node, "load_inference_manifest", lambda *_: manifest)
        monkeypatch.setattr(cls, "_load_contract", lambda *_: None)
        monkeypatch.setattr(cls, "_setup_observation_subscriptions", lambda *_: None)

        def manager(*args, **kwargs):
            events.append("loaded")
            return owner

        monkeypatch.setattr(pipeline_policy_node, "create_pipeline_manager", manager)
        config = pipeline_policy_node.PipelineNodeConfig(
            pipeline_id="test",
            model_path="unused",
            deployment="cpu",
            execution_mode="monolithic",
            request_timeout=1.0,
            default_task="",
            runtime_options_json="{}",
            robot_config_path="unused",
            use_sim=False,
            action_server="/test_action",
            reset_service="/test_reset",
            health_topic="/health",
            action_topic="/action",
            request_topic="",
            result_topic="",
            heartbeat_topic="",
        )

        def construct():
            return cls(
                config, node_name="constructor_test", registry_set=dependencies.registry_set, providers=providers
            )

    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            "service_type:=ibrobot_msgs/srv/RecognizeTags",
            "-p",
            "service_endpoint:=/test_cleanup",
        ]
    )
    try:
        with pytest.raises(RuntimeError, match="ROS entity creation failed"):
            construct()
        assert events == ["loaded", "model_close", "ros_destroy"]
        providers.close()
        assert events == ["loaded", "model_close", "ros_destroy"] + (["model_close"] if close_pending else []) + [
            "provider_close"
        ]
    finally:
        providers.close()
        rclpy.try_shutdown()


@pytest.mark.parametrize("kind", ["model_service", "policy"])
@pytest.mark.parametrize("failed_owner", ["node", "provider"])
def test_service_main_attempts_all_shutdown_owners(monkeypatch, kind, failed_owner):
    from inference_service import model_service_node, pipeline_policy_node

    module = model_service_node if kind == "model_service" else pipeline_policy_node
    events = []

    def close(name):
        def run():
            events.append(name)
            if failed_owner == name:
                raise RuntimeError(f"{name} shutdown failed")

        return run

    def spin():
        raise KeyboardInterrupt

    node = SimpleNamespace(destroy_node=close("node"))
    dependencies = SimpleNamespace(registry_set=object(), providers=SimpleNamespace(close=close("provider")))
    monkeypatch.setattr(module.rclpy, "init", lambda **_: None)
    monkeypatch.setattr(module.rclpy, "ok", lambda: True)
    monkeypatch.setattr(module.rclpy, "shutdown", close("ros"))
    monkeypatch.setattr(module.rclpy, "try_shutdown", close("ros"))
    monkeypatch.setattr(
        module, "MultiThreadedExecutor", lambda **_: SimpleNamespace(add_node=lambda _: None, spin=spin)
    )
    if kind == "model_service":
        monkeypatch.setattr(module, "build_model_service_runtime_dependencies", lambda: dependencies)
        monkeypatch.setattr(module, "ModelServiceNode", lambda **_: node)
    else:
        monkeypatch.setattr(module, "_read_config", lambda: (object(), "test"))
        monkeypatch.setattr(module, "_pipeline_executor_threads", lambda _: 2)
        monkeypatch.setattr(module, "build_policy_runtime_dependencies", lambda: dependencies)
        monkeypatch.setattr(module, "PipelinePolicyNode", lambda *_args, **_kwargs: node)
    with pytest.raises(RuntimeError, match=f"{failed_owner} shutdown failed"):
        module.main()
    assert events == ["node", "provider", "ros"]


@pytest.mark.parametrize("failure_at", ["load", "start", "video"])
@pytest.mark.parametrize("cleanup_failure", ["session", "runtime", None])
def test_distributed_construction_failure_attempts_all_cleanup(monkeypatch, failure_at, cleanup_failure):
    from rclpy.node import Node

    from inference_service import pipeline_policy_node as module
    from inference_service.runtime_composition import build_runtime_dependencies
    from inference_service.unified_runtime import RuntimeProviders

    events = []
    original = RuntimeError(f"{failure_at} constructor failed")
    blocked = [True]

    def operation(name):
        def run(*_args, **_kwargs):
            events.append(name)
            if name == failure_at:
                raise original
            if name == f"{cleanup_failure}_close" and blocked[0]:
                raise RuntimeError("cleanup pending")
            return []

        return run

    runtime = SimpleNamespace(load=operation("load"), close=operation("runtime_close"))
    session = SimpleNamespace(start=operation("start"), close=operation("session_close"))
    providers = RuntimeProviders(SimpleNamespace(close=lambda: None), object())
    dependencies = build_runtime_dependencies(providers=providers)
    monkeypatch.setattr(Node, "__init__", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(Node, "destroy_node", operation("ros_destroy"))
    monkeypatch.setattr(Node, "get_logger", lambda _: SimpleNamespace(error=lambda *_: None))
    cls = module.PipelinePolicyNode
    monkeypatch.setattr(cls, "_load_contract", lambda *_: None)
    monkeypatch.setattr(cls, "_setup_observation_subscriptions", lambda *_: None)
    monkeypatch.setattr(cls, "_invalidate_visual_trigger", lambda *_: None)
    monkeypatch.setattr(cls, "_complete_invalidated", lambda *_: None)
    monkeypatch.setattr(
        module,
        "load_inference_manifest_metadata",
        lambda *_: SimpleNamespace(policy=SimpleNamespace(n_obs_steps=1), fingerprint="fingerprint"),
    )
    monkeypatch.setattr(module, "EdgeProcessorRuntime", lambda *_args, **_kwargs: runtime)
    monkeypatch.setattr(module, "EdgeSession", lambda *_: session)
    monkeypatch.setattr(module, "build_pipeline_identity", lambda *_: object())
    monkeypatch.setattr(module, "contract_fingerprint", lambda *_: "contract")
    monkeypatch.setattr(module, "_ExternalVideoProducerView", operation("video"))
    config = module.PipelineNodeConfig(
        pipeline_id="review",
        model_path="unused",
        deployment="cpu",
        execution_mode="distributed",
        request_timeout=1.0,
        default_task="",
        runtime_options_json="{}",
        robot_config_path="unused",
        use_sim=False,
        action_server="/action",
        reset_service="/reset",
        health_topic="/health",
        action_topic="/action",
        request_topic="/request",
        result_topic="/result",
        heartbeat_topic="/heartbeat",
        external_video_producer=True,
    )
    try:
        with pytest.raises(RuntimeError) as caught:
            cls(config, node_name="review", registry_set=dependencies.registry_set, providers=providers)
        assert caught.value is original
        assert "runtime_close" in events and "ros_destroy" in events
        if failure_at != "load":
            assert "session_close" in events
        assert events.index("runtime_close") < events.index("ros_destroy")
        if cleanup_failure == "runtime":
            blocked[0] = False
            providers.close()
            assert events.count("runtime_close") == 2
    finally:
        blocked[0] = False
        providers.close()
