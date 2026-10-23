"""Live ROS closure tests owned by the visual-game Agent control plane."""

import json
import os
import threading
import time
import uuid

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter

from embodied_agent.visual_game_gateway_node import VisualGameGatewayNode
from embodied_common.visual_game_contracts import build_visual_game_capability_view
from ibrobot_msgs.msg import SceneAnalysisRequest, SceneAnalysisResult, VisualGameEvent
from ibrobot_msgs.srv import GetVisualGameResult, StartVisualGame


def _assert_isolated_ros_domain() -> None:
    allocated = os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID", "")
    assert allocated.isdecimal() and os.environ.get("ROS_DOMAIN_ID") == allocated
    assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"


def _wait_until(predicate, *, timeout_sec: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _wait_future(future, *, timeout_sec: float = 2.0):
    assert _wait_until(future.done, timeout_sec=timeout_sec)
    return future.result()


def _game_view(games, *, start_service: str, result_service: str, event_topic: str):
    return build_visual_game_capability_view(
        "test_robot",
        games,
        timeout_sec=2.0,
        result_retention_sec=10.0,
        result_capacity=128,
        start_service=start_service,
        result_service=result_service,
        event_topic=event_topic,
    )


def test_gateway_services_and_perception_topics_form_one_live_chain():
    _assert_isolated_ros_domain()
    rclpy.init()
    suffix = f"game_e2e_{os.getpid()}_{uuid.uuid4().hex}"
    request_topic = f"/{suffix}/perception_request"
    result_topic = f"/{suffix}/perception_result"
    start_service = f"/{suffix}/start"
    result_service = f"/{suffix}/result"
    event_topic = f"/{suffix}/events"
    games = {
        "sorting_hat": {
            "enabled": True,
            "handler": "sorting_hat_v1",
            "summary": "Choose a Hogwarts house.",
        }
    }
    gateway = VisualGameGatewayNode(
        parameter_overrides=[
            Parameter("robot_name", Parameter.Type.STRING, "test_robot"),
            Parameter("perception_enabled", Parameter.Type.BOOL, True),
            Parameter("visual_games_json", Parameter.Type.STRING, json.dumps(games)),
            Parameter("perception_request_topic", Parameter.Type.STRING, request_topic),
            Parameter("perception_result_topic", Parameter.Type.STRING, result_topic),
            Parameter("start_service", Parameter.Type.STRING, start_service),
            Parameter("result_service", Parameter.Type.STRING, result_service),
            Parameter("event_topic", Parameter.Type.STRING, event_topic),
            Parameter("visual_game_timeout_sec", Parameter.Type.DOUBLE, 2.0),
            Parameter("result_retention_sec", Parameter.Type.DOUBLE, 10.0),
        ]
    )
    client_node = rclpy.create_node(f"visual_game_client_{suffix}")
    start_client = client_node.create_client(StartVisualGame, start_service)
    result_client = client_node.create_client(GetVisualGameResult, result_service)
    result_publisher = client_node.create_publisher(SceneAnalysisResult, result_topic, 10)
    observed_requests = []
    observed_events = []

    def respond(request: SceneAnalysisRequest) -> None:
        observed_requests.append(request)
        result = SceneAnalysisResult()
        result.request_id = request.request_id
        result.source = request.source
        result.success = True
        result.scene_summary = "赫奇帕奇"
        result.message = "scene analysis completed"
        result_publisher.publish(result)

    client_node.create_subscription(SceneAnalysisRequest, request_topic, respond, 10)
    client_node.create_subscription(VisualGameEvent, event_topic, observed_events.append, 10)
    executor = MultiThreadedExecutor(num_threads=3)
    for node in (gateway, client_node):
        executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        assert start_client.wait_for_service(timeout_sec=1.0)
        assert result_client.wait_for_service(timeout_sec=1.0)
        assert _wait_until(lambda: gateway._request_publisher.get_subscription_count() > 0)  # noqa: SLF001
        view = _game_view(
            games,
            start_service=start_service,
            result_service=result_service,
            event_topic=event_topic,
        )
        request = StartVisualGame.Request()
        request.request_id = "game-e2e-1"
        request.game_name = "sorting_hat"
        request.expected_config_digest = view["config_digest"]

        started = _wait_future(start_client.call_async(request))
        assert started.accepted is True
        assert _wait_until(lambda: len(observed_requests) == 1)

        result_request = GetVisualGameResult.Request()
        result_request.request_id = request.request_id
        result = None
        while result is None or not result.terminal:
            result = _wait_future(result_client.call_async(result_request))
            if not result.terminal:
                time.sleep(0.01)

        assert observed_requests[0].request_id != request.request_id
        assert observed_requests[0].request_id == gateway._records[request.request_id].execution_id  # noqa: SLF001
        assert result.success is True
        assert result.scene_summary == "赫奇帕奇"
        assert result.config_digest == view["config_digest"]
        assert _wait_until(
            lambda: any(
                event.request_id == request.request_id and event.state == "succeeded" for event in observed_events
            )
        )
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        gateway.destroy_node()
        client_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
