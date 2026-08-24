"""Live ROS contract test owned by the generic perception consumer."""

import json
import os
import threading
import time
import uuid

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter

from embodied_common.scene_analysis import SceneAnalysis
from ibrobot_msgs.msg import SceneAnalysisRequest, SceneAnalysisResult
from perception_service.perception_service_node import PerceptionServiceNode


def _assert_isolated_ros_domain() -> None:
    allocated = os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID", "")
    assert allocated.isdecimal() and os.environ.get("ROS_DOMAIN_ID") == allocated
    assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"


def test_real_perception_node_enforces_request_response_contract():
    _assert_isolated_ros_domain()
    rclpy.init()
    suffix = f"perception_contract_{os.getpid()}_{uuid.uuid4().hex}"
    request_topic = f"/{suffix}/request"
    result_topic = f"/{suffix}/result"
    perception = PerceptionServiceNode(
        parameter_overrides=[
            Parameter("request_topic", Parameter.Type.STRING, request_topic),
            Parameter("result_topic", Parameter.Type.STRING, result_topic),
        ]
    )
    perception._analyze = lambda request: (  # noqa: SLF001
        SceneAnalysis("赫奇帕奇", [], "", "", [], 0.9),
        '{"scene_summary": "赫奇帕奇"}',
        {},
        json.loads(request.context_json),
    )
    client = rclpy.create_node(f"perception_contract_client_{suffix}")
    request_publisher = client.create_publisher(SceneAnalysisRequest, request_topic, 10)
    observed_results = []
    client.create_subscription(SceneAnalysisResult, result_topic, observed_results.append, 10)
    executor = MultiThreadedExecutor(num_threads=3)
    for node in (perception, client):
        executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and request_publisher.get_subscription_count() == 0:
            time.sleep(0.01)
        assert request_publisher.get_subscription_count() > 0
        request = SceneAnalysisRequest()
        request.request_id = "perception-contract-1"
        request.source = "game.sorting_hat"
        request.context_json = json.dumps(
            {
                "response_contract": {
                    "field": "scene_summary",
                    "kind": "enum",
                    "allowed_values": ["斯莱特林", "格兰芬多", "拉文克劳", "赫奇帕奇", "无法判断"],
                }
            }
        )
        request_publisher.publish(request)

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not observed_results:
            time.sleep(0.01)
        assert len(observed_results) == 1
        result = observed_results[0]
        assert result.request_id == request.request_id
        assert result.source == request.source
        assert result.success is True
        assert result.scene_summary == "赫奇帕奇"
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        perception.destroy_node()
        client.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
