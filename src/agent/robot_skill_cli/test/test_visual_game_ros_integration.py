"""Live ROS service tests owned by the visual-game CLI consumer."""

import json
import os
import threading
import uuid
from types import SimpleNamespace

import rclpy
from rclpy.executors import MultiThreadedExecutor

from ibrobot_msgs.srv import GetVisualGameResult, StartVisualGame
from robot_skill_cli import catalog as robot_skill_catalog
from robot_skill_cli.catalog import GatewayTransport
from robot_skill_cli.cli import main as robot_skill_main


def _assert_isolated_ros_domain() -> None:
    allocated = os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID", "")
    assert allocated.isdecimal() and os.environ.get("ROS_DOMAIN_ID") == allocated
    assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"


def test_visual_game_commands_use_public_gateway_services(capsys, monkeypatch):
    """Exercise CLI parsing, context binding, RosBridge, and public services."""
    _assert_isolated_ros_domain()
    rclpy.init()
    suffix = f"cli_visual_game_{os.getpid()}_{uuid.uuid4().hex}"
    start_service = f"/{suffix}/start"
    result_service = f"/{suffix}/result"
    game_digest = "game-digest"
    request_id = f"game-{suffix}"
    observed_start_requests = []
    observed_result_requests = []
    server = rclpy.create_node(f"visual_game_cli_server_{suffix}")

    def start_game(request, response):
        observed_start_requests.append(request)
        response.accepted = True
        response.duplicate = False
        response.request_id = request.request_id
        response.config_digest = game_digest
        response.message = "accepted"
        return response

    def get_game_result(request, response):
        observed_result_requests.append(request)
        response.found = True
        response.terminal = True
        response.success = True
        response.game_name = "sorting_hat"
        response.scene_summary = "赫奇帕奇"
        response.result_json = '{"scene_summary":"赫奇帕奇"}'
        response.config_digest = game_digest
        response.message = "completed"
        return response

    server.create_service(StartVisualGame, start_service, start_game)
    server.create_service(GetVisualGameResult, result_service, get_game_result)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(server)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    context = SimpleNamespace(
        game_view={
            "config_digest": game_digest,
            "games": [
                {
                    "name": "sorting_hat",
                }
            ],
        },
        view={"timeout_policy": {"rpc_timeout_sec": 1.0}},
    )
    transport = GatewayTransport(
        status_service=f"/{suffix}/unused_status",
        validate_skill_service=f"/{suffix}/unused_validate",
        skill_action_name=f"/{suffix}/unused_skill",
        start_visual_game_service=start_service,
        get_visual_game_result_service=result_service,
    )
    monkeypatch.setattr(
        robot_skill_catalog,
        "load_visual_game_runtime_context",
        lambda **_kwargs: (context, transport),
    )
    try:
        assert robot_skill_main(["start-game", "sorting_hat", "--request-id", request_id]) == 0
        started = json.loads(capsys.readouterr().out)
        assert started["ok"] is True
        assert started["command"] == "start-game"
        assert started["data"] == {
            "accepted": True,
            "duplicate": False,
            "request_id": request_id,
            "config_digest": game_digest,
            "error_code": "",
            "message": "accepted",
        }

        assert robot_skill_main(["game-result", "--request-id", request_id]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["ok"] is True
        assert result["command"] == "game-result"
        assert result["data"]["request_id"] == request_id
        assert result["data"]["terminal"] is True
        assert result["data"]["success"] is True
        assert result["data"]["scene_summary"] == "赫奇帕奇"
        assert result["data"]["result_json"] == '{"scene_summary":"赫奇帕奇"}'

        assert len(observed_start_requests) == 1
        assert observed_start_requests[0].request_id == request_id
        assert observed_start_requests[0].game_name == "sorting_hat"
        assert observed_start_requests[0].expected_config_digest == game_digest
        assert len(observed_result_requests) == 1
        assert observed_result_requests[0].request_id == request_id
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        server.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
