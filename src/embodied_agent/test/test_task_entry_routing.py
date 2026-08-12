"""Node-level routing tests for the generic ASR task entry."""

from unittest.mock import Mock

import rclpy
from std_msgs.msg import String

from embodied_agent.task_entry_node import TaskEntryNode


def _make_node() -> tuple[TaskEntryNode, list]:
    node = TaskEntryNode()
    collected = []
    node._publisher = Mock()  # noqa: SLF001
    node._publisher.publish.side_effect = collected.append  # noqa: SLF001
    return node, collected


def test_asr_text_is_forwarded_as_an_unplanned_task():
    rclpy.init()
    node = None
    try:
        node, collected = _make_node()
        node._handle_text_command(String(data="向前移动"))  # noqa: SLF001

        assert len(collected) == 1
        assert collected[0].source == "voice_asr"
        assert collected[0].raw_command == "向前移动"
        assert collected[0].task_type == "unplanned"
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_sorting_hat_text_has_no_visual_game_route():
    rclpy.init()
    node = None
    try:
        node, collected = _make_node()
        node._handle_text_command(String(data="分院帽"))  # noqa: SLF001

        assert len(collected) == 1
        assert collected[0].source == "voice_asr"
        assert collected[0].raw_command == "分院帽"
        assert not hasattr(node, "_visual_game_client")
        assert not hasattr(node, "_perception_publisher")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_empty_asr_text_is_ignored():
    rclpy.init()
    node = None
    try:
        node, collected = _make_node()
        node._handle_text_command(String(data="  "))  # noqa: SLF001

        assert collected == []
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_visual_game_routing_parameters_are_removed():
    rclpy.init()
    node = None
    try:
        node, _ = _make_node()

        for parameter_name in (
            "entry_visual_games_json",
            "perception_enabled",
            "perception_request_topic",
            "visual_games_json",
            "visual_game_aliases_json",
            "visual_game_start_service",
            "visual_game_config_digest",
        ):
            assert not node.has_parameter(parameter_name)
        assert node.has_parameter("output_topic")
        assert node.has_parameter("default_task_timeout_sec")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
