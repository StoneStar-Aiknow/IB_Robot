"""Node-level routing tests for task_entry_node.

Asserts the mutual-exclusion guarantee: one utterance belongs to exactly one
business domain. A matched visual-game trigger produces exactly one
SceneAnalysisRequest and zero TaskCommand / planned_task / task_status, while a
normal motion command keeps the original planning route and produces no game
request.

Follows the repo's single-node test idiom (see
perception_service/test/test_perception_node_observation.py): construct the real
node with parameter overrides, mock its publishers to collect messages, then
drive the handler directly.
"""

import json
from unittest.mock import Mock

import rclpy
from rclpy.parameter import Parameter
from std_msgs.msg import String

from embodied_agent.task_entry_node import TaskEntryNode

_SORTING_HAT_GAMES = json.dumps(
    {"sorting_hat": {"enabled": True, "trigger_aliases": ["分院帽"]}}
)


def _make_node(*, perception_enabled: bool) -> tuple[TaskEntryNode, dict]:
    node = TaskEntryNode(
        parameter_overrides=[
            Parameter("entry_visual_games_json", Parameter.Type.STRING, _SORTING_HAT_GAMES),
            Parameter("perception_enabled", Parameter.Type.BOOL, perception_enabled),
        ]
    )
    collected = {"task": [], "planned": [], "status": [], "perception": []}
    node._publisher = Mock()  # noqa: SLF001
    node._publisher.publish.side_effect = collected["task"].append  # noqa: SLF001
    node._planned_publisher = Mock()  # noqa: SLF001
    node._planned_publisher.publish.side_effect = collected["planned"].append  # noqa: SLF001
    node._status_publisher = Mock()  # noqa: SLF001
    node._status_publisher.publish.side_effect = collected["status"].append  # noqa: SLF001
    node._perception_publisher = Mock()  # noqa: SLF001
    node._perception_publisher.publish.side_effect = collected["perception"].append  # noqa: SLF001
    return node, collected


def test_game_trigger_produces_only_perception_request():
    rclpy.init()
    try:
        node, collected = _make_node(perception_enabled=True)
        node._handle_text_command(String(data="来玩分院帽"))  # noqa: SLF001

        assert len(collected["perception"]) == 1
        assert collected["perception"][0].source == "game.sorting_hat"
        # Mutually exclusive: no task planning artifacts for the same utterance.
        assert collected["task"] == []
        assert collected["planned"] == []
        assert collected["status"] == []
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_motion_command_keeps_task_route_and_emits_no_game():
    rclpy.init()
    try:
        node, collected = _make_node(perception_enabled=True)
        node._handle_text_command(String(data="向前移动"))  # noqa: SLF001

        assert collected["perception"] == []
        # A recognized motion command routes to the planned-task path; either way
        # it must reach the task domain, never the game domain.
        assert (len(collected["planned"]) + len(collected["task"])) == 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_game_dropped_when_perception_disabled():
    rclpy.init()
    try:
        node, collected = _make_node(perception_enabled=False)
        node._handle_text_command(String(data="分院帽"))  # noqa: SLF001

        # Matched trigger returns early; perception is off so nothing is published,
        # and the utterance must not fall through to task planning either.
        assert collected["perception"] == []
        assert collected["task"] == []
        assert collected["planned"] == []
        assert collected["status"] == []
    finally:
        node.destroy_node()
        rclpy.shutdown()
