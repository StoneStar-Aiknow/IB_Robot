"""Tests for the Agent visual game control plane."""

import json
from unittest.mock import Mock

import rclpy
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy

from embodied_agent.visual_game_gateway_node import VisualGameGatewayNode
from embodied_agent.visual_game_qos import visual_game_event_qos
from embodied_common.visual_game_contracts import build_visual_game_capability_view
from ibrobot_msgs.msg import SceneAnalysisResult
from ibrobot_msgs.srv import GetVisualGameResult, StartVisualGame


def test_visual_game_event_qos_is_reliable_replayable_and_retention_bounded():
    profile = visual_game_event_qos(depth=0, lifespan_sec=12.5)

    assert profile.depth == 1
    assert profile.reliability == ReliabilityPolicy.RELIABLE
    assert profile.durability == DurabilityPolicy.TRANSIENT_LOCAL
    assert profile.lifespan.nanoseconds == 12_500_000_000


_GAMES = json.dumps(
    {
        "sorting_hat": {
            "enabled": True,
            "handler": "sorting_hat_v1",
            "summary": "Choose a Hogwarts house.",
        }
    }
)

_ANNOUNCING_GAMES = json.dumps(
    {
        "sorting_hat": {
            "enabled": True,
            "handler": "sorting_hat_v1",
            "summary": "Choose a Hogwarts house.",
            "announce": True,
        }
    }
)


def _make_node(
    *,
    perception_enabled: bool = True,
    capacity: int = 128,
    timeout_sec: float = 130.0,
    retention_sec: float = 300.0,
    games_json: str = _GAMES,
    event_topic: str = "/embodied/visual_game_events",
    debug: bool = False,
):
    node = VisualGameGatewayNode(
        parameter_overrides=[
            Parameter("perception_enabled", Parameter.Type.BOOL, perception_enabled),
            Parameter("robot_name", Parameter.Type.STRING, "test_robot"),
            Parameter("visual_games_json", Parameter.Type.STRING, games_json),
            Parameter("result_capacity", Parameter.Type.INTEGER, capacity),
            Parameter("visual_game_timeout_sec", Parameter.Type.DOUBLE, timeout_sec),
            Parameter("result_retention_sec", Parameter.Type.DOUBLE, retention_sec),
            Parameter("event_topic", Parameter.Type.STRING, event_topic),
            Parameter("debug_tracing", Parameter.Type.BOOL, debug),
        ]
    )
    published = []
    node._request_publisher = Mock()  # noqa: SLF001
    node._request_publisher.get_subscription_count.return_value = 1  # noqa: SLF001
    node._request_publisher.publish.side_effect = published.append  # noqa: SLF001
    node._event_publisher = Mock()  # noqa: SLF001
    return node, published


def test_custom_event_topic_is_included_in_gateway_config_digest():
    rclpy.init()
    node = None
    try:
        node, _published = _make_node(event_topic="/custom/visual_game_events")
        expected = build_visual_game_capability_view(
            "test_robot",
            json.loads(_GAMES),
            timeout_sec=130.0,
            result_retention_sec=300.0,
            result_capacity=128,
            start_service="/embodied/start_visual_game",
            result_service="/embodied/get_visual_game_result",
            event_topic="/custom/visual_game_events",
        )

        assert node._config_digest == expected["config_digest"]  # noqa: SLF001
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def _start(node, game_name: str = "sorting_hat", request_id: str = "game-test-1", digest: str | None = None):
    request = StartVisualGame.Request()
    request.request_id = request_id
    request.game_name = game_name
    request.expected_config_digest = node._config_digest if digest is None else digest  # noqa: SLF001
    return node._handle_start_game(request, StartVisualGame.Response())  # noqa: SLF001


def _get(node, request_id: str):
    request = GetVisualGameResult.Request()
    request.request_id = request_id
    return node._handle_get_result(request, GetVisualGameResult.Response())  # noqa: SLF001


def test_oversized_request_id_is_rejected_before_side_effects():
    rclpy.init()
    node = None
    try:
        node, published = _make_node()
        response = _start(node, request_id="x" * 129)

        assert response.accepted is False
        assert response.error_code == "INVALID_REQUEST_ID"
        assert "at most 128" in response.message
        assert published == []
        node._event_publisher.publish.assert_not_called()  # noqa: SLF001
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def _published_events(node):
    return [call.args[0] for call in node._event_publisher.publish.call_args_list]  # noqa: SLF001


def _assert_rejection_event(node, *, request_id: str, error_code: str):
    events = _published_events(node)
    assert events[-1].request_id == request_id
    assert events[-1].game_name == "sorting_hat"
    assert events[-1].handler == "sorting_hat_v1"
    assert events[-1].state == "failed"
    assert events[-1].success is False
    assert events[-1].error_code == error_code


def test_start_returns_request_id_then_result_reaches_terminal():
    rclpy.init()
    node = None
    try:
        node, published = _make_node()
        started = _start(node)

        assert started.accepted is True
        assert started.duplicate is False
        assert started.request_id == "game-test-1"
        assert started.config_digest == node._config_digest  # noqa: SLF001
        assert len(published) == 1
        assert published[0].request_id == node._records[started.request_id].execution_id  # noqa: SLF001
        assert published[0].source == "game.sorting_hat"
        assert published[0].timeout_sec == 120.0

        pending = _get(node, started.request_id)
        assert pending.found is True
        assert pending.terminal is False
        assert pending.scene_summary == ""

        result = SceneAnalysisResult()
        result.request_id = node._records[started.request_id].execution_id  # noqa: SLF001
        result.source = "game.sorting_hat"
        result.success = True
        result.scene_summary = "拉文克劳"
        result.message = "scene analysis completed"
        node._handle_perception_result(result)  # noqa: SLF001

        terminal = _get(node, started.request_id)
        assert terminal.found is True
        assert terminal.terminal is True
        assert terminal.success is True
        assert terminal.game_name == "sorting_hat"
        assert terminal.scene_summary == "拉文克劳"
        assert json.loads(terminal.result_json)["scene_summary"] == "拉文克劳"
        assert terminal.message == "scene analysis completed"
        events = [call.args[0] for call in node._event_publisher.publish.call_args_list]  # noqa: SLF001
        assert [event.state for event in events] == ["accepted", "succeeded"]
        assert events[-1].request_id == started.request_id
        assert events[-1].handler == "sorting_hat_v1"
        assert events[0].announce is False
        assert events[-1].announce is False
        assert events[-1].message == "scene analysis completed"
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_dropped_perception_results_are_logged_by_drop_reason():
    rclpy.init()
    node = None
    try:
        node, _published = _make_node(debug=True)
        logger = Mock()
        node.get_logger = Mock(return_value=logger)  # noqa: SLF001
        started = _start(node)
        execution_id = node._records[started.request_id].execution_id  # noqa: SLF001

        unknown = SceneAnalysisResult()
        unknown.request_id = "unknown-request"
        unknown.source = "game.sorting_hat"
        node._handle_perception_result(unknown)  # noqa: SLF001

        wrong_source = SceneAnalysisResult()
        wrong_source.request_id = execution_id
        wrong_source.source = "vlm_planner"
        node._handle_perception_result(wrong_source)  # noqa: SLF001

        terminal = SceneAnalysisResult()
        terminal.request_id = execution_id
        terminal.source = "game.sorting_hat"
        terminal.success = True
        terminal.scene_summary = "拉文克劳"
        node._handle_perception_result(terminal)  # noqa: SLF001
        node._handle_perception_result(terminal)  # noqa: SLF001

        messages = [call.args[0] for call in logger.info.call_args_list]
        assert any("unknown request_id=unknown-request" in message for message in messages)
        assert any("expected source=game.sorting_hat" in message for message in messages)
        assert any("already terminal" in message for message in messages)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_disabled_perception_rejects_without_perception_request_or_ledger_record():
    rclpy.init()
    node = None
    try:
        node, published = _make_node(perception_enabled=False)

        started = _start(node)

        assert started.accepted is False
        assert started.error_code == "PERCEPTION_DISABLED"
        assert published == []
        assert _get(node, "game-test-1").found is False
        _assert_rejection_event(node, request_id="game-test-1", error_code="PERCEPTION_DISABLED")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_missing_perception_subscriber_rejects_without_reserving_request_id():
    rclpy.init()
    node = None
    try:
        node, published = _make_node()
        node._request_publisher.get_subscription_count.return_value = 0  # noqa: SLF001

        started = _start(node)

        assert started.accepted is False
        assert started.error_code == "PERCEPTION_UNAVAILABLE"
        assert published == []
        assert _get(node, "game-test-1").found is False
        _assert_rejection_event(node, request_id="game-test-1", error_code="PERCEPTION_UNAVAILABLE")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_second_request_is_rejected_while_a_game_is_running():
    rclpy.init()
    node = None
    try:
        node, published = _make_node(capacity=1)
        first = _start(node, request_id="game-test-1")

        second = _start(node, request_id="game-test-2")

        assert first.accepted is True
        assert second.accepted is False
        assert second.error_code == "GAME_BUSY"
        assert len(published) == 1
        assert _get(node, first.request_id).found is True
        assert _get(node, second.request_id).found is False
        assert [event.state for event in _published_events(node)] == ["accepted", "failed"]
        _assert_rejection_event(node, request_id="game-test-2", error_code="GAME_BUSY")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_game_busy_rejection_is_not_announced_for_an_announcing_game():
    rclpy.init()
    node = None
    try:
        node, _published = _make_node(games_json=_ANNOUNCING_GAMES, capacity=1)
        first = _start(node, request_id="game-test-1")
        second = _start(node, request_id="game-test-2")

        assert first.accepted is True
        assert second.accepted is False
        assert second.error_code == "GAME_BUSY"
        events = _published_events(node)
        accepted_event, busy_event = events
        # The admitted request still announces (the game's own terminal result
        # is worth speaking). The busy rejection must not: it uses a fresh
        # request_id every collision, so announcing it would repeat TTS under
        # rapid retry.
        assert accepted_event.announce is True
        assert busy_event.announce is False
        assert busy_event.state == "failed"
        assert busy_event.error_code == "GAME_BUSY"
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_non_busy_rejection_still_announces_for_an_announcing_game():
    rclpy.init()
    node = None
    try:
        node, _published = _make_node(games_json=_ANNOUNCING_GAMES, perception_enabled=False)

        started = _start(node)

        assert started.accepted is False
        assert started.error_code == "PERCEPTION_DISABLED"
        events = _published_events(node)
        assert len(events) == 1
        # One-shot rejections (config/perception) are not collision-driven, so
        # they keep announcing their diagnostic text.
        assert events[0].announce is True
        assert events[0].error_code == "PERCEPTION_DISABLED"
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_new_request_accepted_after_prior_game_terminal():
    rclpy.init()
    node = None
    try:
        node, published = _make_node(capacity=2)
        first = _start(node, request_id="game-test-1")
        result = SceneAnalysisResult()
        result.request_id = node._records[first.request_id].execution_id  # noqa: SLF001
        result.source = "game.sorting_hat"
        result.success = True
        result.scene_summary = "斯莱特林"
        node._handle_perception_result(result)  # noqa: SLF001

        second = _start(node, request_id="game-test-2")

        assert second.accepted is True
        assert second.error_code == ""
        assert len(published) == 2
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_unexpired_terminal_request_is_not_evicted_for_a_new_start():
    rclpy.init()
    node = None
    try:
        node, published = _make_node(capacity=1)
        first = _start(node, request_id="game-test-1")
        result = SceneAnalysisResult()
        result.request_id = node._records[first.request_id].execution_id  # noqa: SLF001
        result.source = "game.sorting_hat"
        result.success = True
        result.scene_summary = "斯莱特林"
        node._handle_perception_result(result)  # noqa: SLF001

        second = _start(node, request_id="game-test-2")

        assert second.accepted is False
        assert second.error_code == "GAME_CAPACITY_EXHAUSTED"
        assert len(published) == 1
        assert _get(node, first.request_id).found is True
        assert _get(node, second.request_id).found is False
        _assert_rejection_event(node, request_id="game-test-2", error_code="GAME_CAPACITY_EXHAUSTED")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_unrelated_perception_result_is_ignored():
    rclpy.init()
    node = None
    try:
        node, _published = _make_node()
        started = _start(node)
        unrelated = SceneAnalysisResult()
        unrelated.request_id = node._records[started.request_id].execution_id  # noqa: SLF001
        unrelated.source = "vlm_planner"
        unrelated.success = True
        unrelated.scene_summary = "not a house"

        node._handle_perception_result(unrelated)  # noqa: SLF001

        assert _get(node, started.request_id).terminal is False
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_perception_runtime_failure_reaches_terminal_without_waiting_for_gateway_timeout():
    rclpy.init()
    node = None
    try:
        node, _published = _make_node()
        started = _start(node)
        failed = SceneAnalysisResult()
        failed.request_id = node._records[started.request_id].execution_id  # noqa: SLF001
        failed.source = "game.sorting_hat"
        failed.success = False
        failed.error_code = "SCENE_ANALYSIS_FAILED"
        failed.message = "primary camera image is unavailable"

        node._handle_perception_result(failed)  # noqa: SLF001

        terminal = _get(node, started.request_id)
        assert terminal.terminal is True
        assert terminal.success is False
        # Perception-internal codes collapse onto the Gateway public surface so
        # callers never see undocumented codes; the original code is preserved in
        # the message for diagnostics.
        assert terminal.error_code == "PERCEPTION_FAILED"
        assert "SCENE_ANALYSIS_FAILED" in terminal.message
        assert "primary camera image is unavailable" in terminal.message
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_perception_failure_with_empty_error_code_is_failed_closed():
    rclpy.init()
    node = None
    try:
        node, _published = _make_node()
        started = _start(node)
        failed = SceneAnalysisResult()
        failed.request_id = node._records[started.request_id].execution_id  # noqa: SLF001
        failed.source = "game.sorting_hat"
        failed.success = False
        failed.error_code = ""
        failed.message = ""

        node._handle_perception_result(failed)  # noqa: SLF001

        terminal = _get(node, started.request_id)
        assert terminal.terminal is True
        assert terminal.success is False
        assert terminal.error_code == "PERCEPTION_FAILED"
        assert terminal.message == "PERCEPTION_FAILED"
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_invalid_success_result_is_failed_closed_at_gateway_boundary():
    rclpy.init()
    node = None
    try:
        node, _published = _make_node()
        started = _start(node)
        invalid = SceneAnalysisResult()
        invalid.request_id = node._records[started.request_id].execution_id  # noqa: SLF001
        invalid.source = "game.sorting_hat"
        invalid.success = True
        invalid.scene_summary = "不存在的学院"

        node._handle_perception_result(invalid)  # noqa: SLF001

        terminal = _get(node, started.request_id)
        assert terminal.terminal is True
        assert terminal.success is False
        assert terminal.error_code == "INVALID_GAME_RESULT"
        assert terminal.scene_summary == ""
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_undetermined_success_result_becomes_no_person_terminal_failure():
    rclpy.init()
    node = None
    try:
        node, _published = _make_node()
        started = _start(node)
        undetermined = SceneAnalysisResult()
        undetermined.request_id = node._records[started.request_id].execution_id  # noqa: SLF001
        undetermined.source = "game.sorting_hat"
        undetermined.success = True
        undetermined.scene_summary = "无法判断"

        node._handle_perception_result(undetermined)  # noqa: SLF001

        terminal = _get(node, started.request_id)
        assert terminal.terminal is True
        assert terminal.success is False
        assert terminal.error_code == "NO_PERSON"
        assert terminal.scene_summary == ""
        assert terminal.result_json == ""
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_pending_duplicate_start_is_idempotent_without_new_side_effects():
    rclpy.init()
    node = None
    try:
        node, published = _make_node()

        first = _start(node)
        duplicate = _start(node)

        assert first.accepted is True
        assert duplicate.accepted is True
        assert duplicate.duplicate is True
        assert duplicate.request_id == first.request_id
        assert len(published) == 1
        assert node._event_publisher.publish.call_count == 1  # noqa: SLF001
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_terminal_duplicate_start_does_not_publish_state_regression_event():
    rclpy.init()
    node = None
    try:
        node, published = _make_node()
        first = _start(node)
        result = SceneAnalysisResult()
        result.request_id = node._records[first.request_id].execution_id  # noqa: SLF001
        result.source = "game.sorting_hat"
        result.success = True
        result.scene_summary = "拉文克劳"
        node._handle_perception_result(result)  # noqa: SLF001

        duplicate = _start(node)

        assert duplicate.accepted is True
        assert duplicate.duplicate is True
        assert len(published) == 1
        events = [call.args[0] for call in node._event_publisher.publish.call_args_list]  # noqa: SLF001
        assert [event.state for event in events] == ["accepted", "succeeded"]
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_config_mismatch_rejects_before_perception_without_reserving_request_id():
    rclpy.init()
    node = None
    try:
        node, published = _make_node()

        started = _start(node, digest="stale-digest")

        assert started.accepted is False
        assert started.error_code == "CONFIG_MISMATCH"
        assert started.config_digest == node._config_digest  # noqa: SLF001
        assert published == []
        assert _get(node, "game-test-1").found is False
        _assert_rejection_event(node, request_id="game-test-1", error_code="CONFIG_MISMATCH")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_pending_request_reaches_timeout_terminal_and_late_result_is_ignored():
    rclpy.init()
    node = None
    try:
        node, _published = _make_node(timeout_sec=1.0)
        started = _start(node)
        record = node._records[started.request_id]  # noqa: SLF001
        record.deadline_monotonic = 0.0

        late = SceneAnalysisResult()
        late.request_id = record.execution_id
        late.source = "game.sorting_hat"
        late.success = True
        late.scene_summary = "格兰芬多"
        node._handle_perception_result(late)  # noqa: SLF001

        still_timed_out = _get(node, started.request_id)
        assert still_timed_out.terminal is True
        assert still_timed_out.success is False
        assert still_timed_out.error_code == "GAME_RESULT_TIMEOUT"
        assert still_timed_out.scene_summary == ""
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_timeout_terminal_event_is_not_announced():
    rclpy.init()
    node = None
    try:
        node, _published = _make_node(games_json=_ANNOUNCING_GAMES, timeout_sec=1.0)
        started = _start(node)
        record = node._records[started.request_id]  # noqa: SLF001
        # Force the deadline into the past so the next expiry sweep times it out.
        record.deadline_monotonic = 0.0
        node._expire_records(now=1.0)  # noqa: SLF001

        timed_out_event = _published_events(node)[-1]
        assert timed_out_event.state == "failed"
        assert timed_out_event.success is False
        assert timed_out_event.announce is True
        assert timed_out_event.error_code == "GAME_RESULT_TIMEOUT"
        from embodied_common.visual_game_contracts import get_visual_game_announcement

        text = get_visual_game_announcement(
            timed_out_event.handler,
            state=timed_out_event.state,
            success=timed_out_event.success,
            error_code=timed_out_event.error_code,
            result={},
        )
        assert text is None
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_terminal_record_expires_after_retention():
    rclpy.init()
    node = None
    try:
        node, _published = _make_node(retention_sec=1.0)
        started = _start(node)
        result = SceneAnalysisResult()
        result.request_id = node._records[started.request_id].execution_id  # noqa: SLF001
        result.source = "game.sorting_hat"
        result.success = True
        result.scene_summary = "赫奇帕奇"
        node._handle_perception_result(result)  # noqa: SLF001
        record = node._records[started.request_id]  # noqa: SLF001

        node._expire_records(record.terminal_at_monotonic + 1.0)  # noqa: SLF001

        assert _get(node, started.request_id).found is False
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_request_id_can_be_reused_only_after_terminal_record_expires():
    rclpy.init()
    node = None
    try:
        node, published = _make_node(retention_sec=1.0)
        started = _start(node)
        result = SceneAnalysisResult()
        result.request_id = node._records[started.request_id].execution_id  # noqa: SLF001
        result.source = "game.sorting_hat"
        result.success = True
        result.scene_summary = "赫奇帕奇"
        node._handle_perception_result(result)  # noqa: SLF001
        record = node._records[started.request_id]  # noqa: SLF001
        first_execution_id = record.execution_id
        node._expire_records(record.terminal_at_monotonic + 1.0)  # noqa: SLF001

        restarted = _start(node)
        second_execution_id = node._records[restarted.request_id].execution_id  # noqa: SLF001

        assert restarted.accepted is True
        assert restarted.duplicate is False
        assert len(published) == 2
        assert first_execution_id
        assert second_execution_id
        assert first_execution_id != second_execution_id

        late_result = SceneAnalysisResult()
        late_result.request_id = first_execution_id
        late_result.source = "game.sorting_hat"
        late_result.success = True
        late_result.scene_summary = "斯莱特林"
        node._handle_perception_result(late_result)  # noqa: SLF001
        assert _get(node, restarted.request_id).terminal is False

        current_result = SceneAnalysisResult()
        current_result.request_id = second_execution_id
        current_result.source = "game.sorting_hat"
        current_result.success = True
        current_result.scene_summary = "赫奇帕奇"
        node._handle_perception_result(current_result)  # noqa: SLF001
        current = _get(node, restarted.request_id)
        assert current.terminal is True
        assert current.scene_summary == "赫奇帕奇"
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
