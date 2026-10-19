"""Tests for visual-game TTS request ownership and deduplication."""

import io
import wave
from array import array
from pathlib import Path
from unittest.mock import Mock

import pytest
import rclpy
from rclpy.parameter import Parameter

from embodied_agent.visual_game_announcer_node import (
    VisualGameAnnouncerNode,
    _Announcement,
    announcement_text,
)
from ibrobot_msgs.msg import SynthesizedAudio, VisualGameEvent
from ibrobot_msgs.srv import PlayAudioFile, SynthesizeSpeech


def _event(
    *,
    state="succeeded",
    success=True,
    summary="拉文克劳",
    error_code="",
    announce=True,
    request_id="game-test-1",
    execution_id="execution-1",
):
    event = VisualGameEvent()
    event.request_id = request_id
    event.execution_id = execution_id
    event.game_name = "sorting_hat"
    event.handler = "sorting_hat_v1"
    event.announce = announce
    event.state = state
    event.success = success
    event.scene_summary = summary
    event.error_code = error_code
    return event


def test_announcement_text_is_handler_owned():
    assert announcement_text(_event()) == "拉文克劳"


def test_no_person_uses_non_triggering_guidance():
    text = announcement_text(_event(state="failed", success=False, summary="", error_code="NO_PERSON"))
    assert text == "暂未识别到新生，请走入画面中央"
    assert "分院帽" not in text


@pytest.mark.parametrize(
    ("error_code", "expected"),
    [
        ("CONFIG_MISMATCH", "视觉游戏配置不一致，请联系管理员"),
        ("PERCEPTION_DISABLED", "视觉感知服务尚未启用"),
        ("PERCEPTION_UNAVAILABLE", "视觉感知服务暂不可用，请稍后再试"),
        ("GAME_CAPACITY_EXHAUSTED", "视觉游戏结果空间已满，请稍后再试"),
    ],
)
def test_admission_failures_have_handler_owned_announcements(error_code, expected):
    assert announcement_text(_event(state="failed", success=False, summary="", error_code=error_code)) == expected


def test_non_terminal_and_unknown_failures_are_silent():
    assert announcement_text(_event(announce=False)) is None
    assert announcement_text(_event(state="accepted")) is None
    assert (
        announcement_text(_event(state="failed", success=False, summary="", error_code="GAME_RESULT_TIMEOUT")) is None
    )
    assert announcement_text(_event(state="failed", success=False, summary="", error_code="PERCEPTION_BUSY")) is None
    assert announcement_text(_event(state="failed", success=False, summary="", error_code="GAME_BUSY")) is None
    assert announcement_text(_event(summary="不存在的学院")) is None
    assert announcement_text(_event(state="failed", success=True)) is None
    event = _event()
    event.handler = "missing_v1"
    assert announcement_text(event) is None


def _node(*, service_ready=True, max_attempts=3):
    node = VisualGameAnnouncerNode(
        parameter_overrides=[Parameter("max_delivery_attempts", Parameter.Type.INTEGER, max_attempts)]
    )
    client = Mock()
    client.service_is_ready.return_value = service_ready
    future = Mock()
    future.add_done_callback = Mock()
    client.call_async.return_value = future
    node._tts_client = client  # noqa: SLF001
    playback_client = Mock()
    playback_client.service_is_ready.return_value = True
    playback_future = Mock()
    playback_future.add_done_callback = Mock()
    playback_client.call_async.return_value = playback_future
    node._playback_client = playback_client  # noqa: SLF001
    return node, client, future


def _wav_bytes():
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * 16)
    return output.getvalue()


def _audio_segment(index=0, *, pause_after_ms=0):
    return SynthesizedAudio(
        index=index,
        text="segment",
        audio_data=array("B", _wav_bytes()),
        audio_format="wav_pcm_s16le",
        sample_rate=16000,
        channels=1,
        duration_sec=0.001,
        pause_after_ms=pause_after_ms,
    )


def test_ready_tts_called_once_and_duplicate_request_is_suppressed():
    rclpy.init()
    node = None
    try:
        node, client, _future = _node()
        event = _event()
        node._handle_event(event)  # noqa: SLF001
        node._handle_event(event)  # noqa: SLF001
        assert client.call_async.call_count == 1
        assert client.call_async.call_args.args[0].text == "拉文克劳"
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_new_game_supersedes_pending_announcement_from_previous_round():
    rclpy.init()
    node = None
    try:
        node, client, _future = _node(service_ready=False)
        node._handle_event(_event(request_id="game-old", execution_id="execution-old"))  # noqa: SLF001

        node._handle_event(  # noqa: SLF001
            _event(state="accepted", success=False, summary="", request_id="game-new", execution_id="execution-new")
        )

        assert "execution-old" not in node._pending_announcements  # noqa: SLF001
        assert "execution-old" in node._completed_announcements  # noqa: SLF001
        assert "execution-old" not in node._active_announcements  # noqa: SLF001
        assert client.call_async.call_count == 0
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_new_game_cancels_previous_round_tts_before_playback():
    rclpy.init()
    node = None
    try:
        node, client, future = _node()
        future.done.return_value = False
        node._handle_event(_event(request_id="game-old", execution_id="execution-old"))  # noqa: SLF001
        old_generation = node._tts_generation  # noqa: SLF001

        node._handle_event(  # noqa: SLF001
            _event(state="accepted", success=False, summary="", request_id="game-new", execution_id="execution-new")
        )

        future.cancel.assert_called_once_with()
        assert node._tts_generation == old_generation + 1  # noqa: SLF001
        assert node._tts_inflight is False  # noqa: SLF001
        assert node._tts_announcement is None  # noqa: SLF001
        assert "execution-old" in node._completed_announcements  # noqa: SLF001
        assert client.call_async.call_count == 1
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_new_game_does_not_interrupt_previous_round_playback():
    rclpy.init()
    node = None
    try:
        node, _client, tts_future = _node()
        node._handle_event(_event(request_id="game-old", execution_id="execution-old"))  # noqa: SLF001
        tts_future.result.return_value = SynthesizeSpeech.Response(
            success=True,
            audio_segments=[_audio_segment()],
        )
        node._handle_tts_response(node._tts_announcement, tts_future, node._tts_generation)  # noqa: SLF001
        playback_announcement = node._playback_announcement  # noqa: SLF001
        playback_future = node._playback_future  # noqa: SLF001

        node._handle_event(  # noqa: SLF001
            _event(state="accepted", success=False, summary="", request_id="game-new", execution_id="execution-new")
        )

        assert node._playback_announcement is playback_announcement  # noqa: SLF001
        assert node._playback_future is playback_future  # noqa: SLF001
        assert node._playback_inflight is True  # noqa: SLF001
        playback_future.cancel.assert_not_called()
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.shutdown()


def test_superseded_tts_late_response_cannot_start_playback():
    rclpy.init()
    node = None
    try:
        node, _client, old_future = _node()
        old_future.done.return_value = False
        node._handle_event(_event(request_id="game-old", execution_id="execution-old"))  # noqa: SLF001
        old_announcement = node._tts_announcement  # noqa: SLF001
        old_generation = node._tts_generation  # noqa: SLF001

        node._handle_event(  # noqa: SLF001
            _event(state="accepted", success=False, summary="", request_id="game-new", execution_id="execution-new")
        )
        old_future.result.return_value = SynthesizeSpeech.Response(
            success=True,
            audio_segments=[_audio_segment()],
        )
        node._handle_tts_response(old_announcement, old_future, old_generation)  # noqa: SLF001

        assert node._playback_announcement is None  # noqa: SLF001
        assert node._playback_client.call_async.call_count == 0  # noqa: SLF001
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_unavailable_tts_is_delivered_if_service_recovers_within_grace_period(monkeypatch):
    rclpy.init()
    node = None
    try:
        now = 100.0
        monkeypatch.setattr("embodied_agent.visual_game_announcer_node.time.monotonic", lambda: now)
        node, client, _future = _node(service_ready=False)
        error_code = node._handle_event(_event())  # noqa: SLF001
        assert client.call_async.call_count == 0
        assert error_code is None
        assert "execution-1" in node._pending_announcements  # noqa: SLF001
        assert "execution-1" not in node._completed_announcements  # noqa: SLF001

        now = 102.9
        client.service_is_ready.return_value = True
        node._service_announcement_queues()  # noqa: SLF001
        assert client.call_async.call_count == 1
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_unavailable_tts_keeps_pending_batch_and_recovers_in_fifo_order(monkeypatch):
    rclpy.init()
    node = None
    try:
        now = 100.0
        monkeypatch.setattr("embodied_agent.visual_game_announcer_node.time.monotonic", lambda: now)
        node, client, _future = _node(service_ready=False)
        node._handle_event(_event(request_id="game-test-1", execution_id="execution-1"))  # noqa: SLF001
        node._handle_event(_event(request_id="game-test-2", execution_id="execution-2"))  # noqa: SLF001

        assert tuple(node._pending_announcements) == ("execution-1", "execution-2")  # noqa: SLF001
        assert not node._completed_announcements  # noqa: SLF001

        now = 102.9
        client.service_is_ready.return_value = True
        node._service_announcement_queues()  # noqa: SLF001

        assert client.call_async.call_count == 1
        assert node._tts_announcement.request_id == "game-test-1"  # noqa: SLF001
        assert tuple(node._pending_announcements) == ("execution-2",)  # noqa: SLF001

        synth_future = Mock()
        synth_future.result.return_value = SynthesizeSpeech.Response(
            success=True,
            audio_segments=[_audio_segment()],
        )
        node._handle_tts_response(node._tts_announcement, synth_future)  # noqa: SLF001
        playback_future = Mock()
        playback_future.result.return_value = PlayAudioFile.Response(success=True)
        now = 104.0
        node._handle_playback_response(playback_future)  # noqa: SLF001

        assert client.call_async.call_count == 2
        assert node._tts_announcement.request_id == "game-test-2"  # noqa: SLF001
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_unavailable_tts_is_not_delivered_if_service_recovers_after_grace_period(monkeypatch):
    rclpy.init()
    node = None
    try:
        now = 100.0
        monkeypatch.setattr("embodied_agent.visual_game_announcer_node.time.monotonic", lambda: now)
        node, client, _future = _node(service_ready=False)
        node._handle_event(_event())  # noqa: SLF001

        now = 103.0
        client.service_is_ready.return_value = True
        node._service_announcement_queues()  # noqa: SLF001

        assert client.call_async.call_count == 0
        assert "execution-1" not in node._pending_announcements  # noqa: SLF001
        assert "execution-1" in node._completed_announcements  # noqa: SLF001
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_reused_request_id_with_new_execution_is_not_suppressed():
    rclpy.init()
    node = None
    try:
        node, client, _future = _node(service_ready=False)
        node._handle_event(_event(execution_id="execution-1"))  # noqa: SLF001
        assert client.call_async.call_count == 0

        client.service_is_ready.return_value = True
        node._handle_event(_event(execution_id="execution-2"))  # noqa: SLF001

        assert client.call_async.call_count == 1
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_unavailable_playback_is_skipped_after_three_second_grace_period(monkeypatch):
    rclpy.init()
    node = None
    try:
        now = 100.0
        monkeypatch.setattr("embodied_agent.visual_game_announcer_node.time.monotonic", lambda: now)
        node, client, _future = _node()
        node._playback_client.service_is_ready.return_value = False  # noqa: SLF001

        error_code = node._handle_event(_event())  # noqa: SLF001

        assert error_code is None
        assert client.call_async.call_count == 0
        assert "execution-1" in node._pending_announcements  # noqa: SLF001
        assert "execution-1" not in node._completed_announcements  # noqa: SLF001

        now = 102.9
        node._service_announcement_queues()  # noqa: SLF001
        assert "execution-1" in node._pending_announcements  # noqa: SLF001

        now = 103.0
        node._service_announcement_queues()  # noqa: SLF001
        assert "execution-1" not in node._pending_announcements  # noqa: SLF001
        assert "execution-1" in node._completed_announcements  # noqa: SLF001
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_retry_pending_is_skipped_if_tts_stays_unavailable_for_three_seconds(monkeypatch):
    rclpy.init()
    node = None
    try:
        now = 100.0
        monkeypatch.setattr("embodied_agent.visual_game_announcer_node.time.monotonic", lambda: now)
        node, client, _future = _node(service_ready=True, max_attempts=2)
        announcement = _Announcement("game-test-1", "拉文克劳", attempts=1)
        node._active_announcements.add(announcement.deduplication_key)  # noqa: SLF001
        node._pending_announcements[announcement.deduplication_key] = announcement  # noqa: SLF001
        client.service_is_ready.return_value = False

        node._service_announcement_queues()  # noqa: SLF001

        assert client.call_async.call_count == 0
        assert announcement.deduplication_key in node._pending_announcements  # noqa: SLF001
        assert announcement.deduplication_key not in node._completed_announcements  # noqa: SLF001

        now = 103.0
        node._service_announcement_queues()  # noqa: SLF001

        assert announcement.deduplication_key not in node._pending_announcements  # noqa: SLF001
        assert announcement.deduplication_key in node._completed_announcements  # noqa: SLF001
        assert announcement.deduplication_key not in node._active_announcements  # noqa: SLF001
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_tts_failure_is_retried_without_duplicate_admission():
    rclpy.init()
    node = None
    try:
        node, client, _future = _node(max_attempts=2)
        announcement = _Announcement("game-test-1", "拉文克劳", attempts=1)
        node._active_announcements.add(announcement.deduplication_key)  # noqa: SLF001
        response = SynthesizeSpeech.Response(success=False, error_code="TTS_BUSY", message="try again")
        future = Mock()
        future.result.return_value = response
        node._handle_tts_response(announcement, future)  # noqa: SLF001
        retried = node._tts_announcement  # noqa: SLF001
        assert retried.request_id == announcement.request_id
        assert retried.attempts == 2
        assert client.call_async.call_count == 1
        assert announcement.deduplication_key in node._active_announcements  # noqa: SLF001
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_successful_tts_response_is_played_and_temp_file_is_removed():
    rclpy.init()
    node = None
    try:
        node, _client, _future = _node()
        announcement = _Announcement("game-test-1", "拉文克劳", attempts=1)
        node._active_announcements.add(announcement.request_id)  # noqa: SLF001
        response = SynthesizeSpeech.Response(success=True, audio_segments=[_audio_segment()])
        future = Mock()
        future.result.return_value = response
        node._handle_tts_response(announcement, future)  # noqa: SLF001

        playback_request = node._playback_client.call_async.call_args.args[0]  # noqa: SLF001
        playback_path = Path(playback_request.file_path)
        assert playback_path.is_file()
        playback_response = PlayAudioFile.Response(success=True)
        playback_future = Mock()
        playback_future.result.return_value = playback_response
        node._handle_playback_response(playback_future)  # noqa: SLF001

        assert announcement.deduplication_key in node._completed_announcements  # noqa: SLF001
        assert not playback_path.exists()
        assert node._temp_dir is None  # noqa: SLF001
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.shutdown()


def test_multiple_tts_segments_are_played_in_index_order():
    rclpy.init()
    node = None
    try:
        node, _client, _future = _node()
        announcement = _Announcement("game-test-1", "拉文克劳", attempts=1)
        node._active_announcements.add(announcement.deduplication_key)  # noqa: SLF001
        response = SynthesizeSpeech.Response(
            success=True,
            audio_segments=[_audio_segment(1), _audio_segment(0)],
        )
        future = Mock()
        future.result.return_value = response

        node._handle_tts_response(announcement, future)  # noqa: SLF001
        first_path = Path(node._playback_client.call_async.call_args.args[0].file_path)  # noqa: SLF001
        first_response = Mock()
        first_response.result.return_value = PlayAudioFile.Response(success=True)
        node._handle_playback_response(first_response)  # noqa: SLF001
        second_path = Path(node._playback_client.call_async.call_args.args[0].file_path)  # noqa: SLF001

        assert first_path.name.endswith("-0000.wav")
        assert second_path.name.endswith("-0001.wav")
        assert node._playback_client.call_async.call_count == 2  # noqa: SLF001
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.shutdown()


def test_segment_pause_delays_next_playback_request(monkeypatch):
    rclpy.init()
    node = None
    try:
        now = 100.0
        monkeypatch.setattr("embodied_agent.visual_game_announcer_node.time.monotonic", lambda: now)
        node, _client, _future = _node()
        announcement = _Announcement("game-test-1", "拉文克劳", attempts=1)
        node._active_announcements.add(announcement.deduplication_key)  # noqa: SLF001
        synth_future = Mock()
        synth_future.result.return_value = SynthesizeSpeech.Response(
            success=True,
            audio_segments=[_audio_segment(0, pause_after_ms=500), _audio_segment(1)],
        )
        node._handle_tts_response(announcement, synth_future)  # noqa: SLF001

        first_response = Mock()
        first_response.result.return_value = PlayAudioFile.Response(success=True)
        node._handle_playback_response(first_response)  # noqa: SLF001
        assert node._playback_client.call_async.call_count == 1  # noqa: SLF001

        now = 100.4
        node._service_announcement_queues()  # noqa: SLF001
        assert node._playback_client.call_async.call_count == 1  # noqa: SLF001

        now = 100.5
        node._service_announcement_queues()  # noqa: SLF001
        assert node._playback_client.call_async.call_count == 2  # noqa: SLF001
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.shutdown()


def test_playback_failure_does_not_retry_synthesis_and_is_deduplicated():
    rclpy.init()
    node = None
    try:
        node, tts_client, _future = _node()
        announcement = _Announcement("game-test-1", "拉文克劳", attempts=1)
        node._active_announcements.add(announcement.deduplication_key)  # noqa: SLF001
        synth_future = Mock()
        synth_future.result.return_value = SynthesizeSpeech.Response(
            success=True,
            audio_segments=[_audio_segment()],
        )
        node._handle_tts_response(announcement, synth_future)  # noqa: SLF001
        playback_path = Path(node._playback_client.call_async.call_args.args[0].file_path)  # noqa: SLF001

        playback_future = Mock()
        playback_future.result.return_value = PlayAudioFile.Response(
            success=False,
            error_code="PLAYBACK_FAILED",
            message="speaker unavailable",
        )
        node._handle_playback_response(playback_future)  # noqa: SLF001

        assert tts_client.call_async.call_count == 0
        assert announcement.deduplication_key in node._completed_announcements  # noqa: SLF001
        assert announcement.deduplication_key not in node._active_announcements  # noqa: SLF001
        assert not playback_path.exists()
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.shutdown()


def test_playback_timeout_cleans_files_without_replaying(monkeypatch):
    rclpy.init()
    node = None
    try:
        now = 100.0
        monkeypatch.setattr("embodied_agent.visual_game_announcer_node.time.monotonic", lambda: now)
        node, tts_client, _future = _node()
        node._playback_timeout_sec = 1.0  # noqa: SLF001
        announcement = _Announcement("game-test-1", "拉文克劳", attempts=1)
        node._active_announcements.add(announcement.deduplication_key)  # noqa: SLF001
        synth_future = Mock()
        synth_future.result.return_value = SynthesizeSpeech.Response(
            success=True,
            audio_segments=[_audio_segment()],
        )
        node._handle_tts_response(announcement, synth_future)  # noqa: SLF001
        playback_path = Path(node._playback_client.call_async.call_args.args[0].file_path)  # noqa: SLF001

        now = 101.1
        node._expire_playback_request()  # noqa: SLF001

        assert tts_client.call_async.call_count == 0
        assert announcement.deduplication_key in node._completed_announcements  # noqa: SLF001
        assert not playback_path.exists()
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.shutdown()


def test_exhausted_tts_failure_is_deduplicated():
    rclpy.init()
    node = None
    try:
        node, client, _future = _node(max_attempts=2)
        announcement = _Announcement("game-test-1", "拉文克劳", attempts=2, execution_id="execution-1")
        node._active_announcements.add(announcement.deduplication_key)  # noqa: SLF001

        node._retry_or_finish(announcement, "TTS failed")  # noqa: SLF001
        node._handle_event(_event())  # noqa: SLF001

        assert announcement.deduplication_key in node._completed_announcements  # noqa: SLF001
        assert announcement.deduplication_key not in node._active_announcements  # noqa: SLF001
        assert client.call_async.call_count == 0
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
