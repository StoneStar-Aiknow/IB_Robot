"""X2 wake event -> opener anchor -> wake-free speaker checks, without ROS/hardware.

Embeddings are controlled unit vectors: these tests verify routing, not microphone
quality or CAMPPlus accuracy. Use on_heard (the transport's entry point), including
cross-topic arrival ordering rather than bypassing the grace-period queue.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from ibrobot_agent.addressee import Voiceprint
from ibrobot_agent.voice_bridge import Log, PublishControl, PublishRequest, VoiceBridgeConfig, VoiceBridgeCore

MODEL = "campplus@flow-test"
OWNER = Voiceprint((1.0, 0.0, 0.0), MODEL)
OWNER_FOLLOWUP = Voiceprint((0.8, 0.6, 0.0), MODEL)
OTHER = Voiceprint((0.0, 0.0, 1.0), MODEL)


def _segment(sequence):
    return ("x2-test-source", sequence)


def _core():
    return VoiceBridgeCore(
        VoiceBridgeConfig(
            channel_id="voice_asr",
            principal_id="local_operator",
            robot_scope="aimdk_x2_skills",
            confirm_mode="auto",
            addressee_trigger="vendor_session",
            session_gap_sec=25.0,
            speaker_check_enabled=True,
            speaker_min_similarity=0.60,
            speaker_max_age_sec=6.0,
            speaker_merge_margin=0.10,
            speaker_reject_short=True,
            measurement_grace_sec=1.5,
        )
    )


def _requests(actions):
    return [a for a in actions if isinstance(a, PublishRequest)]


def _finish(core, actions, now):
    requests = _requests(actions)
    assert len(requests) == 1
    core.on_event(
        {
            "request_key": {
                "channel_id": "voice_asr",
                "principal_id": "local_operator",
                "robot_scope": "aimdk_x2_skills",
                "request_id": requests[0].payload["request_id"],
            },
            "event_type": "terminal",
            "state": "SUCCEEDED",
        },
        now=now,
    )
    assert not core.interruptible


@pytest.mark.parametrize("opener_embedding_first", [True, False])
@pytest.mark.parametrize("followup_embedding_first", [True, False])
def test_x2_wake_anchor_followups_takeover_and_expiry(opener_embedding_first, followup_embedding_first):
    core = _core()
    # Before the vendor wake event, speech alone must not start the Agent.
    assert not _requests(core.on_heard("请挥个手", now=90.0, segment=_segment(1)))
    assert not _requests(core.tick(now=92.0))

    # X2 consumes the wake phrase: the opener text contains no wake word.
    core.on_wake(now=100.0, segment=_segment(2))
    if opener_embedding_first:
        core.on_measurement("speaker", OWNER, now=100.1, segment=_segment(2))
    opened = core.on_heard("请挥个手", now=100.2, segment=_segment(2))
    assert [r.payload["text"] for r in _requests(opened)] == ["请挥个手"]
    if not opener_embedding_first:
        core.on_measurement("speaker", OWNER, now=100.3, segment=_segment(2))
    _finish(core, opened, 100.4)

    # No new wake event: the owner's follow-up passes, even if its vector is late.
    if followup_embedding_first:
        core.on_measurement("speaker", OWNER_FOLLOWUP, now=104.9, segment=_segment(3))
    accepted = core.on_heard("再帮我鼓个掌", now=105.0, segment=_segment(3))
    if not followup_embedding_first:
        assert not _requests(accepted)
        accepted = core.on_measurement("speaker", OWNER_FOLLOWUP, now=105.4, segment=_segment(3))
    assert [r.payload["text"] for r in _requests(accepted)] == ["再帮我鼓个掌"]
    _finish(core, accepted, 105.5)

    # Another person, without waking: reject both before and after an accepted turn.
    core.on_measurement("speaker", OTHER, now=108.0, segment=_segment(4))
    rejected = core.on_heard("请举起手", now=108.1, segment=_segment(4))
    assert not _requests(rejected)
    assert any(isinstance(a, Log) and "speaker: similarity" in a.message for a in rejected)

    # Re-waking explicitly transfers the session to the other speaker.
    core.on_wake(now=110.0, segment=_segment(5))
    core.on_measurement("speaker", OTHER, now=110.1, segment=_segment(5))
    takeover = core.on_heard("请举起手", now=110.2, segment=_segment(5))
    _finish(core, takeover, 110.3)
    core.on_measurement("speaker", OWNER, now=112.0, segment=_segment(6))
    assert not _requests(core.on_heard("请挥个手", now=112.1, segment=_segment(6)))
    core.on_measurement("speaker", OTHER, now=114.0, segment=_segment(7))
    _finish(core, core.on_heard("请鼓个掌", now=114.1, segment=_segment(7)), 114.2)

    # Even the new owner needs a fresh wake after the configured silence gap.
    core.on_measurement("speaker", OTHER, now=145.0, segment=_segment(8))
    assert not _requests(core.on_heard("请挥个手", now=145.1, segment=_segment(8)))
    assert not _requests(core.tick(now=147.0))


def test_x2_missing_voiceprint_still_fails_open_but_explicit_short_is_rejected():
    """Document the actual policy; enabling speaker checks is not authentication."""
    core = _core()
    core.on_wake(now=100.0, segment=_segment(1))
    core.on_measurement("speaker", OWNER, now=100.1, segment=_segment(1))
    _finish(core, core.on_heard("请挥个手", now=100.2, segment=_segment(1)), 100.3)
    core.on_measurement("speaker", None, now=104.0, segment=_segment(2))
    short = core.on_heard("鼓个掌", now=104.1, segment=_segment(2))
    assert not _requests(short)
    assert any(isinstance(a, Log) and "too short" in a.message for a in short)
    assert not _requests(core.on_heard("请举起手", now=106.0, segment=_segment(3)))
    assert not _requests(core.tick(now=107.4))
    released = core.tick(now=107.6)
    assert [r.payload["text"] for r in _requests(released)] == ["请举起手"]
    # A stop from any voice must still interrupt the resulting active request.
    stop = core.on_heard("停止", now=108.0, segment=_segment(4))
    assert any(isinstance(a, PublishControl) and a.payload["operation"] == "stop" for a in stop)


def test_x2_short_opener_leaves_followups_unchecked_until_next_wake():
    core = _core()
    core.on_wake(now=100.0, segment=_segment(1))
    core.on_measurement("speaker", None, now=100.1, segment=_segment(1))
    _finish(core, core.on_heard("请挥手", now=100.2, segment=_segment(1)), 100.3)
    core.on_measurement("speaker", OTHER, now=104.0, segment=_segment(2))
    # This is intentional existing behavior, not a strict same-speaker guarantee.
    accepted = core.on_heard("请举起手", now=104.1, segment=_segment(2))
    assert [r.payload["text"] for r in _requests(accepted)] == ["请举起手"]


def _text_core():
    config = _core().config
    return VoiceBridgeCore(
        replace(config, addressee_trigger="wake_phrase", wake_phrases=("小欧小欧",), wake_window_sec=25.0)
    )


@pytest.mark.parametrize("opener_first", [True, False])
@pytest.mark.parametrize("followup_first", [True, False])
def test_xiaoou_text_wake_speaker_flow_without_vendor_event(opener_first, followup_first):
    core = _text_core()
    assert not _requests(core.on_heard("请挥个手", now=90.0, segment=_segment(1)))
    if opener_first:
        core.on_measurement("speaker", OWNER, now=100.0, segment=_segment(2))
    opened = core.on_heard("小欧小欧，现在请你挥个手", now=100.1, segment=_segment(2))
    assert [r.payload["text"] for r in _requests(opened)] == ["现在请你挥个手"]
    if not opener_first:
        core.on_measurement("speaker", OWNER, now=100.2, segment=_segment(2))
    _finish(core, opened, 100.3)
    if followup_first:
        core.on_measurement("speaker", OWNER_FOLLOWUP, now=120.0, segment=_segment(3))
    accepted = core.on_heard("接下来请你鼓掌", now=120.1, segment=_segment(3))
    if not followup_first:
        assert not _requests(accepted)
        accepted = core.on_measurement("speaker", OWNER_FOLLOWUP, now=120.4, segment=_segment(3))
    _finish(core, accepted, 120.5)
    core.on_measurement("speaker", OTHER, now=130.0, segment=_segment(4))
    rejected = core.on_heard("请举起手", now=130.1, segment=_segment(4))
    assert not _requests(rejected)
    assert any(isinstance(a, Log) and "speaker: similarity" in a.message for a in rejected)
    # The owner's accepted turn renewed the window past the original 125.1 deadline.
    core.on_measurement("speaker", OWNER, now=140.0, segment=_segment(5))
    _finish(core, core.on_heard("请挥个手", now=140.1, segment=_segment(5)), 140.2)
    core.on_measurement("speaker", OTHER, now=142.0, segment=_segment(6))
    takeover = core.on_heard("小欧小欧，请举起手", now=142.1, segment=_segment(6))
    assert [r.payload["text"] for r in _requests(takeover)] == ["请举起手"]
    _finish(core, takeover, 142.2)
    core.on_measurement("speaker", OWNER, now=144.0, segment=_segment(7))
    assert not _requests(core.on_heard("请鼓掌", now=144.1, segment=_segment(7)))
    core.on_measurement("speaker", OTHER, now=170.0, segment=_segment(8))
    assert not _requests(core.on_heard("请挥手", now=170.1, segment=_segment(8)))


def test_xiaoou_stop_cancels_followup_waiting_for_voiceprint():
    core = _text_core()
    core.on_measurement("speaker", OWNER, now=100.0, segment=_segment(1))
    _finish(core, core.on_heard("小欧小欧，请挥手", now=100.1, segment=_segment(1)), 100.2)
    assert not _requests(core.on_heard("请举手", now=104.0, segment=_segment(2)))
    core.on_heard("停止", now=104.1, segment=_segment(3))
    assert not _requests(core.on_measurement("speaker", OWNER, now=104.2, segment=_segment(2)))
    assert not _requests(core.tick(now=106.0))


@pytest.mark.parametrize("phrase", [f"小{a}小{b}" for a in "欧鸥偶" for b in "欧鸥偶"])
def test_xiaoou_variants_are_prefix_only_and_stripped(phrase):
    config = replace(_text_core().config, wake_phrases=tuple(f"小{a}小{b}" for a in "欧鸥偶" for b in "欧鸥偶"))
    core = VoiceBridgeCore(config)
    assert not _requests(core.on_heard(f"我刚才说了{phrase}请挥手", now=90.0, segment=_segment(1)))
    core.on_measurement("speaker", OWNER, now=100.0, segment=_segment(2))
    actions = core.on_heard(f"{phrase}，请挥手", now=100.1, segment=_segment(2))
    assert [r.payload["text"] for r in _requests(actions)] == ["请挥手"]


def test_disabled_wake_and_speaker_check_accepts_plain_speech_without_waiting():
    core = VoiceBridgeCore(replace(_text_core().config, wake_phrases=(), speaker_check_enabled=False))
    assert core.required_measurements() == ()
    assert not core.config.pairs_by_segment
    actions = core.on_heard("请挥个手", now=100.0)
    assert [r.payload["text"] for r in _requests(actions)] == ["请挥个手"]
    _finish(core, actions, 100.1)
    later = core.on_heard("请鼓掌", now=200.0)
    assert [r.payload["text"] for r in _requests(later)] == ["请鼓掌"]
