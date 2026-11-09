"""Keeping the owner's words of a segment that holds two voices (speaker_split + the bridge).

Vectors are controlled unit vectors: these tests pin the routing and the time-to-text
cut, not CAM++ accuracy, which tools/x2_speaker_split measures on real recordings.
"""

from __future__ import annotations

import math

import pytest

from ibrobot_agent.addressee import Voiceprint
from ibrobot_agent.speaker_split import TokenAlignment, WindowVoiceprints, split_by_speaker
from ibrobot_agent.voice_bridge import Log, PublishRequest, VoiceBridgeConfig, VoiceBridgeCore

MODEL = "campplus@split-test"
OWNER = (1.0, 0.0, 0.0)
OTHER = (0.0, 0.0, 1.0)
UNSURE = (0.35, 0.0, math.sqrt(1 - 0.35**2))  # cosine 0.35 with the owner: between the two lines


def _windows(*labels, window=2.0, hop=1.0):
    vectors = tuple({"o": OWNER, "x": OTHER, "?": UNSURE}[label] for label in labels)
    return WindowVoiceprints(window, tuple(index * hop for index in range(len(labels))), vectors)


def _aligned(text, starts):
    """One token per character, starting at ``starts``."""
    return TokenAlignment.build(starts, range(1, len(text) + 1), text)


def _split(windows, text="", alignment=None):
    return split_by_speaker(OWNER, windows, text, alignment, owner_similarity=0.45, other_similarity=0.25)


# --- the verdict ----------------------------------------------------------------------------


def test_a_bystander_then_the_owner_keeps_the_owner_s_tokens():
    # The 2026-11-09 case: someone else for ~3.5 s, then the owner's request.
    text = "这不是跟他说的你还会做什么"
    starts = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 3.7, 3.9, 4.1, 4.3, 4.5, 4.7, 4.9]
    split = _split(_windows("x", "x", "x", "o", "o"), text, _aligned(text, starts))

    assert split.kind == "mixed"
    # Window 2 (2-4 s) is someone else, window 3 (3-5 s) the owner: the margin crosses
    # zero halfway between their middles (3 s and 4 s), so the owner starts at 3.5 s.
    assert split.span == (3.5, math.inf)
    # Token times trail the voice by 0.35 s: "的" at 3.7 s was heard before 3.5 s.
    assert split.text == "你还会做什么"
    assert split.dropped == "这不是跟他说的"
    assert split.vector == pytest.approx(OWNER)


def test_the_owner_then_a_bystander_keeps_the_beginning():
    text = "挥个手吧我们下午再测"
    # "挥" timed at 0.2 s: less than the token delay, still the owner's first word.
    starts = [0.2, 0.9, 1.2, 1.5, 2.9, 3.2, 3.5, 3.8, 4.1, 4.4]
    split = _split(_windows("o", "o", "x", "x"), text, _aligned(text, starts))

    assert split.kind == "mixed"
    assert split.span == (-math.inf, 2.5)
    assert split.text == "挥个手吧"


@pytest.mark.parametrize("labels", [("o", "o", "o"), ("o", "?", "o"), ("?", "?", "o")])
def test_no_window_that_is_clearly_someone_else_is_a_single_voice(labels):
    # Undecided windows (a quiet or noisy stretch of the owner's own speech) never split.
    assert _split(_windows(*labels), "挥个手").kind == "single"


def test_no_window_that_is_clearly_the_owner_is_left_to_the_speaker_check():
    assert _split(_windows("x", "?", "x"), "下午再测").kind == "absent"


def test_a_low_window_of_the_owner_s_own_voice_splits_nothing():
    """Against the anchor one window may dip; against the segment's own windows it is the owner."""
    here = (0.6, 0.8, 0.0)  # the owner in this room: 0.60 with the anchor
    low = (0.2, math.sqrt(1 - 0.2**2), 0.0)  # 0.20 with the anchor, 0.90 with ``here``
    windows = WindowVoiceprints(2.0, (0.0, 1.0, 2.0), (here, low, here))
    near = (0.9, 0.0, math.sqrt(1 - 0.81))
    assert _split(windows).kind == "single"
    assert _split(WindowVoiceprints(2.0, (0.0, 1.0), (OWNER, near))).kind == "single"


def test_a_mixed_segment_without_token_times_keeps_no_text():
    # Without an alignment the owner's words cannot be told from the bystander's.
    split = _split(_windows("x", "x", "o", "o"), "随便说说挥个手", None)
    assert split.kind == "mixed"
    assert split.text == ""


# --- in the bridge --------------------------------------------------------------------------


def _segment(sequence):
    return ("split-test-source", sequence)


def _core(**overrides):
    values = dict(
        channel_id="voice_asr",
        principal_id="local_operator",
        robot_scope="aimdk_x2_skills",
        confirm_mode="auto",
        addressee_trigger="wake_phrase",
        wake_phrases=("欧拉欧拉",),
        wake_window_sec=60.0,
        speaker_check_enabled=True,
        speaker_min_similarity=0.60,
        speaker_max_age_sec=6.0,
        measurement_grace_sec=1.5,
        speaker_split_enabled=True,
    )
    values.update(overrides)
    return VoiceBridgeCore(VoiceBridgeConfig(**values))


def _requests(actions):
    return [action.payload["text"] for action in actions if isinstance(action, PublishRequest)]


def _logs(actions):
    return [action.message for action in actions if isinstance(action, Log)]


def _finish(core, actions, now):
    (request,) = [action for action in actions if isinstance(action, PublishRequest)]
    core.on_event(
        {
            "request_key": {
                "channel_id": "voice_asr",
                "principal_id": "local_operator",
                "robot_scope": "aimdk_x2_skills",
                "request_id": request.payload["request_id"],
            },
            "event_type": "terminal",
            "state": "SUCCEEDED",
        },
        now=now,
    )


MIXED_TEXT = "这都不是跟他说的话你还会做什么动作"
MIXED_STARTS = [0.3 + 0.3 * index for index in range(9)] + [4.9 + 0.2 * index for index in range(8)]


def _woken(core):
    core.on_measurement("speaker", Voiceprint(OWNER, MODEL), now=1.0, segment=_segment(1))
    _finish(core, core.on_heard("欧拉欧拉你好", now=1.0, segment=_segment(1)), now=2.0)


def _mixed_measurement():
    # The utterance vector is an average of both voices; the windows tell them apart.
    blend = (0.5, 0.0, math.sqrt(0.75))
    return Voiceprint(blend, MODEL, _windows("x", "x", "x", "x", "o", "o"))


def test_the_bridge_sends_only_the_owner_s_words_of_a_mixed_segment():
    core = _core()
    _woken(core)
    core.on_measurement("speaker", _mixed_measurement(), now=10.0, segment=_segment(2))

    actions = core.on_heard(MIXED_TEXT, now=10.1, segment=_segment(2), alignment=_aligned(MIXED_TEXT, MIXED_STARTS))

    assert _requests(actions) == ["你还会做什么动作"]
    assert any("mixed" in message and "dropped '这都不是跟他说的话'" in message for message in _logs(actions))


def test_with_the_split_off_the_windows_are_only_logged():
    core = _core(speaker_split_enabled=False)
    _woken(core)
    core.on_measurement("speaker", _mixed_measurement(), now=10.0, segment=_segment(2))

    actions = core.on_heard(MIXED_TEXT, now=10.1, segment=_segment(2), alignment=_aligned(MIXED_TEXT, MIXED_STARTS))

    # The averaged vector (0.50 < 0.60) rejects the whole segment, as before this change.
    assert _requests(actions) == []
    assert any("split off" in message for message in _logs(actions))


def test_a_wake_utterance_is_never_split():
    # The opener sets the anchor; there is nothing yet to compare it with.
    core = _core()
    text = "欧拉欧拉挥个手"
    core.on_measurement("speaker", Voiceprint(OWNER, MODEL, _windows("o", "x", "x")), now=1.0, segment=_segment(1))

    actions = core.on_heard(text, now=1.0, segment=_segment(1), alignment=_aligned(text, [0.2 * i for i in range(7)]))

    assert _requests(actions) == ["挥个手"]
    assert not any("speaker windows" in message for message in _logs(actions))
