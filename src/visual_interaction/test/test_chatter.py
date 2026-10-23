"""Chatter: what is said, and that something is said every 5–10 s."""

from __future__ import annotations

import random
import re
import statistics

from embodied_common.visual_interaction_contract import UTTERANCE_KINDS
from visual_interaction.session.chatter import (
    ACTIONS,
    FILLERS,
    OBJECTS,
    Chatter,
    ChatterConfig,
    chatter_prompt,
    classify,
)


def test_phrases_map_to_subjects():
    assert classify("双手比心") == ("action", "heart")
    assert classify("竖起大拇指点赞") == ("action", "thumbs_up")
    assert classify("比出V字手势") == ("action", "victory")
    assert classify("正在挥手打招呼") == ("action", "wave")
    assert classify("双手举高伸懒腰") == ("action", "stretch")
    assert classify("拿着一个玩偶") == ("object", "toy")
    # Said on the board, 10-23.
    assert classify("画面里离镜头最近的人正在用右手做手势。") == ("action", "gesture")
    assert classify("无") is None
    assert classify("") is None
    assert classify(None) is None


def test_every_line_is_speakable():
    lines = [line for subject in ACTIONS + OBJECTS for line in subject.lines] + list(FILLERS)
    for line in lines:
        assert not re.search(r"[A-Za-z0-9]", line), line
        assert 0 < len(line) <= 20, line


def test_kinds_are_admitted_by_the_voice():
    for kind in ("action", "object", "chatter"):
        assert kind in UTTERANCE_KINDS


def test_prompt_is_short():
    assert len(chatter_prompt(2)) < 200


def _run(chatter: Chatter, seconds: float, phrase=lambda t: "无", step=1.6):
    said = []
    now = 0.0
    while now < seconds:
        chatter.observe(now, phrase(now))
        if chatter.due(now, True):
            said.append((now, chatter.pick(now)))
        now += step
    return said


def test_speaks_every_five_to_ten_seconds_after_the_line_ends():
    config = ChatterConfig()
    said = _run(Chatter(config, rng=random.Random(1)), 120.0)
    assert len(said) >= 9
    gaps = []
    for (t0, line), (t1, _) in zip(said, said[1:], strict=False):
        end = t0 + config.speech_lead_sec + config.speech_per_char_sec * len(line.text)
        gaps.append(t1 - end)
    assert all(5.0 <= gap <= 10.0 + 1.6 for gap in gaps), gaps
    assert statistics.pstdev(gaps) > 0.5


def test_a_fresh_gesture_wins_and_cools_down():
    said = _run(Chatter(rng=random.Random(2)), 40.0, phrase=lambda t: "双手比心")
    kinds = [line.subject for _, line in said]
    assert kinds[0] == "heart"
    # Not twice within the 15 s cooldown.
    hearts = [t for t, line in said if line.subject == "heart"]
    assert all(b - a >= 15.0 for a, b in zip(hearts, hearts[1:], strict=False))


def test_busy_voice_holds_the_next_line():
    chatter = Chatter(rng=random.Random(3))
    for now in range(0, 30):
        chatter.voice(float(now), True)
        assert not chatter.due(float(now), True)


def test_nothing_while_not_mirroring():
    chatter = Chatter(rng=random.Random(4))
    assert not any(chatter.due(float(now), False) for now in range(60))
