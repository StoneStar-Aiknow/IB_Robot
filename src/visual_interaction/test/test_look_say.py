"""look_say: the mapping, the questions' triggers, and when and what the look says.

Pure Python: the clock is a float passed in, the RNG is seeded, Q3 is a
callable. Nothing here needs ROS, the model or the board.
"""

from __future__ import annotations

import random

import pytest

from visual_interaction.session import look_say
from visual_interaction.session.look_say import (
    DANCE,
    DOWN,
    HAND_ANY,
    HEART,
    NONE,
    OPEN,
    OTHER,
    POSE,
    QUESTIONS,
    THUMB,
    WAVE,
    YEAH,
    LookConfig,
    LookSay,
    classify,
    needs_refine,
    parse_yes,
    refine,
)

LOOK_POOLS = (
    "hand.heart",
    "hand.thumb",
    "hand.yeah",
    "hand.any",
    "big.wave",
    "big.open",
    "big.any",
    "pose.any",
    "dance.any",
    "other.any",
    "seq.repeat",
    "seq.still",
    "seq.switch",
    "seq.invite",
)


def pools(count: int = 3, **sizes: int) -> dict[str, tuple[tuple[str, str], ...]]:
    """``count`` clips per look pool; ``pose_any=1`` and the like override one pool."""
    result = {}
    for pool in LOOK_POOLS:
        group, sub = pool.split(".")
        size = sizes.get(f"{group}_{sub}", count)
        result[pool] = tuple((f"{group}_{sub}_{index:02d}", f"{pool} 第{index}句") for index in range(1, size + 1))
    return result


def make(seed: int = 0, clip_pools=None, **overrides) -> LookSay:
    settings = {"first_after_sec": (3.0, 3.0), "gap_sec": (4.0, 4.0)}
    settings.update(overrides)
    return LookSay(LookConfig(**settings), clip_pools if clip_pools is not None else pools(), random.Random(seed))


def played(look: LookSay, at: float, seconds: float = 0.5) -> None:
    """The line said at ``at`` plays for ``seconds``, as the voice's status reports it."""
    look.voice(at + 0.05, busy=True, heard=True)
    look.voice(at + seconds, busy=False, heard=True)


def start(look: LookSay, at: float = 0.0) -> LookSay:
    """Enter MIRRORING with a fresh, idle voice status, as the node reports it every tick."""
    look.enter(at)
    look.voice(at, busy=False, heard=True)
    return look


class Q3:
    """A scripted Q3: answers in order, and records what it was asked."""

    def __init__(self, *answers: bool | None) -> None:
        self.answers = list(answers)
        self.asked: list[tuple[str, object]] = []

    def __call__(self, key: str, ref: object) -> bool | None:
        self.asked.append((key, ref))
        return self.answers.pop(0) if self.answers else True


# ---------------------------------------------------------------------------
# The mapping, on the short answers the 1B model actually gives.


@pytest.mark.parametrize(
    ("answer", "label"),
    [
        ("比心", HEART),
        ("用手比了个爱心。", HEART),
        ("双手比心", HEART),
        ("竖起大拇指", THUMB),
        ("点赞", THUMB),
        ("比剪刀手", YEAH),
        ("比出“V”字", YEAH),
        ("比耶", YEAH),
        ("伸出两根手指", YEAH),
        ("OK手势", HAND_ANY),
        ("在做手势", HAND_ANY),
        ("比划着", HAND_ANY),
        ("伸出手指", HAND_ANY),
        ("举着手指", HAND_ANY),
        ("挥手", WAVE),
        ("站着挥手", WAVE),
        ("张开双臂", OPEN),
        ("伸懒腰", look_say.UP),
        ("双手叉腰", POSE),
        ("双手抱胸", POSE),
        ("跳舞", DANCE),
        ("鼓掌", OTHER),
        ("抱拳", OTHER),
        ("双手合十", OTHER),
        ("鞠躬", OTHER),
        ("指向镜头", OTHER),
        ("站着", DOWN),
        ("双手下垂", DOWN),
        ("站立不动", DOWN),
        ("没有动作", DOWN),
        ("无", NONE),
        ("无。", NONE),
        ("", NONE),
        ("没有人", NONE),
        ("看不清", NONE),
    ],
)
def test_answers_map_to_the_plan_table(answer, label):
    assert classify(answer) is label


@pytest.mark.parametrize("answer", ["微笑", "拿着玩偶", "戴着帽子", "手里拿着手机", "看镜头"])
def test_objects_faces_and_unknowns_are_no_candidate(answer):
    assert classify(answer) is None


def test_the_labels_carry_the_pools_and_downgrades_of_the_plan():
    assert (HEART.pool, HEART.coarse_pool, HEART.verify_key) == ("hand.heart", "hand.any", "q3.heart")
    assert (THUMB.pool, THUMB.coarse_pool, THUMB.verify_key) == ("hand.thumb", "hand.any", "q3.thumb")
    assert (YEAH.pool, YEAH.coarse_pool, YEAH.verify_key) == ("hand.yeah", "hand.any", "q3.yeah")
    assert (WAVE.pool, WAVE.coarse_pool, WAVE.verify_key) == ("big.wave", "big.any", "q3.wave")
    assert (OPEN.pool, OPEN.coarse_pool, OPEN.verify_key) == ("big.open", "big.any", "q3.open")
    assert look_say.UP.pool == "big.any" and not look_say.UP.named
    assert (HAND_ANY.pool, POSE.pool, DANCE.pool, OTHER.pool) == ("hand.any", "pose.any", "dance.any", "other.any")
    assert DOWN.pool is None and NONE.pool is None


def test_a_specific_action_wins_over_standing_still():
    # "站着" is only read as still when no action matched.
    assert classify("站着比心") is HEART
    assert classify("站着鼓掌") is OTHER


def test_the_questions_are_the_plan_s_with_their_budgets():
    assert set(QUESTIONS) == {"q1", "q2", "q3.heart", "q3.thumb", "q3.yeah", "q3.wave", "q3.open"}
    assert QUESTIONS["q1"][1] == 16
    assert QUESTIONS["q2"] == ("这个人的手指比的是什么手势？用不超过五个字回答。", 14)
    assert QUESTIONS["q3.heart"] == ("这个人是在用手比爱心吗？只回答是或不是。", 3)
    assert "竖大拇指" in QUESTIONS["q3.thumb"][0]
    assert "剪刀手" in QUESTIONS["q3.yeah"][0]
    assert "挥手打招呼" in QUESTIONS["q3.wave"][0]
    assert "张开双臂" in QUESTIONS["q3.open"][0]
    # Every Q3 key a label can ask is a question.
    assert {label.verify_key for label in look_say.LABELS if label.named} <= set(QUESTIONS)


# ---------------------------------------------------------------------------
# Q2 and Q3.


def test_q2_is_asked_only_for_a_vague_hand_gesture():
    assert needs_refine(HAND_ANY)
    for label in (HEART, THUMB, YEAH, WAVE, POSE, OTHER, DOWN, NONE, None):
        assert not needs_refine(label)
    assert make().wants_refine(HAND_ANY)
    assert not make(refine_hand=False).wants_refine(HAND_ANY)


@pytest.mark.parametrize(
    ("answer", "label"),
    [("比心", HEART), ("爱心", HEART), ("大拇指", THUMB), ("点赞", THUMB), ("V字", YEAH), ("剪刀手", YEAH)],
)
def test_q2_names_one_of_the_three_hand_gestures(answer, label):
    assert refine(HAND_ANY, answer) is label


@pytest.mark.parametrize(
    ("answer", "label"),
    [
        ("这个人的手指比的是心形手势。", HEART),
        ('这个人的手指比的是"V"字手势。', YEAH),
        ('这个人的手指比的是"三"字手势。', HAND_ANY),
        ("这个人的手指比的是“", HAND_ANY),
    ],
)
def test_q2_answers_read_on_the_board_with_the_question_repeated(answer, label):
    assert refine(HAND_ANY, answer) is label


@pytest.mark.parametrize("answer", ["OK", "不知道", "挥手", "鼓掌", "", "握拳"])
def test_q2_naming_anything_else_keeps_hand_any(answer):
    # Only the three hand rows are taken from Q2; a wave or a clap is not.
    assert refine(HAND_ANY, answer) is HAND_ANY


@pytest.mark.parametrize(
    ("answer", "yes"),
    [
        ("是", True),
        ("是的", True),
        ("是。", True),
        ("不是", False),
        ("不", False),
        ("否", False),
        ("没有", False),
        ("是不是", False),
        ("嗯", False),
        ("对", False),
        ("", False),
    ],
)
def test_q3_reads_yes_only_from_a_plain_yes(answer, yes):
    assert parse_yes(answer) is yes


def test_a_named_gesture_is_said_by_name_after_q3_says_yes():
    look = make()
    start(look, 0.0)
    look.looked(1.0, HEART, ref="seen-1")
    q3 = Q3(True)
    pick = look.decide(3.0, q3)
    assert (pick.pool, pick.reason) == ("hand.heart", "named")
    assert q3.asked == [("q3.heart", "seen-1")]


@pytest.mark.parametrize(
    ("label", "coarse"), [(HEART, "hand.any"), (THUMB, "hand.any"), (WAVE, "big.any"), (OPEN, "big.any")]
)
def test_q3_no_downgrades_to_the_family_s_generic_pool(label, coarse):
    look = make()
    start(look, 0.0)
    look.looked(1.0, label, ref="seen")
    pick = look.decide(3.0, Q3(False))
    assert (pick.pool, pick.reason) == (coarse, "general")


def test_an_unanswered_q3_downgrades_too():
    look = make()
    start(look, 0.0)
    look.looked(1.0, YEAH, ref="seen")
    assert look.decide(3.0, Q3(None)).pool == "hand.any"


def test_without_a_q3_a_named_gesture_is_not_said_by_name():
    look = make()
    start(look, 0.0)
    look.looked(1.0, HEART, ref="seen")
    assert look.decide(3.0, None).pool == "hand.any"


def test_verify_off_says_the_named_line_without_asking():
    look = make(verify=False)
    start(look, 0.0)
    look.looked(1.0, HEART)
    q3 = Q3()
    assert look.decide(3.0, q3).pool == "hand.heart"
    assert q3.asked == []


def test_at_most_one_q3_per_decision():
    look = make()
    start(look, 0.0)
    look.looked(1.0, HEART, ref="a")
    look.looked(2.0, WAVE, ref="b")
    q3 = Q3(False, True)
    pick = look.decide(3.0, q3)
    assert len(q3.asked) == 1
    assert q3.asked[0][0] == "q3.wave"
    # The wave was refused; the heart was not asked this time and counts by its family.
    assert pick.pool in ("big.any", "hand.any")


def test_a_q3_verdict_is_kept_for_its_look():
    look = make(gap_sec=(0.5, 0.5), pool_cooldown_sec=0.0)
    start(look, 0.0)
    look.looked(1.0, HEART, ref="a")
    q3 = Q3(False)
    look.decide(3.0, q3)
    look.decide(3.6, q3)
    assert len(q3.asked) == 1


# ---------------------------------------------------------------------------
# Priority, freshness, cooldown, fallback.


def test_named_beats_general():
    look = make()
    start(look, 0.0)
    look.looked(1.0, HEART, ref="a")
    look.looked(2.0, POSE)
    assert look.decide(3.0, Q3(True)).pool == "hand.heart"


def test_general_beats_other_even_when_other_is_newer():
    look = make()
    start(look, 0.0)
    look.looked(1.0, POSE)
    look.looked(2.0, OTHER)
    assert look.decide(3.0).pool == "pose.any"


def test_the_newest_general_look_wins():
    look = make()
    start(look, 0.0)
    look.looked(1.0, POSE)
    look.looked(2.0, DANCE)
    assert look.decide(3.0).pool == "dance.any"


def test_other_beats_the_sequence_rules():
    look = make()
    start(look, 0.0)
    look.looked(1.0, DOWN)
    look.looked(1.5, DOWN)
    look.looked(2.0, OTHER)
    assert (look.decide(3.0).pool, look.labels[-1]) == ("other.any", OTHER)


def test_a_look_older_than_fresh_sec_is_no_candidate():
    look = make(first_after_sec=(6.5, 6.5))
    start(look, 0.0)
    look.looked(1.0, POSE)
    assert look.decide(6.5) is None
    assert look.decide(9.0).reason == "fallback"


def test_seq_still_after_two_still_looks():
    look = make()
    start(look, 0.0)
    look.looked(1.0, DOWN)
    look.looked(2.0, DOWN)
    assert look.decide(3.0).pool == "seq.still"


def test_seq_switch_when_the_last_two_families_differ():
    look = make(first_after_sec=(8.0, 8.0))
    start(look, 0.0)
    look.looked(1.0, POSE)
    look.looked(2.0, look_say.UP)
    # Both stale by now, so neither is a candidate of its own.
    assert look.decide(8.0).pool == "seq.switch"


def test_none_and_still_do_not_count_as_a_switch():
    look = make(first_after_sec=(8.0, 8.0))
    start(look, 0.0)
    look.looked(1.0, POSE)
    look.looked(2.0, NONE)
    assert look.decide(8.0) is None


def test_seq_repeat_when_the_same_label_keeps_coming_and_its_pool_cools():
    look = make()
    start(look, 0.0)
    look.looked(1.0, POSE)
    assert look.decide(3.0).pool == "pose.any"
    played(look, 3.0, 0.5)
    for at in (4.0, 5.0, 6.0):
        look.looked(at, POSE)
    assert look.decide(7.5).pool == "seq.repeat"


def test_a_cooling_pool_is_skipped():
    look = make(pool_cooldown_sec=15.0)
    start(look, 0.0)
    look.looked(1.0, POSE)
    assert look.decide(3.0).pool == "pose.any"
    played(look, 3.0, 0.5)
    look.looked(6.5, OTHER)
    look.looked(7.0, POSE)
    assert look.decide(7.5).pool == "other.any"


def test_the_fallback_waits_fallback_after_sec_then_invites():
    look = make()
    start(look, 0.0)
    assert look.decide(3.0) is None
    assert look.decide(5.4) is None
    pick = look.decide(5.5)
    assert (pick.pool, pick.reason) == ("seq.invite", "fallback")


def test_the_fallback_takes_other_while_invite_cools():
    look = make()
    start(look, 0.0)
    assert look.decide(5.5).pool == "seq.invite"
    played(look, 5.5, 0.5)
    # Next due at 6.0 + 4, fallback 2.5 later; seq.invite is still cooling.
    assert look.decide(12.5).pool == "other.any"


def test_the_fallback_ignores_cooldown_rather_than_stay_silent():
    look = make(clip_pools=pools(other_any=0), pool_cooldown_sec=100.0)
    start(look, 0.0)
    assert look.decide(5.5).pool == "seq.invite"
    played(look, 5.5, 0.5)
    assert look.decide(12.5).pool == "seq.invite"


def test_no_line_twice_in_a_round():
    look = make(clip_pools=pools(seq_invite=2, other_any=0), pool_cooldown_sec=0.0)
    start(look, 0.0)
    first = look.decide(5.5)
    played(look, 5.5, 0.5)
    second = look.decide(12.5)
    played(look, 12.5, 0.5)
    assert first.clip_id != second.clip_id
    assert look.decide(19.5) is None


def test_the_first_line_of_a_round_differs_from_the_last_of_the_previous():
    for seed in range(40):
        look = make(seed=seed, clip_pools=pools(seq_invite=2, other_any=0))
        start(look, 0.0)
        last = look.decide(5.5).clip_id
        look.leave()
        start(look, 100.0)
        assert look.decide(105.5).clip_id != last


def test_each_round_shuffles_its_pools():
    look = make(seed=3, clip_pools=pools(other_any=0))
    firsts = set()
    for round_index in range(20):
        start = round_index * 100.0
        look.enter(start)
        firsts.add(look.decide(start + 5.5).clip_id)
        look.leave()
    assert len(firsts) > 1


# ---------------------------------------------------------------------------
# Rhythm: first line, gaps, voice, quick reply, window, limits.


def test_the_first_line_waits_first_after_sec_from_the_mirroring_clip_s_end():
    look = make()
    start(look, 0.0)
    look.looked(0.5, POSE)
    # evt.mirroring plays from 0.3 to 2.0.
    look.voice(0.3, busy=True, heard=True)
    look.voice(1.0, busy=True, heard=True)
    look.voice(2.0, busy=False, heard=True)
    look.looked(4.0, POSE)
    assert look.decide(4.9) is None
    assert look.due_at == pytest.approx(5.0)
    assert look.decide(5.0).pool == "pose.any"


def test_first_after_and_gap_are_drawn_from_their_intervals():
    for seed in range(20):
        look = make(seed=seed, first_after_sec=(2.0, 4.0), gap_sec=(3.0, 6.0))
        start(look, 0.0)
        assert 2.0 <= look.due_at <= 4.0
        said_at = look.due_at
        look.looked(said_at, POSE)
        assert look.decide(said_at) is not None
        look.voice(said_at + 0.1, busy=True, heard=True)
        look.voice(10.0, busy=False, heard=True)
        assert 13.0 <= look.due_at <= 16.0


def test_nothing_is_said_while_the_voice_is_busy():
    look = make()
    start(look, 0.0)
    look.looked(2.5, POSE)
    look.voice(3.0, busy=True, heard=True)
    assert look.decide(3.0) is None
    assert look.decide(4.0) is None


def test_the_gap_runs_from_the_end_of_the_line():
    look = make()
    start(look, 0.0)
    look.looked(2.9, POSE)
    assert look.decide(3.0) is not None
    look.voice(3.2, busy=True, heard=True)
    look.voice(6.0, busy=False, heard=True)
    assert look.due_at == pytest.approx(10.0)


def test_a_stale_voice_status_falls_back_to_the_length_estimate():
    look = make()
    look.enter(0.0)
    look.looked(2.9, POSE)
    pick = look.decide(3.0)
    estimate = 1.0 + 0.22 * len(pick.text)
    assert look.busy(3.0 + estimate + 0.5 - 0.1)
    assert not look.busy(3.0 + estimate + 0.5 + 0.1)
    assert look.due_at == pytest.approx(3.0 + estimate + 4.0)


def test_a_status_from_before_the_line_does_not_end_it():
    """The double line: a fresh idle status that predates our publish says nothing about our line."""
    look = make(clip_pools=pools(seq_invite=0, other_any=0))
    start(look, 0.0)
    look.looked(2.9, HEART, ref="seen")
    first = look.decide(3.0, verify=Q3())
    assert first.pool == "hand.heart"
    # Received at 2.99, before the publish; reported again at the next ticks.
    for now in (3.05, 3.5, 5.1, 6.0):
        look.voice(now, busy=False, heard=True, at=2.99)
        look.looked(now, HEART, ref="seen")
        assert look.decide(now, verify=Q3()) is None
    assert look.busy(6.0)


def test_our_line_is_over_once_a_later_status_says_busy_then_idle():
    look = make()
    start(look, 0.0)
    look.looked(2.9, POSE)
    look.decide(3.0)
    look.voice(3.2, busy=False, heard=True)
    assert look.busy(3.2)
    look.voice(3.3, busy=True, heard=True)
    look.voice(4.0, busy=False, heard=True)
    assert not look.busy(4.0)
    assert look.due_at == pytest.approx(8.0)


def test_our_line_the_voice_never_played_is_over_at_its_estimate():
    """A dropped line (latest_start_at passed) must not hold the look forever."""
    look = make()
    start(look, 0.0)
    look.looked(2.9, POSE)
    pick = look.decide(3.0)
    end = 3.0 + 1.0 + 0.22 * len(pick.text)
    look.voice(end + 0.4, busy=False, heard=True)
    assert look.busy(end + 0.4)
    look.voice(end + 0.6, busy=False, heard=True)
    assert not look.busy(end + 0.6)
    assert look.due_at == pytest.approx(end + 4.0)


def test_the_look_keeps_looking_while_a_clip_plays():
    look = make()
    start(look, 0.0)
    look.voice(0.5, busy=True, heard=True)
    assert look.wants_look(0.5)


def test_quick_reply_to_a_verified_named_gesture_after_quick_gap():
    look = make(gap_sec=(5.0, 5.0))
    start(look, 0.0)
    look.looked(2.0, POSE)
    assert look.decide(3.0).pool == "pose.any"
    look.voice(3.2, busy=True, heard=True)
    look.looked(4.0, HEART, ref="seen")
    look.voice(6.0, busy=False, heard=True)
    # Due at 11; the heart may come at 6 + quick_gap 2.
    assert look.decide(7.9, Q3(True)) is None
    pick = look.decide(8.0, Q3(True))
    assert (pick.pool, pick.reason, pick.quick) == ("hand.heart", "named", True)


def test_quick_reply_needs_q3_s_yes():
    look = make(gap_sec=(5.0, 5.0))
    start(look, 0.0)
    look.looked(2.0, POSE)
    look.decide(3.0)
    look.voice(3.2, busy=True, heard=True)
    look.voice(6.0, busy=False, heard=True)
    look.looked(7.0, HEART, ref="seen")
    assert look.decide(8.0, Q3(False)) is None
    # At its due time the refused heart still counts, by its family.
    look.looked(10.5, NONE)
    assert look.decide(11.0).pool == "hand.any"


def test_quick_reply_is_only_for_named_gestures():
    look = make(gap_sec=(5.0, 5.0))
    start(look, 0.0)
    look.looked(2.0, POSE)
    look.decide(3.0)
    look.voice(3.2, busy=True, heard=True)
    look.voice(6.0, busy=False, heard=True)
    look.looked(7.0, DANCE)
    assert look.decide(8.0) is None


def test_quick_reply_ignores_what_was_seen_before_the_last_line():
    look = make(gap_sec=(5.0, 5.0))
    start(look, 0.0)
    look.looked(2.0, POSE)
    look.looked(2.5, HEART, ref="seen")
    assert look.decide(3.0, Q3(False)).pool == "hand.any"
    look.voice(3.2, busy=True, heard=True)
    look.voice(6.0, busy=False, heard=True)
    assert look.decide(8.0, Q3(True)) is None


def test_quick_reply_never_comes_while_the_voice_is_busy():
    look = make(gap_sec=(5.0, 5.0))
    start(look, 0.0)
    look.looked(2.0, POSE)
    look.decide(3.0)
    look.voice(3.5, busy=True, heard=True)
    look.looked(4.0, HEART, ref="seen")
    look.voice(8.0, busy=True, heard=True)
    assert look.decide(8.0, Q3(True)) is None


def test_window_zero_looks_through_the_whole_round():
    look = make(window_sec=0.0)
    start(look, 0.0)
    assert look.wants_look(0.0)
    assert look.wants_look(2.9)


def test_window_mode_looks_only_shortly_before_a_line_is_due():
    look = make(window_sec=1.0)
    start(look, 0.0)
    assert not look.wants_look(0.0)
    assert not look.wants_look(1.9)
    assert look.wants_look(2.0)
    look.looked(2.5, POSE)
    look.decide(3.0)
    look.voice(3.1, busy=True, heard=True)
    look.voice(5.0, busy=False, heard=True)
    # Next due at 9.
    assert not look.wants_look(7.9)
    assert look.wants_look(8.0)


def test_at_most_max_per_round_lines_then_no_more_looking():
    look = make(max_per_round=2, pool_cooldown_sec=0.0)
    start(look, 0.0)
    assert look.decide(5.5) is not None
    played(look, 5.5, 0.5)
    assert look.decide(12.5) is not None
    assert look.done
    assert not look.wants_look(13.0)
    assert look.decide(30.0) is None


def test_outside_mirroring_nothing_happens():
    look = make()
    assert not look.wants_look(0.0)
    assert look.decide(100.0) is None
    look.enter(0.0)
    look.leave()
    assert not look.wants_look(1.0)
    assert look.decide(10.0) is None


def test_three_slow_looks_stop_looking_but_the_fallback_goes_on():
    look = make()
    start(look, 0.0)
    look.slow()
    look.slow()
    look.looked(1.0, NONE)
    look.slow()
    look.slow()
    assert not look.stopped
    look.slow()
    assert look.stopped
    assert not look.wants_look(1.0)
    assert look.decide(5.5).reason == "fallback"


def test_entering_mirroring_resets_the_round():
    look = make(max_per_round=2, pool_cooldown_sec=100.0)
    start(look, 0.0)
    look.looked(1.0, POSE)
    look.decide(3.0)
    played(look, 3.0, 0.5)
    look.decide(10.0)
    for _ in range(3):
        look.slow()
    assert look.done and look.stopped
    look.leave()
    start(look, 50.0)
    assert (look.spoken, look.done, look.stopped, look.labels) == (0, False, False, [])
    assert look.wants_look(50.0)
    # Cooldowns are per round: pose.any is free again.
    look.looked(51.0, POSE)
    assert look.decide(53.0).pool == "pose.any"
    assert look.total == 3


# ---------------------------------------------------------------------------
# The clip table the deployment ships.


def test_every_pool_a_look_can_choose_is_in_the_shipped_table():
    from pathlib import Path

    from embodied_common.internvl_voice_contract import Settings, load_clip_table, look_json, parse_look_json

    path = Path(__file__).resolve().parents[2] / "robot_config" / "config" / "mirror_voice" / "x2_clips_v1.yaml"
    if not path.exists():
        pytest.skip("robot_config's clip table is not beside this package")
    table, _ = load_clip_table(str(path))
    assert table is not None
    parsed, warnings = parse_look_json(look_json(Settings(), table))
    assert warnings == []
    reachable = {pool for label in look_say.LABELS for pool in (label.pool, label.coarse_pool) if pool}
    reachable |= {"seq.repeat", "seq.still", "seq.switch", "seq.invite"}
    assert reachable <= set(parsed["pools"])

    look = LookSay(LookConfig.from_look(parsed), parsed["pools"], random.Random(1))
    assert look.config.window_sec == parsed["window_sec"]
    start(look, 0.0)
    look.looked(1.0, HEART, ref="seen")
    pick = look.decide(5.0, Q3(True))
    assert pick.pool == "hand.heart" and pick.text
