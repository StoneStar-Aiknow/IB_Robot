"""The hand gesture vote and the one-template latch (preset_motions/gesture_vote.py)."""

import json

import pytest

from manipulation_execution.preset_motions import gesture_vote as gv
from manipulation_execution.preset_motions.match_signal import _ACTIVE_REASONS, _POSE_REASONS

MS = 1_000_000
T0 = 100_000_000_000  # camera stamp, ns (100 s)
ROS0 = 100.0


def msg(stamp, cls="holy", score=0.9, actor=3, crop="both"):
    return {"schema": gv.SCHEMA, "stamp_ns": stamp, "actor_id": actor, "cls": cls, "score": score,
            "crop": crop, "roi": [0, 0, 64, 64], "infer_ms": 5.0}  # fmt: skip


class Feed:
    """Runs every 66 ms (15 Hz) by camera time; receive time and ROS time follow the stamps."""

    def __init__(self, vote=None, actor=3):
        self.vote = vote or gv.GestureVote()
        self.vote.set_actor(actor)
        self.stamp = T0

    def mono(self, stamp=None):
        return ((self.stamp if stamp is None else stamp) - T0) / 1e9

    def run(self, cls="holy", score=0.9, actor=3, step_ms=66, crop="both"):
        ok = self.vote.observe(
            msg(self.stamp, cls, score, actor, crop), now_ros_s=ROS0 + self.mono(), now_mono_s=self.mono()
        )
        state = self.vote.update(now_mono_s=self.mono())
        self.stamp += int(step_ms * MS)
        return ok, state

    def ticks(self, until_stamp):
        """The node's 50 ms state ticks (receive time) up to just before ``until_stamp``; the states."""
        out, t = [], self.mono(self.stamp - 1)
        while t + 0.05 < self.mono(until_stamp):
            t += 0.05
            out.append(self.vote.update(now_mono_s=t))
        return out


def test_four_runs_in_the_window_enter():
    f = Feed()
    for _ in range(3):
        _, s = f.run()
        assert not s["holy"].active and s["holy"].reason == "confirming"
    _, s = f.run()
    assert s["holy"].active and s["holy"].entered and s["holy"].reason == "matched"
    _, s = f.run()
    assert s["holy"].active and not s["holy"].entered and s["holy"].reason == "holding"
    assert s["heart"].reason == "pose_not_matched" and not s["heart"].active


def test_low_scores_are_none_runs():
    f = Feed()
    for _ in range(6):
        _, s = f.run(score=0.49)
    assert not s["holy"].active and s["holy"].votes == 0 and s["holy"].runs == 6


def test_four_is_not_enough_without_the_majority():
    f = Feed()
    for cls in ("none", "holy", "none", "holy", "none", "holy", "none", "holy"):
        _, s = f.run(cls=cls, step_ms=40)  # 8 runs inside 0.6 s: 4 holy of 8 is not more than half
    assert s["holy"].votes == 4 and s["holy"].runs == 8 and not s["holy"].active
    _, s = f.run(cls="holy", step_ms=40)  # oldest (none) drops out: 5 of 8
    assert s["holy"].active


def test_window_is_0_6_s_of_camera_time():
    f = Feed()
    for _ in range(3):
        f.run(step_ms=250)  # 3 holy over 0.5 s
    _, s = f.run(step_ms=250)  # 4th at 0.75 s: the first is out of the window
    assert s["holy"].votes == 3 and not s["holy"].active


def test_exit_after_0_3_s_without_a_run_by_tick_time():
    f = Feed()
    for _ in range(5):
        f.run()
    last = f.mono(f.stamp - 66 * MS)
    s = f.vote.update(now_mono_s=last + 0.29)
    assert s["holy"].active
    s = f.vote.update(now_mono_s=last + 0.31)
    assert not s["holy"].active and s["holy"].exited and s["holy"].reason == "pose_exited"
    s = f.vote.update(now_mono_s=last + 0.32)
    assert s["holy"].reason == "pose_not_matched" and not s["holy"].exited  # its runs left the window
    s = f.vote.update(now_mono_s=last + 0.35)
    assert not s["holy"].active


def test_exit_by_camera_time_when_other_runs_keep_coming():
    f = Feed()
    for _ in range(5):
        f.run()
    states = [f.run(cls="none", step_ms=66)[1]["holy"] for _ in range(6)]
    # none runs 66, 132, ... ms after the last holy: the one 330 ms after it ends it
    assert [s.active for s in states] == [True, True, True, True, False, False]
    assert states[4].reason == "pose_exited" and states[4].exited


def test_exit_by_stamp_even_if_received_together():
    vote = gv.GestureVote()
    vote.set_actor(3)
    for i in range(5):
        assert vote.observe(msg(T0 + i * 66 * MS), now_ros_s=ROS0 + 0.3, now_mono_s=0.0)
    assert vote.update(now_mono_s=0.0)["holy"].active
    vote.observe(msg(T0 + (4 * 66 + 310) * MS, cls="none"), now_ros_s=ROS0 + 0.6, now_mono_s=0.01)
    assert not vote.update(now_mono_s=0.01)["holy"].active


def test_receive_time_window_ages_out_a_stopped_stream():
    f = Feed()
    for _ in range(3):
        f.run()
    s = f.vote.update(now_mono_s=f.mono() + 1.0)
    assert s["holy"].runs == 0 and s["holy"].reason == "pose_not_matched"


def test_only_the_current_actor_counts_and_a_change_resets():
    f = Feed()
    for _ in range(5):
        ok, s = f.run(actor=9)
        assert not ok
    assert f.vote.last_reason == "other_actor" and s["holy"].runs == 0
    for _ in range(5):
        f.run()
    f.vote.set_actor(4)
    s = f.vote.update(now_mono_s=f.mono())
    assert not s["holy"].active and s["holy"].exited and s["holy"].runs == 0
    f.vote.set_actor(None)
    assert not f.vote.observe(msg(f.stamp, actor=4), now_ros_s=ROS0 + f.mono(), now_mono_s=f.mono())


def test_stamp_going_back_resets_and_a_repeat_is_ignored():
    f = Feed()
    for _ in range(5):
        f.run()
    back = f.stamp - 2_000 * MS
    assert not f.vote.observe(msg(f.stamp - 66 * MS), now_ros_s=ROS0 + f.mono(), now_mono_s=f.mono())
    assert f.vote.last_reason == "duplicate"
    assert f.vote.observe(msg(back, cls="none"), now_ros_s=ROS0 + f.mono(back) + 0.1, now_mono_s=f.mono())
    s = f.vote.update(now_mono_s=f.mono())
    assert not s["holy"].active and s["holy"].exited and s["holy"].runs == 1


def test_stale_and_future_runs_do_not_count():
    vote = gv.GestureVote()
    vote.set_actor(3)
    assert not vote.observe(msg(T0), now_ros_s=ROS0 + 0.81, now_mono_s=0.0)
    assert vote.last_reason == "source_stale"
    assert not vote.observe(msg(T0), now_ros_s=ROS0 - 0.11, now_mono_s=0.0)
    assert vote.last_reason == "source_future"


def test_barrier_bars_runs_seen_before_a_session_change():
    f = Feed()
    for _ in range(3):
        f.run()
    seen = f.stamp - 66 * MS
    f.vote.reset(barrier=True)
    assert not f.vote.observe(msg(seen), now_ros_s=ROS0 + f.mono(), now_mono_s=f.mono())
    assert f.vote.last_reason == "before_session"
    assert f.run()[0]


@pytest.mark.parametrize(
    "bad",
    [None, "x", {}, msg(T0) | {"schema": "v0"}, msg(T0) | {"cls": "wave"}, msg(T0) | {"score": 1.5},
     msg(T0) | {"score": True}, msg(T0) | {"stamp_ns": 1.0}, msg(T0) | {"actor_id": "3"}],
)  # fmt: skip
def test_bad_messages_do_not_count(bad):
    vote = gv.GestureVote()
    vote.set_actor(3)
    assert not vote.observe(bad, now_ros_s=ROS0, now_mono_s=0.0) and vote.last_reason == "message_invalid"


def test_numbers_are_parameters():
    f = Feed(gv.GestureVote(gv.VoteConfig(min_frames=2, min_score=0.95)))
    f.run(score=0.9)
    f.run(score=0.9)
    assert not f.vote.update(now_mono_s=f.mono())["holy"].active
    f.run(score=0.96)
    _, s = f.run(score=0.96)
    assert not s["holy"].active  # 2 of 4: not the majority
    _, s = f.run(score=0.96)
    assert s["holy"].active


def test_reentry_after_a_short_dropout_takes_min_frames_fresh_runs():
    f = Feed()
    for _ in range(8):
        f.run(cls="heart")  # 15 Hz heart: active
    assert f.vote.update(now_mono_s=f.mono(f.stamp - 66 * MS))["heart"].active
    resume = f.stamp - 66 * MS + 350 * MS  # a 0.35 s gap, ticks in between
    gap = f.ticks(resume)
    assert [s["heart"].exited for s in gap].count(True) == 1 and not gap[-1]["heart"].active
    f.stamp = resume
    states = [f.run(cls="heart")[1]["heart"] for _ in range(4)]
    # the old window's votes (the last 0.25 s before the gap) no longer count: 4 fresh runs to enter
    assert [s.active for s in states] == [False, False, False, True]
    assert [s.entered for s in states] == [False, False, False, True]
    assert states[0].votes == 1 and states[0].reason == "confirming"


def test_an_exit_only_clears_that_class():
    f = Feed()
    for _ in range(5):
        f.run(cls="holy")
    for _ in range(3):
        f.run(cls="xsign", step_ms=110)
    _, s = f.run(cls="xsign")  # holy last seen 330+ ms ago: exits; the xsign runs stay and now hold the majority
    assert s["holy"].exited and s["holy"].votes == 0
    assert s["xsign"].votes == 4 and s["xsign"].active


def _jitter(base_ms, n, spread=0.15, seed=7):
    import random

    rng = random.Random(seed)
    return [base_ms * (1 + rng.uniform(-spread, spread)) for _ in range(n)]


@pytest.mark.parametrize("seed", range(5))
def test_two_hands_at_15_hz_with_jitter_enter_within_the_window(seed):
    f = Feed()
    t0 = f.stamp
    for step in _jitter(66.7, 20, seed=seed):
        stamp = f.stamp
        _, s = f.run(cls="heart", step_ms=step)
        if s["heart"].entered:
            break
    assert s["heart"].entered and stamp - t0 <= 0.6e9


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("spread", [0.10, 0.15])
def test_one_hand_at_5_hz_with_jitter_enters_and_holds(seed, spread):
    f = Feed()
    t0, entered_at, held = f.stamp, None, []
    for step in _jitter(200.0, 15, spread=spread, seed=seed):  # 3 s of one-hand heart runs
        stamp = f.stamp
        _, s = f.run(cls="heart", step_ms=step, crop="one")
        if entered_at is None and s["heart"].entered:
            entered_at = stamp
        ticks = f.ticks(f.stamp)
        if entered_at is not None:
            held += [s["heart"].active] + [t["heart"].active for t in ticks]
    assert entered_at is not None and entered_at - t0 <= 0.6e9  # the 3rd run
    assert all(held)  # no exit between one-hand runs 170-230 ms apart


def test_one_hand_runs_exit_after_exit_s_one_hand():
    f = Feed()
    for _ in range(4):
        f.run(cls="heart", step_ms=200, crop="one")
    last = f.mono(f.stamp - 200 * MS)
    assert f.vote.update(now_mono_s=last + 0.45)["heart"].active  # past exit_s 0.3, within 0.5
    assert not f.vote.update(now_mono_s=last + 0.51)["heart"].active


def test_one_hand_entry_needs_its_own_count():
    f = Feed(gv.GestureVote(gv.VoteConfig(min_frames_one_hand=4)))
    for _ in range(6):
        _, s = f.run(cls="heart", step_ms=230, crop="one")  # 3 runs fit the 0.6 s window
        assert not s["heart"].active
    f = Feed()
    for _ in range(2):
        f.run(cls="heart", crop="one")
    _, s = f.run(cls="heart")  # 2 one-hand + 1 two-hand: neither count is reached
    assert not s["heart"].active
    _, s = f.run(cls="heart")  # 4 runs in all
    assert s["heart"].active


@pytest.mark.parametrize(
    "kw",
    [dict(window_s=0), dict(min_frames=2.5), dict(min_frames_one_hand=3.0), dict(min_score=1.5),
     dict(exit_s=float("nan")), dict(exit_s=True), dict(exit_s_one_hand=0)],
)  # fmt: skip
def test_vote_config_is_checked(kw):
    with pytest.raises(ValueError):
        gv.VoteConfig(**kw)


def test_reasons_are_pose_reasons():
    f = Feed()
    seen = set()
    for cls in ("holy",) * 6 + ("none",) * 8:
        _, s = f.run(cls=cls)
        for state in s.values():
            seen.add(state.reason)
            assert state.reason in _POSE_REASONS
            assert not state.active or state.reason in _ACTIVE_REASONS
    assert seen == {"pose_not_matched", "confirming", "matched", "holding", "pose_exited"}


# -- latch --------------------------------------------------------------------------------------


def test_latch_first_to_activate_keeps_it():
    latch = gv.TemplateLatch()
    assert latch.update({"holy": True}) == {"overhead_heart": False, "heart": False, "holy": True, "xsign": False}
    assert latch.update({"holy": True, "xsign": True})["holy"]
    got = latch.update({"holy": False, "xsign": True})
    assert got["xsign"] and not got["holy"]
    assert sum(latch.update({"heart": True, "holy": True, "xsign": True}).values()) == 1


def test_latch_tie_goes_to_the_earlier_name():
    assert gv.TemplateLatch().update({"heart": True, "holy": True, "xsign": True})["heart"]


def test_overhead_heart_is_the_rule_and_beats_the_hand_heart():
    latch = gv.TemplateLatch()
    got = latch.update({"overhead_heart": True, "heart": True})
    assert got["overhead_heart"] and not got["heart"]
    got = latch.update({"heart": True})  # heart was held back, not latched: now it is
    assert got["heart"] and not got["overhead_heart"]
    got = latch.update({"overhead_heart": True, "heart": True})
    assert got["overhead_heart"] and not got["heart"]
    assert latch.update({"overhead_heart": True, "holy": True}) == {
        "overhead_heart": True, "heart": False, "holy": False, "xsign": False}  # fmt: skip
    assert latch.update({}) == dict.fromkeys(gv.TEMPLATE_ORDER, False) and latch.latched is None


# -- fusion -------------------------------------------------------------------------------------


class Actor:
    def __init__(self, actor_id=3, present=True, ready=True):
        self.actor_id, self.actor_present, self.start_ready = actor_id, present, ready


def fusion_with_votes(n=5, cls="holy"):
    fu = gv.GestureFusion()
    fu.set_session(1, True)
    fu.observe_actor(Actor(), now_mono_s=0.0)
    for i in range(n):
        fu.observe(json.dumps(msg(T0 + i * 66 * MS, cls)), now_ros_s=ROS0 + i * 0.066, now_mono_s=i * 0.066)
    return fu


def by_name(results):
    return {r.template_name: r for r in results}


def test_fusion_results_are_match_state_ready():
    fu = fusion_with_votes()
    first = by_name(fu.results(overhead_active=False, session_active=True, epoch=1, now_mono_s=0.3))
    assert list(first) == ["heart", "holy", "xsign"]
    holy = first["holy"]
    assert holy.active and holy.reason == "matched" and holy.actor_id == 3 and holy.session_epoch == 1
    assert holy.source_stamp_ns == T0 + 4 * 66 * MS
    second = by_name(fu.results(overhead_active=False, session_active=True, epoch=1, now_mono_s=0.31))
    assert all(second[n].sequence == first[n].sequence + 1 for n in first)
    assert second["holy"].reason == "holding"


def test_fusion_heart_yields_to_the_overhead_heart():
    fu = fusion_with_votes(cls="heart")
    heart = by_name(fu.results(overhead_active=True, session_active=True, epoch=1, now_mono_s=0.3))["heart"]
    assert not heart.active and heart.reason == "confirming"
    heart = by_name(fu.results(overhead_active=False, session_active=True, epoch=1, now_mono_s=0.31))["heart"]
    assert heart.active


def test_fusion_ignores_runs_outside_an_active_session_and_resets_on_a_change():
    fu = fusion_with_votes()
    fu.set_session(2, True)  # new round: the vote starts over, earlier stamps are barred
    holy = by_name(fu.results(overhead_active=False, session_active=True, epoch=2, now_mono_s=0.3))["holy"]
    assert not holy.active and holy.source_stamp_ns is None
    fu.set_session(2, False)
    assert not fu.observe(msg(T0 + 10**9), now_ros_s=ROS0 + 1.0, now_mono_s=1.0)


def test_fusion_actor_must_be_present_ready_and_fresh():
    fu = fusion_with_votes()
    assert fu.current_actor(0.5) == 3 and fu.current_actor(1.01) is None
    fu.observe_actor(Actor(ready=False), now_mono_s=0.5)
    assert fu.current_actor(0.5) is None
    results = by_name(fu.results(overhead_active=False, session_active=True, epoch=1, now_mono_s=0.5))
    assert not results["holy"].active and results["holy"].actor_id is None
    fu.observe_actor(Actor(actor_id=True), now_mono_s=0.6)
    assert fu.current_actor(0.6) is None


def test_stats_count_gaps_enters_exits_and_fast_reentries():
    f = Feed()
    for _ in range(5):
        f.run()  # 66 ms gaps: the 0.1 s bin
    last = f.mono(f.stamp - 66 * MS)
    s = f.vote.update(now_mono_s=last + 0.31)
    assert s["holy"].exited and s["holy"].held_s == pytest.approx(0.066 + 0.31)  # entered on run 4
    assert s["holy"].since_pos_s == pytest.approx(0.31) and not s["holy"].one_hand
    assert gv.gesture_transition_lines(s) == ["gesture exit holy held=0.38 since_pos=0.31 last=two"]
    f.stamp += int(0.4e9)  # back 0.4 s after the last run (0.09 s after the exit)
    for _ in range(4):
        _, s = f.run()
    assert s["holy"].entered
    stats = f.vote.stats["holy"]
    assert stats["runs"] == 9 and stats["enters"] == 2 and stats["exits"] == 1 and stats["reenters"] == 1
    assert stats["gaps"][0] == 7 and sum(stats["gaps"]) == 8 and stats["gaps"][4] == 1  # 0.466 s: the 0.5 s bin
    assert stats["gap_max_active_s"] == pytest.approx(0.066)  # the long gap came while inactive
    assert f.vote.stats["heart"]["runs"] == 0


def test_stats_one_hand_gaps_and_a_reset_exit_have_their_own_fields():
    f = Feed()
    for _ in range(4):
        _, s = f.run(crop="one", step_ms=200)
    assert s["holy"].active and s["holy"].one_hand
    assert f.vote.stats["holy"]["gaps_one_hand"][1] == 3 and sum(f.vote.stats["holy"]["gaps"]) == 0
    f.vote.set_actor(4)
    s = f.vote.update(now_mono_s=f.mono())
    assert s["holy"].exited and s["holy"].held_s is not None and f.vote.stats["holy"]["exits"] == 1
