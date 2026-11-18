"""The hand gesture vote and the one-template latch (preset_motions/gesture_vote.py)."""

import json

import pytest

from manipulation_execution.preset_motions import gesture_vote as gv
from manipulation_execution.preset_motions.match_signal import _ACTIVE_REASONS, _POSE_REASONS

MS = 1_000_000
T0 = 100_000_000_000  # camera stamp, ns (100 s)
ROS0 = 100.0


def msg(stamp, cls="holy", score=0.9, actor=3):
    return {"schema": gv.SCHEMA, "stamp_ns": stamp, "actor_id": actor, "cls": cls, "score": score,
            "crop": "both", "roi": [0, 0, 64, 64], "infer_ms": 5.0}  # fmt: skip


class Feed:
    """Runs every 66 ms (15 Hz) by camera time; receive time and ROS time follow the stamps."""

    def __init__(self, vote=None, actor=3):
        self.vote = vote or gv.GestureVote()
        self.vote.set_actor(actor)
        self.stamp = T0

    def mono(self, stamp=None):
        return ((self.stamp if stamp is None else stamp) - T0) / 1e9

    def run(self, cls="holy", score=0.9, actor=3, step_ms=66):
        ok = self.vote.observe(
            msg(self.stamp, cls, score, actor), now_ros_s=ROS0 + self.mono(), now_mono_s=self.mono()
        )
        state = self.vote.update(now_mono_s=self.mono())
        self.stamp += step_ms * MS
        return ok, state


def test_four_runs_in_the_window_enter():
    f = Feed()
    for i in range(3):
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
    assert s["holy"].reason == "confirming" and not s["holy"].exited  # older votes alone do not re-enter
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


@pytest.mark.parametrize(
    "kw", [dict(window_s=0), dict(min_frames=2.5), dict(min_score=1.5), dict(exit_s=float("nan")), dict(exit_s=True)]
)
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
