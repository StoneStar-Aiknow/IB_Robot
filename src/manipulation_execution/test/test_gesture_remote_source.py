"""Source lifetime is independent of vote lifetime; recovery never reuses old votes."""

from types import SimpleNamespace

import pytest

from embodied_common.hand_gesture_contract import GESTURE_SCHEMA, SourceState
from manipulation_execution.preset_motions.gesture_vote import GestureFusion

CLIENT = "00000000-0000-4000-8000-000000000001"
SESSION = "00000000-0000-4000-8000-000000000002"
SERVER = "00000000-0000-4000-8000-000000000003"
ROS0 = 1_000.0


def fusion(start=0.0):
    f = GestureFusion(remote_source=True, source_started_ns=int(start * 1e9))
    f.set_session(1, True, now_mono_s=start)
    f.set_source_session_token(SESSION)
    actor(f, start)
    return f


def actor(f, now):
    f.observe_actor(SimpleNamespace(actor_id=3, actor_present=True, start_ready=True), now_mono_s=now)


def state(f, now, *, generation=1, seq=1, available=True, published=None, start=0):
    actor(f, now)
    return f.observe_source(
        SourceState(
            CLIENT,
            start,
            generation,
            seq,
            available,
            int((now if published is None else published) * 1e9),
            actor_id=3,
            session_token=SESSION,
            server_boot_id=SERVER,
            engine_sha256="a" * 64,
        ).to_json(),
        now_mono_s=now,
    )


def result(now, *, generation=1, seq=1, published=None, crop="both"):
    return dict(
        schema=GESTURE_SCHEMA,
        stamp_ns=int((ROS0 + now - 0.02) * 1e9),
        actor_id=3,
        cls="holy",
        score=0.9,
        crop=crop,
        roi=[0, 0, 320, 320],
        infer_ms=5.0,
        requester_boot_id=CLIENT,
        producer_start_mono_ns=0,
        stream_generation=generation,
        result_seq=seq,
        published_mono_ns=int((now if published is None else published) * 1e9),
        session_token=SESSION,
    )


def feed(f, now, **kw):
    actor(f, now)
    accepted = f.observe(result(now, **kw), now_ros_s=ROS0 + now, now_mono_s=now)
    return accepted, outputs(f, now)["holy"]


def outputs(f, now, overhead=False):
    return {
        r.template_name: r
        for r in f.results(
            overhead_active=overhead,
            session_active=True,
            epoch=1,
            now_mono_s=now,
        )
    }


def test_two_old_plus_two_new_votes_do_not_enter_but_four_new_do():
    f = fusion()
    assert state(f, 0.0)
    for seq, t in enumerate((0.0, 0.05), 1):
        assert feed(f, t, seq=seq)[0]
    assert state(f, 0.10, generation=2, seq=2, available=False)
    assert state(f, 0.15, generation=2, seq=3)
    for seq, t in enumerate((0.20, 0.25, 0.30), 1):
        accepted, voted = feed(f, t, generation=2, seq=seq)
        assert accepted and not voted.active
    assert feed(f, 0.35, generation=2, seq=4)[1].active


def test_counter_control_without_the_reset_would_enter_on_two_new_votes(monkeypatch):
    f = fusion()
    state(f, 0.0)
    feed(f, 0.0, seq=1)
    feed(f, 0.05, seq=2)
    monkeypatch.setattr(f, "_reset_source_votes", lambda: None)
    state(f, 0.10, generation=2, seq=2, available=False)
    state(f, 0.15, generation=2, seq=3)
    assert not feed(f, 0.20, generation=2, seq=1)[1].active
    assert feed(f, 0.25, generation=2, seq=2)[1].active


def test_late_labels_and_reordered_ready_cannot_reverse_the_fence():
    f = fusion()
    state(f, 0.0, seq=10)
    feed(f, 0.01, seq=1)
    state(f, 0.10, generation=2, seq=11, available=False)
    assert not state(f, 0.11, generation=1, seq=10)
    assert not feed(f, 0.12, seq=2)[0]  # source age is only 20 ms: not a TTL rejection
    assert f.source_gate.state.stream_generation == 2
    assert not f.source_gate.available


def test_unregistered_or_pre_registration_labels_are_never_buffered():
    f = fusion()
    assert not feed(f, 0.0)[0]
    assert state(f, 0.10)
    assert not feed(f, 0.11, published=0.0)[0]
    assert len(f.vote._frames) == 0
    assert feed(f, 0.12, seq=2)[0]


def test_matcher_restart_requires_post_start_state_and_post_registration_labels():
    f = fusion(0.5)
    assert not state(f, 0.6, published=0.4)
    assert not feed(f, 0.61, published=0.4)[0]
    assert state(f, 0.7)
    assert not feed(f, 0.71, published=0.6)[0]
    assert feed(f, 0.72, seq=2)[0]


@pytest.mark.parametrize(
    "crop,times,exit_at", [("both", (0, 0.05, 0.10, 0.15), 0.451), ("one", (0, 0.20, 0.40), 0.901)]
)
def test_fresh_ready_heartbeats_never_extend_the_last_positive(crop, times, exit_at):
    f = fusion()
    state(f, 0.0)
    for seq, t in enumerate(times, 1):
        accepted, voted = feed(f, t, seq=seq, crop=crop)
        assert accepted
    assert voted.active
    assert state(f, exit_at - 0.01, seq=2)
    assert not outputs(f, exit_at)["holy"].active


def test_duplicate_ready_and_lost_are_idempotent_and_overhead_survives(monkeypatch):
    f = fusion()
    resets = []
    original = f._reset_source_votes
    monkeypatch.setattr(f, "_reset_source_votes", lambda: (resets.append(True), original()))
    state(f, 0.0)
    before = len(resets)
    for seq, t in enumerate((0.0, 0.05, 0.10, 0.15), 1):
        feed(f, t, seq=seq)
    assert state(f, 0.2, seq=2)
    assert not state(f, 0.21, seq=2)
    assert len(resets) == before
    outputs(f, 0.22, overhead=True)
    assert f.latch.latched == "overhead_heart"
    state(f, 0.23, generation=2, seq=3, available=False)
    state(f, 0.24, generation=2, seq=4, available=False)
    outputs(f, 0.25, overhead=True)
    assert len(resets) == before + 1
    assert f.latch.latched == "overhead_heart"


def test_health_expiry_only_resets_once():
    f = fusion()
    state(f, 0)
    assert f.source_gate.tick(2_600_000_000)
    assert not f.source_gate.tick(2_700_000_000)


def test_canonical_session_change_clears_votes_even_if_a_stop_heartbeat_was_missed():
    f = fusion()
    state(f, 0)
    for seq, t in enumerate((0, 0.05, 0.10, 0.15), 1):
        feed(f, t, seq=seq)
    assert outputs(f, 0.16)["holy"].active
    f.set_source_session_token("00000000-0000-4000-8000-000000000099", now_mono_s=0.17)
    assert not outputs(f, 0.18)["holy"].active
    assert not feed(f, 0.19, seq=5)[0]


def test_oversize_formal_input_does_not_escape_or_advance_the_source():
    f = fusion()
    state(f, 0)
    assert not f.observe(" " * 4097, now_ros_s=ROS0, now_mono_s=0)
    assert f.source_gate.result_seq == 0


def test_remote_marked_results_cannot_bypass_the_fence_with_a_local_matcher_config():
    f = GestureFusion()
    f.set_session(1, True)
    actor(f, 0)
    assert not f.observe(result(0), now_ros_s=ROS0, now_mono_s=0)
    assert f.vote.last_reason == "remote_source_not_configured"
    local = {k: v for k, v in result(0).items() if k in {"schema", "stamp_ns", "actor_id", "cls", "score", "crop"}}
    assert f.observe(local, now_ros_s=ROS0, now_mono_s=0)
