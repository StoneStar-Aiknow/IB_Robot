"""Session heartbeat admission must not replay old rounds into pose events."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from manipulation_execution.preset_motions import PoseMatcher
from manipulation_execution.preset_motions.session_gate import SessionGate


def _state(seq, stamp, *, state="MIRRORING", phase="running", holding=True, releasing=False):
    return json.dumps(
        {"seq": seq, "stamp": stamp, "state": state, "phase": phase, "holding": holding, "releasing": releasing}
    )


def test_only_fresh_owned_running_state_enters_and_heartbeats_keep_epoch():
    gate = SessionGate()
    assert gate.permit(now_ros_s=100.0, now_mono_s=0.0) == (0, False)
    assert gate.observe(_state(1, 100.0), now_ros_s=100.0, now_mono_s=0.0) == (1, True)
    assert gate.observe(_state(2, 101.0), now_ros_s=101.0, now_mono_s=1.0) == (1, True)
    assert gate.permit(now_ros_s=101.5, now_mono_s=1.5) == (1, True)


@pytest.mark.parametrize(
    "overrides",
    [
        {"state": "WAIT", "phase": "none", "holding": False},
        {"state": "STANDBY", "phase": "ready"},
        {"phase": "wrap_up"},
        {"holding": False},
        {"releasing": True},
    ],
)
def test_leaving_running_revokes_then_new_round_uses_new_epoch(overrides):
    gate = SessionGate()
    first = gate.observe(_state(1, 100.0), now_ros_s=100.0, now_mono_s=0.0)
    assert gate.observe(_state(2, 100.1, **overrides), now_ros_s=100.1, now_mono_s=0.1) == (first[0], False)
    assert gate.observe(_state(3, 100.2), now_ros_s=100.2, now_mono_s=0.2) == (first[0] + 1, True)


def test_duplicate_state_does_not_refresh_either_freshness_clock():
    gate = SessionGate()
    state = _state(1, 100.0)
    gate.observe(state, now_ros_s=100.0, now_mono_s=0.0)
    assert gate.observe(state, now_ros_s=101.9, now_mono_s=1.9)[1]
    assert not gate.permit(now_ros_s=102.01, now_mono_s=2.01)[1]
    assert gate.reason == "session_stale"


def test_monotonic_freshness_revokes_even_when_ros_clock_is_frozen():
    gate = SessionGate()
    gate.observe(_state(1, 100.0), now_ros_s=100.0, now_mono_s=0.0)
    assert not gate.permit(now_ros_s=100.0, now_mono_s=2.01)[1]
    assert gate.reason == "session_receive_stale"


def test_sequence_restart_revokes_and_requires_a_later_increasing_state():
    gate = SessionGate()
    old_epoch, _ = gate.observe(_state(40, 100.0), now_ros_s=100.0, now_mono_s=0.0)
    restart_epoch, active = gate.observe(_state(2, 100.1), now_ros_s=100.1, now_mono_s=0.1)
    assert restart_epoch > old_epoch and not active
    assert gate.permit(now_ros_s=100.15, now_mono_s=0.15) == (restart_epoch, False)
    assert gate.observe(_state(2, 100.1), now_ros_s=100.15, now_mono_s=0.15) == (restart_epoch, False)
    new_epoch, active = gate.observe(_state(3, 100.2), now_ros_s=100.2, now_mono_s=0.2)
    assert new_epoch > old_epoch and active


@pytest.mark.parametrize(
    "state",
    [
        "{",
        "[]",
        "{}",
        _state(True, 100.1),
        _state(1.5, 100.1),
        _state(2, float("nan")),
        _state(2, 100.1, holding=1),
        _state(2, 100.1, releasing=0),
        _state(2, 100.1, state="UNKNOWN"),
    ],
)
def test_invalid_message_revokes_without_reusing_cached_permission(state):
    gate = SessionGate()
    gate.observe(_state(1, 100.0), now_ros_s=100.0, now_mono_s=0.0)
    assert not gate.observe(state, now_ros_s=100.1, now_mono_s=0.1)[1]
    assert not gate.permit(now_ros_s=100.2, now_mono_s=0.2)[1]


def test_stale_and_future_states_cannot_enable_pose_events():
    gate = SessionGate()
    assert not gate.observe(_state(1, 97.0), now_ros_s=100.0, now_mono_s=0.0)[1]
    assert not gate.observe(_state(2, 100.2), now_ros_s=100.0, now_mono_s=0.1)[1]
    assert gate.observe(_state(3, 100.05), now_ros_s=100.0, now_mono_s=0.2)[1]


def test_monotonic_reset_revokes_the_previous_epoch():
    gate = SessionGate()
    epoch, _ = gate.observe(_state(1, 100.0), now_ros_s=100.0, now_mono_s=10.0)
    new_epoch, active = gate.permit(now_ros_s=100.1, now_mono_s=0.0)
    assert new_epoch > epoch and not active


def test_gate_epoch_and_matcher_barrier_reject_old_pear_after_restart():
    gate, matcher = SessionGate(), PoseMatcher()
    epoch, active = gate.observe(_state(1, 100.0), now_ros_s=100.0, now_mono_s=0.0)
    matcher.set_session(epoch=epoch, active=active)
    matcher.tick(now_ros_s=100.0, now_mono_s=0.0)
    epoch, active = gate.observe(
        _state(2, 100.5, state="WAIT", phase="none", holding=False), now_ros_s=100.5, now_mono_s=0.5
    )
    matcher.set_session(epoch=epoch, active=active)
    epoch, active = gate.observe(_state(3, 101.0), now_ros_s=101.0, now_mono_s=1.0)
    matcher.set_session(epoch=epoch, active=active)
    matcher.tick(now_ros_s=101.0, now_mono_s=1.0)
    header = SimpleNamespace(stamp=SimpleNamespace(sec=100, nanosec=800_000_000))
    matcher.observe_actor(
        SimpleNamespace(header=header, actor_id=0, actor_present=True, start_ready=True), now_mono_s=1.0
    )
    old = SimpleNamespace(
        header=header,
        sequence=2,
        schema="pear_parameter_v1",
        source="yolox",
        actor_id=0,
        smplx_pose_raw=np.zeros(312),
    )
    result = matcher.observe_pear(old, now_ros_s=101.0, now_mono_s=1.0)
    assert not result.active and result.reason == "source_before_session"
