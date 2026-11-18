"""A fault or a lost runtime status sends the robot back to idle (zll 11-18).

The session used to enter WAIT and leave ``policy_stream`` standing: the
runtime admits no stream while it is not ACTIVE, so the arm stayed where the
fault caught it. Now the next tick releases, once per WAIT, unless the fault
says the robot is in someone else's hands. Real core, isolated runtime port.
"""

import ast
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
from test_mirror_session_preset_core import Clock, Cover, Hooks, Runtime, dispatch, make_rig, tick

from manipulation_execution.mirror_session.core import MirrorSession, Phase, SessionConfig, SessionState
from manipulation_execution.mirror_session.runtime import NOT_OURS_FAULTS, WaitReason

IDENTITY = {"id": "fake", "type": "fake", "runtime_name": "isolated", "runtime_version": "1"}
RUNTIME_NODE = Path(__file__).resolve().parents[2] / "robots/aimdk/aimdk_robot/aimdk_robot/runtime_node.py"

ARBITRATION = (
    "command arbitration held by 'pad' at priority 90 (this runtime registers 40); streamed commands are not applied"
)
REGISTRATION = "vendor input source registration failed; commands would be discarded"
ESTOP = "platform system mode 'EStop' disables the interfaces this runtime depends on"


def idle_requests(rig):
    return [call for call in rig.runtime.calls if call == ("set_mode", "idle")]


def fault(rig, *faults, lifecycle="DEGRADED"):
    """The runtime reports a problem; only the status callback runs."""
    rig.runtime.status = replace(rig.runtime.status, lifecycle=lifecycle, faults=tuple(faults))
    rig.session.observe_status(rig.runtime.status)


def standby_rig():
    """A session in STANDBY/approach: taken over, not yet ticked."""
    clock, runtime, hooks, sent = Clock(), Runtime(), Hooks(), []
    session = MirrorSession(
        SessionConfig("policy_stream", IDENTITY), runtime, sent.append, hooks, Cover(), clock, clock.sleep
    )
    assert session.startup().idle
    runtime.status = replace(runtime.status, active_mode="policy_stream")
    session.observe_arm_position((0.0,) * 14)
    assert session.request("start")[0]
    runtime.calls.clear()
    return type("Rig", (), {"session": session, "runtime": runtime, "clock": clock, "sent": sent})()


def test_a_fault_while_mirroring_is_released_to_idle_on_the_next_tick_not_in_the_callback():
    rig = make_rig(with_preset=False)
    fault(rig, "joint feedback stale: arm")
    assert rig.session.state == SessionState.WAIT and rig.session.wait_reason == WaitReason.FAULT
    assert not rig.session.holding
    assert idle_requests(rig) == []  # the status callback never blocks on a release
    rig.session.tick()
    assert len(idle_requests(rig)) == 1
    assert rig.runtime.status.active_mode == "idle"
    assert rig.session.last_release.idle
    assert rig.session.detail.startswith("runtime DEGRADED: joint feedback stale: arm; fault release: ")
    assert "runtime returned to idle" in rig.session.detail
    assert not any(call[0] == "stop" for call in rig.runtime.calls)


@pytest.mark.parametrize(
    "faults,lifecycle",
    [
        (("clock skew 0.31 s exceeds 0.1 s",), "FAULTED"),
        (("right hand expected 1 (nimble_hands), platform reports 0 (none)",), "DEGRADED"),
        (("arm joint fault: left_elbow",), "DEGRADED"),
        (("head joint fault: head_yaw",), "DEGRADED"),
        (("named hold ownership unproven",), "DEGRADED"),
        ((), "FAULTED"),
        (("joint feedback stale: arm", "clock skew 0.31 s exceeds 0.1 s"), "FAULTED"),
    ],
)
def test_every_fault_that_leaves_the_robot_ours_is_released(faults, lifecycle):
    rig = make_rig(with_preset=False)
    fault(rig, *faults, lifecycle=lifecycle)
    assert rig.session.wait_reason == WaitReason.FAULT
    rig.session.tick()
    assert len(idle_requests(rig)) == 1
    assert "fault release: runtime returned to idle" in rig.session.detail


@pytest.mark.parametrize(
    "faults",
    [
        (ARBITRATION,),
        (REGISTRATION,),
        (ESTOP,),
        ("platform system mode 'OTA' disables the interfaces this runtime depends on",),
        ("joint feedback stale: arm", ARBITRATION),  # one owner elsewhere is enough
    ],
)
def test_a_fault_that_puts_the_robot_in_other_hands_is_left_alone(faults):
    rig = make_rig(with_preset=False)
    fault(rig, *faults)
    assert rig.session.state == SessionState.WAIT and rig.session.wait_reason == WaitReason.FAULT
    for _ in range(5):
        tick(rig, seconds=0.1)
    assert not any(call[0] in ("set_mode", "stop") for call in rig.runtime.calls)
    assert "fault release" not in rig.session.detail


def test_every_marker_is_still_what_the_runtime_writes_for_those_sources():
    """Mention-level contract: catches a reworded fault, not a new fault source."""
    source = RUNTIME_NODE.read_text()
    details = {"arbitration": [], "system": []}
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "_set_fault" and node.args):
            continue
        name = node.args[0]
        if isinstance(name, ast.Constant) and name.value in details:
            details[name.value].append(ast.get_source_segment(source, node.args[1]))
    assert len(details["arbitration"]) >= 3 and len(details["system"]) >= 2, details
    for text in (t for texts in details.values() for t in texts):
        if text != "None":
            assert any(marker in text for marker in NOT_OURS_FAULTS), text
    for marker in NOT_OURS_FAULTS:
        assert any(marker in (text or "") for texts in details.values() for text in texts), marker


def test_a_stop_that_is_waiting_makes_the_only_release():
    rig = make_rig(with_preset=False)
    answers = []
    stopper = threading.Thread(target=lambda: answers.append(rig.session.request("stop")))
    stopper.start()
    deadline = time.monotonic() + 5.0
    while rig.session._stop_waiters == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert rig.session._stop_waiters == 1
    with rig.session._release_lock:
        # The stop cannot reach its release yet; a tick now must not start a second one.
        fault(rig, "joint feedback stale: arm")
        ticker = threading.Thread(target=rig.session.tick)
        ticker.start()
        ticker.join(timeout=1.0)
        assert not ticker.is_alive(), "the tick waited to release beside the stop"
    stopper.join(timeout=10.0)
    assert not stopper.is_alive() and answers and answers[0][0]
    for _ in range(3):
        tick(rig, seconds=0.1)
    assert len(idle_requests(rig)) == 1
    assert "fault release" not in rig.session.detail


def test_a_start_that_takes_over_before_the_tick_is_never_released_under():
    rig = make_rig(with_preset=False)
    fault(rig, "joint feedback stale: arm")
    rig.runtime.status = replace(rig.runtime.status, lifecycle="ACTIVE", faults=())
    assert rig.session.request("start")[0]
    assert rig.session.holding and rig.session.state == SessionState.STANDBY
    rig.session.tick()
    assert idle_requests(rig) == []
    # The pending release died with the takeover: a later revocation is not answered with one.
    rig.runtime.status = replace(rig.runtime.status, active_mode="grasp_stream")
    rig.session.observe_status(rig.runtime.status)
    assert rig.session.wait_reason == WaitReason.REVOKED
    for _ in range(3):
        rig.session.tick()
    assert idle_requests(rig) == []
    assert "fault release" not in rig.session.detail


def test_a_fault_that_persists_is_released_once():
    rig = make_rig(with_preset=False)
    for _ in range(5):
        fault(rig, "clock skew 0.31 s exceeds 0.1 s", lifecycle="FAULTED")
        rig.clock.advance(0.1)
        rig.session.tick()
    assert len(idle_requests(rig)) == 1
    assert rig.session.detail.count("fault release") == 1


def test_a_fault_during_a_preset_still_only_holds():
    rig = make_rig()
    dispatch(rig)
    fault(rig, "joint feedback stale: arm")
    assert rig.session.wait_reason == WaitReason.FAULT
    tick(rig, matcher=False)
    tick(rig, matcher=False, seconds=0.1)
    assert rig.runtime.calls.count(("stop", "HOLD")) == 1
    assert idle_requests(rig) == []
    assert "fault release: " in rig.session.detail and "idle not requested" in rig.session.detail


def test_a_fault_in_standby_approach_is_released():
    rig = standby_rig()
    assert (rig.session.state, rig.session.phase) == (SessionState.STANDBY, Phase.APPROACH)
    rig.runtime.status = replace(rig.runtime.status, lifecycle="DEGRADED", faults=("arm joint fault: left_elbow",))
    rig.session.observe_status(rig.runtime.status)
    assert idle_requests(rig) == []
    rig.session.tick()
    assert rig.session.state == SessionState.WAIT and len(idle_requests(rig)) == 1


@pytest.mark.parametrize("where", ["cancel_wrap_up", "cooldown"])
def test_a_fault_while_a_round_ends_or_cools_down_is_released(where):
    rig = make_rig(with_preset=False)
    assert rig.session.operator_touch()[0]
    assert rig.session.phase == Phase.WRAP_UP
    if where == "cooldown":
        tick(rig)
        assert (rig.session.state, rig.session.phase) == (SessionState.STANDBY, Phase.COOLDOWN)
    fault(rig, "joint feedback stale: arm")
    rig.session.tick()
    assert rig.session.state == SessionState.WAIT and len(idle_requests(rig)) == 1


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"active_mode": "grasp_stream"}, WaitReason.REVOKED),
        ({"runtime_name": "someone_else"}, WaitReason.REVOKED),
        ({"stop_latched": True, "active_mode": "idle", "lifecycle": "STOPPED"}, WaitReason.STOP_LATCHED),
    ],
)
def test_another_mode_or_a_stop_latch_is_still_not_answered(change, reason):
    rig = make_rig(with_preset=False)
    rig.runtime.status = replace(rig.runtime.status, **change)
    rig.session.observe_status(rig.runtime.status)
    assert rig.session.wait_reason == reason
    for _ in range(3):
        tick(rig, seconds=0.1)
    assert not any(call[0] in ("set_mode", "stop") for call in rig.runtime.calls)


def test_a_silent_runtime_is_released_in_the_tick_that_notices():
    rig = make_rig(with_preset=False)
    rig.clock.advance(2.5)  # no status for longer than status_max_age_s
    rig.session.tick()
    assert rig.session.wait_reason == WaitReason.RUNTIME_LOST
    assert len(idle_requests(rig)) == 1
    assert rig.session.detail.startswith("no fresh runtime status; fault release: runtime returned to idle")


def test_a_stale_status_stamp_is_released_on_the_next_tick():
    rig = make_rig(with_preset=False)
    rig.session.observe_status(replace(rig.runtime.status, stamp_age_s=5.0))
    assert rig.session.wait_reason == WaitReason.RUNTIME_LOST and idle_requests(rig) == []
    rig.session.tick()
    assert len(idle_requests(rig)) == 1


def test_a_runtime_that_does_not_answer_gets_no_request():
    rig = make_rig(with_preset=False)
    rig.runtime.get_status = lambda timeout_s: None
    rig.clock.advance(2.5)
    rig.session.tick()
    assert rig.session.wait_reason == WaitReason.RUNTIME_LOST
    assert idle_requests(rig) == []
    assert rig.session.detail.endswith("fault release: runtime status unavailable; idle not requested")
