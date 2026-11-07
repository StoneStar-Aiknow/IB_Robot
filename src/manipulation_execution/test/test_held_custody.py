"""Custody fencing and expiry must never let two executors move one arm."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from manipulation_execution.grasp_pipeline.contracts import EffectorAction, GraspFailure, HeldObject, Move
from manipulation_execution.grasp_pipeline.sequence import GraspSequence, SequenceResult
from manipulation_execution.held_custody import HeldCustody, HeldCustodyServer, held_from_msg, held_to_msg


def held():
    return HeldObject(
        "grip-1",
        "a red apple",
        "left",
        ("joint",),
        (0.2,),
        (0.0,),
        0.05,
        (0.0, 0.0, 0.0),
        "{}",
        lifted_at_ns=1234567890,
        frame_id="base",
    )


def custody():
    now, calls = [0.0], []
    owner = HeldCustody(clock=lambda: now[0])
    owner.accept(
        held(),
        recover=lambda: calls.append("recover") or True,
        release=lambda: calls.append("yield"),
        holding=lambda: "",
    )
    return owner, now, calls


def test_wire_preserves_grip_geometry_pose_and_capture_time():
    assert held_from_msg(held_to_msg(held())) == held()


def test_claim_is_exclusive_and_return_is_fenced():
    owner, _, calls = custody()
    token, value = owner.claim("grip-1", "a red apple", "place")
    assert value == held() and calls == ["yield"]
    with pytest.raises(GraspFailure, match="claimed"):
        owner.claim("grip-1", "a red apple", "other")
    with pytest.raises(GraspFailure, match="does not own"):
        owner.returned("grip-1", "stale", "released")
    owner.returned("grip-1", token, "released")
    assert not owner.occupied


def test_unclaimed_expiry_puts_back_but_claimed_expiry_only_alarms():
    owner, now, calls = custody()
    now[0] = 31
    owner.tick()
    assert calls == ["recover"] and owner.state == "released"
    owner, now, calls = custody()
    token, _ = owner.claim("grip-1", "a red apple", "place", lease_s=10)
    now[0] = 11
    owner.tick()
    assert calls == ["yield"] and owner.state == "uncertain" and owner.occupied
    with pytest.raises(GraspFailure):
        owner.returned("grip-1", token, "released")


@pytest.mark.parametrize("drop", [False, True])
def test_failed_sequence_is_not_truthy_success_and_preserves_recovery_evidence(drop):
    owner, now, _ = custody()
    reports = []
    owner.diagnostic = reports.append
    failed = SequenceResult(
        message="tracking anomaly",
        error_code="MOTION_FAILED",
        phases=[{"name": "lift_reverse", "state": "failed", "reason": "tracking anomaly"}],
        execution_report={"last_sent_at": 30.5},
    )
    if drop:
        owner.holding, owner.drop = lambda: "lost", lambda: failed
    else:
        owner.recover = lambda: failed
    now[0] = 31
    owner.tick()
    assert owner.state == "uncertain" and owner.occupied
    assert owner.reason == ("DROP_RECOVERY_FAILED" if drop else "RECOVERY_FAILED")
    assert owner.recovery_report["result"]["message"] == "tracking anomaly"
    assert owner.recovery_report["result"]["phases"][0]["name"] == "lift_reverse"
    assert owner.recovery_report["result"]["execution_report"]["last_sent_at"] == 30.5
    assert reports == [owner.recovery_report]


def test_recovery_exception_keeps_type_message_and_traceback():
    owner, now, _ = custody()

    def broken_recovery():
        raise RuntimeError("transport disconnected")

    owner.recover = broken_recovery
    now[0] = 31
    owner.tick()
    error = owner.recovery_report["exception"]
    assert owner.state == "uncertain"
    assert error["type"] == "RuntimeError" and error["message"] == "transport disconnected"
    assert "broken_recovery" in error["traceback"]


def test_recovery_log_preserves_unstamped_feedback_as_stale():
    logs = []
    server = object.__new__(HeldCustodyServer)
    server.node = SimpleNamespace(get_logger=lambda: SimpleNamespace(info=logs.append, error=logs.append))
    server.log_recovery({"succeeded": False, "feedback": {"received_at": float("-inf")}})
    logged = json.loads(logs[0].removeprefix("held custody recovery: "))
    assert logged["feedback"]["received_at"] == "-inf"


@pytest.mark.parametrize("first_succeeded", [False, True])
def test_recovery_log_allows_failure_and_success_on_the_same_server(first_succeeded, capfd):
    from rclpy.logging import LoggingSeverity, get_logger

    logger = get_logger(f"held_recovery_severity_{first_succeeded}")
    logger.set_level(LoggingSeverity.INFO)
    server = object.__new__(HeldCustodyServer)
    server.node = SimpleNamespace(get_logger=lambda: logger)
    outcomes = [first_succeeded, not first_succeeded, first_succeeded]
    for succeeded in outcomes:
        server.log_recovery({"succeeded": succeeded, "operation": "severity_transition"})
    output = capfd.readouterr()
    lines = [line for line in (output.out + output.err).splitlines() if "held custody recovery: " in line]
    reports = [json.loads(line.split("held custody recovery: ", 1)[1].split("\x1b", 1)[0]) for line in lines]
    assert [report["succeeded"] for report in reports] == outcomes


@pytest.mark.parametrize("lease", [0, -1, float("nan"), float("inf"), 121])
def test_renewal_rejects_invalid_lease(lease):
    owner, _, _ = custody()
    token, _ = owner.claim("grip-1", "a red apple", "place")
    with pytest.raises(GraspFailure):
        owner.claim("grip-1", "a red apple", "place", token=token, lease_s=lease)


def test_unknown_grip_blocks_claim_and_prevents_expiry_recovery():
    owner, now, calls = custody()
    owner.holding = lambda: None
    with pytest.raises(GraspFailure) as error:
        owner.claim("grip-1", "a red apple", "place")
    assert error.value.code == "GRASP_UNCERTAIN"
    owner.tick()
    now[0] = 31
    owner.tick()
    assert owner.state == "uncertain" and not calls


def test_unknown_grip_before_a_claim_does_not_count_toward_the_returned_debounce():
    owner, now, _ = custody()
    grip = [None]
    owner.holding = lambda: grip[0]
    owner.tick()  # one unknown read while picked starts the 1 s debounce
    grip[0] = ""
    now[0] = 0.05
    token, _ = owner.claim("grip-1", "a red apple", "place")
    now[0] = 2.5  # place held the arm for longer than the debounce
    owner.returned("grip-1", token, "held")
    grip[0] = None
    owner.tick()
    assert owner.state == "picked"
    now[0] = 3.6
    owner.tick()
    assert (owner.state, owner.reason) == ("uncertain", "GRASP_UNCERTAIN")


def test_returned_held_object_can_only_be_claimed_by_its_target():
    owner, _, _ = custody()
    token, _ = owner.claim("grip-1", "a red apple", "place")
    owner.returned("grip-1", token, "held")
    with pytest.raises(GraspFailure) as error:
        owner.claim("grip-1", "a green apple", "place")
    assert error.value.code == "HELD_OBJECT_MISMATCH"
    owner.held = replace(owner.held, custody_id="grip-2")
    with pytest.raises(GraspFailure):
        owner.returned("grip-1", token, "released")


@pytest.mark.parametrize(
    ("home", "state", "reason"), [(True, "lost", "GRIP_LOST"), (False, "uncertain", "DROP_RECOVERY_FAILED")]
)
def test_unclaimed_drop_opens_the_hand_and_goes_home_or_waits_for_a_manual_check(home, state, reason):
    owner, _, calls = custody()
    states = []
    owner.drop = lambda: states.append(owner.state) or calls.append("drop") or home
    owner.holding = lambda: "thumb lost contact"
    owner.tick()
    assert calls == ["drop"] and states == ["recovering"]
    assert (owner.state, owner.reason) == (state, reason)
    assert owner.occupied is (state == "uncertain")
    with pytest.raises(GraspFailure):
        owner.claim("grip-1", "a red apple", "place")


def dropped_home_failed(settled):
    """A custody whose drop opened the hand, but whose way home failed (live 2026-11-07, a lemon)."""
    owner, now, calls = custody()
    reports = []
    owner.diagnostic, owner.settled = reports.append, settled
    owner.holding = lambda: "thumb lost contact"
    owner.drop = lambda: SequenceResult(
        error_code="HOME_PLAN_FAILED",
        completed=["release"],
        phases=[{"name": "release", "state": "succeeded", "reason": ""}],
        at_rest=False,
    )
    owner.tick()
    return owner, now, reports


def test_a_drop_whose_hand_opened_but_could_not_go_home_waits_as_drop_home_failed():
    owner, _, reports = dropped_home_failed(lambda: False)
    assert (owner.state, owner.reason) == ("uncertain", "DROP_HOME_FAILED") and owner.occupied
    assert reports[-1]["operation"] == "DROP_HOME_FAILED" and not reports[-1]["succeeded"]


def test_drop_home_failed_ends_once_the_open_hand_stays_home():
    settled = [False]
    owner, now, reports = dropped_home_failed(lambda: settled[0])
    owner.tick()
    assert owner.state == "uncertain"
    settled[0] = True  # a recovery pose brought the arm back
    now[0] = 1.0
    owner.tick()
    now[0] = 1.5
    owner.tick()
    assert owner.state == "uncertain"  # not yet still for SETTLED_S
    settled[0] = False  # moving again: the wait restarts
    now[0] = 1.9
    owner.tick()
    settled[0] = True
    now[0] = 2.0
    owner.tick()
    now[0] = 2.9
    owner.tick()
    assert owner.state == "uncertain"
    now[0] = 3.0
    owner.tick()
    assert (owner.state, owner.reason) == ("lost", "GRIP_LOST") and not owner.occupied
    assert reports[-1]["operation"] == "DROP_HOME_RECOVERED" and reports[-1]["succeeded"]
    owner.accept(held(), recover=lambda: True, release=lambda: None, holding=lambda: "")  # the next pick
    assert owner.state == "picked"


def test_unreadable_feedback_never_ends_a_drop_home_failed_custody():
    def settled():
        raise RuntimeError("stale arm feedback")

    owner, now, _ = dropped_home_failed(settled)
    for step in range(1, 5):
        now[0] = float(step)
        owner.tick()
    assert (owner.state, owner.reason) == ("uncertain", "DROP_HOME_FAILED")


def test_a_drop_whose_hand_did_not_open_never_settles():
    """Something may still be in the hand: it stays for a manual check whatever the arm does."""
    owner, now, calls = custody()
    asked = []
    owner.settled = lambda: asked.append(1) or True
    owner.holding = lambda: "thumb lost contact"
    owner.drop = lambda: SequenceResult(error_code="MOTION_FAILED", phases=[{"name": "release", "state": "failed"}])
    owner.tick()
    for step in range(1, 4):
        now[0] = float(step)
        owner.tick()
    assert (owner.state, owner.reason) == ("uncertain", "DROP_RECOVERY_FAILED") and not asked


def test_drop_without_a_way_home_still_stops_for_a_manual_check():
    owner, _, calls = custody()
    owner.holding = lambda: "thumb lost contact"
    owner.tick()
    assert calls == [] and (owner.state, owner.reason) == ("uncertain", "GRIP_LOST")


def test_operator_authorized_empty_hand_reset_clears_uncertain_after_home_is_verified():
    owner, now, calls = custody()
    reports, at_home = [], [False]
    owner.diagnostic = reports.append
    owner.settled = lambda: at_home[0]
    token, _ = owner.claim("grip-1", "a red apple", "place")
    owner.returned("grip-1", token, "uncertain", "GRASP_UNCERTAIN")
    assert (
        owner.authorize_reset(
            "grip-1", "operator-check-1", operator_confirmed_empty=True, operator_note="object removed; hand empty"
        )
        > 0
    )
    owner.tick()
    assert owner.state == "uncertain"  # Authorization permits recovery, but does not invent feedback.
    at_home[0] = True
    owner.tick()
    now[0] += 1.1
    owner.tick()
    assert (owner.state, owner.reason, owner.occupied) == ("released", "MANUAL_RESET_COMPLETE", False)
    assert calls == ["yield"]  # Recording authorization and verifying home issue no motion commands.
    assert reports[-1]["operation"] == "MANUAL_RESET_COMPLETE"
    assert reports[-1]["original_reason"] == "GRASP_UNCERTAIN"


def test_expired_or_stale_manual_check_cannot_clear_an_uncertain_custody():
    owner, now, _ = custody()
    owner.state, owner.reason = "uncertain", "GRASP_UNCERTAIN"
    owner.settled = lambda: True
    with pytest.raises(GraspFailure, match="does not match"):
        owner.authorize_reset("old-custody", "check-1", operator_confirmed_empty=True, operator_note="empty")
    owner.authorize_reset("grip-1", "check-1", operator_confirmed_empty=True, operator_note="empty")
    owner.tick()
    now[0] = 61.0
    owner.tick()
    assert owner.state == "uncertain" and owner.occupied and owner.manual_reset is None


def test_a_drop_after_place_gave_the_object_back_also_goes_home():
    owner, _, calls = custody()
    owner.drop = lambda: calls.append("drop") or True
    token, _ = owner.claim("grip-1", "a red apple", "place")
    owner.returned("grip-1", token, "held")  # place failed before moving
    owner.holding = lambda: "thumb lost contact"
    owner.tick()
    assert calls == ["yield", "drop"] and owner.state == "lost" and not owner.occupied


def lifted_hand():
    from test_grasp_pipeline import Effector, Executor

    executor, effector = Executor(), Effector(gripped=True)
    executor.position = np.full(3, 0.4)
    return SimpleNamespace(executor=executor, effector=effector, watch=None)


def test_home_after_drop_opens_the_hand_before_driving_home_on_the_fresh_plan():
    from test_grasp_pipeline import trajectory

    backends = lifted_hand()
    plans = []

    def go_home():
        plans.append(list(backends.effector.log))
        return Move(trajectory("home", np.full(3, 0.4), np.zeros(3)))

    result = GraspSequence(backends).home_after_drop(go_home)
    assert result.succeeded and result.at_rest and not result.held
    assert plans == [[EffectorAction.RELEASE]]  # planned with the hand already open
    assert [row[0] for row in backends.executor.log] == ["home"] and np.allclose(backends.executor.position, 0)


def test_home_after_drop_without_a_plan_stops_with_the_hand_open():
    backends = lifted_hand()

    def go_home():
        raise GraspFailure("HOME_PLAN_FAILED", "start state in collision")

    result = GraspSequence(backends).home_after_drop(go_home)
    assert not result.succeeded and not result.at_rest and result.error_code == "HOME_PLAN_FAILED"
    assert backends.effector.log == [EffectorAction.RELEASE] and backends.executor.stopped
    assert not backends.executor.log


def test_drop_recovery_keeps_the_failed_segment_and_report_before_finish():
    from test_grasp_pipeline import trajectory

    backends = lifted_hand()
    backends.executor.fail_on = "home"
    backends.executor.report = {"segments": [{"name": "home", "last_sent_at": 3.0}]}
    backends.executor.finish = backends.executor.report.clear
    result = GraspSequence(backends).home_after_drop(lambda: Move(trajectory("home", np.full(3, 0.4), np.zeros(3))))
    assert not result.succeeded and "tracking error 0.31 rad" in result.message
    assert result.phases[-1]["name"] == "home" and result.phases[-1]["state"] == "failed"
    assert result.execution_report["segments"][0]["last_sent_at"] == 3.0
    assert not backends.executor.report
    assert result.exception["type"] == "_Stopped"
    assert "_expect" in result.exception["traceback"]
