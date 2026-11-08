"""Text -> real Agent -> Gateway -> grasp pipeline -> query/events -> TTS."""

import json
import time

import pytest
import rclpy
from std_msgs.msg import String

from embodied_common.execution_detail import ExecutionDetail
from ibrobot_agent.speech_policy import GraspSpeechPolicy

from .grasp_contract_rig import ContractRig, wait_until


@pytest.fixture(scope="module")
def ros_context():
    # Keep one DDS participant for this process's cases, as the live pipeline
    # does. Each rig still creates and destroys its own nodes and executor.
    rclpy.init()
    try:
        yield
    finally:
        rclpy.shutdown()


@pytest.fixture
def rig(request, tmp_path, ros_context):
    options = getattr(request, "param", {})
    instance = ContractRig(tmp_path, **options)
    try:
        yield instance
    finally:
        instance.close()


def terminal_speech(terminal):
    """Check TTS transport against the current policy; wording has Agent unit coverage."""
    notices = GraspSpeechPolicy().observe(terminal, now=0.0)
    assert len(notices) == 1 and notices[0].terminal
    return notices[0].text


def assert_ledger_matches(rig, request_id, terminal):
    query = rig.query(request_id)
    context = terminal["detail"]
    assert query["execution_id"] == context["execution_id"]
    assert query["execution_detail"] == context["execution_detail"]
    assert query["current_skill"] == context["current_skill"] == "pick_object"
    assert query["workflow_step_index"] == context["workflow_step_index"]
    response = rig.agent._bridge.get_execution_events(
        request_id=request_id, execution_id=query["execution_id"], after_sequence=0, timeout_sec=2.0
    )
    assert response["found"]
    events = response["events"]
    assert events
    assert query["last_event_sequence"] == context["gateway_sequence"] == events[-1].sequence
    assert [event.sequence for event in events] == sorted({event.sequence for event in events})
    assert ExecutionDetail.from_value(events[-1].execution_detail).to_dict() == query["execution_detail"]
    assert events[-1].current_skill == query["current_skill"]
    assert len(query["message"]) <= 300
    return query, events


def test_text_reaches_real_skill_and_success_is_spoken(rig):
    request_id, envelope = rig.submit()
    terminal = rig.terminal(request_id)
    assert terminal["state"] == "SUCCEEDED", terminal
    assert rig.requests and "帮我抓取苹果" in json.dumps(rig.requests, ensure_ascii=False)
    assert rig.mock.queries == ["an apple"] and rig.mock.closes == 1
    query, events = assert_ledger_matches(rig, request_id, terminal)
    assert query["execution_detail"]["phase_state"] == "SUCCEEDED"
    wait_until(lambda: terminal_speech(terminal) in rig.speech)
    wait_until(lambda: rig.played)
    # Same id is replayed, not dispatched twice; no new model call or grasp.
    rig.input.publish(String(data=json.dumps(envelope, ensure_ascii=False)))
    time.sleep(0.2)
    assert len(rig.requests) == 1 and rig.mock.closes == 1
    assert rig.speech.count(terminal_speech(terminal)) == 1


FAILURES = [
    ("INVALID_TARGET", "perceive", "WAIT_USER"),
    ("TARGET_NOT_VISIBLE", "perceive", "WAIT_USER"),
    ("OBSERVATION_UNAVAILABLE", "observe", "WAIT_SERVICE"),
    ("RPC_FAILED", "observe", "WAIT_SERVICE"),
    ("RPC_TIMEOUT", "observe", "WAIT_SERVICE"),
    ("NOT_READY", "observe", "WAIT_SERVICE"),
    ("SERVICE_UNAVAILABLE", "observe", "WAIT_SERVICE"),
    ("TASK_TIMEOUT", "observe", "WAIT_SERVICE"),
    ("NO_EXECUTABLE_CANDIDATE", "plan", "WAIT_USER"),
    ("NO_COMPLETE_PATH", "plan", "WAIT_USER"),
    ("OUT_OF_REACH", "plan", "WAIT_USER"),
    ("PLAN_TIMEOUT", "plan", "WAIT_SERVICE"),
    ("PLAN_REJECTED", "plan", "MANUAL_CHECK"),
    ("PLANNER_FAULT", "plan", "MANUAL_CHECK"),
    ("REOBSERVE_REQUIRED", "plan", "WAIT_USER"),
    ("EXECUTION_BUDGET_INSUFFICIENT", "plan", "WAIT_SERVICE"),
    ("MOTION_FAILED", "approach", "MANUAL_CHECK"),
    ("EFFECTOR_FAILED", "preshape", "MANUAL_CHECK"),
    ("WATCH_FAILED", "approach", "MANUAL_CHECK"),
    ("INTERNAL_ERROR", "observe", "MANUAL_CHECK"),
    ("EMPTY_GRASP", "close", "WAIT_USER"),
    # Grip checks run during lift, before the terminal verification.
    ("GRIP_LOST", "lift", "WAIT_USER"),
    ("GRASP_UNCERTAIN", "close", "MANUAL_CHECK"),
    ("TARGET_MOVED", "approach", "WAIT_USER"),
]


@pytest.mark.parametrize(
    "rig,reason,phase,action",
    [({"scenario": reason}, reason, phase, action) for reason, phase, action in FAILURES],
    indirect=["rig"],
)
def test_each_failure_reaches_text_query_and_tts_without_agent_retry(rig, reason, phase, action):
    request_id, _ = rig.submit()
    terminal = rig.terminal(request_id)
    assert terminal["state"] == "FAILED", terminal
    query, events = assert_ledger_matches(rig, request_id, terminal)
    detail = query["execution_detail"]
    assert detail["reason_code"] == reason
    assert detail["failed_phase"] == phase
    assert detail["suggested_action"] == action
    assert detail["retry_count"] <= 2
    if detail["recovery_state"] == "COMPLETED":
        assert detail["phase"] == "recover" and detail["phase_state"] == "SUCCEEDED"
    wait_until(lambda: terminal_speech(terminal) in rig.speech)
    assert len(rig.requests) == 1
    captures = rig.mock.captures
    time.sleep(0.1)
    assert rig.mock.captures == captures
    assert query["state"] == "FAILED"


@pytest.mark.parametrize("rig", [{"scenario": "retry_success"}], indirect=True)
def test_internal_retry_keeps_history_and_clears_final_failure(rig):
    request_id, _ = rig.submit()
    terminal = rig.terminal(request_id)
    assert terminal["state"] == "SUCCEEDED", terminal
    query, events = assert_ledger_matches(rig, request_id, terminal)
    assert rig.mock.closes == 2 and len(rig.requests) == 1
    assert query["execution_detail"]["retry_count"] == 1
    assert query["execution_detail"]["reason_code"] == ""
    assert query["execution_detail"]["failed_phase"] == ""
    assert any(event.execution_detail.reason_code == "EMPTY_GRASP" for event in events)
    assert any(event.execution_detail.recovery_state == "COMPLETED" for event in events)


@pytest.mark.parametrize("rig", [{"scenario": "recovery_failed"}], indirect=True)
def test_recovery_failure_keeps_original_cause_and_requires_operator(rig):
    request_id, _ = rig.submit()
    terminal = rig.terminal(request_id)
    query, _ = assert_ledger_matches(rig, request_id, terminal)
    detail = query["execution_detail"]
    assert detail["reason_code"] == "EMPTY_GRASP" and detail["failed_phase"] == "close"
    assert detail["recovery_state"] == "FAILED" and detail["suggested_action"] == "MANUAL_CHECK"
    assert rig.mock.closes == 1 and "操作员" in terminal["user_message"]


@pytest.mark.parametrize("rig", [{"scenario": "stop_unknown"}], indirect=True)
def test_unknown_stop_quarantines_new_requests(rig):
    request_id, _ = rig.submit()
    terminal = rig.terminal(request_id)
    assert terminal["state"] == "UNKNOWN", terminal
    query, _ = assert_ledger_matches(rig, request_id, terminal)
    assert query["execution_detail"]["recovery_state"] == "UNKNOWN"
    count = rig.mock.captures
    second, _ = rig.submit()
    wait_until(
        lambda: any(response.get("request_id") == second and not response.get("accepted") for response in rig.responses)
    )
    assert rig.mock.captures == count and "操作员" in terminal["user_message"]


@pytest.mark.parametrize("rig", [{"scenario": "cancel"}], indirect=True)
def test_text_stop_preserves_cancellation_details(rig):
    request_id, _ = rig.submit()
    assert rig.mock.started.wait(10.0)
    second, _ = rig.submit()
    wait_until(
        lambda: any(
            response.get("request_id") == second and response.get("reason_code") == "BUSY" for response in rig.responses
        )
    )
    rig.control.publish(String(data=json.dumps({"operation": "stop", "request_id": request_id})))
    terminal = rig.terminal(request_id)
    assert terminal["state"] == "CANCELLED", terminal
    query, _ = assert_ledger_matches(rig, request_id, terminal)
    assert query["execution_detail"]["failed_phase"] == "approach"
    assert rig.mock.stops and rig.mock.closes == 0 and rig.mock.captures == 1
    wait_until(lambda: terminal_speech(terminal) in rig.speech)


@pytest.mark.parametrize("rig", [{"scenario": "deadline"}], indirect=True)
def test_deadline_converges_to_a_terminal_with_stop_facts(rig):
    request_id, _ = rig.submit()
    terminal = rig.terminal(request_id)
    assert terminal["state"] == "FAILED", terminal
    query, _ = assert_ledger_matches(rig, request_id, terminal)
    assert query["error_code"] == "SKILL_TIMEOUT"
    assert query["execution_detail"]["failed_phase"] == "approach"
    assert rig.mock.stops and rig.mock.closes == 0


@pytest.mark.parametrize("rig", [{"scenario": "server_unavailable"}, {"scenario": "goal_rejected"}], indirect=True)
def test_dispatch_failure_is_reported_without_starting_the_pipeline(rig):
    if rig.pick is None:
        expected = "PICK_SERVER_UNAVAILABLE"
    else:
        rig.pick._ready.clear()  # backend warmup has not confirmed readiness
        expected = "PICK_GOAL_REJECTED"
    request_id, _ = rig.submit()
    terminal = rig.terminal(request_id)
    assert terminal["state"] == "FAILED", terminal
    query, _ = assert_ledger_matches(rig, request_id, terminal)
    assert query["error_code"] == expected
    assert query["execution_detail"]["failed_phase"] == "dispatch"
    assert query["execution_detail"]["suggested_action"] == "WAIT_SERVICE"
    assert rig.mock.captures == 0


@pytest.mark.parametrize("rig", [{"tts_block": True}], indirect=True)
def test_slow_tts_is_not_a_motion_gate_and_terminal_drops_queued_stages(rig):
    request_id, _ = rig.submit()
    terminal = rig.terminal(request_id)
    assert terminal["state"] == "SUCCEEDED" and not rig.speech_gate.is_set()
    wait_until(lambda: rig.speech)
    wait_until(lambda: any(entry[1] for entry in rig.voice._core._feedback._seen.values()))
    rig.speech_gate.set()
    wait_until(lambda: terminal_speech(terminal) in rig.speech)
    # Stale in-flight synthesis cannot revive a stage after the terminal.
    wait_until(lambda: rig.played)
    assert len(rig.played) == 1
    assert rig.mock.closes == 1


@pytest.mark.parametrize("rig", [{"tts_failure": True}], indirect=True)
def test_tts_failure_never_changes_execution_or_retries(rig):
    request_id, _ = rig.submit()
    terminal = rig.terminal(request_id)
    assert terminal["state"] == "SUCCEEDED", terminal
    wait_until(lambda: terminal_speech(terminal) in rig.speech)
    assert not rig.played
    assert rig.query(request_id)["state"] == "SUCCEEDED" and rig.mock.closes == 1


@pytest.mark.parametrize("rig", [{"steps": 2}], indirect=True)
def test_multi_step_result_names_the_last_child_and_preserves_prior_events(rig):
    request_id, _ = rig.submit()
    terminal = rig.terminal(request_id)
    assert terminal["state"] == "SUCCEEDED", terminal
    query, events = assert_ledger_matches(rig, request_id, terminal)
    assert query["workflow_step_index"] == 1 and rig.mock.closes == 2
    assert {event.workflow_step_index for event in events if event.current_skill} == {0, 1}
