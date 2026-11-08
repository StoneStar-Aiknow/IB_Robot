"""Local state-evidence mocks; no ROS graph, placement executor, or Gateway calls."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/agent/ibrobot_agent"))

from ibrobot_agent.gift_workflow import GiftWorkflowSpeech, Notice  # noqa: E402


def steps(target="an apple"):
    return [
        {"skill_name": "pick_object", "target_name": target},
        {"skill_name": "place_in_container", "target_name": target, "container_name": "gift box"},
    ]


def event(
    sequence=1,
    *,
    skill="pick_object",
    index=0,
    root="ACTIVE",
    state="RUNNING",
    phase="approach",
    phase_state="RUNNING",
    kind="execution_progress",
    retry=0,
    recovery="NOT_NEEDED",
    request="r1",
    execution="e1",
    **evidence,
):
    return {
        "event_type": kind,
        "state": root,
        "sequence": sequence,
        "request_key": {"request_id": request},
        "detail": {
            "execution_id": execution,
            "current_skill": skill,
            "workflow_step_index": index,
            "gateway_sequence": sequence,
            "state": state,
            "execution_detail": {
                "schema_version": 2,
                "phase": phase,
                "phase_state": phase_state,
                "retry_count": retry,
                "recovery_state": recovery,
                **evidence,
            },
        },
    }


def place(sequence=2, **changes):
    return event(sequence, **{"skill": "place_in_container", "index": 1, "phase": "transfer", **changes})


def terminal(sequence=3, **changes):
    return place(sequence, root="SUCCEEDED", state="SUCCEEDED", kind="terminal", phase_state="SUCCEEDED", **changes)


@pytest.fixture
def policy():
    value = GiftWorkflowSpeech()
    assert value.register("r1", "s1", steps())
    return value


def test_complete_mock_trace_separates_step_success_from_root_terminal(policy):
    (start,) = policy.accept(event(), 0)
    assert start.text == "好，我来拿苹果，再放到指定容器。请稍等。"
    assert start.token.startswith("F05:") and not start.terminal
    assert policy.accept(event(2, state="SUCCEEDED", phase="verify", phase_state="SUCCEEDED"), 1) == []
    assert not policy.is_current(start, 1)
    (moving,) = policy.accept(place(3), 2)
    assert moving.token.startswith("F08:") and not moving.terminal
    # Individual place SUCCEEDED with root ACTIVE is still not task completion.
    assert policy.accept(place(4, state="SUCCEEDED", phase_state="SUCCEEDED", release_status="RELEASED"), 3) == []
    assert not policy.is_current(moving, 3)
    (complete,) = policy.accept(terminal(5, release_status="RELEASED"), 4)
    assert complete == Notice("放置步骤已结束，请工作人员确认现场状态。", "r1", "s1", "e1", True, True, complete.token)
    assert policy.is_current(complete, 4)
    assert policy.accept(terminal(6, release_status="RELEASED"), 5) == []
    assert policy.owns("r1")


@pytest.mark.parametrize(
    "target,word",
    [("a lemon", "柠檬"), ("an apple", "苹果"), ("an orange", "橙子"), ("a banana", "这个物体"), ("苹果", "这个物体")],
)
def test_only_explicit_target_mapping(target, word):
    policy = GiftWorkflowSpeech()
    assert policy.register("r1", "s1", steps(target))
    (notice,) = policy.accept(event(), 0)
    assert word in notice.text


@pytest.mark.parametrize(
    "workflow",
    [
        [],
        steps()[:1],
        steps()[::-1],
        [steps()[0], {"skill_name": "nod_yes"}, steps()[1]],
        [steps()[0], {**steps()[1], "target_name": "a lemon"}],
        [{"steps": steps()}],
        [None, steps()[1]],
        "pick_object",
    ],
)
def test_only_adjacent_same_target_flat_workflows_owned(workflow):
    policy = GiftWorkflowSpeech()
    assert not policy.register("r1", "s1", workflow)
    assert not policy.owns("r1")
    assert policy.accept(event(), 0) == []


def test_actual_flat_position_not_optional_display_index(policy):
    workflow = [{"skill_name": "nod_yes"}, *steps()]
    workflow[1]["index"] = 99
    assert policy.register("r2", "s1", workflow)
    assert policy.accept(event(request="r2", index=0), 0) == []
    (notice,) = policy.accept(event(2, request="r2", index=1), 1)
    assert notice.token.startswith("F05:")


@pytest.mark.parametrize("release", ["RELEASED"])
def test_structured_release_evidence_with_root_success(policy, release):
    (result,) = policy.accept(terminal(release_status=release), 0)
    assert result.text == "放置步骤已结束，请工作人员确认现场状态。"


@pytest.mark.parametrize("facts", [{}, {"message": "released"}, {"place_succeeded": True}, {"success": True}])
def test_generic_success_does_not_infer_release(policy, facts):
    (result,) = policy.accept(terminal(**facts), 0)
    assert result.text == "放置步骤已结束，请工作人员确认现场状态。"


def test_earlier_release_fact_is_not_reused_for_generic_terminal(policy):
    assert policy.accept(place(release_status="RELEASED", phase_state="SUCCEEDED"), 0) == []
    (result,) = policy.accept(terminal(), 1)
    assert result.text == "放置步骤已结束，请工作人员确认现场状态。"


@pytest.mark.parametrize(
    "facts",
    [
        {"release_status": 2},
        {"release_status": "RELEASE_UNKNOWN"},
        {"release_status": "future"},
        {"release_status": True},
        {"recovery_state": "UNKNOWN"},
        {"recovery_state": "FAILED"},
    ],
)
def test_uncertain_release_or_recovery_overrides_success(policy, facts):
    payload = terminal(release_status="RELEASED")
    payload["detail"]["execution_detail"].update(facts)
    (result,) = policy.accept(payload, 0)
    assert result.token.startswith("E06:") and result.important and result.terminal
    assert "领取" not in result.text


def test_place_failure_is_never_success(policy):
    payload = place(root="FAILED", state="FAILED", kind="terminal", release_status="RELEASED")
    (result,) = policy.accept(payload, 0)
    assert result.text == "物体已经松开，动作还未全部完成，请先别靠近。"


@pytest.mark.parametrize("release", ["NOT_RELEASED"])
def test_explicit_not_released_overrides_inconsistent_root_success(policy, release):
    (result,) = policy.accept(terminal(release_status=release), 0)
    assert result.text == "这次没能完成放置，请工作人员协助。"


def test_pick_failure_blocks_later_contradictory_place_success(policy):
    (failure,) = policy.accept(event(state="FAILED", phase_state="FAILED"), 0)
    assert "没能完成抓取" in failure.text and not failure.terminal
    (result,) = policy.accept(terminal(release_status="RELEASED"), 1)
    assert "领取" not in result.text and "没能完成放置" in result.text


def test_retry_only_on_count_increase_once_per_attempt(policy):
    assert not policy.accept(event(state="FAILED", phase_state="FAILED"), 0)[0].token.startswith("F12:")
    (retry,) = policy.accept(event(2, retry=1, phase="observe"), 1)
    assert retry.token.startswith("F12:")
    assert policy.accept(event(3, retry=1, phase="approach"), 2) == []
    (retry_again,) = policy.accept(event(4, retry=2, phase="observe"), 3)
    assert retry_again.token.startswith("F12:")
    assert policy.accept(event(5, retry=1, phase="observe"), 4) == []
    assert policy.accept(event(6, retry=2, state="SUCCEEDED"), 5) == []
    (result,) = policy.accept(
        terminal(7, release_status="RELEASED", verification_status="SUCCESS", place_succeeded=True), 6
    )
    assert "领取" in result.text


@pytest.mark.parametrize("phase_state", ["SUCCEEDED", "FAILED", "UNKNOWN", ""])
def test_transfer_requires_running_phase(policy, phase_state):
    assert policy.accept(place(phase_state=phase_state), 0) == []


def test_late_transfer_does_not_backfill_after_release(policy):
    assert policy.accept(place(phase="release"), 0) == []
    assert policy.accept(place(3), 1) == []


def test_start_and_transfer_each_once_and_queued_phase_speech_expires(policy):
    (first,) = policy.accept(event(), 0)
    assert policy.accept(event(2), 1) == []
    assert policy.is_current(first, 1)
    (moving,) = policy.accept(place(3), 2)
    assert not policy.is_current(first, 2)
    assert policy.accept(place(4), 3) == []
    assert policy.is_current(moving, 3)
    assert policy.accept(place(5, phase="release"), 4) == []
    assert not policy.is_current(moving, 4)


def test_gateway_order_duplicates_foreign_execution_and_request(policy):
    (first,) = policy.accept(event(3), 0)
    assert policy.accept(event(3), 1) == []
    assert policy.accept(event(2), 1) == []
    assert policy.accept(event(4, request="foreign"), 1) == []
    assert policy.accept(event(4, execution="foreign"), 1) == []
    assert policy.accept(place(4), -1) == []
    assert policy.is_current(first, 1)
    assert policy.accept(event(5, index=0, skill="place_in_container"), 2) == []


def test_root_terminal_can_promote_same_gateway_sequence_with_new_agent_sequence(policy):
    assert policy.accept(place(3, state="SUCCEEDED", phase_state="SUCCEEDED"), 0) == []
    payload = terminal(3, release_status="RELEASED", verification_status="SUCCESS", place_succeeded=True)
    assert policy.accept(payload, 1) == []
    payload["sequence"] = 4
    (result,) = policy.accept(payload, 1)
    assert result.terminal and "领取" in result.text


def test_terminal_event_without_known_root_outcome_is_conservative(policy):
    payload = place(kind="terminal", release_status="RELEASED")
    (result,) = policy.accept(payload, 0)
    assert result.terminal and "领取" not in result.text


def test_terminal_root_state_does_not_require_terminal_event_type(policy):
    (result,) = policy.accept(
        place(root="SUCCEEDED", release_status="RELEASED", verification_status="SUCCESS", place_succeeded=True), 0
    )
    assert result.terminal and "领取" in result.text


def test_current_skill_alone_never_proves_place_finished(policy):
    assert policy.accept(place(state="SUCCEEDED", phase_state="SUCCEEDED", release_status="RELEASED"), 0) == []
    assert policy.accept(place(3, state="SUCCEEDED", phase_state="SUCCEEDED", release_status="RELEASED"), 1) == []


def test_request_terminal_after_pick_never_claims_workflow_success(policy):
    (notice,) = policy.accept(event(root="SUCCEEDED", state="SUCCEEDED", kind="terminal"), 0)
    assert notice.terminal and "放置动作完成" not in notice.text and "领取" not in notice.text


def test_later_unrelated_step_invalidates_queued_progress(policy):
    (notice,) = policy.accept(place(), 0)
    assert policy.accept(event(3, skill="nod_yes", index=2), 1) == []
    assert not policy.is_current(notice, 1)
    assert policy.accept(place(4), 2) == []


def test_new_request_and_ended_session_invalidate_old_terminal(policy):
    (notice,) = policy.accept(terminal(release_status="RELEASED"), 0)
    assert policy.register("r2", "s1", steps())
    assert not policy.is_current(notice, 1)
    assert policy.accept(terminal(4, release_status="RELEASED"), 1) == []
    assert policy.owns("r1")
    (current,) = policy.accept(terminal(request="r2", release_status="RELEASED"), 2)
    policy.end_session("s1")
    assert not policy.is_current(current, 3)
    assert policy.accept(terminal(4, request="r2", release_status="RELEASED"), 3) == []
    assert not policy.register("r3", "s1", steps())


def test_non_gift_new_request_also_invalidates_previous_speech(policy):
    (notice,) = policy.accept(terminal(), 0)
    assert not policy.register("r2", "s1", [{"skill_name": "nod_yes"}])
    assert not policy.is_current(notice, 1)
    assert policy.owns("r1") and not policy.owns("r2")


def test_reregister_does_not_restart_same_request(policy):
    (start,) = policy.accept(event(), 0)
    assert policy.register("r1", "s1", steps())
    assert policy.is_current(start, 1)
    assert policy.accept(event(2), 1) == []


def test_notice_token_identity_age_and_no_timeout_actions(policy):
    (notice,) = policy.accept(event(), 0)
    assert not policy.is_current(replace(notice, text="forged"), 1)
    assert not policy.is_current(notice, -1)
    assert not policy.is_current(notice, 15)
    assert policy.accept(event(2), 1000) == []
    (result,) = policy.accept(terminal(release_status="RELEASED"), 1001)
    assert not policy.is_current(result, 1061)
    assert not policy.is_current(result, float("nan"))


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        [],
        {"state": []},
        {"request_key": None},
        {"request_key": {"request_id": "r1"}, "event_type": [], "state": []},
    ],
)
def test_malformed_payloads_ignored(policy, payload):
    assert policy.accept(payload, 0) == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("execution_id", ""),
        ("gateway_sequence", 0),
        ("gateway_sequence", True),
        ("workflow_step_index", True),
        ("workflow_step_index", -1),
    ],
)
def test_missing_or_malformed_correlation_is_not_evidence(policy, field, value):
    payload = terminal(release_status="RELEASED")
    payload["detail"][field] = value
    assert policy.accept(payload, 0) == []


def test_missing_execution_facts_preserve_only_generic_terminal(policy):
    payload = terminal()
    payload["detail"]["execution_detail"] = None
    (notice,) = policy.accept(payload, 0)
    assert notice.text == "放置步骤已结束，请工作人员确认现场状态。"


@pytest.mark.parametrize("phase", ["release", "retreat", "future_next_phase"])
def test_any_later_place_phase_prevents_old_transfer_backfill(policy, phase):
    assert policy.accept(place(phase=phase), 0) == []
    assert policy.accept(place(3), 1) == []


def test_finished_transfer_never_restarts_transfer_narration(policy):
    assert policy.accept(place(phase_state="SUCCEEDED"), 0) == []
    assert policy.accept(place(3), 1) == []


def test_public_unknown_release_reason_is_e06_without_new_ros_field(policy):
    (result,) = policy.accept(terminal(reason_code="PLACE_RELEASE_STATE_UNKNOWN"), 0)
    assert result.token.startswith("E06:")


def test_failed_place_phase_overrides_contradictory_success(policy):
    payload = terminal(release_status="RELEASED")
    payload["detail"]["execution_detail"]["phase_state"] = "FAILED"
    (result,) = policy.accept(payload, 0)
    assert result.text == "这次没能完成放置，请工作人员协助。"


def test_two_pairs_with_same_target_are_matched_by_step_index():
    policy = GiftWorkflowSpeech()
    assert policy.register("r1", "s1", steps() + steps())
    assert policy.accept(event(state="SUCCEEDED"), 0) == []
    assert policy.accept(event(2, index=2, state="SUCCEEDED"), 1) == []
    final = event(
        3,
        index=3,
        skill="place_in_container",
        root="SUCCEEDED",
        kind="terminal",
        release_status="RELEASED",
        verification_status="SUCCESS",
        place_succeeded=True,
    )
    (result,) = policy.accept(final, 2)
    assert result.terminal and "领取" in result.text


@pytest.mark.parametrize("verification", ["NOT_RUN", "UNKNOWN", "FAILED", "UNCERTAIN"])
def test_release_alone_never_claims_verified_placement(policy, verification):
    (notice,) = policy.accept(
        terminal(release_status="RELEASED", verification_status=verification, place_succeeded=True), 0
    )
    assert "领取" not in notice.text


def test_all_production_facts_are_required_for_verified_success(policy):
    (notice,) = policy.accept(
        terminal(release_status="RELEASED", verification_status="SUCCESS", place_succeeded=True), 0
    )
    assert "已确认" in notice.text and "工作人员确认后领取" in notice.text


def test_real_container_preparation_does_not_swallow_transfer_and_verification(policy):
    for seq, phase in enumerate(("observe_container", "perceive_container", "plan_place"), 1):
        assert (
            policy.accept(place(seq, phase=phase, release_status="NOT_RELEASED", verification_status="NOT_RUN"), seq)
            == []
        )
    (moving,) = policy.accept(place(4, release_status="NOT_RELEASED"), 4)
    assert moving.token.startswith("F08:")
    (checking,) = policy.accept(
        place(5, phase="verify_place", release_status="RELEASED", verification_status="NOT_RUN"), 5
    )
    assert "正在确认" in checking.text
    assert not policy.is_current(moving, 5)
    assert (
        policy.accept(place(6, phase="verify_place", release_status="RELEASED", verification_status="NOT_RUN"), 6) == []
    )


@pytest.mark.parametrize("verification", ["FAILED", "UNCERTAIN"])
def test_failure_preserves_release_but_does_not_authorize_collection(policy, verification):
    (notice,) = policy.accept(
        place(root="FAILED", kind="terminal", release_status="RELEASED", verification_status=verification), 0
    )
    assert "已经松开" in notice.text and "还没确认放置成功" in notice.text and "领取" not in notice.text


def test_place_preparation_retry_is_once_per_actual_attempt(policy):
    (retry,) = policy.accept(place(retry=1, phase="observe_container", release_status="NOT_RELEASED"), 0)
    assert retry.token.startswith("F12:")
    assert policy.accept(place(3, retry=1, phase="plan_place", release_status="NOT_RELEASED"), 1) == []
    assert policy.accept(place(4, retry=2, phase="observe_container", release_status="NOT_RELEASED"), 2)[
        0
    ].token.startswith("F12:")


def test_configured_label_is_used_only_for_matching_detector_query():
    policy = GiftWorkflowSpeech(container_name="an orange bowl", container_label="橙色的碗")
    workflow = steps()
    workflow[1]["container_name"] = "an orange bowl"
    assert policy.register("r1", "s1", workflow)
    assert "橙色的碗" in policy.accept(event(), 0)[0].text
