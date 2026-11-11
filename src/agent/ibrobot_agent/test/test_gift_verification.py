"""Production versioned facts govern gift speech, not release-phase names."""

import pytest

from embodied_common.execution_detail import ExecutionDetail
from ibrobot_agent.gift_workflow import GiftWorkflowSpeech


def policy(**options):
    speech = GiftWorkflowSpeech(container_name="an orange bowl", container_label="橙色的碗", **options)
    assert speech.register(
        "r",
        "s",
        [
            {"skill_name": "pick_object", "target_name": "an apple"},
            {"skill_name": "place_in_container", "target_name": "an apple", "container_name": "an orange bowl"},
        ],
    )
    return speech


def event(sequence=1, *, terminal=False, root=None, **facts):
    detail = ExecutionDetail(
        schema_version=2,
        phase=facts.pop("phase", "verify_place"),
        phase_state="SUCCEEDED" if terminal else "RUNNING",
        recovery_state=facts.pop("recovery_state", "NOT_NEEDED"),
        suggested_action="NONE",
        **facts,
    )
    return {
        "request_key": {"request_id": "r"},
        "sequence": sequence,
        "event_type": "terminal" if terminal else "execution_progress",
        "state": root or ("SUCCEEDED" if terminal else "ACTIVE"),
        "detail": {
            "current_skill": "place_in_container",
            "workflow_step_index": 1,
            "execution_id": "e",
            "gateway_sequence": sequence,
            "state": "SUCCEEDED" if terminal else "RUNNING",
            "execution_detail": detail.to_dict(),
        },
    }


def test_valid_typed_facts_confirm_completion_and_expire_progress():
    speech = policy()
    (progress,) = speech.accept(event(release_status="RELEASED", verification_status="NOT_RUN"), 1)
    assert "正在确认" in progress.text
    (done,) = speech.accept(
        event(2, terminal=True, release_status="RELEASED", verification_status="SUCCESS", place_succeeded=True), 2
    )
    assert "橙色的碗" in done.text and "确认后领取" in done.text
    assert not speech.is_current(progress, 2)
    assert speech.is_current(done, 2)


@pytest.mark.parametrize("verification", ["NOT_RUN", "UNKNOWN", "FAILED", "UNCERTAIN"])
def test_release_without_visual_success_cannot_announce_collection(verification):
    (done,) = policy().accept(event(terminal=True, release_status="RELEASED", verification_status=verification), 1)
    assert "领取" not in done.text


@pytest.mark.parametrize("verification", ["FAILED", "UNCERTAIN"])
def test_failed_visual_check_preserves_release_fact(verification):
    (done,) = policy().accept(
        event(terminal=True, root="FAILED", release_status="RELEASED", verification_status=verification), 1
    )
    assert "已经松开" in done.text and "还没确认" in done.text


def test_unknown_release_does_not_turn_into_verified_success():
    (done,) = policy().accept(
        event(terminal=True, release_status="UNKNOWN", verification_status="SUCCESS", place_succeeded=False), 1
    )
    assert done.token.startswith("E06:") and "领取" not in done.text


def test_recovered_drop_is_announced_as_failed_but_ready_for_a_new_request():
    (done,) = policy().accept(
        event(
            terminal=True,
            root="FAILED",
            release_status="RELEASED",
            reason_code="GRIP_LOST_IN_TRANSFER",
            recovery_state="COMPLETED",
        ),
        1,
    )
    assert "已完成收尾" in done.text and "重新下达指令" in done.text
    assert "领取" not in done.text and "尚未确认" not in done.text


def test_failed_cleanup_with_confirmed_release_is_not_called_an_unknown_execution():
    (done,) = policy().accept(
        event(terminal=True, root="FAILED", release_status="RELEASED", recovery_state="FAILED"), 1
    )
    assert "自动收尾未完成" in done.text and "尚未确认" not in done.text


def test_admission_gate_names_the_prior_custody_without_claiming_a_new_pick_failed():
    payload = event(terminal=True, root="FAILED")
    payload["detail"].update(error_code="HELD_OBJECT_RECOVERY_REQUIRED", current_skill="", execution_detail={})
    (done,) = policy().accept(payload, 1)
    assert "上一次任务" in done.text and "需要先恢复" in done.text
    assert "没能完成抓取" not in done.text and "执行状态尚未确认" not in done.text


def test_real_preparation_phases_do_not_advance_beyond_transfer():
    speech = policy()
    for sequence, phase in enumerate(("observe_container", "perceive_container", "plan_place"), 1):
        assert (
            speech.accept(
                event(sequence, phase=phase, release_status="NOT_RELEASED", verification_status="NOT_RUN"), sequence
            )
            == []
        )
    (moving,) = speech.accept(event(4, phase="transfer", release_status="NOT_RELEASED"), 4)
    assert "橙色的碗" in moving.text and moving.token.startswith("F08:")


def test_voice_bridge_parameters_preserve_deployment_container():
    from ibrobot_agent.voice_bridge import config_from_parameters, node_parameter_defaults

    values = node_parameter_defaults()
    values.update(robot_scope="aimdk_x2_summit", gift_container_name="an orange bowl", gift_container_label="橙色的碗")
    config = config_from_parameters(values)
    assert config.gift_container_name == "an orange bowl"
    assert config.gift_container_label == "橙色的碗"


def test_gateway_admission_placeholder_is_not_a_physical_warning():
    speech = policy()
    payload = event()
    payload["detail"].update(current_skill="pick_object", workflow_step_index=0)
    payload["detail"]["execution_detail"] = ExecutionDetail().to_dict()
    assert speech.accept(payload, 1) == []
    payload["sequence"] = payload["detail"]["gateway_sequence"] = 2
    payload["detail"]["execution_detail"] = ExecutionDetail(
        1, "observe", "RUNNING", "", "", "NOT_NEEDED", 0, "NONE"
    ).to_dict()
    assert speech.accept(payload, 2)[0].token.startswith("F05:")


def test_deployment_gift_line_uses_the_registered_fruit_and_container():
    speech = policy(messages={"F05": "给你拿{fruit}，送到{container}。"})
    payload = event(phase="observe")
    payload["detail"].update(current_skill="pick_object", workflow_step_index=0)
    (notice,) = speech.accept(payload, 1)
    assert notice.text == "给你拿苹果，送到橙色的碗。"


@pytest.mark.parametrize("messages", [[], {"unknown": "提示"}, {"F05": ""}, {"F05": "{missing}"}])
def test_invalid_gift_overrides_fail_at_startup(messages):
    with pytest.raises(ValueError, match="gift message"):
        policy(messages=messages)


def test_release_delivery_requires_opt_in_and_completed_typed_facts():
    payload = event(terminal=True, release_status="RELEASED", verification_status="NOT_RUN", place_succeeded=True)
    (ordinary,) = policy().accept(payload, 1)
    assert ordinary.token.startswith("place_ended:")
    (delivered,) = policy(release_is_delivery=True, messages={"placed": "礼物已送到{container}。"}).accept(payload, 1)
    assert delivered.text == "礼物已送到橙色的碗。"


@pytest.mark.parametrize(
    "facts",
    [
        {"release_status": "UNKNOWN"},
        {"place_succeeded": False},
        {"verification_status": "FAILED"},
        {"recovery_state": "FAILED"},
    ],
)
def test_release_delivery_never_promotes_incomplete_or_failed_facts(facts):
    detail = {"release_status": "RELEASED", "verification_status": "NOT_RUN", "place_succeeded": True, **facts}
    (notice,) = policy(release_is_delivery=True).accept(event(terminal=True, **detail), 1)
    assert not notice.token.startswith("placed:")


def test_inflight_release_is_silent_until_a_confirmed_result_or_explicit_fault():
    speech = policy()
    assert speech.accept(event(phase="release", release_status="UNKNOWN"), 1) == []
    (fault,) = speech.accept(
        event(2, phase="release", release_status="UNKNOWN", reason_code="PLACE_RELEASE_STATE_UNKNOWN"), 2
    )
    assert fault.token.startswith("E06:")


def test_normal_pick_phase_without_recovery_telemetry_does_not_announce_a_fault():
    speech = policy()
    payload = event(phase="observe")
    payload["detail"].update(current_skill="pick_object", workflow_step_index=0)
    speech.accept(payload, 1)
    payload = event(2, phase="preshape", recovery_state="UNKNOWN")
    payload["detail"].update(current_skill="pick_object", workflow_step_index=0)
    assert speech.accept(payload, 2) == []


def test_successful_return_placeholder_is_not_a_physical_recovery_fault():
    payload = event(terminal=True)
    payload["detail"].update(current_skill="return_to_task_pose", workflow_step_index=2)
    payload["detail"]["execution_detail"] = ExecutionDetail().to_dict()
    (done,) = policy().accept(payload, 1)
    assert done.token.startswith("completed:")
    assert done.text == "好啦，这次任务完成啦！"
    assert "领取" not in done.text


@pytest.mark.parametrize(
    ("state", "facts", "key"),
    [
        ("UNKNOWN", {}, "E06"),
        ("FAILED", {}, "pick_failed"),
        ("CANCELED", {}, "cancelled"),
        ("SUCCEEDED", {"schema_version": 2, "recovery_state": "UNKNOWN"}, "E06"),
        ("SUCCEEDED", {"schema_version": 2, "recovery_state": "FAILED"}, "recovery_failed"),
        ("ACTIVE", {}, "ended"),
    ],
)
def test_return_without_confirmed_workflow_success_keeps_caution(state, facts, key):
    payload = event(terminal=True, root=state)
    payload["detail"].update(current_skill="return_to_task_pose", workflow_step_index=2, execution_detail=facts)
    (notice,) = policy().accept(payload, 1)
    assert notice.token.startswith(f"{key}:")
    assert "完成啦" not in notice.text and "领取" not in notice.text


def test_completed_override_does_not_replace_unconfirmed_placement():
    speech = policy(messages={"completed": "好啦，这次任务完成啦！"})
    (notice,) = speech.accept(event(terminal=True, release_status="RELEASED", verification_status="UNCERTAIN"), 1)
    assert notice.token.startswith("place_ended:")
    assert "工作人员" in notice.text


def pick_event(sequence=1, phase="observe", retry=0, reason="", **fields):
    payload = event(sequence, phase=phase, retry_count=retry, reason_code=reason, **fields)
    payload["detail"].update(current_skill="pick_object", workflow_step_index=0)
    return payload


def test_start_survives_phases_but_retry_and_place_replace_it():
    speech = policy()
    (start,) = speech.accept(pick_event(), 0)
    speech.accept(pick_event(2, "plan"), 1)
    assert speech.is_current(start, 29.9)
    assert not speech.is_current(start, 30)
    speech.accept(pick_event(3, "close", reason="EMPTY_GRASP", recovery_state="IN_PROGRESS"), 2)
    (retry,) = speech.accept(pick_event(4, "observe", retry=1), 3)
    assert retry.text == "刚才没抓住，我再试一次。"
    assert not speech.is_current(start, 3)
    speech.accept(pick_event(5, "plan", retry=1), 4)
    assert speech.is_current(retry, 32.9)
    assert not speech.is_current(retry, 33)
    speech.accept(event(6, phase="observe_container"), 5)
    assert not speech.is_current(retry, 5)


def test_normal_failed_phase_before_recovery_does_not_alarm_but_unknown_recovery_does():
    speech = policy()
    payload = pick_event(phase="plan", recovery_state="UNKNOWN", reason="PLAN_TIMEOUT")
    payload["detail"]["execution_detail"]["phase_state"] = "FAILED"
    assert not speech.accept(payload, 0)
    payload = pick_event(2, "recover", recovery_state="UNKNOWN")
    assert all(not notice.token.startswith("E06:") for notice in speech.accept(payload, 1))
    payload = pick_event(3, "recover", recovery_state="UNKNOWN")
    payload["detail"]["execution_detail"]["phase_state"] = "FAILED"
    (alarm,) = speech.accept(payload, 2)
    assert alarm.token.startswith("E06:")


def test_gift_wait_uses_playback_end_and_global_throttle_and_releases_failed_pending():
    speech = policy()
    (start,) = speech.accept(pick_event(), 0)
    speech.note_playback(1, notice=start)
    assert not speech.tick(40)
    speech.note_playback(41, finished=True, notice=start)
    assert not speech.tick(52.9)
    (waiting,) = speech.tick(53)
    assert waiting.token.startswith("waiting_pick:")
    assert not speech.tick(54)
    speech.discard(waiting)  # synthesis failure, not playback
    speech.accept(pick_event(2, "plan"), 55)
    assert not speech.tick(72.9)
    assert speech.tick(73)


def test_expired_wait_does_not_permanently_block_next_wait():
    speech = policy()
    (start,) = speech.accept(pick_event(), 0)
    speech.discard(start)
    (waiting,) = speech.tick(12)
    assert not speech.tick(13)
    assert not speech.is_current(waiting, 27)
    assert speech.tick(32)


def test_duplicate_transfer_preserves_its_valid_notice_and_new_retry_replaces_old():
    speech = policy()
    (first,) = speech.accept(pick_event(retry=1), 0)
    (second,) = speech.accept(pick_event(2, retry=2), 1)
    assert not speech.is_current(first, 1)
    assert speech.is_current(second, 1)
    (transfer,) = speech.accept(event(3, phase="transfer"), 2)
    assert not speech.is_current(second, 2)
    assert speech.accept(event(4, phase="transfer"), 3) == []
    assert speech.is_current(transfer, 3)


def test_gift_stop_and_session_replacement_invalidate_all_progress():
    speech = policy()
    (start,) = speech.accept(pick_event(), 0)
    speech.end_session("s")
    assert not speech.is_current(start, 1)
    assert speech.tick(40) == []
