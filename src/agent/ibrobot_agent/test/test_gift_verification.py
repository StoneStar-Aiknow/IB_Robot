"""Production versioned facts govern gift speech, not release-phase names."""

import pytest

from embodied_common.execution_detail import ExecutionDetail
from ibrobot_agent.gift_workflow import GiftWorkflowSpeech


def policy():
    speech = GiftWorkflowSpeech(container_name="an orange bowl", container_label="橙色的碗")
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
