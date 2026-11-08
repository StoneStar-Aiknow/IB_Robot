from dataclasses import replace

import pytest

from embodied_common.execution_detail import ExecutionDetail
from ibrobot_agent.execution_feedback import FeedbackTracker, execution_reply
from ibrobot_agent.voice_bridge import Speak, VoiceBridgeConfig, VoiceBridgeCore


def event(sequence=1, state="EXECUTING", **changes):
    detail = ExecutionDetail(1, "observe", "RUNNING", "", "", "NOT_NEEDED", 0, "NONE")
    detail = replace(detail, **changes)
    return {
        "event_type": "terminal" if state in {"FAILED", "SUCCEEDED"} else "execution_progress",
        "state": state,
        "sequence": sequence,
        "request_key": {"request_id": "r1", "channel_id": "text", "principal_id": "operator", "robot_scope": "robot"},
        "detail": {
            "execution_id": "e1",
            "gateway_sequence": sequence,
            "current_skill": "pick_object",
            "workflow_step_index": 0,
            "execution_detail": detail.to_dict(),
        },
    }


@pytest.mark.parametrize(
    "reason,action,fragment",
    [
        ("TARGET_NOT_VISIBLE", "WAIT_USER", "放到机器人面前"),
        ("OUT_OF_REACH", "WAIT_USER", "放近"),
        ("PLAN_TIMEOUT", "WAIT_SERVICE", "检查规划服务"),
        ("EMPTY_GRASP", "WAIT_USER", "没有抓到"),
        ("GRIP_LOST", "WAIT_USER", "滑落"),
        ("FUTURE_CODE", "MANUAL_CHECK", "操作员"),
    ],
)
def test_reason_controls_feedback_without_executing_a_retry(reason, action, fragment):
    context = event(reason_code=reason, suggested_action=action)["detail"]
    assert fragment in execution_reply("FAILED", context)
    assert "操作员" in execution_reply(
        "FAILED", {**context, "execution_detail": {**context["execution_detail"], "recovery_state": "UNKNOWN"}}
    )


def test_tracker_discards_duplicates_stale_and_post_terminal_events():
    tracker = FeedbackTracker()
    assert tracker.accept(event(2))[1] == "正在观察目标。"
    assert tracker.accept(event(2)) is None
    assert tracker.accept(event(1)) is None
    assert tracker.accept(event(3)) is None  # identical phase is silent
    assert tracker.accept(
        event(4, "FAILED", phase_state="FAILED", reason_code="TARGET_NOT_VISIBLE", suggested_action="WAIT_USER")
    )[2]
    assert tracker.accept(event(5, phase="plan")) is None


def test_admission_recovery_gate_is_explained_even_without_executor_detail():
    message = execution_reply("FAILED", {"error_code": "HELD_OBJECT_RECOVERY_REQUIRED"})
    assert "上一次任务" in message and "需要先恢复" in message
    assert "查看执行记录" not in message


def test_completed_drop_cleanup_is_a_failed_task_that_can_accept_a_new_command():
    detail = ExecutionDetail(
        schema_version=2,
        phase="recover",
        phase_state="FAILED",
        failed_phase="observe_container",
        reason_code="GRIP_LOST_IN_TRANSFER",
        recovery_state="COMPLETED",
        release_status="RELEASED",
        verification_status="NOT_RUN",
        suggested_action="WAIT_USER",
    )
    text = execution_reply("FAILED", {"current_skill": "place_in_container", "execution_detail": detail.to_dict()})
    assert "没能完成放置" in text and "已完成收尾" in text and "重新下达指令" in text
    assert "未知" not in text and "放置成功" not in text


def test_text_feedback_bridge_checks_scope_and_never_submits_or_acknowledges():
    # ``detailed`` narrates the observe phase; ``standard`` deliberately stays quiet until the
    # plan, so this test asks for the mode that is supposed to speak here.
    core = VoiceBridgeCore(
        VoiceBridgeConfig(
            channel_id="text", principal_id="operator", robot_scope="robot", feedback_only=True, speech_mode="detailed"
        )
    )
    payload = event()
    assert core.on_heard("抓取苹果", 0) == []
    assert core.on_partial("停止", 0) == []
    assert core.on_event({**payload, "event_type": "presentation"}, 0) == []
    for name in ("channel_id", "principal_id", "robot_scope"):
        assert core.on_event({**payload, "request_key": {**payload["request_key"], name: "foreign"}}, 0) == []
    actions = core.on_event(payload, 0)
    assert len(actions) == 1 and isinstance(actions[0], Speak) and not actions[0].then
    assert core.on_event(payload, 0) == []


def test_legacy_and_non_grasp_success_do_not_claim_a_grasp():
    assert execution_reply("SUCCEEDED", {"current_skill": "pick_object"}) == "执行完成。"
    context = event()["detail"]
    context["current_skill"] = "navigate"
    assert execution_reply("SUCCEEDED", context) == "执行完成。"


@pytest.mark.parametrize(
    "detail",
    [
        {"schema_version": 1},
        ExecutionDetail(1, "verify", "RUNNING", "", "", "NOT_NEEDED", 0, "NONE").to_dict(),
    ],
)
def test_incomplete_display_facts_preserve_success_without_claiming_a_verified_grasp(detail):
    assert execution_reply("SUCCEEDED", {"current_skill": "pick_object", "execution_detail": detail}) == "执行完成。"


@pytest.mark.parametrize("skill", ["free_pose", "return_to_task_pose"])
def test_pose_arrival_failure_reports_motion_end_without_claiming_verified_success(skill):
    context = {
        "current_skill": skill,
        "execution_detail": ExecutionDetail(
            1, "verify", "FAILED", "verify", "TASK_POSE_NOT_REACHED", "NOT_NEEDED", 0, "MANUAL_CHECK"
        ).to_dict(),
    }
    assert execution_reply("FAILED", context) == "动作执行结束，未校验通过。"
    assert "结果未知" in execution_reply("UNKNOWN", context)
    assert execution_reply("CANCELLED", context) == "已停止。"
    context["current_skill"] = "pick_object"
    assert "操作员" in execution_reply("FAILED", context)


def test_explicit_unknown_recovery_overrides_success():
    detail = ExecutionDetail(schema_version=1).to_dict()
    assert "检查" in execution_reply("SUCCEEDED", {"current_skill": "pick_object", "execution_detail": detail})


def test_place_success_requires_complete_versioned_physical_evidence():
    detail = ExecutionDetail(
        schema_version=2,
        phase="verify_place",
        phase_state="SUCCEEDED",
        recovery_state="NOT_NEEDED",
        suggested_action="NONE",
        release_status="RELEASED",
        verification_status="SUCCESS",
        place_succeeded=True,
    )
    context = {"current_skill": "place_in_container", "execution_detail": detail.to_dict()}
    assert "通过验证" in execution_reply("SUCCEEDED", context)
    for overrides in ({"place_succeeded": False}, {"release_status": "UNKNOWN"}, {"verification_status": "NOT_RUN"}):
        context["execution_detail"] = replace(detail, **overrides).to_dict()
        assert "通过验证" not in execution_reply("SUCCEEDED", context)


def test_place_failure_reports_known_release_without_claiming_delivery():
    detail = ExecutionDetail(
        schema_version=2,
        phase="verify_place",
        phase_state="FAILED",
        recovery_state="NOT_NEEDED",
        release_status="RELEASED",
        verification_status="FAILED",
    )
    reply = execution_reply("FAILED", {"current_skill": "place_in_container", "execution_detail": detail.to_dict()})
    assert "已经松开" in reply and "还没确认放置成功" in reply
