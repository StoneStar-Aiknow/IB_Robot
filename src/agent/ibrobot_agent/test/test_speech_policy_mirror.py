"""Mirror admission/release feedback uses facts without starting robot actions."""

import pytest

from embodied_common.execution_detail import ExecutionDetail
from ibrobot_agent.speech_policy import GraspSpeechPolicy

_REASONS = [
    "MIRROR_SESSION_BUSY",
    "MIRROR_SESSION_STATE_UNKNOWN",
    "MIRROR_SESSION_RELEASE_UNCONFIRMED",
]


def mirror_event(state="FAILED", **changes):
    facts = {
        "schema_version": 1,
        "phase": "admission",
        "phase_state": "FAILED",
        "reason_code": "MIRROR_SESSION_BUSY",
        "recovery_state": "NOT_NEEDED",
        "suggested_action": "MANUAL_CHECK",
        **changes,
    }
    return {
        "state": state,
        "sequence": 1,
        "request_key": {"request_id": "r1"},
        "detail": {
            "execution_id": "e1",
            "current_skill": "pick_object",
            "execution_detail": ExecutionDetail(**facts).to_dict(),
        },
    }


@pytest.mark.parametrize(
    "mode,reason,phase,version",
    [
        ("concise", _REASONS[0], "admission", 1),
        ("standard", _REASONS[1], "admission", 2),
        ("detailed", _REASONS[2], "release", 2),
    ],
)
def test_mirror_failure_announces_unexecuted_actions_in_every_mode(mode, reason, phase, version):
    policy = GraspSpeechPolicy(mode)
    policy.last_played = 0.0
    payload = mirror_event(schema_version=version, reason_code=reason, phase=phase)
    notices = policy.observe(payload, 0.1)
    assert len(notices) == 1
    notice = notices[0]
    assert notice.terminal and notice.important
    assert "照镜子" in notice.text
    action = "后续动作" if reason.endswith("RELEASE_UNCONFIRMED") else "本次动作"
    assert action + "未执行" in notice.text
    assert "不会排队" in notice.text
    assert "恢复" not in notice.text
    assert policy.is_current(notice, 0.1)
    assert policy.observe(payload, 0.2) == []
    assert policy.tick(30.0) == []


@pytest.mark.parametrize(
    "state,fragment",
    [
        ("UNKNOWN", "执行结果还不能确认"),
        ("CANCELED", "已经停止"),
        ("CANCELLED", "已经停止"),
        ("CANCELLED_BEFORE_EXECUTION", "任务取消"),
        ("SUCCEEDED", "这一步已经完成"),
    ],
)
def test_authoritative_terminal_state_precedes_mirror_reason(state, fragment):
    notice = GraspSpeechPolicy().observe(mirror_event(state), 0.0)[0]
    assert fragment in notice.text
    assert "照镜子" not in notice.text


@pytest.mark.parametrize("recovery", ["FAILED", "UNKNOWN"])
def test_real_recovery_failure_precedes_mirror_reason(recovery):
    notice = GraspSpeechPolicy().observe(mirror_event(recovery_state=recovery), 0.0)[0]
    assert "恢复还没有完成" in notice.text
    assert "照镜子" not in notice.text


@pytest.mark.parametrize(
    "changes",
    [
        {"reason_code": "SKILL_BUSY"},
        {"reason_code": "FUTURE_MIRROR_REASON"},
        {"schema_version": 0},
        {"phase": "verify"},
        {"recovery_state": "IN_PROGRESS"},
        {"recovery_state": "COMPLETED"},
    ],
)
def test_unrelated_busy_and_out_of_contract_details_keep_manual_check(changes):
    notice = GraspSpeechPolicy().observe(mirror_event(**changes), 0.0)[0]
    assert "操作员检查机器人状态" in notice.text
    if changes.get("schema_version") == 0:
        # Preserve the current branch's tolerant v0 handling: absent recovery
        # facts must not be presented as an actual failed recovery.
        assert "恢复还没有完成" not in notice.text
    assert "照镜子" not in notice.text
