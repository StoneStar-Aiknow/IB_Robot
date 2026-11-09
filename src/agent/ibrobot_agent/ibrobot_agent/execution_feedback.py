"""Chinese presentation of authoritative facts; never an execution policy."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

from embodied_common.execution_detail import ExecutionDetail, short_message

_TERMINALS = {"SUCCEEDED", "FAILED", "CANCELED", "CANCELLED", "UNKNOWN", "CANCELLED_BEFORE_EXECUTION"}
_GRASP_REASONS = {
    "INVALID_TARGET": "目标名称不明确，请说明要抓取的物体。",
    "TARGET_NOT_VISIBLE": "没有看到目标物体，请把它放到机器人面前后重新下达指令。",
    "OUT_OF_REACH": "目标超出抓取范围，请放近后重新下达指令。",
    "NO_COMPLETE_PATH": "没有找到可执行的抓取路径，请调整物体位置后重新下达指令。",
    "NO_EXECUTABLE_CANDIDATE": "没有找到可执行的抓取路径，请调整物体位置后重新下达指令。",
    "PLAN_TIMEOUT": "抓取规划超时，请检查规划服务后再试。",
    "OBSERVATION_UNAVAILABLE": "暂时无法获取图像或机器人状态，请等待服务恢复。",
    "REOBSERVE_REQUIRED": "规划期间场景发生变化，请保持物体静止后重新下达指令。",
    "TARGET_MOVED": "目标在接近时移动了，请保持物体静止后重新下达指令。",
    "EMPTY_GRASP": "这次没有抓到物体，请重新摆放后再让我抓取。",
    "GRIP_LOST": "物体滑落了，请检查物体位置后重新下达指令。",
    "GRASP_VERIFICATION_FAILED": "这次未通过抓取验证，请检查物体位置后重新下达指令。",
    "EXECUTION_BUDGET_INSUFFICIENT": "剩余时间不足，本次没有开始新的抓取动作，请检查任务预算。",
}
_MIRROR_REASONS = {
    "MIRROR_SESSION_BUSY": "照镜子正在进行，本次动作未执行，也不会排队。请结束照镜子后重新下达指令。",
    "MIRROR_SESSION_STATE_UNKNOWN": "照镜子状态尚未确认，本次动作未执行，也不会排队。请操作员确认照镜子状态后重新下达指令。",
    "MIRROR_SESSION_RELEASE_UNCONFIRMED": "照镜子结束后的控制权释放尚未确认，后续动作未执行，也不会排队。请操作员确认机器人状态后重新下达指令。",
}
_ACTIONS = {
    "WAIT_USER": "本次任务未完成，请调整现场或补充信息后重新下达指令。",
    "WAIT_SERVICE": "服务暂时不可用，请等待服务恢复后重新下达指令。",
    "MANUAL_CHECK": "本次任务未完成，请操作员检查机器人状态。",
    "NONE": "本次任务未完成，请查看执行记录。",
}
_PLACE_REASONS = {
    "NOT_HOLDING": "当前没有已确认的持物，请先抓取目标。",
    "HELD_OBJECT_MISMATCH": "持物与放置请求不匹配，请操作员检查。",
    "CONTAINER_NOT_FOUND": "没有看到指定容器，请调整容器位置后重新下达指令。",
    "CONTAINER_NOT_OPEN": "未确认容器开口，请使用开口可见的容器。",
    "CONTAINER_TOO_SMALL": "容器开口无法容纳当前持物，请更换容器。",
    "CONTAINER_OUT_OF_REACH": "指定容器超出持物手臂的范围，请把容器放近。",
    "PLACE_PLAN_FAILED": "没有找到可执行的放置路径。",
    "GRIP_LOST_IN_TRANSFER": "搬运中物体滑落了，请检查物体位置。",
    "PLACED_OUTSIDE": "物体已释放，但观测结果未通过容器内放置检查。",
    "PLACE_UNSEEN": "物体已释放，但当前视角无法确认放置结果。",
}
_PLACE_PHASES = {
    "claim": "持物交接",
    "observe_container": "容器观测",
    "perceive_container": "容器识别",
    "plan_place": "放置规划",
    "transfer": "搬运",
    "descend": "下降",
    "release": "释放",
    "retreat": "退离",
    "home": "返回",
    "verify_place": "放置验证",
}
_GRASP_PHASES = {
    "dispatch": "调度",
    "observe": "观测",
    "perceive": "识别",
    "plan": "规划",
    "preshape": "准备手型",
    "approach": "接近目标",
    "grasp": "接近抓取点",
    "close": "闭合手爪",
    "lift": "抬升",
    "verify": "持握验证",
    "recover": "恢复",
}


def execution_reply(state: str, context: Mapping | None = None, fallback: str = "") -> str:
    context = context or {}
    detail = ExecutionDetail.from_value(context.get("execution_detail"))
    state = str(state).upper()
    placing = context.get("current_skill") == "place_in_container"
    phase = (
        _GRASP_PHASES.get(detail.failed_phase or detail.phase, "")
        if context.get("current_skill") == "pick_object" and detail.schema_version in {1, 2}
        else ""
    )
    failure_stage = f"抓取在{phase}阶段失败。" if phase else ""
    if placing and detail.schema_version in {1, 2}:
        phase = _PLACE_PHASES.get(detail.failed_phase or detail.phase, "")
        failure_stage = f"放置在{phase}阶段失败。" if phase else ""
    if state == "UNKNOWN":
        last_stage = f"最后记录为{phase}阶段。" if phase else ""
        return (
            f"执行结果未知。{last_stage}请操作员确认机器人状态。" if phase else "执行结果未知，请操作员确认机器人状态。"
        )
    if state in {"CANCELED", "CANCELLED", "CANCELLED_BEFORE_EXECUTION"}:
        return "已停止。" if state != "CANCELLED_BEFORE_EXECUTION" else "任务已取消，尚未开始执行。"
    if state == "FAILED" and context.get("error_code") == "HELD_OBJECT_RECOVERY_REQUIRED":
        return "上一次任务的持物状态尚未确认，需要先恢复后才能执行动作。"
    if detail.schema_version in {1, 2} and state in _TERMINALS:
        if detail.recovery_state == "UNKNOWN":
            return failure_stage + "恢复状态尚未确认，请操作员检查机器人。"
        if detail.recovery_state == "FAILED":
            return failure_stage + "自动收尾未完成，请操作员检查机器人。"
        if (
            state == "FAILED"
            and placing
            and detail.reason_code == "GRIP_LOST_IN_TRANSFER"
            and detail.recovery_state == "COMPLETED"
            and detail.release_status == "RELEASED"
        ):
            return "这次没能完成放置，已完成收尾，可以重新下达指令。"
    if state == "SUCCEEDED":
        if placing:
            if (
                detail.schema_version == 2
                and detail.release_status == "RELEASED"
                and detail.verification_status == "SUCCESS"
                and detail.place_succeeded
            ):
                return "物体已放入指定容器并通过验证，请工作人员确认后领取。"
            return "放置步骤已结束，请工作人员确认现场状态。"
        if (
            context.get("current_skill") == "pick_object"
            and detail.schema_version in {1, 2}
            and detail.phase
            and detail.phase_state == "SUCCEEDED"
        ):
            if detail.phase == "observe":
                return "观测完成。"
            if detail.phase in {"perceive", "plan"}:
                return "抓取规划完成，尚未执行抓取。"
            return "物体已经抓起并通过验证。"
        return "执行完成。"
    if placing and state == "FAILED" and detail.schema_version == 2 and detail.release_status == "RELEASED":
        if detail.verification_status in {"FAILED", "UNCERTAIN"}:
            return "物体已经松开，但还没确认放置成功，请工作人员检查。"
        return "物体已经松开，动作还未全部完成，请先别靠近。"
    if state == "FAILED":
        if (
            context.get("current_skill") in {"free_pose", "return_to_task_pose"}
            and detail.schema_version in {1, 2}
            and detail.failed_phase == "verify"
            and detail.reason_code == "TASK_POSE_NOT_REACHED"
        ):
            return "动作执行结束，未校验通过。"
        if detail.schema_version not in {1, 2}:
            return "执行失败，请查看执行记录或联系操作员。"
        if (
            detail.phase in {"admission", "release"}
            and detail.recovery_state == "NOT_NEEDED"
            and detail.reason_code in _MIRROR_REASONS
        ):
            return _MIRROR_REASONS[detail.reason_code]
        if detail.suggested_action == "MANUAL_CHECK":
            return failure_stage + _ACTIONS["MANUAL_CHECK"]
        if context.get("current_skill") == "pick_object":
            return _GRASP_REASONS.get(detail.reason_code, _ACTIONS[detail.suggested_action])
        if placing:
            return _PLACE_REASONS.get(detail.reason_code, _ACTIONS[detail.suggested_action])
        return _ACTIONS[detail.suggested_action]
    if state == "STOPPING":
        return "正在停止。"
    if detail.schema_version in {1, 2} and detail.phase_state == "RUNNING":
        if placing:
            return {
                "observe_container": "正在观察指定容器。",
                "plan_place": "正在规划放置。",
                "transfer": "正在搬运物体。",
                "verify_place": "正在验证放置结果。",
            }.get(detail.phase, "")
        if detail.recovery_state == "IN_PROGRESS":
            return "正在恢复，请等待。"
        if detail.retry_count and detail.phase == "observe":
            return "正在重新尝试。"
        if context.get("current_skill") == "pick_object":
            return {"observe": "正在观察目标。", "plan": "正在规划抓取。", "verify": "正在验证抓取结果。"}.get(
                detail.phase, ""
            )
    return short_message(fallback) if state not in {"ACTIVE", "EXECUTING", "RUNNING"} else ""


class FeedbackTracker:
    """Deduplicate one trusted event stream; discard all post-terminal stages."""

    def __init__(self, limit: int = 256) -> None:
        self._seen: OrderedDict[str, tuple[int, bool, str]] = OrderedDict()
        self._limit = limit

    def accept(self, payload: Mapping) -> tuple[str, str, bool] | None:
        context = payload.get("detail")
        if not isinstance(context, Mapping):
            context = {}
        key = str(context.get("execution_id") or payload.get("request_key", {}).get("request_id", ""))
        sequence = context.get("gateway_sequence") or payload.get("sequence", 0)
        if not key or type(sequence) is not int or sequence <= 0:
            return None
        previous, finished, spoken = self._seen.get(key, (0, False, ""))
        if finished or sequence <= previous:
            return None
        state = str(payload.get("state", ""))
        terminal = state in _TERMINALS
        text = execution_reply(state, context, str(payload.get("user_message", "")))
        self._seen[key] = (sequence, terminal, text or spoken)
        self._seen.move_to_end(key)
        while len(self._seen) > self._limit:
            self._seen.popitem(last=False)
        if not text or (text == spoken and not terminal):
            return None
        return key, text, terminal
