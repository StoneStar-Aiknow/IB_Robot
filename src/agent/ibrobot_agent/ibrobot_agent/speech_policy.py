"""ROS-free, deterministic grasp narration; callers check validity before TTS and playback."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from string import Formatter

from embodied_common.control_phrases import STOP_PHRASES
from embodied_common.execution_detail import ExecutionDetail

_TERMINALS = {"SUCCEEDED", "FAILED", "UNKNOWN", "CANCELED", "CANCELLED", "CANCELLED_BEFORE_EXECUTION"}
_MESSAGES = {
    "start": "好呀，我来拿{target}。",
    "success": "{target}拿起来啦，已经确认抓住了。",
    "completed": "这一步已经完成啦。",
    "observed": "观察完成啦，还没有执行抓取。",
    "planned": "抓取规划完成啦，还没有执行抓取。",
    "canceled": "已经停止啦，请确认一下现场状态。",
    "not_started": "任务取消啦，还没有开始执行。",
    "unknown": "这次执行结果还不能确认，请操作员检查机器人和物体吧。",
    "recovery_failed": "{stage}恢复还没有完成，或状态尚未确认，请操作员检查机器人吧。",
    "manual_check": "{stage}这次没有完成，请操作员检查机器人状态吧。",
    "failed": "这次没有完成抓取，请查看执行记录，再确认一下现场吧。",
    "observe": "我先看看物体的位置呀。",
    "perceive": "我在辨认物体，找找合适的抓取位置。",
    "plan": "我在想一条合适的抓取路线，再等我一下呀。",
    "preshape": "我来调整一下手型，准备抓取。",
    "approach": "我正慢慢靠近物体啦。",
    "close": "我来合拢手爪，试着握住物体。",
    "lift": "我正试着把物体抬起来呀。",
    "verify": "我再确认一下有没有抓住。",
    "waiting": "我还在处理这一步，再等我一下呀。",
    "recover": "我在做恢复处理，请给我一点时间呀。",
    "recovered": "恢复动作完成啦，我再确认一下接下来的步骤。",
    "retry": "这次需要重新尝试，我再试一次呀。",
    "retry_recovered": "退回来啦，我再试一次。",
    "retry_TARGET_NOT_VISIBLE": "刚才没看清物体，我再观察一次呀。",
    "retry_EMPTY_GRASP": "刚才没有抓到物体，我再试一次呀。",
    "retry_TARGET_MOVED": "刚才物体移动了，我再确认一下位置呀。",
    "reason_INVALID_TARGET": "物体名称还不明确，请再告诉我要抓哪个吧。",
    "reason_TARGET_NOT_VISIBLE": "我还没看到物体，请把它放到机器人面前，再让我试试吧。",
    "reason_OUT_OF_REACH": "物体有点远啦，请放近一些，再让我试试吧。",
    "reason_NO_COMPLETE_PATH": "这次没找到合适的抓取路线，请调整一下物体位置吧。",
    "reason_NO_EXECUTABLE_CANDIDATE": "这次没找到可执行的抓取路线，请调整一下物体位置吧。",
    "reason_PLAN_TIMEOUT": "这次规划用时太长啦，请检查规划服务后再试吧。",
    "reason_OBSERVATION_UNAVAILABLE": "暂时拿不到图像或机器人状态，请等服务恢复后再试吧。",
    "reason_REOBSERVE_REQUIRED": "规划时场景发生变化了，请保持物体静止，再让我试试吧。",
    "reason_TARGET_MOVED": "靠近时物体移动了，请保持物体静止，再让我试试吧。",
    "reason_EMPTY_GRASP": "这次没有抓到物体，请重新摆放后再让我试试吧。",
    "reason_GRIP_LOST": "物体滑落了，请先检查物体的位置吧。",
    "reason_GRASP_VERIFICATION_FAILED": "这次还没能确认抓稳，请检查一下物体位置吧。",
    "reason_EXECUTION_BUDGET_INSUFFICIENT": "剩余时间不够啦，这次没有开始新的抓取动作，请检查任务预算吧。",
}
_PHASES = {
    "dispatch": "调度",
    "observe": "观察",
    "perceive": "识别",
    "plan": "规划",
    "preshape": "准备手型",
    "approach": "接近物体",
    "close": "合拢手爪",
    "lift": "抬升",
    "verify": "持握验证",
    "recover": "恢复",
}
_IMPORTANT_KEYS = {"canceled", "not_started", "unknown", "recovery_failed", "manual_check", "failed"}
_IMPORTANT_KEYS.update(key for key in _MESSAGES if key.startswith("reason_"))


@dataclass(frozen=True)
class Notice:
    text: str
    execution_id: str
    token: str
    terminal: bool
    important: bool
    request_id: str = ""


@dataclass
class _Execution:
    phase: str = ""
    state: str = ""
    phase_state: str = ""
    retry_count: int = 0
    reason: str = ""
    recovery: str = ""
    entered: float = 0.0
    generation: int = 0
    emitted: float | None = None
    terminal: bool = False
    sequence: int = 0
    request_id: str = ""


class GraspSpeechPolicy:
    """Flat ``messages`` overrides use the keys in ``DEFAULT_MESSAGES``.

    ``now`` is monotonic seconds. Terminal notices ignore phase and throttling;
    ordinary notices follow the current phase generation and playback interval.
    Retention is bounded to 256 executions/targets and the latest 2048 notices.
    ``played`` means playback STARTED, not completed; it updates ``last_played``.
    """

    DEFAULT_MESSAGES = dict(_MESSAGES)

    def __init__(self, mode: str = "standard", messages: Mapping | None = None):
        if mode not in {"concise", "standard", "detailed"}:
            raise ValueError("unknown speech mode")
        if messages is not None and not isinstance(messages, Mapping):
            raise ValueError("messages must be a mapping")
        self.messages = dict(_MESSAGES)
        for key, text in (messages or {}).items():
            if key not in self.messages:
                raise ValueError(f"unknown message key: {key}")
            self.messages[key] = text
        for key, text in self.messages.items():
            self._validate(key, text)
        self.mode = mode
        self.interval = {"concise": 15.0, "standard": 4.0, "detailed": 3.0}[mode]
        self.last_played = float("-inf")
        self._executions: OrderedDict[str, _Execution] = OrderedDict()
        self._notices: OrderedDict[str, tuple[Notice, float, int]] = OrderedDict()
        self._played: set[str] = set()
        self._targets: OrderedDict[str, str] = OrderedDict()
        self._serial = 0

    def _remember(self, notice: Notice, now: float, generation: int) -> None:
        self._notices[notice.token] = (notice, now, generation)
        self._notices.move_to_end(notice.token)
        while len(self._notices) > 2048:
            token, _ = self._notices.popitem(last=False)
            self._played.discard(token)

    @staticmethod
    def _validate(key: str, text: str) -> None:
        if not isinstance(text, str) or not text.strip() or len(text) > 180:
            raise ValueError(f"invalid message: {key}")
        if not any("一" <= char <= "鿿" for char in text):
            raise ValueError(f"message must contain Chinese: {key}")
        try:
            fields = list(Formatter().parse(text))
        except ValueError as exc:
            raise ValueError(f"invalid template: {key}") from exc
        allowed = (
            {"target"}
            if key in {"start", "success"}
            else {"stage"}
            if key in {"recovery_failed", "manual_check"}
            else set()
        )
        if any(
            name is not None and (name not in allowed or spec or conversion) for _, name, spec, conversion in fields
        ):
            raise ValueError(f"invalid template field: {key}")
        if len(text.format(target="物" * 40, stage="抓取在接近物体阶段未完成。")) > 180:
            raise ValueError(f"rendered message can exceed 180 characters: {key}")
        if key not in _IMPORTANT_KEYS and any(word in text.lower() for word in STOP_PHRASES):
            raise ValueError(f"ordinary message contains a stop phrase: {key}")

    @staticmethod
    def _target(value: object) -> str:
        if not isinstance(value, str):
            return "这个东西"
        text = value.strip()
        if not text or len(text) > 40 or not any("一" <= char <= "鿿" for char in text):
            return "这个东西"
        if any(word in text.lower() for word in STOP_PHRASES) or any(c in text for c in "{}\n\r"):
            return "这个东西"
        return text

    def _notice(self, key, execution_id, now, *, terminal=False, important=False, request_id="", stage=""):
        self._serial += 1
        token = f"speech:{self._serial}"
        item = self._executions.get(execution_id)
        text = self.messages[key].format(target=self._targets.get(request_id, "这个东西"), stage=stage)
        if len(text) > 180:
            raise ValueError("rendered message exceeds 180 characters")
        notice = Notice(text, execution_id, token, terminal, important, request_id)
        self._remember(notice, now, item.generation if item else 0)
        return notice

    def start(self, request_id: str, target: object, now: float) -> Notice:
        self._targets[request_id] = self._target(target)
        self._targets.move_to_end(request_id)
        while len(self._targets) > 256:
            self._targets.popitem(last=False)
        notice = Notice(
            self.messages["start"].format(target=self._targets[request_id]),
            "",
            "start:" + request_id,
            False,
            True,
            request_id,
        )
        if len(notice.text) > 180:
            raise ValueError("rendered message exceeds 180 characters")
        self._remember(notice, now, 0)
        return notice

    def observe(self, payload: Mapping, now: float) -> list[Notice]:
        if not isinstance(payload, Mapping):
            return []
        context = payload.get("detail")
        if not isinstance(context, Mapping) or context.get("current_skill") != "pick_object":
            return []
        execution_id = context.get("execution_id")
        if not isinstance(execution_id, str) or not execution_id:
            return []
        item = self._executions.setdefault(execution_id, _Execution(entered=now))
        self._executions.move_to_end(execution_id)
        while len(self._executions) > 256:
            self._executions.popitem(last=False)
        if item.terminal:
            return []
        sequence = context.get("gateway_sequence", payload.get("sequence", 0))
        if type(sequence) is int and sequence > 0:
            if sequence <= item.sequence:
                return []
            item.sequence = sequence
        request = payload.get("request_key", {})
        if isinstance(request, Mapping) and isinstance(request.get("request_id"), str):
            item.request_id = request["request_id"]
        detail = context.get("execution_detail", {})
        if not isinstance(detail, Mapping):
            detail = {}
        phase = str(detail.get("phase", ""))
        phase = {"hold": "verify", "verify_hold": "verify", "grasp": "approach"}.get(phase, phase)
        retry = detail.get("retry_count", 0)
        retry = retry if type(retry) is int and retry >= 0 else item.retry_count
        changed = phase != item.phase or retry > item.retry_count
        if changed:
            item.phase, item.entered = phase, now
            item.generation += 1
            item.emitted = None
        increased = retry > item.retry_count
        old_recovery = item.recovery
        item.retry_count = max(retry, item.retry_count)
        item.reason = str(detail.get("reason_code") or item.reason)
        item.recovery = str(detail.get("recovery_state", ""))
        item.phase_state = str(detail.get("phase_state", "UNKNOWN")).upper()
        item.state = str(payload.get("state", "")).upper()
        if item.state in _TERMINALS:
            item.terminal = True
            key = self._terminal_key(item, ExecutionDetail.from_value(detail))
            stage = f"抓取在{_PHASES.get(str(detail.get('failed_phase') or phase), '当前')}阶段未完成。"
            return [
                self._notice(
                    key, execution_id, now, terminal=True, important=True, request_id=item.request_id, stage=stage
                )
            ]
        if item.state not in {"ACTIVE", "RUNNING", "EXECUTING"} or item.phase_state != "RUNNING":
            return []
        key = ""
        if item.recovery == "IN_PROGRESS" and old_recovery != "IN_PROGRESS":
            key = "recover"
        elif increased:
            key = (
                "retry_recovered"
                if "COMPLETED" in {item.recovery, old_recovery}
                else ("retry_" + item.reason if "retry_" + item.reason in self.messages else "retry")
            )
        elif item.recovery == "COMPLETED" and old_recovery == "IN_PROGRESS":
            key = "recovered"
        if key:
            item.emitted = now
            return [self._notice(key, execution_id, now, important=True, request_id=item.request_id)]
        return self._progress(execution_id, item, now)

    @staticmethod
    def _terminal_key(item: _Execution, detail: ExecutionDetail) -> str:
        if item.state == "UNKNOWN":
            return "unknown"
        if item.state == "CANCELLED_BEFORE_EXECUTION":
            return "not_started"
        if item.state in {"CANCELED", "CANCELLED"}:
            return "canceled"
        if item.state == "SUCCEEDED":
            if detail.schema_version == 1 and detail.phase_state == "SUCCEEDED" and detail.phase:
                if item.phase == "observe":
                    return "observed"
                if item.phase in {"perceive", "plan"}:
                    return "planned"
                if item.phase in {"lift", "verify", "close", "approach"}:
                    return "success"
            return "completed"
        if detail.recovery_state in {"FAILED", "UNKNOWN"}:
            return "recovery_failed"
        if detail.suggested_action == "MANUAL_CHECK":
            return "manual_check"
        key = "reason_" + detail.reason_code
        return key if key in _MESSAGES else "failed"

    def _progress(self, execution_id: str, item: _Execution, now: float) -> list[Notice]:
        if item.terminal or item.state not in {"ACTIVE", "EXECUTING", "RUNNING"} or item.phase_state != "RUNNING":
            return []
        if item.phase not in _PHASES or item.phase in {"dispatch", "recover"}:
            return []
        if now - self.last_played < self.interval:
            return []
        delay = 10.0 if self.mode == "concise" else 4.0 if self.mode == "standard" else 0.0
        if self.mode == "standard" and item.phase != "plan":
            if item.phase != "approach":
                return []
            delay = 0.0
        if now - item.entered < delay:
            return []
        if item.emitted is not None and now - item.emitted < 15.0:
            return []
        key = "waiting" if item.emitted is not None or self.mode == "concise" else item.phase
        item.emitted = now
        # A repeated long-wait notice supersedes a queued earlier notice in this phase.
        item.generation += 1
        return [self._notice(key, execution_id, now, request_id=item.request_id)]

    def tick(self, now: float) -> list[Notice]:
        return [notice for key, item in self._executions.items() for notice in self._progress(key, item, now)]

    def is_current(self, notice: Notice, now: float) -> bool:
        record = self._notices.get(notice.token)
        if record is None or record[0] != notice:
            return False
        if notice.terminal:
            return True
        if notice.token.startswith("start:"):
            return 0 <= now - record[1] < 10.0
        item = self._executions.get(notice.execution_id)
        if item is None or item.terminal or item.generation != record[2]:
            return False
        if item.phase_state != "RUNNING" or item.state not in {"ACTIVE", "EXECUTING", "RUNNING"}:
            return False
        return notice.important or notice.token in self._played or now - self.last_played >= self.interval

    def played(self, notice: Notice, now: float) -> None:
        if notice.token in self._played or not self.is_current(notice, now):
            return
        self._played.add(notice.token)
        self.last_played = now
