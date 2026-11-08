"""ROS-free speech facts for adjacent, same-target pick -> place workflows.

Integration: register the validated presentation's flat ``steps`` (``skill_name`` /
``target_name``; list positions are zero-based Gateway step indices). ``owns``
selects this policy instead of single-grasp speech, including after completion.
Feed scoped Agent events to ``accept``; check ``is_current`` both before synthesis
and immediately before playback. End a wake session with ``end_session``. A new
registration, even a non-gift request, invalidates old queued speech.

``now`` is finite monotonic seconds, not wall clock. Notices expire after 15 s
(60 s for terminals); expiry only drops speech and never initiates robot actions.
The bridge must validate channel/principal/robot scope before calling this class.

Placement evidence comes from the versioned production ExecutionDetail contract.
A terminal must carry its own release, verification and full-completion facts;
phase names, free text and generic success never prove release or placement.
Missing facts stay conservative. This policy never issues recovery commands.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

_TERMINAL = {"SUCCEEDED", "FAILED", "UNKNOWN", "CANCELED", "CANCELLED", "CANCELLED_BEFORE_EXECUTION"}
_RUNNING = {"ACTIVE", "RUNNING", "EXECUTING"}
_TARGETS = {"a lemon": "柠檬", "an apple": "苹果", "an orange": "橙子"}


@dataclass(frozen=True)
class Notice:
    text: str
    request_id: str
    session_id: str
    execution_id: str
    terminal: bool
    important: bool
    token: str


@dataclass
class _Request:
    request_id: str
    session_id: str
    pairs: dict[int, str]
    containers: dict[int, str] = field(default_factory=dict)
    execution_id: str = ""
    sequence: int = 0
    event_sequence: int = 0
    index: int = -1
    scope: tuple = ()
    generation: int = 0
    last_now: float = float("-inf")
    started: bool = False
    finished: bool = False
    failed: set[int] = field(default_factory=set)
    retries: dict[int, int] = field(default_factory=dict)
    emitted: set[tuple] = field(default_factory=set)
    place_phase_rank: dict[int, int] = field(default_factory=dict)


def _mapping(value: object) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _clock(now: object) -> bool:
    return type(now) in {int, float} and math.isfinite(now)


def _positive(value: object) -> bool:
    return type(value) is int and value > 0


class GiftWorkflowSpeech:
    """One current dialogue request; no timers, I/O, dependencies, or robot actions.

    F05/F08/F12/E06 are encoded as the first component of opaque notice tokens.
    Other tokens are descriptive. Callers must not infer execution state from
    token/text. ``register`` returns whether the workflow is owned; repeated
    registration of the same current request is idempotent.
    """

    def __init__(self, *, container_name: str = "", container_label: str = "指定容器") -> None:
        self._container_name = container_name
        self._container_label = container_label
        self._current: _Request | None = None
        self._owned: OrderedDict[str, None] = OrderedDict()
        self._ended: set[str] = set()
        self._notices: OrderedDict[str, tuple[Notice, float, int]] = OrderedDict()
        self._serial = 0

    def register(self, request_id: str, session_id: str, steps: Sequence) -> bool:
        if not isinstance(request_id, str) or not request_id or not isinstance(session_id, str) or not session_id:
            raise ValueError("request_id and session_id must be nonempty strings")
        current = self._current
        if current and (current.request_id, current.session_id) == (request_id, session_id):
            return True
        self._current = None
        self._notices.clear()
        if request_id in self._owned or session_id in self._ended:
            return False  # Never resurrect a replaced request or ended session.
        if not isinstance(steps, Sequence) or isinstance(steps, str | bytes):
            return False
        pairs = {}
        containers = {}
        for index in range(len(steps) - 1):
            pick, place = _mapping(steps[index]), _mapping(steps[index + 1])
            target = pick.get("target_name")
            if (
                pick.get("skill_name") == "pick_object"
                and place.get("skill_name") == "place_in_container"
                and isinstance(target, str)
                and target.strip()
                and target == place.get("target_name")
                and isinstance(place.get("container_name"), str)
                and place["container_name"].strip()
            ):
                pairs[index] = _TARGETS.get(target, "这个物体")
                # A generic label avoids reading an English detector query aloud.
                containers[index] = (
                    self._container_label if place["container_name"] == self._container_name else "指定容器"
                )
        if not pairs:
            return False
        self._current = _Request(request_id, session_id, pairs, containers)
        self._owned[request_id] = None
        while len(self._owned) > 256:
            self._owned.popitem(last=False)
        return True

    def owns(self, request_id: str) -> bool:
        """Retain ownership of the latest 256 registrations to suppress legacy speech."""
        return request_id in self._owned

    def end_session(self, session_id: str) -> None:
        self._ended.add(session_id)
        if self._current and self._current.session_id == session_id:
            self._current = None
            self._notices.clear()

    def _notice(self, item: _Request, key: str, text: str, now: float, *, terminal=False) -> list[Notice]:
        marker = (key, item.index, item.retries.get(item.index, 0) if key == "F12" else 0)
        if marker in item.emitted:
            return []
        item.emitted.add(marker)
        self._serial += 1
        notice = Notice(
            text,
            item.request_id,
            item.session_id,
            item.execution_id,
            terminal,
            terminal or key in {"E06", "F12", "pick_failed", "place_failed"},
            f"{key}:{self._serial}",
        )
        self._notices[notice.token] = (notice, now, item.generation)
        while len(self._notices) > 256:
            self._notices.popitem(last=False)
        return [notice]

    def accept(self, payload: Mapping, now: float) -> list[Notice]:
        item = self._current
        if item is None or item.finished or not _clock(now) or now < item.last_now:
            return []
        payload = _mapping(payload)
        if _mapping(payload.get("request_key")).get("request_id") != item.request_id:
            return []
        event_type = payload.get("event_type")
        root_state = payload.get("state")
        if not isinstance(event_type, str) or not isinstance(root_state, str):
            return []
        terminal = event_type == "terminal" or root_state in _TERMINAL
        if event_type != "execution_progress" and not terminal:
            return []
        context = _mapping(payload.get("detail"))
        execution_id = context.get("execution_id")
        if not isinstance(execution_id, str) or not execution_id:
            return []
        if item.execution_id and execution_id != item.execution_id:
            return []
        sequence, event_sequence = context.get("gateway_sequence"), payload.get("sequence", 0)
        if not _positive(sequence) or sequence < item.sequence:
            return []
        # A terminal projection can repeat the final Gateway sequence. Accept
        # that promotion only with a newer Agent sequence, never ordinary repeats.
        if sequence == item.sequence and not (
            terminal and _positive(event_sequence) and event_sequence > item.event_sequence
        ):
            return []
        if _positive(event_sequence) and event_sequence <= item.event_sequence:
            return []
        index, skill = context.get("workflow_step_index"), context.get("current_skill")
        if type(index) is not int or index < 0 or index < item.index:
            return []
        pick_index = index if skill == "pick_object" else index - 1 if skill == "place_in_container" else -1
        in_pair = pick_index in item.pairs
        if not in_pair and not terminal:
            # Even an unrelated later step invalidates queued progress narration.
            if index >= 0:
                item.index = index
                item.generation += 1
                item.sequence = sequence
                item.event_sequence = event_sequence if _positive(event_sequence) else item.event_sequence
                item.last_now = now
            return []
        facts = _mapping(context.get("execution_detail"))
        phase, phase_state = facts.get("phase", ""), facts.get("phase_state", "")
        phase = phase if isinstance(phase, str) else ""
        phase_state = phase_state if isinstance(phase_state, str) else ""
        step_state = context.get("state", "")
        step_state = step_state if isinstance(step_state, str) else ""
        retry = facts.get("retry_count", 0)
        if type(retry) is not int or retry < item.retries.get(index, 0):
            return []
        scope = (index, str(phase), str(phase_state), str(step_state), retry)
        if scope != item.scope or terminal:
            item.generation += 1
            item.scope = scope
        item.execution_id = execution_id
        item.sequence = sequence
        item.event_sequence = event_sequence if _positive(event_sequence) else item.event_sequence
        item.index, item.last_now = index, now
        previous_retry = item.retries.get(index, 0)
        item.retries[index] = retry
        if terminal:
            item.finished = True
            return self._terminal(item, root_state, skill, in_pair, pick_index, facts, now)
        if root_state not in _RUNNING or not in_pair:
            return []
        # Schema 0 is the Gateway admission placeholder, not a physical warning.
        # Wait for executor facts rather than alarming on normal dispatch.
        if facts.get("schema_version") not in {1, 2}:
            return []
        if facts.get("recovery_state") in ("UNKNOWN", "FAILED") or (
            skill == "place_in_container" and self._release(facts) == "UNKNOWN"
        ):
            return self._notice(item, "E06", "执行状态尚未确认，请工作人员检查机器人和物体。", now)
        if skill == "pick_object":
            if retry > previous_retry:
                item.failed.discard(index)
                return self._notice(item, "F12", "刚才没能拿稳，我再尝试一次。", now)
            if step_state == "FAILED":
                item.failed.add(index)
                return self._notice(item, "pick_failed", "这次没能完成抓取，请工作人员协助。", now)
            if (
                not item.started
                and retry == 0
                and step_state in _RUNNING
                and phase_state == "RUNNING"
                and len(item.emitted) == 0
            ):
                item.started = True
                return self._notice(
                    item, "F05", f"好，我来拿{item.pairs[index]}，再放到{item.containers[index]}。请稍等。", now
                )
        elif skill == "place_in_container":
            if retry > previous_retry:
                item.failed.discard(index)
                item.place_phase_rank[index] = 0
                return self._notice(item, "F12", "刚才没能完成放置准备，我再试一次。", now)
            if step_state == "FAILED":
                item.failed.add(index)
                return self._notice(item, "place_failed", self._place_failure(facts), now)
            rank = {
                "observe": 0,
                "perceive": 0,
                "plan": 0,
                "approach": 0,
                "observe_container": 0,
                "perceive_container": 0,
                "plan_place": 0,
                "transfer": 1,
            }.get(phase, 2)
            if phase == "transfer" and phase_state != "RUNNING":
                rank = 2
            previous_rank = item.place_phase_rank.get(index, 0)
            item.place_phase_rank[index] = max(rank, previous_rank)
            if (
                phase == "transfer"
                and phase_state == "RUNNING"
                and step_state in _RUNNING
                and previous_rank <= 1
                and pick_index not in item.failed
                and index not in item.failed
            ):
                return self._notice(item, "F08", f"拿到了，我正在把它送到{item.containers[pick_index]}。", now)
            if (
                phase in {"observe_placed", "verify_place"}
                and phase_state == "RUNNING"
                and self._release(facts) == "RELEASED"
            ):
                return self._notice(item, "verifying", "正在确认物体是否放进容器。", now)
        return []

    @staticmethod
    def _release(facts: Mapping) -> str:
        value = facts.get("release_status")
        if facts.get("reason_code") == "PLACE_RELEASE_STATE_UNKNOWN":
            return "UNKNOWN"
        if value is None:
            return ""
        return value if isinstance(value, str) and value in {"NOT_RELEASED", "RELEASED", "UNKNOWN"} else "UNKNOWN"

    @staticmethod
    def _verified(facts: Mapping) -> bool:
        return (
            facts.get("schema_version") == 2
            and facts.get("release_status") == "RELEASED"
            and facts.get("verification_status") == "SUCCESS"
            and facts.get("place_succeeded") is True
        )

    @classmethod
    def _place_failure(cls, facts: Mapping) -> str:
        if cls._release(facts) == "RELEASED":
            if facts.get("verification_status") in {"FAILED", "UNCERTAIN"}:
                return "物体已经松开，但还没确认放置成功，请工作人员检查。"
            return "物体已经松开，动作还未全部完成，请先别靠近。"
        return "这次没能完成放置，请工作人员协助。"

    def _terminal(self, item, state, skill, in_pair, pick_index, facts, now):
        if (
            state == "UNKNOWN"
            or facts.get("recovery_state") in ("UNKNOWN", "FAILED")
            or (skill == "place_in_container" and self._release(facts) == "UNKNOWN")
        ):
            key, text = "E06", "执行状态尚未确认，请工作人员检查机器人和物体。"
        elif state in {"CANCELED", "CANCELLED", "CANCELLED_BEFORE_EXECUTION"}:
            key, text = "cancelled", "任务已停止，请工作人员确认现场状态。"
        elif state == "FAILED":
            key, text = (
                ("place_failed", self._place_failure(facts))
                if skill == "place_in_container"
                else ("pick_failed", "这次没能完成抓取，请工作人员协助。")
            )
        elif state == "SUCCEEDED" and in_pair and skill == "place_in_container":
            if (
                pick_index in item.failed
                or item.index in item.failed
                or self._release(facts) == "NOT_RELEASED"
                or facts.get("phase_state") == "FAILED"
            ):
                key, text = "place_failed", "这次没能完成放置，请工作人员协助。"
            elif self._verified(facts):
                key, text = "placed", f"已确认物体放进{item.containers[pick_index]}，请工作人员确认后领取。"
            else:
                key, text = "place_ended", "放置步骤已结束，请工作人员确认现场状态。"
        else:
            key, text = "ended", "任务已结束，请工作人员确认现场状态。"
        # Step-failure narration must not suppress the request's authoritative terminal.
        item.emitted.discard((key, item.index, 0))
        return self._notice(item, key, text, now, terminal=True)

    def is_current(self, notice: Notice, now: float) -> bool:
        record = self._notices.get(notice.token)
        item = self._current
        if record is None or record[0] != notice or item is None or not _clock(now):
            return False
        if (notice.request_id, notice.session_id) != (item.request_id, item.session_id):
            return False
        age = now - record[1]
        return (
            0 <= age < (60.0 if notice.terminal else 15.0)
            and record[2] == item.generation
            and (notice.terminal or not item.finished)
        )
