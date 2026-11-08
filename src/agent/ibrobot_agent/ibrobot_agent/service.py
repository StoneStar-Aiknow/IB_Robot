"""ROS-free Agent orchestration for the IB-Robot incubation node."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from embodied_common.tracing import create_trace_logger, trace_scope, trace_stage
from embodied_common.workflow_contracts import normalize_workflow_steps
from ibrobot_agent.continuation import (
    CONTINUE_REFUSED,
    ContinuationDecision,
    decide_continuation,
    is_continue_request,
    needs_gateway,
    resume_execution_id,
    select_anchor,
)
from ibrobot_agent.contracts import (
    ASYNC_PRESENTATION_MODE,
    CATALOG_SKILL_NAMES_ZH,
    AgentEvent,
    AgentRequest,
    CatalogPort,
    ConversationStore,
    EventSink,
    ExecutionPort,
    ExecutionResult,
    Planner,
    PlannerIdentity,
    PlannerOutcome,
    PlanProposal,
    PoseMemoryPort,
    PosePreviewPort,
    PresentationPort,
    RegistryIdentity,
    RequestKey,
    RequestStore,
    StopReceipt,
    planner_outcome_to_mapping,
    request_hash_text,
)
from ibrobot_agent.execution_feedback import execution_reply
from ibrobot_agent.gift_workflow import fruit_label_zh
from ibrobot_agent.planner import (
    _HYPOTHETICAL_MARKERS,
    _NEGATION_MARKERS,
    RulePlanner,
    direct_command_outcome,
    fruit_delivery_from_config,
    planner_outcome_from_payload,
    read_only_outcome,
    validate_fruit_delivery,
)
from ibrobot_agent.planning_timing import planning_report, planning_stage
from ibrobot_agent.planning_timing import snapshot as planning_timing_snapshot

_LOGGER = logging.getLogger(__name__)
_trace = create_trace_logger("ib_trace.agent")

# Proposals that resume an interrupted step from the Gateway ledger were not
# planned by a model; the proposal snapshot says so.
LEDGER_REPLAY_IDENTITY = PlannerIdentity(
    route="continuation_ledger_replay",
    returned_model="none",
    protocol="gateway-canonical-steps-v1",
    prompt_version="none",
    schema_version="1",
    config_digest="none",
)


def _chinese_skill_name(catalog: Mapping[str, Any], skill_name: str) -> str | None:
    """The skill's Chinese catalog name, or None when the catalog does not give one."""
    names = catalog.get(CATALOG_SKILL_NAMES_ZH)
    name = names.get(skill_name) if isinstance(names, Mapping) else None
    return name.strip() if isinstance(name, str) and name.strip() else None


def _planner_catalog(catalog: Mapping[str, Any]) -> Mapping[str, Any]:
    """The catalog as the planner sees it: Chinese skill names word replies, never plans."""
    return {key: value for key, value in catalog.items() if key != CATALOG_SKILL_NAMES_ZH}


def _safe_error_message(exc: Exception) -> str:
    message = str(exc).strip() or type(exc).__name__
    lowered = message.casefold()
    if "bearer " in lowered or "api key" in lowered or "apikey" in lowered:
        return type(exc).__name__
    return message[:300]


@dataclass(frozen=True)
class AcceptedResponse:
    accepted: bool
    request_id: str
    state: str
    reason_code: str = ""
    message: str = ""


class InMemoryConversationStore:
    """Bounded session memory used by the incubator until persistent memory is added."""

    def __init__(self, *, max_turns: int = 12, clarification_ttl_sec: float = 300.0) -> None:
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns <= 0:
            raise ValueError("max_turns must be a positive integer")
        if clarification_ttl_sec <= 0.0:
            raise ValueError("clarification_ttl_sec must be positive")
        self._max_turns = max_turns
        self._clarification_ttl_sec = clarification_ttl_sec
        self._lock = threading.RLock()
        self._messages: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
        self._clarifications: dict[tuple[str, str, str, str, str], dict[str, object]] = {}

    def append(self, request: AgentRequest, *, role: str = "user") -> None:
        key = (request.session_id, request.principal_id, request.channel_id, request.robot_scope)
        with self._lock:
            messages = self._messages.setdefault(key, [])
            messages.append({"role": role, "content": request.text})
            del messages[: -self._max_turns]

    def append_assistant(self, request: AgentRequest, content: str) -> None:
        key = (request.session_id, request.principal_id, request.channel_id, request.robot_scope)
        with self._lock:
            messages = self._messages.setdefault(key, [])
            messages.append({"role": "assistant", "content": content})
            del messages[: -self._max_turns]

    def context(self, request: AgentRequest) -> list[dict[str, str]]:
        key = (request.session_id, request.principal_id, request.channel_id, request.robot_scope)
        with self._lock:
            return [dict(item) for item in self._messages.get(key, [])]

    def save_clarification(self, request: AgentRequest, missing_fields) -> None:
        key = (*self._session_key(request), request.request_id)
        with self._lock:
            self._clarifications[key] = {
                "request_id": request.request_id,
                "original_text": request.text,
                "missing_fields": list(missing_fields),
                "expires_at": time.monotonic() + self._clarification_ttl_sec,
                "consumed": False,
            }

    def consume_clarification(self, request: AgentRequest) -> dict[str, object] | None:
        if request.reply_to_request_id is None:
            return None
        key = (*self._session_key(request), request.reply_to_request_id)
        with self._lock:
            value = self._clarifications.get(key)
            if value is None or value["consumed"] is True or float(value["expires_at"]) <= time.monotonic():
                return None
            value["consumed"] = True
            return dict(value)

    def close(self) -> None:
        return

    @staticmethod
    def _session_key(request: AgentRequest) -> tuple[str, str, str, str]:
        return request.session_id, request.principal_id, request.channel_id, request.robot_scope


class AgentService:
    """Coordinate planning and optional execution without importing ROS."""

    def __init__(
        self,
        *,
        planner: Planner,
        catalog: CatalogPort,
        store: RequestStore,
        execution: ExecutionPort | None = None,
        presentation: PresentationPort | None = None,
        event_sink: EventSink | None = None,
        conversation: ConversationStore | None = None,
        execution_enabled: bool = False,
        allowed_skills: set[str] | None = None,
        max_planning_workers: int = 1,
        robot_scope: str = "so101_single_arm",
        direct_commands: Mapping[str, str] | None = None,
        fruit_delivery: Mapping[str, str] | None = None,
        pose_preview: PosePreviewPort | None = None,
        pose_skills: frozenset[str] = frozenset({"free_pose"}),
        pose_memory: PoseMemoryPort | None = None,
    ) -> None:
        if max_planning_workers != 1:
            raise ValueError("incubation node requires exactly one planning worker")
        self._planner = planner
        self._catalog = catalog
        self._store = store
        self._execution = execution
        self._presentation = presentation
        self._events = event_sink
        self._conversation = conversation or InMemoryConversationStore()
        self._execution_enabled = execution_enabled
        self._allowed_skills = frozenset(allowed_skills or ())
        self._robot_scope = robot_scope
        self._fruit_delivery = fruit_delivery_from_config({"fruit_delivery": fruit_delivery})
        # Utterance -> skill name. Empty by default: no phrase reaches the robot
        # without the planner unless a deployment opted it in.
        self._direct_commands = dict(direct_commands or {})
        # Arm poses described in words are composed before the plan is presented,
        # so the user confirms the pose that will run, and an unreachable or
        # unsafe one is answered without preparing anything.
        self._pose_preview = pose_preview
        self._pose_skills = frozenset(pose_skills)
        # Remembered poses: a confirmed one is recalled without planning, and the
        # user's save intent is interpreted by the existing planner.
        self._pose_memory = pose_memory
        self._pose_memory_identity = PlannerIdentity(
            route="pose_memory",
            returned_model="none",
            protocol="pose_memory_lookup",
            prompt_version="none",
            schema_version=planner.identity.schema_version,
            config_digest=hashlib.sha256(b"pose-memory-v1").hexdigest(),
        )
        # A shortcut is not the model's decision, so a proposal it produced
        # must not be recorded under the model's identity.
        self._direct_identity = PlannerIdentity(
            route="direct_command",
            returned_model="none",
            protocol="phrase_table",
            prompt_version="none",
            schema_version=planner.identity.schema_version,
            config_digest=hashlib.sha256(
                json.dumps(self._direct_commands, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        )
        if execution_enabled and not self._allowed_skills:
            raise ValueError("execution-enabled AgentService requires a non-empty incubation allowlist")
        if execution_enabled and execution is None:
            raise ValueError("execution-enabled AgentService requires an execution port")
        if execution_enabled and presentation is None:
            raise ValueError("execution-enabled AgentService requires a presentation port")
        if getattr(planner, "identity", None) is None:
            raise ValueError("planner must provide a PlannerIdentity")
        store.recover_interrupted_requests(robot_scope)
        self._quarantined = store.is_robot_quarantined(robot_scope)
        self._worker = threading.Thread(target=self._planning_loop, name="agent-planner", daemon=True)
        self._queue: list[AgentRequest] = []
        self._queued_or_active: set[RequestKey] = set()
        self._queue_condition = threading.Condition()
        self._stop = False
        self._active: dict[RequestKey, threading.Event] = {}
        self._active_lock = threading.RLock()
        # Sequence allocation and enqueue must be one step: stop_request emits from
        # the ROS control thread while the worker emits terminal, and subscribers
        # rely on publish order matching ledger sequence order.
        self._emit_lock = threading.Lock()
        self._worker.start()

    @property
    def execution_enabled(self) -> bool:
        return self._execution_enabled

    @property
    def healthy(self) -> bool:
        return not self._quarantined

    def close(self) -> None:
        with self._queue_condition:
            self._stop = True
            self._queue_condition.notify_all()
        with self._active_lock:
            for event in self._active.values():
                event.set()
        if self._execution is not None:
            with contextlib.suppress(Exception):
                self._execution.request_stop(None)
        self._worker.join(timeout=5.0)
        if self._worker.is_alive():
            raise RuntimeError("Agent planning worker did not stop")
        self._conversation.close()
        self._store.close()

    def get_request(self, key: RequestKey):
        return self._store.get_request(key)

    def send_message(self, request: AgentRequest) -> AcceptedResponse:
        if request.robot_scope != self._robot_scope:
            return AcceptedResponse(
                False, request.request_id, "FAILED", "UNSUPPORTED_ROBOT", "robot scope is not bound"
            )
        key = request.to_key()
        try:
            record = self._store.admit(key, input_hash=request_hash_text(request), session_id=request.session_id)
        except Exception as exc:
            return AcceptedResponse(
                False, request.request_id, "FAILED", getattr(exc, "code", "REQUEST_REJECTED"), str(exc)
            )
        if record.state in {"SUCCEEDED", "FAILED", "CANCELLED", "CANCELLED_BEFORE_EXECUTION", "ANSWERED", "UNKNOWN"}:
            terminal = record.terminal
            if terminal is not None and terminal.error_code == "ROBOT_QUARANTINED":
                return AcceptedResponse(False, request.request_id, "UNKNOWN", "ROBOT_QUARANTINED", terminal.message)
            if terminal is not None and terminal.error_code == "BUSY":
                return AcceptedResponse(False, request.request_id, "BUSY", "BUSY", terminal.message)
            return AcceptedResponse(True, request.request_id, record.state)
        if record.state != "RECEIVED":
            return AcceptedResponse(True, request.request_id, record.state)
        rejection: tuple[str, str, str] | None = None
        with self._queue_condition:
            if self._quarantined:
                rejection = ("ROBOT_QUARANTINED", "UNKNOWN", "robot scope is quarantined")
            elif key in self._queued_or_active:
                return AcceptedResponse(True, request.request_id, record.state)
            elif self._queued_or_active:
                rejection = ("BUSY", "BUSY", "robot scope has an active request")
            else:
                self._queued_or_active.add(key)
                self._queue.append(request)
                self._queue_condition.notify()
        if rejection is not None:
            reason_code, state, message = rejection
            try:
                self._store.finish(
                    key,
                    expected_generation=0,
                    result=ExecutionResult(
                        status="failed",
                        task_ref=None,
                        error_code=reason_code,
                        message=message,
                        detail={"admission": False},
                    ),
                )
            except Exception:
                self._quarantined = True
                _LOGGER.exception("Could not persist Agent admission rejection for %s", request.request_id)
                return AcceptedResponse(False, request.request_id, "UNKNOWN", "STORAGE_UNAVAILABLE", message)
            self._emit(key, reason_code.lower(), state, message)
            return AcceptedResponse(False, request.request_id, state, reason_code, message)
        self._conversation.append(request)
        return AcceptedResponse(True, request.request_id, "RECEIVED")

    def stop_request(self, key: RequestKey) -> StopReceipt:
        with self._active_lock:
            event = self._active.get(key)
            if event is not None:
                event.set()
        try:
            record = self._store.get_request(key)
            stopped_record = self._store.mark_stop(key, expected_generation=record.planning_generation)
            if self._execution is not None:
                with contextlib.suppress(Exception):
                    self._execution.request_stop(key)
            self._emit(key, "stop_requested", stopped_record.state, "stop requested")
            return StopReceipt(True, stopped_record, "", "stop requested")
        except Exception as exc:
            try:
                record = self._store.get_request(key)
            except Exception:
                raise
            return StopReceipt(False, record, getattr(exc, "code", "REQUEST_REJECTED"), str(exc))

    def _planning_loop(self) -> None:
        while True:
            with self._queue_condition:
                while not self._queue and not self._stop:
                    self._queue_condition.wait()
                if self._stop:
                    return
                request = self._queue.pop(0)
            try:
                self._plan_one(request)
            except Exception:
                _LOGGER.exception("Agent planner worker failed for request %s", request.request_id)
                self._recover_worker_failure(request)

    def _recover_worker_failure(self, request: AgentRequest) -> None:
        key = request.to_key()
        try:
            record = self._store.get_request(key)
            if record.state not in {
                "SUCCEEDED",
                "FAILED",
                "CANCELLED",
                "CANCELLED_BEFORE_EXECUTION",
                "ANSWERED",
                "UNKNOWN",
            }:
                self._finish_failure(key, record, "Agent planner worker failed")
        except Exception:
            self._quarantined = True
            _LOGGER.exception("Could not persist planner worker failure for request %s", request.request_id)

    def _plan_one(self, request: AgentRequest) -> None:
        with planning_report(), trace_scope(request.request_id):
            self._plan_one_traced(request)

    def _plan_one_traced(self, request: AgentRequest) -> None:
        key = request.to_key()
        try:
            record = self._store.begin_planning(key, expected_generation=0)
            cancel_token = threading.Event()
            with self._active_lock:
                self._active[key] = cancel_token
            gateway_error = None
            try:
                with planning_stage(_trace, "agent.catalog", request_id=request.request_id):
                    status = self._catalog.get_status()
                    catalog = self._catalog.get_catalog(status)
            except Exception as exc:
                status = {}
                catalog = {}
                gateway_error = exc
            if isinstance(self._planner, RulePlanner) and not status:
                status = {
                    "robot_name": self._robot_scope,
                    "active_control_mode": "unknown",
                    "control_plane_ready": False,
                    "motion_authorized": False,
                    "registry_epoch": "offline",
                    "registry_generation": 0,
                    "registry_digest": "offline",
                }
                catalog = {"skills": [], "pose_names": []}
            clarification = self._conversation.consume_clarification(request)
            if request.reply_to_request_id is not None and clarification is None:
                self._finish_answered(
                    key,
                    record.planning_generation,
                    "澄清上下文已过期或不属于当前会话，请重新说明完整请求。",
                    error_code="CLARIFICATION_EXPIRED",
                    event_type="needs_clarification",
                )
                return
            pending_save = (
                dict(self._pose_memory.feedback(_pose_session(request), "status")) if self._pose_memory else {}
            )
            # "继续" keeps its own request id and ledger row; only the text the planner sees
            # becomes the interrupted request's original text, plus how far it got.
            planned_request = request
            planner_context: dict[str, Any] = {
                "messages": self._conversation.context(request),
                "clarification": clarification,
                "pending_pose_save": pending_save if pending_save.get("pose_id") else None,
            }
            replayed: PlannerOutcome | None = None
            if (
                not pending_save.get("pose_id")
                and request.reply_to_request_id is None
                and is_continue_request(request.text)
            ):
                decision = self._resolve_continuation(request)
                if decision.text is None:
                    self._conversation.append_assistant(request, decision.message)
                    self._finish_answered(
                        key,
                        record.planning_generation,
                        decision.message,
                        error_code=decision.error_code,
                        event_type="continuation_refused" if decision.error_code else "answered",
                    )
                    return
                self._store.record_event(
                    key,
                    event_type="continuation_resolved",
                    state="PLANNING",
                    detail_json=json.dumps(dict(decision.hint), ensure_ascii=False, sort_keys=True),
                )
                self._emit(key, "continuation", "PLANNING", decision.message, detail={"continuation": decision.hint})
                planned_request = dataclasses.replace(request, text=decision.text)
                planner_context["continuation"] = dict(decision.hint)
                if decision.steps:
                    replayed = _ledger_replay_outcome(decision.steps)
                    self._store.record_event(
                        key,
                        event_type="continuation_replayed",
                        state="PLANNING",
                        detail_json=json.dumps({"step_count": len(replayed.steps)}, sort_keys=True),
                    )
            query = (
                None if replayed is not None or pending_save.get("pose_id") else read_only_outcome(planned_request.text)
            )
            direct = (
                None
                if replayed is not None or query is not None or pending_save.get("pose_id")
                else direct_command_outcome(planned_request.text, self._direct_commands, catalog)
            )
            remembered = (
                None
                if replayed is not None or query is not None or direct is not None or pending_save.get("pose_id")
                else self._traced_pose_lookup(request, planned_request.text, catalog, planner_context)
            )
            planned_by = self._planner.identity
            if remembered is not None:
                # A confirmed remembered pose: the model is not asked; the Gateway still
                # validates the workflow, and the composer still checks the pose.
                outcome = remembered
                planned_by = self._pose_memory_identity
            elif replayed is not None:
                outcome = replayed
                planned_by = LEDGER_REPLAY_IDENTITY
            elif query is not None:
                outcome = query
            elif direct is not None:
                # Skips the model call entirely; the Gateway still validates the
                # workflow against the same catalog as a planned one.
                _trace.info(
                    "direct command matched",
                    extra={"request_id": request.request_id, "skill": direct.summary},
                )
                outcome = direct
                planned_by = self._direct_identity
            else:
                try:
                    with planning_stage(_trace, "agent.planner", request_id=request.request_id):
                        outcome = self._planner.plan(
                            planned_request,
                            planner_context,
                            _planner_catalog(catalog),
                            cancel_token,
                            # Storage claims are rendered only after the actual commit.
                            on_preview=None
                            if pending_save.get("pose_id")
                            else lambda text: self._emit_reply_preview(key, text, cancel_token),
                        )
                except Exception:
                    if not pending_save.get("pose_id") or cancel_token.is_set():
                        raise
                    _LOGGER.exception("Could not interpret save confirmation; preserving the offer")
                    self._finish_answered(
                        key,
                        record.planning_generation,
                        "暂时没能确认你的意思，请明确同意或拒绝保存。",
                        event_type="pose_feedback",
                    )
                    return
            if cancel_token.is_set():
                self._finish_cancel_before_execution(key, record.planning_generation)
                return
            if self._apply_pose_feedback(key, record.planning_generation, request, outcome, pending_save):
                return
            if outcome.kind != "workflow":
                if outcome.kind == "read_only" and not status:
                    raise RuntimeError("Gateway is unavailable for read-only robot queries") from gateway_error
                user_message = self._resolve_non_workflow_message(outcome, status, catalog)
                if outcome.kind == "needs_clarification":
                    self._conversation.save_clarification(request, outcome.missing_fields)
                detail = planner_outcome_to_mapping(outcome)
                if outcome.repairs:
                    # The audit trail explains why the user got this answer where the strict
                    # contract would have produced a FAILED terminal.
                    detail["planner_repairs"] = [repair.to_dict() for repair in outcome.repairs]
                    _LOGGER.warning("planner reply for %s repaired: %s", request.request_id, detail["planner_repairs"])
                self._store.record_event(
                    key,
                    event_type="planner_outcome",
                    state="PLANNING",
                    detail_json=json.dumps(detail, ensure_ascii=False, sort_keys=True),
                )
                self._conversation.append_assistant(request, user_message)
                self._finish_answered(key, record.planning_generation, user_message, event_type=outcome.kind)
                return
            if gateway_error is not None:
                raise RuntimeError("Gateway is unavailable for workflow validation") from gateway_error
            if not status or not status.get("registry_epoch") or not status.get("registry_digest"):
                raise RuntimeError("Gateway registry is not ready")
            if status.get("control_plane_ready", True) is False:
                raise RuntimeError("Gateway control plane is not ready")
            outcome = self._bind_catalog_targets(outcome, catalog)
            self._validate_workflow(outcome, catalog)
            outcome = self._bind_step_schema_versions(outcome, catalog)
            outcome = self._bind_summary_to_catalog_names(outcome, catalog, pose_skills=self._pose_skills)
            if not self._preview_pose(key, record.planning_generation, request, outcome):
                return
            if cancel_token.is_set():
                self._finish_cancel_before_execution(key, record.planning_generation)
                return
            identity = RegistryIdentity(
                epoch=str(status["registry_epoch"]),
                generation=int(status["registry_generation"]),
                digest=str(status["registry_digest"]),
            )
            proposal = PlanProposal(
                request=planned_request,
                planning_generation=record.planning_generation,
                catalog_identity=identity,
                outcome=outcome,
                planner_identity=planned_by,
            )
            record = self._store.mark_proposal_ready(key, expected_generation=record.planning_generation)
            self._store.record_event(
                key,
                event_type="proposal_snapshot",
                state="PROPOSAL_READY",
                detail_json=json.dumps(proposal.to_dict(), ensure_ascii=False, sort_keys=True),
            )
            self._emit(key, "proposal_ready", "PROPOSAL_READY", outcome.summary)
            if not self._execution_enabled or self._execution is None:
                self._finish_plan_without_execution(key, record.planning_generation, outcome.user_message)
                return
            record = self._store.mark_preparing(key, expected_generation=record.planning_generation)
            observe_progress = getattr(self._execution, "set_progress_callback", None)
            if observe_progress is not None:
                observe_progress(lambda detail: self._execution_progress(key, detail))
            with trace_stage(_trace, "agent.execution", request_id=request.request_id):
                execution_result = self._execution.execute(
                    proposal,
                    expected_registry_identity=identity,
                    presentation_callback=lambda presentation: self._present(
                        key, record.planning_generation, proposal, presentation, cancel_token
                    ),
                    submission_callback=lambda task_ref, detail: self._mark_may_submit(
                        key, record.planning_generation, task_ref, detail
                    ),
                    stop_event=cancel_token,
                    continues_from_execution_id=resume_execution_id(planner_context.get("continuation"), outcome.steps),
                )
            if execution_result.status == "unknown":
                current = self._store.get_request(key)
                self._finish_unknown(
                    key,
                    current,
                    execution_result.message or execution_result.error_code,
                    detail=execution_result.detail,
                )
            else:
                self._store.finish(key, expected_generation=record.planning_generation, result=execution_result)
                terminal_detail = {**execution_result.detail, "error_code": execution_result.error_code}
                terminal_message = execution_reply(self._request_state_for_result(execution_result), terminal_detail)
                save_prompt = ""
                if execution_result.status == "succeeded" and self._pose_memory is not None:
                    try:
                        offer = dict(self._pose_memory.feedback(_pose_session(request), "offer"))
                        if offer.get("pose_id"):
                            save_prompt = str(offer.get("message", ""))
                    except Exception:
                        _LOGGER.exception("Could not offer pose save confirmation")
                try:
                    self._conversation.append_assistant(
                        request, " ".join(filter(None, (terminal_message, save_prompt)))
                    )
                finally:
                    # A conversation-memory failure must not hide an already
                    # persisted execution result from the waiting client.
                    self._emit(
                        key,
                        "terminal",
                        self._request_state_for_result(execution_result),
                        terminal_message,
                        detail={**terminal_detail, **({"save_prompt": save_prompt} if save_prompt else {})},
                    )
        except Exception as exc:
            try:
                record = self._store.get_request(key)
                # Submission callbacks may fail after a durable write. The ledger,
                # not the adapter's exception type, decides whether quarantine is required.
                if record.may_have_submitted:
                    self._finish_unknown(key, record, str(exc))
                elif record.stop_requested:
                    self._finish_cancel_before_execution(key, record.planning_generation)
                else:
                    self._finish_failure(key, record, _safe_error_message(exc))
            except Exception:
                self._quarantined = True
                _LOGGER.exception("Could not persist Agent request failure for %s", request.request_id)
        finally:
            with self._active_lock:
                self._active.pop(key, None)
            with self._queue_condition:
                self._queued_or_active.discard(key)

    def _preview_pose(self, key: RequestKey, generation: int, request: AgentRequest, outcome: PlannerOutcome) -> bool:
        """Compose the plan's leading pose step now; False when the request was answered instead.

        Only a pose that is the first step can be composed from the arm pose
        measured now; a later one depends on the steps before it and is composed
        when it runs (the executor says so in its result).
        """
        if self._pose_preview is None or not outcome.steps or outcome.steps[0].skill_name not in self._pose_skills:
            return True
        text = str(outcome.steps[0].target_name or "").strip()
        with planning_stage(_trace, "agent.pose_preview", request_id=request.request_id):
            composed = dict(self._pose_preview.compose(text, session_id=_pose_session(request), utterance=request.text))
        detail = {
            "text": text,
            "kind": str(composed.get("kind", "")),
            "composition_id": str(composed.get("composition_id", "")),
            "description": _json_or_text(composed.get("description_json", "")),
            "duration_sec": round(float(composed.get("duration_sec", 0.0) or 0.0), 2),
            "approximate": bool(composed.get("approximate", False)),
            "error_code": str(composed.get("error_code", "")),
            "from_memory": bool(composed.get("from_memory", False)),
            "pose_id": str(composed.get("pose_id", "")),
        }
        spoken = str(composed.get("user_message", "")).strip()
        if not composed.get("success"):
            message = f"这个姿势现在做不了：{composed.get('message') or composed.get('error_code')}"
            self._conversation.append_assistant(request, message)
            self._finish_answered(
                key, generation, message, error_code=str(composed.get("error_code") or "POSE_FAILED"),
                event_type="pose_refused",
            )  # fmt: skip
            return False
        if detail["kind"] in ("rejected", "no_motion"):
            message = spoken or "这个姿势我不做。"
            self._conversation.append_assistant(request, message)
            self._finish_answered(key, generation, message, event_type="pose_refused")
            return False
        self._emit(key, "pose_preview", "PLANNING", spoken or text, detail=detail)
        return True

    def _apply_pose_feedback(self, key, generation, request, outcome, pending) -> bool:
        """Apply the planner's decision to the exact pending pose, never its prose."""
        decision = outcome.pose_feedback
        if not pending.get("pose_id") and decision is None:
            return False
        self._store.record_event(
            key,
            event_type="pose_save_decision",
            state="PLANNING",
            detail_json=json.dumps(
                {"decision": decision, "pending_pose_id": pending.get("pose_id")}, ensure_ascii=False
            ),
        )
        session = _pose_session(request)
        message = ""
        allow_followup = False
        if self._pose_memory is None:
            message = "当前没有可用的动作保存服务。"
        elif decision == "offer":
            if pending.get("pose_id"):
                message = str(pending.get("message") or "刚才动作到位、同意保存吗？")
            else:
                offer = dict(self._pose_memory.feedback(session, "offer"))
                message = (
                    str(offer.get("message")) if offer.get("pose_id") else "需要先成功完成两次相近动作，才能确认保存。"
                )
        elif not pending.get("pose_id"):
            message = "当前没有待确认的保存动作。"
        elif decision is None or decision == "clarify":
            message = "刚才动作到位、同意保存吗？请明确确认或拒绝保存。"
        else:
            # One planning worker serializes this session. Recheck after the model
            # call; the Composer additionally checks expiry and achieved targets.
            current = dict(self._pose_memory.feedback(session, "status"))
            if current.get("pose_id") != pending["pose_id"]:
                message = "保存确认已过期或动作已改变，请重新发起保存。"
            elif decision == "new_task":
                self._pose_memory.feedback(session, "next_task")
                return False
            elif decision == "decline":
                self._pose_memory.feedback(session, "next_task")
                message = "好的，这次不保存。"
                allow_followup = True
            elif decision == "accept":
                result = dict(self._pose_memory.feedback(session, "good"))
                allow_followup = (
                    result.get("success")
                    and result.get("pose_id") == pending["pose_id"]
                    and result.get("message") == "confirmed"
                )
                message = (
                    "好的，这个姿势我记住了。"
                    if allow_followup
                    else "保存未完成，确认可能已过期或动作已改变，请重新确认。"
                )
        # Do not let model prose claim successful storage or execute a new task
        # when confirmation is absent, ambiguous or stale.
        if outcome.kind == "workflow" and allow_followup:
            # The write result can be spoken immediately, without gating the new motion.
            self._emit(key, "reply_preview", "PLANNING", message)
            self._conversation.append_assistant(request, message)
            return False
        self._conversation.append_assistant(request, message)
        self._finish_answered(key, generation, message, event_type="pose_feedback")
        return True

    def _traced_pose_lookup(self, request: AgentRequest, text: str, catalog: Mapping[str, Any], context: dict):
        if self._pose_memory is None:
            return None
        with planning_stage(_trace, "agent.pose_lookup", request_id=request.request_id):
            return self._remembered_pose_outcome(text, catalog, context)

    def _remembered_pose_outcome(
        self, text: str, catalog: Mapping[str, Any], context: dict | None = None
    ) -> PlannerOutcome | None:
        """A one-step free_pose workflow for words a confirmed remembered pose answers, else None."""
        if self._pose_memory is None or not self._pose_skills:
            return None
        folded = _fold(text)
        if not folded or any(marker in folded for marker in (*_NEGATION_MARKERS, *_HYPOTHETICAL_MARKERS)):
            return None
        skills = catalog.get("skills")
        entry = next(
            (
                skill
                for skill in (skills if isinstance(skills, list | tuple) else ())
                if isinstance(skill, Mapping)
                and skill.get("name") in self._pose_skills
                and skill.get("planner_visible", True)
                and (not self._allowed_skills or skill.get("name") in self._allowed_skills)
            ),
            None,
        )
        if entry is None:
            return None
        try:
            hit = dict(self._pose_memory.lookup(text))
        except Exception:  # noqa: BLE001 - fall back to planning
            _LOGGER.exception("pose memory lookup failed")
            return None
        if not hit.get("hit"):
            if context is not None:
                context["remembered_poses"] = hit.get("poses", [])
            return None
        schema_version = entry.get("schema_version", 1)
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            schema_version = 1
        return planner_outcome_from_payload(
            {
                "kind": "workflow",
                "user_message": str(hit.get("user_message") or f"好的，{text.strip()}")[:150],
                "summary": f"{entry['name']}: {text.strip()} (remembered)"[:300],
                "steps": [{"schema_version": schema_version, "skill_name": entry["name"], "target_name": text.strip()}],
            }
        )

    def _resolve_continuation(self, request: AgentRequest) -> ContinuationDecision:
        """Read both ledgers for "继续"; any read failure refuses instead of guessing."""
        key = request.to_key()
        try:
            anchor = select_anchor(self._store.recent_session_requests(key, session_id=request.session_id))
            previous_hint = (
                None if anchor is None else self._store.latest_event_detail(anchor.key, "continuation_resolved")
            )
        except Exception:
            _LOGGER.exception("Could not read the Agent ledger for continuation %s", request.request_id)
            return ContinuationDecision("无法读取任务账本，不能继续，请重新下达指令。", error_code=CONTINUE_REFUSED)
        gateway = None
        gateway_error = ""
        if needs_gateway(anchor):
            lookup = getattr(self._execution, "lookup_execution", None)
            if lookup is None:
                gateway_error = "Gateway execution ledger is not available"
            else:
                try:
                    gateway = lookup(anchor.key.request_id)
                except Exception as exc:
                    gateway_error = _safe_error_message(exc)
            if gateway_error:
                _LOGGER.warning("Continuation %s cannot read the Gateway ledger: %s", request.request_id, gateway_error)
        return decide_continuation(anchor, gateway=gateway, gateway_error=gateway_error, previous_hint=previous_hint)

    def _emit_reply_preview(self, key: RequestKey, text: str, cancel_token: threading.Event) -> None:
        """Publish the reply the planner is still producing, ahead of its outcome.

        Only an event: the Agent does not speak or render anything itself. The
        channel that owns the request decides whether to present it, and the
        ledger keeps the record that it was offered before validation.
        """
        if cancel_token.is_set():
            return
        self._emit(key, "reply_preview", "PLANNING", text)

    def _finish_cancel_before_execution(self, key: RequestKey, generation: int) -> None:
        self._store.finish(
            key,
            expected_generation=generation,
            result=ExecutionResult(
                status="cancelled",
                task_ref=None,
                error_code="CANCELLED_BEFORE_EXECUTION",
                message="stopped before execution",
                detail={"submitted": False},
            ),
        )
        self._emit(key, "cancelled", "CANCELLED_BEFORE_EXECUTION", "stopped before execution")

    def _finish_answered(
        self,
        key: RequestKey,
        generation: int,
        message: str,
        *,
        error_code: str = "",
        event_type: str = "answered",
    ) -> None:
        self._store.mark_answered(
            key,
            expected_generation=generation,
            message=message,
            error_code=error_code,
        )
        self._emit(key, event_type, "ANSWERED", message)

    @staticmethod
    def _resolve_non_workflow_message(
        outcome: PlannerOutcome,
        status: Mapping[str, Any],
        catalog: Mapping[str, Any],
    ) -> str:
        if outcome.kind != "read_only":
            return outcome.user_message
        if outcome.query_kind == "status":
            return (
                f"机器人 {status.get('robot_name', 'unknown')}，控制平面"
                f"{'已就绪' if status.get('control_plane_ready') else '未就绪'}，"
                f"当前模式 {status.get('active_control_mode', 'unknown')}，"
                f"{'运动已授权' if status.get('motion_authorized') else '运动未授权'}。"
            )
        if outcome.query_kind == "list_skills":
            names = sorted(
                str(item["name"])
                for item in catalog.get("skills", [])
                if isinstance(item, Mapping) and item.get("planner_visible") is True and item.get("name")
            )
            # Spoken like plan summaries: the Chinese catalog name when there is one, else the skill name.
            return AgentService._bounded_list_message(
                "当前可规划技能", [_chinese_skill_name(catalog, name) or name for name in names]
            )
        if outcome.query_kind == "list_poses":
            poses = [str(name) for name in catalog.get("pose_names", [])]
            return AgentService._bounded_list_message("当前命名姿态", sorted(poses))
        if outcome.query_kind == "describe_skill":
            capability = next(
                (
                    item
                    for item in catalog.get("skills", [])
                    if isinstance(item, Mapping) and item.get("name") == outcome.skill_name
                ),
                None,
            )
            if capability is None:
                return f"当前目录中没有技能 {outcome.skill_name}。"
            return f"{outcome.skill_name}：{capability.get('summary', '无公开说明')}"
        return outcome.user_message

    @staticmethod
    def _bounded_list_message(prefix: str, values: list[str], *, max_length: int = 145) -> str:
        if not values:
            return f"{prefix}：无。"
        visible = []
        used = len(prefix) + 1
        for value in values:
            addition = len(value) + (1 if visible else 0)
            if used + addition + 8 > max_length:
                break
            visible.append(value)
            used += addition
        remaining = len(values) - len(visible)
        suffix = f"，另有 {remaining} 项。" if remaining else "。"
        return f"{prefix}：{'、'.join(visible)}{suffix}"

    @staticmethod
    def _bind_catalog_targets(outcome: PlannerOutcome, catalog: Mapping[str, Any]) -> PlannerOutcome:
        """Resolve only unique explicit aliases; never guess a motion target."""
        by_name = {item.get("name"): item for item in catalog.get("skills", []) if isinstance(item, Mapping)}
        steps = []
        for step in outcome.steps:
            capability = by_name.get(step.skill_name, {})
            schema = capability.get("parameters", {}).get("properties", {}).get("target_name", {})
            allowed = schema.get("enum")
            target = step.target_name
            if isinstance(allowed, list) and target not in allowed:
                matches = {
                    option.get("target_name")
                    for option in capability.get("target_options", [])
                    if isinstance(option, Mapping)
                    and target in option.get("aliases", [])
                    and option.get("target_name") in allowed
                }
                if len(matches) != 1:
                    raise RuntimeError(f"invalid or ambiguous target_name for {step.skill_name}: {target!r}")
                step = dataclasses.replace(step, target_name=next(iter(matches)))
            steps.append(step)
        return dataclasses.replace(outcome, steps=tuple(steps))

    def _validate_workflow(self, outcome: PlannerOutcome, catalog: Mapping[str, Any]) -> None:
        validate_fruit_delivery(
            outcome,
            self._fruit_delivery,
            required=self._robot_scope == "aimdk_x2_summit" or catalog.get("profile_name") == "aimdk_x2_summit",
        )
        skills = catalog.get("skills", [])
        if not isinstance(skills, list):
            raise RuntimeError("Gateway catalog skills are invalid")
        by_name = {item.get("name"): item for item in skills if isinstance(item, Mapping)}
        for step in outcome.steps:
            capability = by_name.get(step.skill_name)
            if not isinstance(capability, Mapping) or capability.get("planner_visible") is not True:
                raise RuntimeError(f"skill is not planner-visible: {step.skill_name}")
            if capability.get("semantic_level") not in {"atomic_operator", "skill"}:
                raise RuntimeError(f"skill is not executable: {step.skill_name}")
            if self._allowed_skills and step.skill_name not in self._allowed_skills:
                raise RuntimeError(f"skill is outside the incubation allowlist: {step.skill_name}")

    @staticmethod
    def _bind_step_schema_versions(outcome: PlannerOutcome, catalog: Mapping[str, Any]) -> PlannerOutcome:
        """Take each step's ``schema_version`` from the Gateway catalog, not from the model.

        The contract version is a property of the selected skill (the Gateway
        rejects any other value), so a planner that picked the wrong one is
        corrected before the proposal, digest and presentation are built.  A
        step that carries navigation parameters for a version-1 skill cannot
        be expressed in that contract and is rejected instead of rewritten.
        """
        versions = {
            item.get("name"): item.get("schema_version", 1)
            for item in catalog.get("skills", [])
            if isinstance(item, Mapping)
        }
        steps = []
        for step in outcome.steps:
            expected = versions[step.skill_name]
            if isinstance(expected, bool) or not isinstance(expected, int):
                raise RuntimeError(f"catalog schema_version of {step.skill_name} is not an integer: {expected!r}")
            if step.schema_version != expected:
                try:
                    step = dataclasses.replace(step, schema_version=expected)
                except (ValueError, TypeError) as exc:
                    raise RuntimeError(
                        f"step cannot be bound to the {step.skill_name} contract (schema_version {expected}): {exc}"
                    ) from exc
                _LOGGER.info("bound %s to catalog schema_version %d", step.skill_name, expected)
            steps.append(step)
        return dataclasses.replace(outcome, steps=tuple(steps))

    @staticmethod
    def _bind_summary_to_catalog_names(
        outcome: PlannerOutcome,
        catalog: Mapping[str, Any],
        *,
        pose_skills: frozenset[str] = frozenset({"free_pose"}),
    ) -> PlannerOutcome:
        """Name the workflow by its Chinese catalog names and validated step targets.

        Clients render this summary before motion. Use catalog action names;
        parameterized picks and poses name their validated targets instead of
        the first alias. When a step has no Chinese name, keep the planner's
        whole summary rather than partially rewriting it.
        """
        # A pose skill's aliases can enumerate opposite seed actions. Its
        # validated Chinese target names the requested pose; the first alias
        # only names the first seed. Keep the planner's summary for opaque IDs.
        names = []
        for step in outcome.steps:
            name = _chinese_skill_name(catalog, step.skill_name)
            if name is not None and step.skill_name == "pick_object":
                # Fruit aliases enumerate choices; the first alias cannot name
                # every pick. Share the exact target labels used by gift speech.
                label = fruit_label_zh(step.target_name)
                name = f"抓取{label}" if label else "抓取物体"
            if name is not None and step.skill_name in pose_skills:
                target = step.target_name
                name = target if any(0x4E00 <= ord(char) <= 0x9FFF for char in target) else None
            names.append(name)
        if not names or None in names:
            return outcome
        summary = "、".join(names)
        try:
            bound = dataclasses.replace(outcome, summary=summary)
        except ValueError:
            # Longer than the summary contract allows.
            return outcome
        if summary != outcome.summary:
            _LOGGER.info("bound workflow summary %r to catalog names %r", outcome.summary, summary)
        return bound

    def _present(
        self,
        key: RequestKey,
        generation: int,
        proposal: PlanProposal,
        presentation: Any,
        cancel_token: threading.Event,
    ) -> None:
        with trace_stage(
            _trace,
            "agent.presentation",
            request_id=key.request_id,
            task_id=presentation.task_ref.task_id,
            plan_id=presentation.task_ref.plan_id,
        ):
            self._store.bind_task(
                key,
                expected_generation=generation,
                task_ref=presentation.task_ref,
                presentation=presentation,
                proposal_json=json.dumps(proposal.to_dict(), ensure_ascii=False, sort_keys=True),
            )
            if self._presentation is None or self._events is None:
                raise RuntimeError("presentation transport is unavailable")
            # The task binding above is the durable pre-submission barrier.
            # Typed Programs then display asynchronously: publishing is
            # best-effort and must not gate, fail, or stop execution.  Legacy
            # Agent Plans keep the receipt gate and its required events.
            async_display = presentation.execution_mode == ASYNC_PRESENTATION_MODE
            self._presentation.present(
                key,
                presentation,
                cancel_token=cancel_token,
                publish=lambda detail: self._emit(
                    key,
                    "presentation",
                    "MAY_EXECUTE",
                    presentation.summary,
                    detail=detail,
                    required=not async_display,
                ),
            )
            if not async_display:
                self._emit(key, "presentation_rendered", "MAY_EXECUTE", "exact plan presentation completed")

    def _mark_may_submit(
        self,
        key: RequestKey,
        generation: int,
        task_ref: Any,
        detail: Mapping[str, object],
    ) -> None:
        with trace_stage(
            _trace,
            "agent.submission",
            request_id=key.request_id,
            task_id=task_ref.task_id,
            plan_id=task_ref.plan_id,
        ):
            self._store.record_confirmation(key, expected_generation=generation, detail=detail)
            self._store.mark_submitted(key, expected_generation=generation, task_ref=task_ref)

    def _finish_failure(self, key: RequestKey, record: Any, message: str) -> None:
        self._store.finish(
            key,
            expected_generation=record.planning_generation,
            result=ExecutionResult(
                status="failed", task_ref=record.task_ref, error_code="AGENT_FAILED", message=message, detail={}
            ),
        )
        self._emit(key, "failed", "FAILED", message)

    def _execution_progress(self, key: RequestKey, detail: Mapping) -> None:
        current = self._store.get_request(key)
        if current.terminal is not None:
            return
        message = execution_reply(str(detail.get("state", "ACTIVE")), detail)
        self._emit(key, "execution_progress", "RUNNING", message, detail=detail)

    def _finish_unknown(self, key: RequestKey, record: Any, message: str, *, detail: Mapping | None = None) -> None:
        reason = (message or "unknown execution outcome")[:300]
        self._store.quarantine(key, expected_generation=record.planning_generation, reason=reason, detail=detail)
        self._quarantined = True
        self._emit(key, "quarantined", "UNKNOWN", execution_reply("UNKNOWN", detail), detail=detail)

    @staticmethod
    def _request_state_for_result(result: Any) -> str:
        if result.status == "succeeded":
            return "SUCCEEDED"
        if result.status == "failed":
            return "FAILED"
        if result.status == "cancelled":
            return "CANCELLED" if result.task_ref is not None else "CANCELLED_BEFORE_EXECUTION"
        return "UNKNOWN"

    def _finish_plan_without_execution(self, key: RequestKey, generation: int, message: str) -> None:
        """Close dry-run planning without claiming that a robot task succeeded."""
        self._store.mark_answered(
            key,
            expected_generation=generation,
            message=message,
            error_code="DRY_RUN_ONLY",
        )
        self._emit(key, "dry_run_complete", "ANSWERED", message)

    def _emit(
        self,
        key: RequestKey,
        event_type: str,
        state: str,
        message: str,
        *,
        detail: Mapping[str, object] | None = None,
        required: bool = False,
    ) -> None:
        if self._events is None:
            if required:
                raise RuntimeError("required event transport is unavailable")
            return
        detail = dict(detail or {})
        timings = planning_timing_snapshot()
        if timings:
            detail["planning_timing"] = timings
        with self._emit_lock:
            sequence = self._store.record_event(
                key,
                event_type=event_type,
                state=state,
                detail_json=json.dumps({"message": message, "detail": detail}, ensure_ascii=False, sort_keys=True),
            )
            try:
                self._events.publish(
                    AgentEvent(
                        schema_version=1,
                        request_key=key,
                        sequence=sequence,
                        event_type=event_type,
                        state=state,
                        user_message=message,
                        detail=detail,
                        created_at=datetime.now(timezone.utc).isoformat(),
                    )
                )
            except Exception:
                if required:
                    raise
                # The ledger is authoritative; a broken transport must not reopen
                # the request through the outer failure handler.
                _LOGGER.exception("Could not publish Agent event for %s", key.request_id)


def _pose_session(request: AgentRequest) -> str:
    return f"{request.channel_id}/{request.principal_id}/{request.session_id}"


def _fold(text: str) -> str:
    return re.sub(r"[\s，。！？、,.!?；;：:～~]+", "", str(text)).casefold()


def _json_or_text(value: Any) -> Any:
    try:
        return json.loads(value) if isinstance(value, str) and value else value
    except ValueError:
        return value


def _ledger_replay_outcome(steps) -> PlannerOutcome:
    """The workflow that resumes an interrupted step: the Gateway's canonical steps, not a plan."""
    canonical = normalize_workflow_steps(list(steps))
    summary = "、".join(step.skill_name for step in canonical)
    return PlannerOutcome(kind="workflow", user_message=f"从中断处继续：{summary}。", summary=summary, steps=canonical)
