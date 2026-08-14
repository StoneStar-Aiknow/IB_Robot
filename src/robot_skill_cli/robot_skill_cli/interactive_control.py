"""Interactive closed-loop controller for the IB-Robot Capability Gateway.

This module is a pure-Python, ROS-free reference implementation of the five-feature
interactive closed loop (catalog discovery, out-of-catalog rejection, in-process
workflow prepare/confirm, ``stop`` to a definite terminal, and ``continue`` on fresh
state). It depends only on an injected ``RosBridge``-like object and the shared
catalog/workflow helpers, so it is unit-testable without a ROS stack.

The controller never parses free natural language into structured steps; it only
recognizes a closed vocabulary (confirm / stop / continue) and drives the existing
Gateway primitives. Structured ``workflow_steps`` remain the caller's responsibility.
"""

from __future__ import annotations

import contextlib
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from embodied_common.workflow_contracts import normalize_workflow_steps

# Closed natural-language grammars. Matched after stripping + lowercasing against
# either the full string or a recognized prefix; unknown text never acts.
_CONFIRM_PHRASES = (
    "确认执行当前计划",
    "确认然后挥手",
    "确认一下计划内容",
    "确认执行",
    "确认",
    "执行吧",
    "好的",
    "好",
    "可以",
    "是的",
    "confirm",
)
_STOP_PHRASES = ("别动了", "别动", "停止", "停下", "停", "stop", "halt")
_CONTINUE_PHRASES = ("继续吧", "继续", "接着来", "go on", "continue")

INTENT_CONFIRM = "confirm"
INTENT_STOP = "stop"
INTENT_CONTINUE = "continue"
INTENT_UNKNOWN = "unknown"

# action_msgs/GoalStatus terminal values.
_GOAL_SUCCEEDED = 4
_GOAL_CANCELED = 5
_GOAL_ABORTED = 6
_TERMINAL_GOAL_STATUSES = {_GOAL_SUCCEEDED, _GOAL_CANCELED, _GOAL_ABORTED}

# Controller states.
IDLE = "idle"
DISCOVERED = "discovered"
PREPARED = "prepared"
CONFIRMED = "confirmed"
EXECUTING = "executing"
STOPPING = "stopping"
STOPPED = "stopped"
SUCCEEDED = "succeeded"
FAILED = "failed"
UNKNOWN = "unknown"

_RPC_TIMEOUT_FLOOR_SEC = 30.0
_STATUS_PREFLIGHT_FLOOR_SEC = 15.0
_CANCEL_POLL_INTERVAL_SEC = 0.02
_EXECUTE_POLL_INTERVAL_SEC = 0.02


def _default_view_resolver(snapshot: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    """Verify a Gateway snapshot and return its public catalog view.

    Imported lazily so this module stays ROS-free at import time; in production the
    bridge is already started (rclpy + ibrobot_msgs available) when this runs.
    """
    from robot_skill_cli.catalog import capability_view_from_snapshot

    return capability_view_from_snapshot(snapshot, status)


class InteractiveControlError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class OutOfCatalogError(InteractiveControlError):
    pass


class NotConfirmedError(InteractiveControlError):
    pass


class UnknownStopError(InteractiveControlError):
    pass


class IllegalStateError(InteractiveControlError):
    pass


def _normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def classify(text: str) -> str:
    """Classify a free-form string against the closed confirm/stop/continue grammar."""
    normalized = _normalize_text(text)
    if not normalized:
        return INTENT_UNKNOWN
    if any(normalized == phrase or normalized.startswith(phrase) for phrase in _STOP_PHRASES):
        return INTENT_STOP
    if any(normalized == phrase or normalized.startswith(phrase) for phrase in _CONTINUE_PHRASES):
        return INTENT_CONTINUE
    if any(normalized == phrase or normalized.startswith(phrase) for phrase in _CONFIRM_PHRASES):
        return INTENT_CONFIRM
    return INTENT_UNKNOWN


def planner_visible_names(view: dict[str, Any]) -> set[str]:
    return {skill["name"] for skill in view.get("skills", []) if skill.get("planner_visible") is True}


class InteractiveController:
    """Drive the five-feature closed loop over an injected Gateway bridge.

    The bridge is duck-typed: it must expose the ``RosBridge`` methods this
    controller calls (``get_status``, ``get_skill_snapshot``, ``plan_agent_command``,
    ``validate_agent_plan``, ``confirm_agent_plan``, ``wait_for_execute_plan_server``,
    ``send_agent_plan_goal``, ``wait_future``, ``cancel_agent_plan``,
    ``get_agent_plan_result``). The module itself never imports ``rclpy``.
    """

    def __init__(
        self,
        bridge: Any,
        *,
        timeout_policy: dict[str, Any],
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        view_resolver: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self._bridge = bridge
        self._rpc_timeout_sec = max(
            float(timeout_policy.get("rpc_timeout_sec", _RPC_TIMEOUT_FLOOR_SEC)), _RPC_TIMEOUT_FLOOR_SEC
        )
        self._status_timeout_sec = max(self._rpc_timeout_sec, _STATUS_PREFLIGHT_FLOOR_SEC)
        self._id_factory = id_factory
        self._monotonic = monotonic
        self._sleep = sleep
        self._resolve_view = view_resolver or _default_view_resolver
        self._fresh_status: dict[str, Any] | None = None
        self._fresh_view: dict[str, Any] | None = None
        self._fresh_identity: tuple[str, int, str] | None = None
        self._pending: dict[str, Any] | None = None
        self._confirmed: dict[str, Any] | None = None
        self._terminal: dict[str, Any] | None = None
        self._state = IDLE
        self._active_task_id: str | None = None
        self._stop_event: threading.Event | None = None

    @property
    def state(self) -> str:
        return self._state

    def _rpc(self) -> float:
        return self._rpc_timeout_sec

    def discover(self) -> dict[str, Any]:
        """Feature 1: read-only query of the current runtime catalog and identity."""
        status = self._bridge.get_status(task_id="", payload_hash="", timeout_sec=self._status_timeout_sec)
        if not status.get("control_plane_ready"):
            raise InteractiveControlError("SERVER_UNAVAILABLE", "Gateway control plane is not ready")
        identity = (
            status.get("registry_epoch", ""),
            int(status.get("registry_generation", 0)),
            status.get("registry_digest", ""),
        )
        if not (identity[0] and identity[2] and identity[1] > 0):
            raise InteractiveControlError("SKILL_REGISTRY_NOT_READY", "Gateway registry identity is not ready")
        snapshot = self._bridge.get_skill_snapshot(
            registry_epoch=identity[0],
            generation=identity[1],
            timeout_sec=status.get("rpc_timeout_sec", self._rpc_timeout_sec),
        )
        view = self._resolve_view(snapshot, status)
        self._fresh_status = status
        self._fresh_view = view
        self._fresh_identity = identity
        self._state = DISCOVERED
        return {
            "robot_name": view["robot_name"],
            "registry_identity": {
                "registry_epoch": identity[0],
                "registry_generation": identity[1],
                "registry_digest": identity[2],
            },
            "capability_digest": view["capability_digest"],
            "planner_visible_names": sorted(planner_visible_names(view)),
            "capabilities": [
                {
                    "name": skill["name"],
                    "semantic_level": skill.get("semantic_level"),
                    "planner_visible": bool(skill.get("planner_visible")),
                    "ready": self._capability_ready(status, skill["name"]),
                }
                for skill in view["skills"]
            ],
        }

    @staticmethod
    def _capability_ready(status: dict[str, Any], skill_name: str) -> bool:
        for capability in status.get("capabilities", []):
            if capability.get("name") == skill_name:
                return bool(capability.get("ready"))
        return False

    def reject_out_of_catalog(self, steps: list[dict[str, Any]], view: dict[str, Any] | None = None) -> None:
        """Feature 2: reject any step whose skill is not planner-visible in the catalog."""
        catalog_view = view if view is not None else self._fresh_view
        if catalog_view is None:
            raise IllegalStateError("ILLEGAL_STATE", "discover() must run before catalog rejection")
        visible = planner_visible_names(catalog_view)
        for step in steps:
            skill_name = str(step.get("skill_name", "")).strip()
            capability = next((s for s in catalog_view["skills"] if s["name"] == skill_name), None)
            if capability is None or skill_name not in visible:
                raise OutOfCatalogError("SKILL_REFERENCE_MISSING", f"skill is not planner-visible: {skill_name}")
            if capability.get("semantic_level") not in {"atomic_operator", "skill"}:
                raise OutOfCatalogError(
                    "SKILL_REFERENCE_MISSING", f"skill semantic level is not plannable: {skill_name}"
                )

    def prepare_workflow(self, raw_command: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
        """Feature 3 (create + present): plan one workflow and bind the pending plan in-session."""
        if self._fresh_view is None:
            raise IllegalStateError("ILLEGAL_STATE", "discover() must run before prepare_workflow()")
        self.reject_out_of_catalog(steps)
        normalized = [step.to_dict() for step in normalize_workflow_steps(steps)]
        request_id = self._id_factory()
        task_id = self._id_factory()
        result = self._bridge.plan_agent_command(
            request_id=request_id,
            raw_command=raw_command,
            workflow_steps=normalized,
            timeout_sec=self._rpc(),
        )
        if not result.get("success"):
            raise InteractiveControlError(
                str(result.get("error_code") or "SKILL_SCHEMA_INVALID"),
                str(result.get("message") or "plan_agent_command failed"),
            )
        plan = result["plan"]
        self._pending = {
            "plan_token": plan["plan_token"],
            "plan_digest": plan["plan_digest"],
            "plan_id": plan["plan_id"],
            "raw_command": raw_command,
            "steps": normalized,
            "registry_identity": self._fresh_identity,
            "task_id": task_id,
        }
        self._confirmed = None
        self._terminal = None
        self._state = PREPARED
        return self._presentation()

    def _presentation(self) -> dict[str, Any]:
        assert self._pending is not None
        assert self._fresh_identity is not None
        return {
            "state": self._state,
            "steps": list(self._pending["steps"]),
            "plan_digest": self._pending["plan_digest"],
            "plan_id": self._pending["plan_id"],
            "registry_identity": {
                "registry_epoch": self._fresh_identity[0],
                "registry_generation": self._fresh_identity[1],
                "registry_digest": self._fresh_identity[2],
            },
            "task_id": self._pending["task_id"],
            "confirm_command": "确认执行当前计划",
        }

    def confirm_plan(self) -> dict[str, Any]:
        """Internal validate + confirm (no user gate).

        Runs ``validate_agent_plan`` + ``confirm_agent_plan`` on the in-session
        pending plan so the Gateway can execute it. The user-facing ``确认`` gate
        is removed: this is called automatically right after presentation so
        execution starts immediately; the user aborts a wrong workflow with
        ``别动`` during execution instead of confirming beforehand.
        """
        if self._pending is None:
            raise IllegalStateError("ILLEGAL_STATE", "no pending workflow to confirm")
        validation = self._bridge.validate_agent_plan(
            plan_token=self._pending["plan_token"],
            timeout_sec=self._status_timeout_sec,
        )
        if not validation.get("allowed"):
            raise InteractiveControlError(
                str(validation.get("error_code") or "SKILL_VALIDATION_FAILED"),
                str(validation.get("message") or "validate_agent_plan failed"),
            )
        task_budget_sec = float(self._fresh_status["task_budget_sec"])
        result = self._bridge.confirm_agent_plan(
            plan_token=self._pending["plan_token"],
            plan_digest=self._pending["plan_digest"],
            task_id=self._pending["task_id"],
            status=self._fresh_status,
            task_budget_sec=task_budget_sec,
            timeout_sec=self._status_timeout_sec,
        )
        if not result.get("confirmed"):
            raise InteractiveControlError(
                str(result.get("error_code") or "SKILL_REQUEST_ID_CONFLICT"),
                str(result.get("message") or "confirm_agent_plan failed"),
            )
        self._confirmed = {
            "confirmation_token": result["confirmation_token"],
            "task_id": self._pending["task_id"],
            "task_budget_sec": float(result.get("confirmed_task_budget_sec", task_budget_sec)),
        }
        self._state = CONFIRMED
        return {
            "state": self._state,
            "task_id": self._confirmed["task_id"],
            "confirmed_task_budget_sec": self._confirmed["task_budget_sec"],
        }

    def confirm(self, confirmation_text: str) -> dict[str, Any]:
        """Optional gated confirm: bind the pending plan via the closed NL ``确认`` grammar, then confirm_plan()."""
        if self._state != PREPARED or self._pending is None:
            raise IllegalStateError("ILLEGAL_STATE", "no pending workflow to confirm")
        if classify(confirmation_text) != INTENT_CONFIRM:
            raise NotConfirmedError("NOT_CONFIRMED", f"not a confirmation phrase: {confirmation_text!r}")
        return self.confirm_plan()

    def run(
        self,
        raw_command: str,
        steps: list[dict[str, Any]],
        *,
        stop_event: threading.Event | None = None,
        feedback_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """No-gate one-shot flow: discover → prepare → auto confirm_plan → execute.

        Removes the user-facing ``确认`` gate: the plan is validated and confirmed
        internally so execution starts immediately after presentation. The user
        interrupts a wrong workflow with ``别动`` (``stop_event`` / ``request_stop``)
        during execution. Callable from ``IDLE`` or any definite terminal.
        """
        if self._state not in {IDLE, DISCOVERED, STOPPED, SUCCEEDED, FAILED}:
            raise IllegalStateError("ILLEGAL_STATE", f"cannot run from state {self._state}")
        if self._fresh_view is None:
            self.discover()
        self.prepare_workflow(raw_command, steps)
        self.confirm_plan()
        return self.execute(stop_event=stop_event, feedback_callback=feedback_callback)

    def execute(
        self,
        *,
        stop_event: threading.Event | None = None,
        feedback_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Feature 4: execute the confirmed plan; interruptible via ``stop_event``."""
        if self._state != CONFIRMED or self._confirmed is None:
            raise IllegalStateError("ILLEGAL_STATE", "no confirmed workflow to execute")
        if not self._bridge.wait_for_execute_plan_server(timeout_sec=self._rpc()):
            raise InteractiveControlError("SERVER_UNAVAILABLE", "agent plan action server unavailable")
        task_id = self._confirmed["task_id"]
        timeout_sec = self._confirmed["task_budget_sec"]
        self._active_task_id = task_id
        self._stop_event = stop_event
        self._state = EXECUTING
        goal_future = self._bridge.send_agent_plan_goal(
            plan_token=self._pending["plan_token"],
            confirmation_token=self._confirmed["confirmation_token"],
            task_id=task_id,
            timeout_sec=timeout_sec,
            feedback_callback=feedback_callback,
        )
        if not self._bridge.wait_future(goal_future, timeout_sec=self._rpc()):
            return self._converge_after_failure(task_id, "goal response timed out")
        goal_handle = goal_future.result()
        if goal_handle is None or not getattr(goal_handle, "accepted", False):
            return self._record_terminal(FAILED, "GOAL_REJECTED", "agent plan goal was rejected")
        try:
            result_future = goal_handle.get_result_async()
        except Exception:
            return self._converge_after_failure(task_id, "result request unavailable")
        deadline = self._monotonic() + timeout_sec + self._rpc_timeout_sec
        while not result_future.done():
            if self._stop_event is not None and self._stop_event.is_set():
                return self._stop_and_converge(task_id)
            if self._monotonic() >= deadline:
                return self._stop_and_converge(task_id, deadline_expired=True)
            self._sleep(_EXECUTE_POLL_INTERVAL_SEC)
        return self._read_terminal(task_id)

    def request_stop(self) -> None:
        """Feature 4: thread-safe stop trigger from outside the execute loop."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._active_task_id is not None:
            self._state = STOPPING
            with contextlib.suppress(Exception):
                self._bridge.cancel_agent_plan(self._active_task_id, timeout_sec=self._rpc())

    def _stop_and_converge(self, task_id: str, *, deadline_expired: bool = False) -> dict[str, Any]:
        self._state = STOPPING
        with contextlib.suppress(Exception):
            self._bridge.cancel_agent_plan(task_id, timeout_sec=self._rpc())
        return self._read_terminal(task_id, deadline_expired=deadline_expired)

    def _converge_after_failure(self, task_id: str, detail: str) -> dict[str, Any]:
        with contextlib.suppress(Exception):
            self._bridge.cancel_agent_plan(task_id, timeout_sec=self._rpc())
        return self._read_terminal(task_id, failure_detail=detail)

    def _read_terminal(
        self,
        task_id: str,
        *,
        deadline_expired: bool = False,
        failure_detail: str = "",
    ) -> dict[str, Any]:
        deadline = self._monotonic() + self._rpc_timeout_sec
        terminal = None
        while self._monotonic() < deadline:
            with contextlib.suppress(Exception):
                terminal = self._bridge.get_agent_plan_result(task_id, timeout_sec=self._rpc())
            status = int(terminal.get("status", 0)) if terminal else 0
            if status in _TERMINAL_GOAL_STATUSES:
                return self._record_terminal_result(task_id, terminal)
            self._sleep(_CANCEL_POLL_INTERVAL_SEC)
        return self._record_unknown(task_id, deadline_expired=deadline_expired, failure_detail=failure_detail)

    @staticmethod
    def _state_for_status(status: int) -> str:
        if status == _GOAL_SUCCEEDED:
            return SUCCEEDED
        if status == _GOAL_CANCELED:
            return STOPPED
        return FAILED

    def _record_terminal_result(self, task_id: str, terminal: dict[str, Any]) -> dict[str, Any]:
        status = int(terminal.get("status", 0))
        result = terminal.get("result", {}) or {}
        return self._record_terminal(
            self._state_for_status(status),
            str(result.get("error_code", "")),
            str(result.get("message", "")),
            result=result,
        )

    def _record_terminal(
        self,
        state: str,
        error_code: str,
        message: str,
        *,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._terminal = {
            "state": state,
            "task_id": self._active_task_id,
            "error_code": error_code,
            "message": message,
            "result": result or {},
        }
        self._state = state
        self._active_task_id = None
        self._stop_event = None
        return dict(self._terminal)

    def _record_unknown(
        self, task_id: str, *, deadline_expired: bool = False, failure_detail: str = ""
    ) -> dict[str, Any]:
        message = "robot stop state is unknown"
        if failure_detail:
            message = f"{message}: {failure_detail}"
        self._terminal = {
            "state": UNKNOWN,
            "task_id": task_id,
            "error_code": "SKILL_CANCEL_TIMEOUT",
            "message": message,
            "result": {},
        }
        self._state = UNKNOWN
        self._active_task_id = None
        self._stop_event = None
        return dict(self._terminal)

    def continue_workflow(
        self,
        raw_command: str,
        steps: list[dict[str, Any]] | None = None,
        *,
        resume: bool = False,
    ) -> dict[str, Any]:
        """Feature 5: continue on fresh state.

        Fresh mode (``resume=False``, default): plan the caller-provided ``steps``
        as a brand-new continuation with a new ``request_id`` / ``task_id``.

        Resume mode (``resume=True``): slice the prior plan's steps to
        ``prior_steps[completed_step_count:]`` and plan only the remaining steps.
        Fully-completed steps are skipped; the step interrupted mid-execution is
        re-run from its start (never resumed mid-skill). Requires a definite prior
        terminal (STOPPED / SUCCEEDED / FAILED); ``UNKNOWN`` is still refused.
        """
        if self._state not in {STOPPED, SUCCEEDED, FAILED}:
            raise UnknownStopError("SKILL_CANCEL_TIMEOUT", "robot stop state is unknown; refuse to continue")
        previous_terminal = dict(self._terminal or {})
        previous_steps = list(self._pending.get("steps", [])) if self._pending else []
        if resume:
            if not previous_steps:
                raise IllegalStateError("ILLEGAL_STATE", "resume requires a prior plan's steps")
            completed = int((previous_terminal.get("result") or {}).get("completed_step_count", 0))
            remaining = previous_steps[completed:]
            if not remaining:
                raise IllegalStateError(
                    "PLAN_ALREADY_COMPLETE",
                    f"prior plan already completed all {len(previous_steps)} step(s); nothing to resume",
                )
            steps_to_plan = remaining
            resumed_from_step = completed
        else:
            if not steps:
                raise IllegalStateError("ILLEGAL_STATE", "non-resume continuation requires steps")
            steps_to_plan = steps
            resumed_from_step = None
        self._fresh_status = None
        self._fresh_view = None
        self._fresh_identity = None
        self._pending = None
        self._confirmed = None
        self._state = IDLE
        self.discover()
        presentation = self.prepare_workflow(raw_command, steps_to_plan)
        presentation["continues_from"] = previous_terminal
        presentation["resume"] = resume
        if resumed_from_step is not None:
            presentation["resumed_from_step"] = resumed_from_step
        return presentation
