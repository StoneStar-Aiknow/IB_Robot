"""Unit tests for the interactive closed-loop controller (no ROS, fake bridge)."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from robot_skill_cli import interactive_control as ic


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, sec: float) -> None:
        self.now += float(sec)


class FakeFuture:
    def __init__(self, result_value: Any, *, done: bool = True) -> None:
        self._result = result_value
        self._done = done

    def done(self) -> bool:
        return self._done

    def result(self) -> Any:
        return self._result


class FakeGoalHandle:
    def __init__(self, *, accepted: bool = True, result_future: FakeFuture | None = None) -> None:
        self.accepted = accepted
        self._result_future = result_future or FakeFuture(None, done=False)

    def get_result_async(self) -> FakeFuture:
        return self._result_future


def _capability_view() -> dict[str, Any]:
    def skill(name: str, *, visible: bool, level: str = "skill") -> dict[str, Any]:
        return {
            "name": name,
            "semantic_level": level,
            "planner_visible": visible,
            "summary": "",
            "domain": "",
            "moves_robot": True,
            "required_control_mode": "moveit_planning",
            "parameters": {},
            "recovery_policy": {},
        }

    return {
        "robot_name": "test_robot",
        "skills": [
            skill("nod_yes", visible=True),
            skill("wave_hand", visible=True),
            skill("internal_only", visible=False, level="atomic_operator"),
        ],
        "pose_names": [],
        "timeout_policy": {"rpc_timeout_sec": 30.0},
        "capability_digest": "capdig",
        "profile_name": "test",
    }


class FakeBridge:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.status = {
            "control_plane_ready": True,
            "registry_epoch": "epoch-1",
            "registry_generation": 1,
            "registry_digest": "regdig",
            "task_budget_sec": 90.0,
            "rpc_timeout_sec": 2.0,
            "capabilities": [
                {
                    "name": "nod_yes",
                    "ready": True,
                    "reason": "",
                    "semantic_level": "skill",
                    "planner_visible": True,
                    "required_control_mode": "moveit_planning",
                },
                {
                    "name": "wave_hand",
                    "ready": True,
                    "reason": "",
                    "semantic_level": "skill",
                    "planner_visible": True,
                    "required_control_mode": "moveit_planning",
                },
            ],
        }
        self.snapshot = {"success": True, "snapshot_json": ""}
        self.plan_result_status = 4
        self.plan_result = {
            "success": True,
            "plan_id": "pid",
            "plan_digest": "wdig",
            "completed_step_count": 1,
            "error_code": "",
            "message": "ok",
            "actual_registry_epoch": "epoch-1",
            "actual_registry_generation": 1,
            "actual_registry_digest": "regdig",
        }
        self.server_ready = True
        self.goal_future = FakeFuture(FakeGoalHandle(), done=True)
        self.result_future = FakeFuture(None, done=False)

    def _record(self, method: str, kwargs: dict[str, Any]) -> None:
        self.calls.append((method, dict(kwargs)))

    def get_status(self, *, task_id, payload_hash, timeout_sec):
        self._record("get_status", {"task_id": task_id, "payload_hash": payload_hash})
        return dict(self.status)

    def get_skill_snapshot(self, *, registry_epoch, generation, timeout_sec):
        self._record("get_skill_snapshot", {"registry_epoch": registry_epoch, "generation": generation})
        return dict(self.snapshot)

    def plan_agent_command(self, *, request_id, raw_command, workflow_steps, timeout_sec):
        self._record("plan_agent_command", {"request_id": request_id, "raw_command": raw_command})
        return {
            "success": True,
            "plan": {"plan_token": "ptok-" + request_id, "plan_digest": "pdig", "plan_id": "pid-" + request_id},
            "error_code": "",
            "message": "",
        }

    def validate_agent_plan(self, *, plan_token, timeout_sec):
        self._record("validate_agent_plan", {"plan_token": plan_token})
        return {"allowed": True, "plan_id": "pid", "plan_digest": "pdig", "error_code": "", "message": ""}

    def confirm_agent_plan(self, *, plan_token, plan_digest, task_id, status, task_budget_sec, timeout_sec):
        self._record("confirm_agent_plan", {"plan_token": plan_token, "plan_digest": plan_digest, "task_id": task_id})
        return {
            "confirmed": True,
            "confirmation_token": "ctok-" + task_id,
            "confirmed_task_budget_sec": float(task_budget_sec),
            "error_code": "",
            "message": "",
        }

    def wait_for_execute_plan_server(self, *, timeout_sec):
        return self.server_ready

    def send_agent_plan_goal(self, *, plan_token, confirmation_token, task_id, timeout_sec, feedback_callback):
        self._record(
            "send_agent_plan_goal",
            {"plan_token": plan_token, "confirmation_token": confirmation_token, "task_id": task_id},
        )
        return self.goal_future

    def wait_future(self, future, *, timeout_sec):
        return future.done()

    def cancel_agent_plan(self, task_id, *, timeout_sec):
        self._record("cancel_agent_plan", {"task_id": task_id})
        return {"accepted": True, "return_code": 0}

    def get_agent_plan_result(self, task_id, *, timeout_sec):
        self._record("get_agent_plan_result", {"task_id": task_id})
        return {"status": self.plan_result_status, "result": dict(self.plan_result)}


@pytest.fixture
def rig(monkeypatch):
    clock = FakeClock()
    bridge = FakeBridge(clock)
    counter = iter(range(1, 100))
    controller = ic.InteractiveController(
        bridge,
        timeout_policy={"rpc_timeout_sec": 30.0},
        id_factory=lambda: f"id-{next(counter)}",
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        view_resolver=lambda snapshot, status: _capability_view(),
    )
    return controller, bridge


def _step(skill_name: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "skill_name": skill_name,
        "target_name": "",
        "place_name": "",
        "motion_direction": "",
        "motion_distance": 0.0,
        "timeout_sec": 0.0,
    }


def test_classify_closed_grammar():
    assert ic.classify("确认") == ic.INTENT_CONFIRM
    assert ic.classify(" 别动 ") == ic.INTENT_STOP
    assert ic.classify("继续吧") == ic.INTENT_CONTINUE
    assert ic.classify("行不行") == ic.INTENT_UNKNOWN
    assert ic.classify("") == ic.INTENT_UNKNOWN


def test_discover_is_readonly_and_queries_catalog(rig):
    controller, bridge = rig
    result = controller.discover()

    methods = [call[0] for call in bridge.calls]
    assert methods == ["get_status", "get_skill_snapshot"]
    assert controller.state == ic.DISCOVERED
    assert result["robot_name"] == "test_robot"
    assert "nod_yes" in result["planner_visible_names"]
    assert "internal_only" not in result["planner_visible_names"]
    # Feature 1: no plan/confirm/execute motion primitives were invoked.
    assert "plan_agent_command" not in methods
    assert "send_agent_plan_goal" not in methods


def test_reject_out_of_catalog_blocks_planning(rig):
    controller, bridge = rig
    controller.discover()

    with pytest.raises(ic.OutOfCatalogError) as exc_info:
        controller.prepare_workflow("ghost motion", [_step("ghost_skill")])
    assert exc_info.value.code == "SKILL_REFERENCE_MISSING"

    # A planner-invisible (internal) skill is also rejected.
    with pytest.raises(ic.OutOfCatalogError):
        controller.prepare_workflow("internal", [_step("internal_only")])

    methods = [call[0] for call in bridge.calls]
    assert "plan_agent_command" not in methods


def test_prepare_then_confirm_presentation_and_nl_grammar(rig):
    controller, bridge = rig
    controller.discover()

    presentation = controller.prepare_workflow("点个头", [_step("nod_yes")])

    assert controller.state == ic.PREPARED
    assert presentation["steps"][0]["skill_name"] == "nod_yes"
    assert presentation["plan_digest"] == "pdig"
    assert presentation["task_id"] == "id-2"
    assert presentation["confirm_command"] == "确认执行当前计划"

    # Open grammar must not confirm.
    with pytest.raises(ic.NotConfirmedError):
        controller.confirm("行不行吧")

    confirmed = controller.confirm("确认执行当前计划")
    assert controller.state == ic.CONFIRMED
    assert confirmed["task_id"] == "id-2"
    confirm_call = next(c for m, c in bridge.calls if m == "confirm_agent_plan")
    assert confirm_call["plan_token"] == "ptok-id-1"
    assert confirm_call["plan_digest"] == "pdig"
    assert confirm_call["task_id"] == "id-2"


def test_execute_success_reaches_succeeded_terminal(rig):
    controller, bridge = rig
    bridge.result_future = FakeFuture(None, done=True)
    bridge.plan_result_status = 4
    controller.discover()
    controller.prepare_workflow("点个头", [_step("nod_yes")])
    controller.confirm("确认")

    result = controller.execute()

    assert controller.state == ic.SUCCEEDED
    assert result["state"] == ic.SUCCEEDED
    assert result["error_code"] == ""
    assert "send_agent_plan_goal" in [m for m, _ in bridge.calls]


def test_execute_stop_reaches_definite_stopped_terminal(rig):
    controller, bridge = rig
    bridge.plan_result_status = 5  # GoalStatus.CANCELED
    stop_event = threading.Event()
    stop_event.set()
    controller.discover()
    controller.prepare_workflow("点个头", [_step("nod_yes")])
    controller.confirm("确认")

    result = controller.execute(stop_event=stop_event)

    assert controller.state == ic.STOPPED
    assert result["state"] == ic.STOPPED
    methods = [m for m, _ in bridge.calls]
    assert "cancel_agent_plan" in methods
    assert "get_agent_plan_result" in methods


def test_execute_stop_unknown_refuses_continue(rig):
    controller, bridge = rig
    bridge.plan_result_status = 0  # never converges to a terminal GoalStatus
    stop_event = threading.Event()
    stop_event.set()
    controller.discover()
    controller.prepare_workflow("点个头", [_step("nod_yes")])
    controller.confirm("确认")

    result = controller.execute(stop_event=stop_event)

    assert controller.state == ic.UNKNOWN
    assert result["state"] == ic.UNKNOWN
    assert result["error_code"] == "SKILL_CANCEL_TIMEOUT"

    # Feature 5 gate: unknown stop state must refuse a continuation.
    with pytest.raises(ic.UnknownStopError) as exc_info:
        controller.continue_workflow("再点一次", [_step("nod_yes")])
    assert exc_info.value.code == "SKILL_CANCEL_TIMEOUT"


def test_continue_uses_fresh_state_and_new_ids(rig):
    controller, bridge = rig
    bridge.result_future = FakeFuture(None, done=True)
    bridge.plan_result_status = 4
    controller.discover()
    controller.prepare_workflow("点个头", [_step("nod_yes")])
    controller.confirm("确认")
    controller.execute()
    assert controller.state == ic.SUCCEEDED

    first_plan_calls = [c for m, c in bridge.calls if m == "plan_agent_command"]
    first_send_calls = [c for m, c in bridge.calls if m == "send_agent_plan_goal"]
    assert len(first_plan_calls) == 1 and len(first_send_calls) == 1
    first_task_id = first_send_calls[0]["task_id"]

    presentation = controller.continue_workflow("再点一次", [_step("wave_hand")])

    assert controller.state == ic.PREPARED
    assert presentation["continues_from"]["state"] == ic.SUCCEEDED
    # Fresh discover happened again (second status + snapshot after the first batch).
    method_counts = {m: sum(1 for mm, _ in bridge.calls if mm == m) for m in ("get_status", "get_skill_snapshot")}
    assert method_counts["get_status"] >= 2
    assert method_counts["get_skill_snapshot"] >= 2
    # New task id, never reused.
    assert presentation["task_id"] != first_task_id
    second_plan_calls = [c for m, c in bridge.calls if m == "plan_agent_command"]
    assert len(second_plan_calls) == 2
    assert second_plan_calls[1]["request_id"] != second_plan_calls[0]["request_id"]


def test_continue_requires_definite_terminal(rig):
    controller, bridge = rig
    controller.discover()
    controller.prepare_workflow("点个头", [_step("nod_yes")])
    controller.confirm("确认")
    assert controller.state == ic.CONFIRMED

    with pytest.raises(ic.UnknownStopError) as exc_info:
        controller.continue_workflow("继续", [_step("nod_yes")])
    assert exc_info.value.code == "SKILL_CANCEL_TIMEOUT"


def test_request_stop_from_another_thread_reaches_stopped(rig):
    controller, bridge = rig
    bridge.plan_result_status = 5
    controller.discover()
    controller.prepare_workflow("点个头", [_step("nod_yes")])
    controller.confirm("确认")

    result_holder: dict[str, Any] = {}

    def run_execute():
        result_holder["result"] = controller.execute()

    thread = threading.Thread(target=run_execute)
    thread.start()
    # Wait for the execute loop to enter EXECUTING, then trigger stop externally.
    for _ in range(200):
        if controller.state == ic.EXECUTING:
            break
        import time

        time.sleep(0.001)
    controller.request_stop()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert controller.state == ic.STOPPED
    assert result_holder["result"]["state"] == ic.STOPPED


def test_continue_resume_slices_remaining_steps(rig):
    controller, bridge = rig
    bridge.plan_result_status = 5  # GoalStatus.CANCELED -> STOPPED
    bridge.plan_result["completed_step_count"] = 2
    plan_steps = [_step("nod_yes"), _step("wave_hand"), _step("nod_yes"), _step("wave_hand")]
    stop_event = threading.Event()
    stop_event.set()
    controller.discover()
    controller.prepare_workflow("点头挥手四步", plan_steps)
    controller.confirm("确认")
    first_task_id = controller._pending["task_id"]
    first_terminal = controller.execute(stop_event=stop_event)

    assert first_terminal["state"] == ic.STOPPED

    presentation = controller.continue_workflow("继续", resume=True)

    assert controller.state == ic.PREPARED
    assert presentation["resume"] is True
    assert presentation["resumed_from_step"] == 2
    assert presentation["continues_from"]["state"] == ic.STOPPED
    # Remaining steps = prior_steps[2:] — the two fully-completed steps are skipped,
    # the interrupted step is re-run from its start.
    assert [s["skill_name"] for s in presentation["steps"]] == ["nod_yes", "wave_hand"]
    # Fresh task id, never reused.
    assert presentation["task_id"] != first_task_id


def test_continue_resume_already_complete_raises(rig):
    controller, bridge = rig
    bridge.plan_result_status = 4  # succeeded
    bridge.plan_result["completed_step_count"] = 4
    bridge.result_future = FakeFuture(None, done=True)
    plan_steps = [_step("nod_yes"), _step("wave_hand"), _step("nod_yes"), _step("wave_hand")]
    controller.discover()
    controller.prepare_workflow("四步", plan_steps)
    controller.confirm("确认")
    controller.execute()
    assert controller.state == ic.SUCCEEDED

    with pytest.raises(ic.IllegalStateError) as exc_info:
        controller.continue_workflow("继续", resume=True)
    assert exc_info.value.code == "PLAN_ALREADY_COMPLETE"


def test_continue_resume_requires_definite_terminal(rig):
    controller, bridge = rig
    controller.discover()
    controller.prepare_workflow("点头", [_step("nod_yes")])
    controller.confirm("确认")
    assert controller.state == ic.CONFIRMED

    with pytest.raises(ic.UnknownStopError) as exc_info:
        controller.continue_workflow("继续", resume=True)
    assert exc_info.value.code == "SKILL_CANCEL_TIMEOUT"


def test_continue_resume_unknown_refused(rig):
    controller, bridge = rig
    bridge.plan_result_status = 0  # never converges -> UNKNOWN
    bridge.plan_result["completed_step_count"] = 2
    stop_event = threading.Event()
    stop_event.set()
    controller.discover()
    controller.prepare_workflow("四步", [_step("nod_yes"), _step("wave_hand"), _step("nod_yes"), _step("wave_hand")])
    controller.confirm("确认")
    controller.execute(stop_event=stop_event)
    assert controller.state == ic.UNKNOWN

    with pytest.raises(ic.UnknownStopError) as exc_info:
        controller.continue_workflow("继续", resume=True)
    assert exc_info.value.code == "SKILL_CANCEL_TIMEOUT"
