import json
import signal
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_config.loader import load_robot_config_dict
from robot_skill_cli.catalog import compile_local_snapshot, load_capability_catalog
from skill_catalog.models import SkillSnapshot

CONFIG_PATH = Path(__file__).parents[2] / "robot_config" / "config" / "robots" / "so101_single_arm.yaml"


class _CancelBridge:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.status_calls = []
        self.cancel_calls = []

    def start(self):
        return True

    def get_status(self, **kwargs):
        self.status_calls.append(kwargs)
        return self.statuses.pop(0)

    def cancel_task(self, task_id, **kwargs):
        self.cancel_calls.append((task_id, kwargs))
        return {"accepted": True}

    def close(self):
        pass


class _Future:
    def __init__(self, value=None, *, done=True, error=None):
        self.value = value
        self.completed = done
        self.error = error

    def done(self):
        return self.completed

    def result(self):
        if self.error is not None:
            raise self.error
        return self.value


class _GoalHandle:
    def __init__(self, *, accepted=True, result=None, result_done=True, result_error=None):
        self.accepted = accepted
        self.result_future = _Future(SimpleNamespace(result=result), done=result_done, error=result_error)

    def get_result_async(self):
        return self.result_future


class _ExecuteBridge:
    def __init__(
        self,
        statuses,
        *,
        accepted=True,
        result_done=True,
        cancel_converges=True,
        goal_response_timeout=False,
        late_feedback_on_close=False,
        result_error=None,
    ):
        self.snapshot = _runtime_snapshot()
        for status in statuses:
            status.update(
                registry_epoch="epoch-1",
                registry_generation=1,
                registry_digest=self.snapshot.registry_digest,
                capability_digest=self.snapshot.capability_digest,
            )
        self.statuses = deque(statuses)
        self.calls = []
        self.sent = []
        self.cancel_converges = cancel_converges
        self.goal_response_timeout = goal_response_timeout
        self.late_feedback_on_close = late_feedback_on_close
        self.feedback_callback = None
        self.goal_wait_hook = None
        result = SimpleNamespace(
            success=True,
            error_code="",
            message="completed",
            executed_primitives=["private_one", "private_two"],
        )
        self.goal_handle = _GoalHandle(
            accepted=accepted,
            result=result,
            result_done=result_done,
            result_error=result_error,
        )

    def start(self):
        self.calls.append("start")
        return True

    def get_status(self, **kwargs):
        self.calls.append(("status", kwargs))
        return self.statuses.popleft()

    def validate_skill(self, payload, **kwargs):
        self.calls.append("validate")
        return {"allowed": True, "reason": "allowed"}

    def get_skill_snapshot(self, **_kwargs):
        return {
            "success": True,
            "registry_epoch": "epoch-1",
            "generation": 1,
            "registry_digest": self.snapshot.registry_digest,
            "capability_digest": self.snapshot.capability_digest,
            "provenance_digest": self.snapshot.provenance_digest,
            "snapshot_json": self.snapshot.snapshot_json,
        }

    def wait_for_skill_server(self, **kwargs):
        self.calls.append("wait_server")
        return True

    def send_skill_goal(self, payload, *, task_id, feedback_callback):
        self.calls.append("send")
        self.sent.append((payload, task_id))
        self.feedback_callback = feedback_callback
        feedback_callback({"state": "executing", "detail": "step 1 of 2"})
        return _Future(self.goal_handle)

    def wait_future(self, future, **kwargs):
        if self.goal_response_timeout and future.value is self.goal_handle:
            if self.goal_wait_hook is not None:
                self.goal_wait_hook()
            return False
        return future.done()

    def cancel_task(self, task_id, **kwargs):
        self.calls.append(("cancel_task", task_id, kwargs))
        return {"accepted": True, "return_code": 0}

    def cancel_goal(self, goal_handle, result_future, **kwargs):
        self.calls.append("cancel_goal")
        if self.cancel_converges:
            result_future.value.result.success = False
            result_future.value.result.error_code = "SKILL_CANCELLED"
            result_future.value.result.message = "cancelled"
            result_future.completed = True
        return self.cancel_converges

    def close(self):
        self.calls.append("close")
        if self.late_feedback_on_close and self.feedback_callback is not None:
            self.feedback_callback({"state": "executing", "detail": "late private_joint"})


def _status(state, *, error_code=""):
    view = load_capability_catalog(config_path=CONFIG_PATH)
    return {
        "schema_version": 1,
        "robot_name": "so101_single_arm",
        "motion_authorized": True,
        "active_control_mode": "moveit_planning",
        "busy": state == "active",
        "active_task_id": "task-1" if state == "active" else "",
        "default_skill_timeout_sec": 30.0,
        "task_budget_sec": 180.0,
        "rpc_timeout_sec": 0.2,
        "config_digest": view["capability_digest"],
        "request_state": state,
        "request_error_code": error_code,
        "capabilities": [],
    }


def _runtime_snapshot() -> SkillSnapshot:
    config = load_robot_config_dict(CONFIG_PATH)
    return compile_local_snapshot(config, CONFIG_PATH)


def _gateway_status(*, request_state="", request_error_code="", ready=True):
    status = _status(request_state, error_code=request_error_code)
    status["capabilities"] = [
        {
            "name": "move_relative_ee",
            "ready": ready,
            "reason": "" if ready else "CAPABILITY_NOT_READY",
            "required_control_mode": "moveit_planning",
        }
    ]
    return status


def test_cancel_uses_task_id_only_lookup(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _CancelBridge([_status("active"), _status("active"), _status("terminal")])
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(["--config-path", str(CONFIG_PATH), "cancel", "--task-id", "task-1"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["data"]["already_terminal"] is False
    assert bridge.cancel_calls[0][0] == "task-1"
    assert all(call["task_id"] == "task-1" and call["payload_hash"] == "" for call in bridge.status_calls)


def test_cancel_terminal_record_is_idempotent(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _CancelBridge([_status("terminal")])
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(["--config-path", str(CONFIG_PATH), "cancel", "--task-id", "task-1"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["data"]["already_terminal"] is True
    assert bridge.cancel_calls == []


def test_cancel_unknown_record_returns_goal_not_found(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _CancelBridge([_status("")])
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(["--config-path", str(CONFIG_PATH), "cancel", "--task-id", "task-1"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 3
    assert payload["error"]["code"] == "GOAL_NOT_FOUND"
    assert bridge.cancel_calls == []


def test_execute_orders_gateway_checks_and_emits_one_public_result(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _ExecuteBridge([_gateway_status(), _gateway_status()])
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "execute",
            "move_relative_ee",
            "--task-id",
            "task-execute",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "0.03",
        ]
    )

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert exit_code == 0
    assert [event["event"] for event in events] == ["feedback", "result"]
    assert len([event for event in events if event["event"] == "result"]) == 1
    assert events[0]["data"] == {"state": "executing", "detail": "step 1 of 2"}
    assert events[1]["data"]["executed_step_count"] == 2
    assert "primitive" not in json.dumps(events)
    status_calls = [call for call in bridge.calls if isinstance(call, tuple) and call[0] == "status"]
    assert status_calls[0][1]["task_id"] == ""
    assert status_calls[0][1]["payload_hash"] == ""
    assert status_calls[1][1]["task_id"] == "task-execute"
    assert status_calls[1][1]["payload_hash"] == events[1]["payload_hash"]
    assert bridge.calls.index("validate") < bridge.calls.index(status_calls[1])
    assert bridge.calls.index(status_calls[1]) < bridge.calls.index("wait_server") < bridge.calls.index("send")


def test_rejected_goal_uses_ledger_to_report_task_conflict(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _ExecuteBridge(
        [
            _gateway_status(),
            _gateway_status(),
            _gateway_status(request_state="terminal", request_error_code="TASK_ID_CONFLICT"),
        ],
        accepted=False,
    )
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "execute",
            "move_relative_ee",
            "--task-id",
            "task-conflict",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "0.03",
        ]
    )

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert exit_code == 3
    assert events[-1]["event"] == "result"
    assert events[-1]["data"]["error_code"] == "TASK_ID_CONFLICT"


def _install_fake_signal_handlers(monkeypatch, cli, bridge, signal_number):
    handlers = {}

    def install(signum, handler):
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        return previous

    triggered = {"value": False}

    def trigger_signal(_seconds):
        if not triggered["value"]:
            triggered["value"] = True
            handlers[signal_number](signal_number, None)
            assert "cancel_goal" not in bridge.calls

    monkeypatch.setattr(cli.signal, "signal", install)
    monkeypatch.setattr(cli.time, "sleep", trigger_signal)
    bridge.goal_wait_hook = lambda: trigger_signal(0.0)


@pytest.mark.parametrize(("signal_number", "expected_exit"), [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
def test_execute_signal_handler_only_sets_flag_and_waits_for_terminal(
    monkeypatch, capsys, signal_number, expected_exit
):
    from robot_skill_cli import cli

    bridge = _ExecuteBridge([_gateway_status(), _gateway_status()], result_done=False)
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)
    _install_fake_signal_handlers(monkeypatch, cli, bridge, signal_number)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "execute",
            "move_relative_ee",
            "--task-id",
            "task-signal",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "0.03",
        ]
    )

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert exit_code == expected_exit
    assert bridge.calls.count("cancel_goal") == 1
    assert events[-1]["event"] == "result"
    assert events[-1]["data"]["error_code"] == "SKILL_CANCELLED"


def test_execute_reports_unknown_stop_state_when_signal_cancel_does_not_converge(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _ExecuteBridge(
        [_gateway_status(), _gateway_status()],
        result_done=False,
        cancel_converges=False,
    )
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)
    _install_fake_signal_handlers(monkeypatch, cli, bridge, signal.SIGINT)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "execute",
            "move_relative_ee",
            "--task-id",
            "task-unknown-stop",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "0.03",
        ]
    )

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert exit_code == 124
    assert events[-1]["data"]["error_code"] == "SKILL_CANCEL_TIMEOUT"
    assert events[-1]["data"]["message"] == "robot stop state is unknown"


def test_execute_goal_response_timeout_cancels_by_task_id_and_waits_for_terminal(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _ExecuteBridge(
        [
            _gateway_status(),
            _gateway_status(),
            _gateway_status(request_state="active"),
            _gateway_status(request_state="terminal"),
        ],
        goal_response_timeout=True,
    )
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "execute",
            "move_relative_ee",
            "--task-id",
            "task-goal-timeout",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "0.03",
        ]
    )

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    cancel_calls = [call for call in bridge.calls if isinstance(call, tuple) and call[0] == "cancel_task"]
    assert exit_code == 124
    assert cancel_calls[0][1] == "task-goal-timeout"
    assert events[-1]["event"] == "result"
    assert events[-1]["data"]["error_code"] == "RESULT_TIMEOUT"
    assert events[-1]["data"]["message"] == "task reached terminal after goal response timeout"


def test_execute_suppresses_feedback_after_terminal_result(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _ExecuteBridge(
        [_gateway_status(), _gateway_status()],
        late_feedback_on_close=True,
    )
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "execute",
            "move_relative_ee",
            "--task-id",
            "task-late-feedback",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "0.03",
        ]
    )

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert exit_code == 0
    assert [event["event"] for event in events] == ["feedback", "result"]
    assert "private_joint" not in json.dumps(events)


def test_execute_signal_during_goal_response_wait_preserves_signal_exit(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _ExecuteBridge(
        [
            _gateway_status(),
            _gateway_status(),
            _gateway_status(request_state="active"),
            _gateway_status(request_state="terminal"),
        ],
        goal_response_timeout=True,
    )
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)
    _install_fake_signal_handlers(monkeypatch, cli, bridge, signal.SIGINT)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "execute",
            "move_relative_ee",
            "--task-id",
            "task-signal-goal-wait",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "0.03",
        ]
    )

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert exit_code == 130
    assert events[-1]["event"] == "result"
    assert events[-1]["data"]["error_code"] == "SKILL_CANCELLED"


def test_execute_result_transport_error_cancels_and_emits_stable_result(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _ExecuteBridge(
        [
            _gateway_status(),
            _gateway_status(),
            _gateway_status(request_state="active"),
            _gateway_status(request_state="terminal"),
        ],
        result_error=RuntimeError("transport failed"),
    )
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "execute",
            "move_relative_ee",
            "--task-id",
            "task-result-error",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "0.03",
        ]
    )

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    cancel_calls = [call for call in bridge.calls if isinstance(call, tuple) and call[0] == "cancel_task"]
    assert exit_code == 4
    assert cancel_calls[0][1] == "task-result-error"
    assert events[-1]["event"] == "result"
    assert events[-1]["data"]["error_code"] == "ROS_UNAVAILABLE"
    assert events[-1]["data"]["message"] == "terminal result unavailable"
