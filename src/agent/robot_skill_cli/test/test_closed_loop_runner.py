import json
from types import SimpleNamespace

from robot_skill_cli import closed_loop_runner


class _Bridge:
    feedback_callback = None

    def __init__(self, **_kwargs):
        pass

    def start(self):
        return True

    def close(self):
        if self.feedback_callback is not None:
            self.feedback_callback({"detail": "late feedback"})


class _Controller:
    terminals = []

    def __init__(self, bridge, **_kwargs):
        self.bridge = bridge

    def discover(self):
        return {"state": "discovered"}

    def prepare_workflow(self, _raw_command, _steps):
        return {"state": "prepared"}

    def confirm_plan(self):
        return {"state": "confirmed"}

    def execute(self, *, feedback_callback=None):
        self.bridge.feedback_callback = feedback_callback
        if feedback_callback is not None:
            feedback_callback({"detail": "executing"})
        return self.terminals.pop(0)

    def continue_workflow(self, _raw_command, _steps=None, *, resume=False):
        return {"state": "prepared", "resume": resume}


def _install_runner(monkeypatch, terminals):
    _Controller.terminals = list(terminals)
    context = SimpleNamespace(view={"timeout_policy": {"rpc_timeout_sec": 1.0}})
    transport = SimpleNamespace(
        status_service="status",
        snapshot_service="snapshot",
        reload_service="reload",
        validate_skill_service="validate",
        skill_action_name="skill",
        plan_service="plan",
        validate_plan_service="validate_plan",
        confirm_plan_service="confirm_plan",
        execute_plan_action="execute_plan",
    )
    monkeypatch.setattr(closed_loop_runner, "load_runtime_context", lambda **_kwargs: (context, transport))
    monkeypatch.setattr(closed_loop_runner, "RosBridge", _Bridge)
    monkeypatch.setattr(closed_loop_runner, "InteractiveController", _Controller)


def test_runner_suppresses_feedback_after_terminal(monkeypatch, capsys):
    _install_runner(monkeypatch, [{"state": "succeeded"}])

    exit_code = closed_loop_runner.main(["--raw-command", "点头", "--skill", "nod_yes"])

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert exit_code == 0
    assert [event["event"] for event in events][-2:] == ["feedback", "execute_terminal"]
    assert "late feedback" not in json.dumps(events)


def test_runner_returns_failed_terminal_exit(monkeypatch, capsys):
    _install_runner(monkeypatch, [{"state": "failed", "error_code": "CAPABILITY_NOT_READY"}])

    exit_code = closed_loop_runner.main(["--raw-command", "点头", "--skill", "nod_yes"])

    assert exit_code == 13
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["event"] == "execute_terminal"


def test_runner_uses_continuation_terminal_exit(monkeypatch, capsys):
    _install_runner(monkeypatch, [{"state": "stopped"}, {"state": "unknown"}])

    exit_code = closed_loop_runner.main(
        [
            "--raw-command",
            "点头",
            "--skill",
            "nod_yes",
            "--continue-command",
            "继续挥手",
            "--continue-skill",
            "wave_hand",
        ]
    )

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert exit_code == 15
    assert events[-1]["event"] == "continue_terminal"
