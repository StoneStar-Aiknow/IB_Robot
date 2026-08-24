from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from robot_skill_cli import hermes_lifecycle_speech as speech


def _payload(event: str, command: str, *, result: str | None = None) -> dict:
    extra = {"turn_id": "turn-1"}
    if result is not None:
        extra.update({"status": "ok", "result": result})
    return {
        "hook_event_name": event,
        "session_id": "session-1",
        "tool_name": "terminal",
        "tool_input": {"command": command},
        "extra": extra,
    }


def test_event_gating() -> None:
    assert speech._event(_payload("pre_tool_call", "robot-skill status")) == "status_check_started"
    assert speech._event(_payload("pre_tool_call", "robot-skill plan-workflow --text x")) == "planning_started"
    success = '{"ok":true,"command":"confirm-plan","data":{"confirmed":true},"error":null}'
    assert (
        speech._event(_payload("post_tool_call", "robot-skill confirm-plan --x", result=success)) == "plan_authorized"
    )
    assert (
        speech._event(_payload("post_tool_call", "robot-skill confirm-plan --x", result='{"confirmed":false}')) is None
    )
    wrapped = f"Process exited with code 0\nFinal output:\n{success}"
    assert (
        speech._event(_payload("post_tool_call", "robot-skill confirm-plan --x", result=wrapped)) == "plan_authorized"
    )


def test_confirmation_requires_successful_confirm_plan_envelope() -> None:
    assert not speech._result_success(
        {"ok": False, "command": "confirm-plan", "data": {"confirmed": True}, "error": {"code": "FAILED"}}
    )
    assert not speech._result_success({"ok": True, "command": "execute-plan", "data": {"success": True}, "error": None})
    assert not speech._result_success('log line {"success":true}\n{"ok":false,"command":"confirm-plan","data":null}')


def test_handle_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("IBROBOT_LIFECYCLE_SPEECH_STATE", str(tmp_path))
    spawned = []
    monkeypatch.setattr(speech, "_spawn_play", lambda *args: spawned.append(args))
    payload = _payload("pre_tool_call", "robot-skill status")
    speech.handle(payload)
    speech.handle(payload)
    assert len(spawned) == 1


def test_pre_llm_remembers_task_without_starting_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("IBROBOT_LIFECYCLE_SPEECH_STATE", str(tmp_path))
    generated = []
    monkeypatch.setattr(speech, "_spawn_model", lambda *args: generated.append(args))
    payload = {
        "hook_event_name": "pre_llm_call",
        "session_id": "session-1",
        "extra": {"turn_id": "turn-1", "user_message": "打开夹爪"},
    }
    speech.handle(payload)
    assert generated == []

    speech.handle(_payload("pre_tool_call", "robot-skill status"))
    assert generated[0][-1] == "打开夹爪"


def test_model_generation_is_spawned_once_per_turn(tmp_path: Path, monkeypatch) -> None:
    spawned = []
    monkeypatch.setattr(speech.subprocess, "Popen", lambda *args, **kwargs: spawned.append((args, kwargs)))

    speech._spawn_model(tmp_path, "turn-1", "打开夹爪")
    speech._spawn_model(tmp_path, "turn-1", "打开夹爪")

    assert len(spawned) == 1


def test_hook_entrypoint_accepts_payload(tmp_path: Path) -> None:
    env = {**os.environ, "IBROBOT_LIFECYCLE_SPEECH_STATE": str(tmp_path), "PYTHONPATH": "src/robot_skill_cli"}
    payload = _payload("pre_tool_call", "robot-skill status")
    result = subprocess.run(
        [sys.executable, "-m", "robot_skill_cli.hermes_lifecycle_speech"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
    )
    assert result.returncode == 0
