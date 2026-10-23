"""Run with Hermes' scripts/run_tests.sh in a patched isolated Hermes checkout.

The checkout must contain the generated plugin fixtures under tests/ibrobot_plugins.
Only the model transport and robot child are test doubles; the conversation loop,
plugin discovery, managed execution, response hooks and SessionDB are real.

The test is intentionally skipped unless ``HERMES_HOME`` points at an isolated
Hermes checkout. It must never silently import the developer's live Hermes profile.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("hermes_cli", reason="requires an isolated, patched Hermes checkout")


def _require_isolated_hermes_home() -> Path:
    raw_home = os.environ.get("HERMES_HOME", "")
    if not raw_home:
        pytest.skip("requires HERMES_HOME for an isolated Hermes checkout")
    home = Path(raw_home).expanduser().resolve()
    if not (home / "config.yaml").is_file() or not (home / "plugins").is_dir():
        pytest.skip("HERMES_HOME must contain config.yaml and plugins/")
    return home


def _require_executable_tmp_path(tmp_path: Path) -> None:
    probe = tmp_path / "exec-probe"
    probe.write_text(f"#!{sys.executable}\nraise SystemExit(0)\n", encoding="utf-8")
    probe.chmod(0o700)
    try:
        subprocess.run([str(probe)], check=True, capture_output=True, timeout=2)
    except OSError as exc:
        if exc.errno == errno.EACCES:
            pytest.skip("temporary test directory is mounted noexec")
        raise


@pytest.mark.parametrize(
    "policy,valid,stop,expected_calls",
    [
        ("after_workflow", True, False, 1),
        ("continue", True, False, 2),
        ("after_workflow", False, False, 2),
        ("after_workflow", True, True, 2),
    ],
)
def test_model_preamble_and_verified_terminal_close_the_real_turn(
    tmp_path, monkeypatch, policy, valid, stop, expected_calls
):
    _require_executable_tmp_path(tmp_path)
    import hermes_cli.plugins as plugins
    from hermes_state import SessionDB
    from run_agent import AIAgent

    fixture = Path(__file__).parents[1] / "ibrobot_plugins"
    assert fixture.is_dir(), "generate plugin fixtures in the isolated checkout first"
    profile = _require_isolated_hermes_home()
    shutil.copytree(fixture, profile / "plugins", dirs_exist_ok=True)
    monkeypatch.setenv("IBROBOT_SPEECH_STATE", str(tmp_path / "speech-state"))
    speech = tmp_path / "speech.jsonl"
    speaker = tmp_path / "speak"
    speaker.write_text(
        f"#!{sys.executable}\nimport sys,json\n"
        f"with open({str(speech)!r}, 'a') as f: f.write(json.dumps(json.load(sys.stdin),ensure_ascii=False)+'\\n')\n"
    )
    speaker.chmod(0o700)
    presentation = {
        "task_id": "task",
        "plan_id": "plan",
        "plan_digest": "digest",
        "registry_identity": {"registry_epoch": "epoch", "registry_generation": 1, "registry_digest": "registry"},
        "steps": [{"schema_version": 1, "skill_name": "close_gripper_skill"}],
    }
    terminal = {
        "task_id": "task",
        "state": "succeeded",
        "goal_status": 4,
        "error_code": "",
        "result": {
            "plan_id": "plan" if valid else "wrong",
            "plan_digest": "digest",
            "actual_registry_epoch": "epoch",
            "actual_registry_generation": 1,
            "actual_registry_digest": "registry",
            "success": True,
            "completed_step_count": 1,
            "error_code": "",
        },
    }
    if stop:
        terminal.update(state="stopped", goal_status=5, error_code="SKILL_CANCELLED")
        terminal["result"].update(success=False, completed_step_count=0, error_code="SKILL_CANCELLED")
    presentation_event = {"command": "run-workflow", "event": "workflow_presentation", "data": presentation}
    terminal_event = {"command": "run-workflow", "event": "workflow_terminal", "data": terminal}
    child = tmp_path / "robot-skill"
    ready = tmp_path / "goal-ready"
    stop_wait = (
        (
            f"Path({str(ready)!r}).touch()\n"
            "import hashlib,os\n"
            "record = Path(os.environ['IBROBOT_STOP_ROOT']) / (hashlib.sha256(os.environ['IBROBOT_STOP_SESSION'].encode()).hexdigest()+'.json')\n"
            "while not json.loads(record.read_text())['owner']['stop_requested'] and time.monotonic()<deadline: time.sleep(0.01)\n"
            "assert json.loads(record.read_text())['owner']['stop_requested']\n"
            "time.sleep(1.5)\n"
        )
        if stop
        else ""
    )
    child.write_text(
        f"#!{sys.executable}\nimport json,time\nfrom pathlib import Path\n"
        f"print(json.dumps({presentation_event!r}),flush=True)\n"
        # Wait for the asynchronous preamble observer, rather than timing a sleep.
        "deadline=time.monotonic()+10\n"
        f"while not Path({str(speech)!r}).exists() and time.monotonic()<deadline: time.sleep(0.01)\n"
        f"{stop_wait}"
        f"print(json.dumps({terminal_event!r}),flush=True)\n"
    )
    child.chmod(0o700)
    import yaml

    (profile / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "plugins": {
                    "enabled": ["ibrobot-motion", "ibrobot-response"],
                    "entries": {
                        "ibrobot-motion": {
                            "settings": {"state_dir": str(tmp_path / "control"), "executable": str(child)}
                        },
                        "ibrobot-response": {"settings": {"speaker": str(speaker)}},
                    },
                },
                "display": {"streaming": False},
                "streaming": {"enabled": False},
                "auxiliary": {"title_generation": {"enabled": False}},
            }
        )
    )
    plugins._reset_plugin_managers_for_tests()
    plugins.discover_plugins(force=True)
    assert plugins.has_hook("finalize_tool_batch")
    assert plugins.has_hook("on_interim_message")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Run a command",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            },
        }
    ]
    with (
        patch("run_agent.get_tool_definitions", return_value=tools),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1/",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=3,
        )
    agent._cached_system_prompt = "Use the response protocol."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.valid_tool_names = {"terminal"}
    agent.client = MagicMock()
    content = json.dumps({"schema_version": 2, "speech": "好的，我来关闭夹爪。", "details": "", "completion": policy})
    command = 'robot-skill run-workflow --text 关闭夹爪 --workflow-json \'[{"schema_version":1,"skill_name":"close_gripper_skill"}]\''
    call = SimpleNamespace(
        id="call",
        type="function",
        function=SimpleNamespace(name="terminal", arguments=json.dumps({"command": command})),
    )

    def response(text, tool_calls=None):
        return SimpleNamespace(
            model="test/model",
            usage=None,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=text, tool_calls=tool_calls),
                    finish_reason="tool_calls" if tool_calls else "stop",
                )
            ],
        )

    fallback_speech = "任务已取消。" if stop else "结果需要进一步确认。"
    fallback = json.dumps({"schema_version": 1, "speech": fallback_speech, "details": ""})
    agent.client.chat.completions.create.side_effect = [response(content, [call]), response(fallback)]
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="speech-e2e", source="cli", model="test/model")
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = "speech-e2e"
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = None
    agent._persist_disabled = False
    stop_results = []

    def send_stop():
        from hermes_cli.robot_control import control_input

        deadline = time.monotonic() + 15.0
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        if ready.exists():
            stop_results.append(control_input("别动", agent.session_id, busy=True, agent=agent))

    stopper = threading.Thread(target=send_stop, daemon=True) if stop else None
    try:
        if stopper:
            stopper.start()
        with patch.object(agent, "_save_trajectory"), patch.object(agent, "_cleanup_task_resources"):
            result = agent.run_conversation("关闭夹爪")
        if stopper:
            stopper.join(timeout=1.0)
            assert stop_results[0]["handled"] is True
            assert "SKILL_CANCELLED" in str(result["messages"])
        assert result["api_calls"] == expected_calls, result
        assert agent.client.chat.completions.create.call_count == expected_calls
        durable = db.get_messages_as_conversation("speech-e2e")
        assert [row["role"] for row in durable] == ["user", "assistant", "tool", "assistant"]
        assert durable[2]["tool_call_id"] == "call"
        expected = "夹爪已经关闭了。" if expected_calls == 1 else fallback_speech
        assert expected in result["final_response"]
        assert json.loads(durable[-1]["content"])["speech"] == expected
        events = [json.loads(line) for line in speech.read_text().splitlines()]
        assert [event["phase"] for event in events] == ["interim", "final"]
        assert events[0]["response"]["speech"] == "好的，我来关闭夹爪。"
        assert events[1]["response"]["speech"] == expected
        assert events[0]["turn_id"] == events[1]["turn_id"]
        if expected_calls == 1:
            assert result["turn_exit_reason"] == "managed_tool_response"
    finally:
        db.close()
