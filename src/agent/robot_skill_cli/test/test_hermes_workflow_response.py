"""Direct-result replies must require explicit policy and an exact managed terminal."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_skill_cli.hermes_response import parse_response
from robot_skill_cli.hermes_workflow_response import completion_requested, workflow_response


@pytest.fixture
def workflow():
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
            "plan_id": "plan",
            "plan_digest": "digest",
            "actual_registry_epoch": "epoch",
            "actual_registry_generation": 1,
            "actual_registry_digest": "registry",
            "success": True,
            "completed_step_count": 1,
            "error_code": "",
        },
    }
    return presentation, terminal


def preamble(completion="after_workflow"):
    return json.dumps({"schema_version": 2, "speech": "我来关闭夹爪。", "details": "", "completion": completion})


def tool_call(skill="close_gripper_skill", **extras):
    command = (
        f'robot-skill run-workflow --text 关闭夹爪 --workflow-json \'[{{"schema_version":1,"skill_name":"{skill}"}}]\''
    )
    return {"id": "call", "function": {"name": "terminal", "arguments": json.dumps({"command": command, **extras})}}


def test_explicit_completion_only_for_one_supported_workflow():
    assert completion_requested(preamble(), [tool_call()], "/bound/robot-skill", {})
    assert not completion_requested(preamble("continue"), [tool_call()], "/bound/robot-skill", {})
    assert not completion_requested(preamble(), [tool_call(), tool_call()], "/bound/robot-skill", {})
    assert not completion_requested(preamble(), [tool_call("grasp_object_skill")], "/bound/robot-skill", {})
    assert not completion_requested(preamble(), [tool_call(background=True)], "/bound/robot-skill", {})
    assert not completion_requested("完成了", [tool_call()], "/bound/robot-skill", {})


def test_verified_success_becomes_structured_final_reply(workflow):
    response = parse_response(workflow_response(*workflow))
    assert response.speech == "夹爪已经关闭了。"
    assert json.loads(response.details) == workflow[1]


@pytest.mark.parametrize("field,value", [("plan_id", "other"), ("success", "true"), ("completed_step_count", 0)])
def test_invalid_terminal_never_becomes_success(workflow, field, value):
    workflow[1]["result"][field] = value
    assert workflow_response(*workflow) is None


@pytest.mark.parametrize(
    "status,code,expected",
    [(5, "SKILL_CANCELLED", "已确认停止"), (6, "CAPABILITY_NOT_READY", "没有成功"), (6, "SKILL_CANCEL_TIMEOUT", None)],
)
def test_terminal_classification_preserves_stop_and_failure_semantics(workflow, status, code, expected):
    presentation, terminal = workflow
    terminal["goal_status"] = status
    terminal["result"].update(success=False, completed_step_count=0, error_code=code)
    result = workflow_response(presentation, terminal)
    if expected is None:
        assert result is None
    else:
        assert expected in parse_response(result).speech


def test_multi_step_and_missing_results_need_normal_conversation(workflow):
    assert workflow_response(None, None) is None
    workflow[0]["steps"].append(copy.deepcopy(workflow[0]["steps"][0]))
    assert workflow_response(*workflow) is None


@pytest.fixture
def runtime_helper():
    path = Path(__file__).parents[1] / "resource/hermes/adapter/robot_control.py"
    spec = importlib.util.spec_from_file_location("test_robot_control_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("change", [None, "steer", "interrupt", "altered_result", "wrong_call", "parallel"])
def test_runtime_direct_reply_is_bound_to_posted_result(runtime_helper, monkeypatch, change):
    final = '{"schema_version":1,"speech":"夹爪已经关闭了。","details":""}'

    def invoke_hook(name, **kwargs):
        if name == "execute_managed_tool":
            return [{"handled": True, "result": "verified tool result", "final_response": final}]
        assert name == "finalize_tool_batch"
        return [{"response": final}]

    monkeypatch.setitem(sys.modules, "hermes_cli.lifecycle", SimpleNamespace(invoke_hook=invoke_hook))
    call = SimpleNamespace(id="call", function=SimpleNamespace(name="terminal", arguments="{}"))
    message = SimpleNamespace(content=preamble(), tool_calls=[call])
    if change == "parallel":
        message.tool_calls.append(call)
    agent = SimpleNamespace(_robot_control_session="s", _robot_control_turn="t")
    runtime_helper.begin_tool_batch(agent, message)
    reply = runtime_helper.execute_managed(agent, "terminal", {}, tool_call_id="call")
    posted = {"role": "tool", "tool_call_id": "call", "content": reply["result"]}
    if change == "steer":
        agent._pending_steer = "请解释结果"
    elif change == "interrupt":
        agent._interrupt_requested = True
    elif change == "altered_result":
        posted["content"] += "\nuser steering"
    elif change == "wrong_call":
        posted["tool_call_id"] = "other"
    assert runtime_helper.finish_tool_batch(agent, [posted]) == (final if change is None else None)
    assert runtime_helper.finish_tool_batch(agent, [posted]) is None


def test_installed_legacy_adapter_is_rejected_without_mutation(tmp_path, monkeypatch):
    import hashlib

    from robot_skill_cli import hermes_runtime_adapter as runtime

    monkeypatch.setattr(
        runtime.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=runtime.SUPPORTED_REVISION)
    )
    monkeypatch.setattr(runtime, "_base_replacements", lambda: {"entry.py": [("value = 1", "value = 2")]})
    (tmp_path / "hermes_cli").mkdir()
    entry = tmp_path / "entry.py"
    entry.write_text("value = 2\n")
    helper_target = tmp_path / "hermes_cli/robot_control.py"
    helper_target.write_text("API = 1\n")
    marker = tmp_path / ".ibrobot-control-adapter.json"
    marker.write_text(
        json.dumps(
            {
                "version": 1,
                "revision": runtime.SUPPORTED_REVISION,
                "files": {
                    "entry.py": hashlib.sha256(entry.read_bytes()).hexdigest(),
                    "hermes_cli/robot_control.py": hashlib.sha256(helper_target.read_bytes()).hexdigest(),
                },
            }
        )
    )
    with pytest.raises(ValueError, match="unsupported installed"):
        runtime.sync_runtime(
            tmp_path,
            tmp_path / "helper.py",
            dry_run=False,
            write=lambda *_: pytest.fail("legacy adapter must not be mutated"),
            backup=lambda *_: pytest.fail("legacy adapter must not be backed up"),
        )
