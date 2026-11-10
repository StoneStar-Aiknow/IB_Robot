import json
from unittest.mock import Mock

import pytest

from robot_skill_cli import catalog, cli, human_output


@pytest.mark.parametrize(
    "tty,requested,managed,expected",
    [
        (True, None, False, True),
        (False, None, False, False),
        (True, "json", False, False),
        (False, "human", False, True),
        (True, None, True, False),
    ],
)
def test_output_selection(monkeypatch, tty, requested, managed, expected):
    monkeypatch.setattr(human_output.sys.stdout, "isatty", lambda: tty)
    monkeypatch.setenv("IBROBOT_HERMES_MANAGED_REQUIRED", "1" if managed else "0")
    assert human_output.use_human_output("status", requested) is expected
    assert not human_output.use_human_output("run-workflow", requested)


@pytest.mark.parametrize("command", ["status", "list-skills"])
@pytest.mark.parametrize("flag", ["--json", "--human"])
def test_cli_uses_format_without_changing_data_or_exit(monkeypatch, capsys, command, flag):
    data = {
        "robot_name": "test_robot",
        "skills": [{"name": "wave", "summary": "physical behavior unverified."}],
        "capabilities": [{"name": "wave", "ready": False, "reason": "STOP_LATCHED"}],
        "busy": False,
        "motion_authorized": True,
        "control_plane_state": "READY",
    }
    # The command is bound to a configured deployment; patch the loader and the
    # Gateway runner so the formatting contract is tested without a live graph.
    monkeypatch.setattr(catalog, "load_catalog_context", Mock(return_value=object()))
    monkeypatch.setattr(catalog, "load_transport_context", Mock(return_value=(object(), object())))
    monkeypatch.setattr(cli, "_run_catalog_command", Mock(return_value=data))
    monkeypatch.setattr(cli, "_run_runtime_command", Mock(return_value=data))
    assert cli.main([command, flag]) == 0
    output = capsys.readouterr().out
    if flag == "--json":
        assert json.loads(output)["data"] == data
    else:
        assert "机器人：test_robot" in output and "wave" in output
        assert ("STOP_LATCHED" if command == "status" else "unverified") in output


def test_human_errors_keep_code_message_and_exit(monkeypatch, capsys):
    monkeypatch.setattr(catalog, "load_transport_context", Mock(side_effect=ValueError("missing snapshot")))
    assert cli.main(["status", "--human"]) == cli.EXIT_INVALID_INPUT
    assert "INVALID_ARGUMENT" in capsys.readouterr().out


def test_tables_keep_unicode_reasons_and_remove_terminal_controls():
    text = human_output.render(
        {
            "ok": True,
            "command": "list-skills",
            "data": {
                "robot_name": "测试",
                "skills": [{"name": "wave", "summary": "未验证\n\x1b[31m厂商动作"}],
            },
        }
    )
    assert "未验证" in text and "厂商动作" in text and "\x1b" not in text


def test_empty_table_and_missing_boolean_are_not_reported_ready():
    text = human_output.render({"ok": True, "command": "status", "data": {}})
    assert "（无条目）" in text and "就绪：未知" in text
