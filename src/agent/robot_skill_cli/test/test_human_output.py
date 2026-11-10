import json
import re
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
    monkeypatch.setattr(cli, "_run_online_catalog_command", Mock(return_value=data))
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


def _preset_skills():
    return {
        "robot_name": "unitree_g1_arm_presets",
        "skills": [
            {
                "name": "release_arm",
                "display_name": "释放手臂控制",
                "tags": ["preset", "control_ownership", "unitree_action_id_99"],
                "required_control_mode": "named_motion",
                "summary": "Release arm control; a control-ownership operation, not a gesture.",
            },
            {
                "name": "arms_horizontal",
                "display_name": "双臂平举",
                "tags": ["preset", "unitree_action_id_15"],
                "required_control_mode": "named_motion",
                "summary": "Extend both arms horizontally.",
            },
            {
                "name": "clamp",
                "display_name": "鼓掌",
                "tags": ["preset", "unitree_action_id_17"],
                "required_control_mode": "named_motion",
                "summary": "Clap both hands to applaud with the G1 preset (action 17); no automatic recovery.",
            },
            {
                "name": "wave_hand",
                "display_name": "挥手",
                "required_control_mode": "joint_trajectory",
                "summary": "挥手致意",
            },
        ],
    }


def test_list_skills_shows_chinese_name_vendor_action_id_and_badges(monkeypatch, capsys):
    monkeypatch.setattr(catalog, "load_transport_context", Mock(return_value=(object(), object())))
    monkeypatch.setattr(cli, "_run_online_catalog_command", Mock(return_value=_preset_skills()))
    assert cli.main(["list-skills", "--human"]) == 0
    output = capsys.readouterr().out

    assert "双臂平举（arms_horizontal）" in output
    assert "ID15" in output
    assert "释放手臂控制（release_arm）" in output
    assert "ID99" in output
    assert "控制权操作" in output
    assert "预置" in output
    # Operator-confirmed ID 17 is applause: display text says so while the
    # stable key stays `clamp` for dispatch and exact firmware matching.
    assert "鼓掌（clamp）" in output
    assert "ID17" in output
    assert "Clap both hands" in output
    assert "夹紧" not in output
    # The number is labelled as a vendor action id, not a sequence or request id.
    assert "厂商动作编号" in output
    assert "不是列表序号、请求 ID 或工作流步骤" in output
    assert "不会自动执行、恢复或收尾" in output


def test_list_skills_hides_vendor_ids_for_non_vendor_skills(monkeypatch, capsys):
    skills = {
        "robot_name": "so101",
        "skills": [{"name": "wave_hand", "display_name": "挥手", "summary": "挥手致意"}],
    }
    monkeypatch.setattr(catalog, "load_transport_context", Mock(return_value=(object(), object())))
    monkeypatch.setattr(cli, "_run_online_catalog_command", Mock(return_value=skills))
    assert cli.main(["list-skills", "--human"]) == 0
    output = capsys.readouterr().out

    assert "挥手（wave_hand）" in output
    assert not re.search(r"ID\d", output)
    assert "厂商动作编号" not in output
    assert "控制权操作" not in output


def test_list_skills_json_keeps_display_tags_machine_readable(monkeypatch, capsys):
    data = _preset_skills()
    monkeypatch.setattr(catalog, "load_transport_context", Mock(return_value=(object(), object())))
    monkeypatch.setattr(cli, "_run_online_catalog_command", Mock(return_value=data))
    assert cli.main(["list-skills", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["data"] == data
    skills = {skill["name"]: skill for skill in payload["data"]["skills"]}
    assert skills["arms_horizontal"]["display_name"] == "双臂平举"
    assert "unitree_action_id_15" in skills["arms_horizontal"]["tags"]
    assert skills["clamp"]["display_name"] == "鼓掌"
    assert "unitree_action_id_17" in skills["clamp"]["tags"]
    assert "clap" in skills["clamp"]["summary"].lower()
    assert "夹紧" not in skills["clamp"]["summary"]
    assert "control_ownership" in skills["release_arm"]["tags"]
    # The tag is traceable display data; the adapter adds no execution field.
    assert "action_id" not in skills["arms_horizontal"]
