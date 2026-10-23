"""Unit tests for the ibrobot-block-raw-ros pre_tool_call hook.

The hook is a script without a .py extension, so it is loaded via importlib.
Tests cover tokenization, direct/indirect ros2 detection, and the end-to-end
main() stdin/stdout JSON protocol. No ROS stack is required.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import io
import json
import sys
from pathlib import Path

_HOOK_PATH = Path(__file__).resolve().parents[1] / "resource" / "hermes" / "hooks" / "ibrobot-block-raw-ros"


def _load_hook_module():
    loader = importlib.machinery.SourceFileLoader("ibrobot_block_raw_ros", str(_HOOK_PATH))
    spec = importlib.util.spec_from_file_location("ibrobot_block_raw_ros", _HOOK_PATH, loader=loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hook = _load_hook_module()


# --- helper function tests -------------------------------------------------


def test_has_direct_ros2_bare() -> None:
    assert hook._has_direct_ros2(["ros2", "topic", "echo"]) is True


def test_has_direct_ros2_path_prefixed() -> None:
    assert hook._has_direct_ros2(["/opt/ros/humble/bin/ros2", "topic", "list"]) is True


def test_has_direct_ros2_negative() -> None:
    assert hook._has_direct_ros2(["echo", "hello"]) is False
    assert hook._has_direct_ros2(["rosbag2", "info"]) is False


def test_has_indirect_marker_rclpy() -> None:
    assert hook._has_indirect_marker("python3 -c 'import rclpy'") is True


def test_has_indirect_marker_roslaunch() -> None:
    assert hook._has_indirect_marker("roslaunch my_pkg my.launch") is True


def test_has_indirect_marker_ros2_substring() -> None:
    assert hook._has_indirect_marker("bash -c 'ros2 topic pub ...'") is True


def test_has_indirect_marker_ros2cli_module() -> None:
    assert hook._has_indirect_marker("python3 -m ros2cli topic echo /joint_states") is True


def test_has_indirect_marker_ros2cli_import() -> None:
    assert hook._has_indirect_marker("python3 -c 'import ros2cli'") is True


def test_has_indirect_marker_empty_or_clean() -> None:
    assert hook._has_indirect_marker("") is False
    assert hook._has_indirect_marker("ls -la") is False
    assert hook._has_indirect_marker("robot-skill list-skills") is False


def test_tokenize_string() -> None:
    assert hook._tokenize("ros2 topic echo --once /t") == ["ros2", "topic", "echo", "--once", "/t"]


def test_tokenize_list() -> None:
    assert hook._tokenize(["ros2 topic echo", "/t"]) == ["ros2", "topic", "echo", "/t"]


def test_tokenize_malformed_falls_back_to_split() -> None:
    tokens = hook._tokenize("echo 'unterminated")
    assert "echo" in tokens


def test_extract_command_present() -> None:
    assert hook._extract_command({"tool_input": {"command": "ros2 topic echo"}}) == "ros2 topic echo"


def test_extract_command_missing() -> None:
    assert hook._extract_command({"tool_input": {}}) is None
    assert hook._extract_command({}) is None


# --- main() protocol tests -------------------------------------------------


def _run_main(payload: dict, capsys, monkeypatch) -> tuple[int, str]:
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    code = 0
    try:
        hook.main()
    except SystemExit as exc:
        code = exc.code
    out = capsys.readouterr().out.strip()
    return code, out


def test_main_blocks_direct_ros2(capsys, monkeypatch) -> None:
    code, out = _run_main(
        {"tool_input": {"command": "ros2 topic echo --once /voice/speech_direction"}}, capsys, monkeypatch
    )
    assert code == 0
    parsed = json.loads(out)
    assert parsed["action"] == "block"
    assert "ibrobot-perceive" in parsed["message"]


def test_main_blocks_path_prefixed_ros2(capsys, monkeypatch) -> None:
    code, out = _run_main({"tool_input": {"command": "/opt/ros/humble/bin/ros2 param list"}}, capsys, monkeypatch)
    assert code == 0
    assert json.loads(out)["action"] == "block"


def test_main_blocks_rclpy_import(capsys, monkeypatch) -> None:
    code, out = _run_main({"tool_input": {"command": "python3 -c 'import rclpy'"}}, capsys, monkeypatch)
    assert code == 0
    assert json.loads(out)["action"] == "block"


def test_main_blocks_ros2cli_module_form(capsys, monkeypatch) -> None:
    code, out = _run_main(
        {"tool_input": {"command": "python3 -m ros2cli topic echo /joint_states"}}, capsys, monkeypatch
    )
    assert code == 0
    parsed = json.loads(out)
    assert parsed["action"] == "block"
    assert "ibrobot-perceive" in parsed["message"]


def test_main_blocks_command_list(capsys, monkeypatch) -> None:
    code, out = _run_main({"tool_input": {"command": ["bash", "-c", "ros2 control list"]}}, capsys, monkeypatch)
    assert code == 0
    assert json.loads(out)["action"] == "block"


def test_main_allows_non_ros_command(capsys, monkeypatch) -> None:
    code, out = _run_main({"tool_input": {"command": "robot-skill list-skills"}}, capsys, monkeypatch)
    assert code == 0
    assert out == ""


def test_main_allows_when_command_missing(capsys, monkeypatch) -> None:
    code, out = _run_main({"tool_input": {}}, capsys, monkeypatch)
    assert code == 0
    assert out == ""


def test_main_fails_open_on_invalid_json(capsys, monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("not json{"))
    code = 0
    try:
        hook.main()
    except SystemExit as exc:
        code = exc.code
    assert code == 0
    assert capsys.readouterr().out.strip() == ""
