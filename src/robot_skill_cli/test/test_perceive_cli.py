"""Unit tests for the ibrobot-perceive allowlist reader.

These tests mock ``subprocess.run`` so no ROS stack is required. The audit log
path is redirected to a tmp_path so tests do not pollute /tmp.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

from robot_skill_cli import perceive_cli


def _ok_run(stdout: str) -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout=stdout, stderr="")


def test_rejects_topic_not_in_allowlist(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(perceive_cli, "LOG_PATH", tmp_path / "perceive.log")
    code = perceive_cli.main(["--topic", "/cmd_vel", "--field", "linear"])
    assert code == 1
    err = capsys.readouterr().err
    assert "not in perception allowlist" in err
    assert "/cmd_vel" in err


def test_rejects_field_not_allowed(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(perceive_cli, "LOG_PATH", tmp_path / "perceive.log")
    code = perceive_cli.main(["--topic", "/voice/speech_direction", "--field", "stamp"])
    assert code == 1
    err = capsys.readouterr().err
    assert "not allowed for /voice/speech_direction" in err
    assert "azimuth_rad" in err
    assert "seq_id" in err


def test_successful_read_prints_literal(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(perceive_cli, "LOG_PATH", tmp_path / "perceive.log")
    yaml_out = "---\nazimuth_rad: 0.5236\nseq_id: 42\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _ok_run(yaml_out))

    code = perceive_cli.main(["--topic", "/voice/speech_direction", "--field", "azimuth_rad"])

    assert code == 0
    out = capsys.readouterr().out.strip()
    assert out == "0.5236"
    log = (tmp_path / "perceive.log").read_text(encoding="utf-8")
    assert "status=ok" in log
    assert "0.5236" in log


def test_timeout_returns_error(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(perceive_cli, "LOG_PATH", tmp_path / "perceive.log")

    def _raise(*a, **k):
        raise subprocess.TimeoutExpired(cmd="ros2", timeout=5)

    monkeypatch.setattr(subprocess, "run", _raise)
    code = perceive_cli.main(["--topic", "/voice/speech_direction", "--field", "azimuth_rad"])
    assert code == 1
    assert "timed out" in capsys.readouterr().err
    assert "timeout" in (tmp_path / "perceive.log").read_text(encoding="utf-8")


def test_ros2_error_returns_error(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(perceive_cli, "LOG_PATH", tmp_path / "perceive.log")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="topic not found\n")
    )
    code = perceive_cli.main(["--topic", "/voice/speech_direction", "--field", "azimuth_rad"])
    assert code == 1
    assert "ros2 topic echo failed" in capsys.readouterr().err


def test_parse_error_returns_error(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(perceive_cli, "LOG_PATH", tmp_path / "perceive.log")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _ok_run("not: : valid: yaml: ["))
    code = perceive_cli.main(["--topic", "/voice/speech_direction", "--field", "azimuth_rad"])
    assert code == 1
    assert "failed to parse" in capsys.readouterr().err


def test_missing_field_returns_error(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(perceive_cli, "LOG_PATH", tmp_path / "perceive.log")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _ok_run("---\nseq_id: 7\n"))
    code = perceive_cli.main(["--topic", "/voice/speech_direction", "--field", "azimuth_rad"])
    assert code == 1
    assert "not found" in capsys.readouterr().err


def test_allowlist_is_hardcoded_and_not_configurable() -> None:
    assert "/voice/speech_direction" in perceive_cli.PERCEPTION_ALLOWLIST
    assert perceive_cli.PERCEPTION_ALLOWLIST["/voice/speech_direction"] == {"azimuth_rad", "seq_id"}
    assert "/cmd_vel" not in perceive_cli.PERCEPTION_ALLOWLIST
    assert "/joint_states" not in perceive_cli.PERCEPTION_ALLOWLIST


def test_extract_field_handles_multiple_yaml_documents() -> None:
    stdout = "---\nseq_id: 1\n---\nazimuth_rad: 0.1\nseq_id: 2\n"
    assert perceive_cli._extract_field(stdout, "azimuth_rad") == 0.1
