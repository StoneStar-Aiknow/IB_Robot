"""The public CLI has no configuration-free dispatch anymore."""

from __future__ import annotations

import json

import pytest

from robot_skill_cli import cli


@pytest.mark.parametrize(
    "argv",
    [
        ["status", "--json"],
        ["list-skills", "--json"],
        ["cancel", "--task-id", "task-1"],
        ["list-games"],
    ],
)
def test_online_commands_need_an_explicit_deployment_selector(argv, monkeypatch, capsys):
    monkeypatch.delenv("ROBOT_CONFIG", raising=False)
    monkeypatch.delenv("ROBOT_NAME", raising=False)
    assert cli.main(argv) == cli.EXIT_INVALID_INPUT
    payload = json.loads(capsys.readouterr().out or "{}")
    message = payload.get("error", {}).get("message", "")
    assert "Robot configuration name or path is required" in message
    assert "ROBOT_CONFIG" in message


@pytest.mark.parametrize(
    "argv",
    [
        ["discover"],
        ["connect", "g1"],
        ["disconnect"],
        ["--target", "g1", "status"],
    ],
)
def test_removed_discovery_commands_are_not_accepted(argv, monkeypatch, capsys):
    monkeypatch.delenv("ROBOT_CONFIG", raising=False)
    monkeypatch.delenv("ROBOT_NAME", raising=False)
    assert cli.main(argv) == cli.EXIT_INVALID_INPUT
    output = capsys.readouterr().out
    assert "discover" not in output.split("choose from", 1)[-1] or "invalid choice" in output
    assert "未配置" not in output


def test_help_no_longer_lists_discovery_commands(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "discover" not in output
    assert "--target" not in output
    assert "--config-name" in output and "--config-path" in output


def test_configuration_selectors_are_mutually_exclusive(monkeypatch, capsys):
    monkeypatch.delenv("ROBOT_CONFIG", raising=False)
    monkeypatch.delenv("ROBOT_NAME", raising=False)
    assert cli.main(["--config-name", "a", "--config-path", "/tmp/b.yaml", "status"]) == cli.EXIT_INVALID_INPUT
    assert "not allowed with argument" in capsys.readouterr().out
