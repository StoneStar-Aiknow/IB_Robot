from __future__ import annotations

import errno

import pytest

from robot_skill_web import app

ENDPOINT_ENV = "IBROBOT_EXECUTION_TOKEN_ROBOT_SKILL_CLI"


@pytest.fixture
def server_state():
    return {
        "session": {
            "connection_id": "connection-1",
            "connection_version": 1,
            "robot_id": "g1",
            "robot_name": "g1",
            "caller_scope": "robot_skill_cli",
            "target_key": "scope:binding",
            "read_only": False,
        }
    }


@pytest.fixture
def server_calls(monkeypatch, server_state):
    calls = []

    class FakeClient:
        @property
        def session(self):
            return server_state["session"]

    class FakeState:
        client = FakeClient()

    class FakeServer:
        state = FakeState()

        def __init__(self, **kwargs):
            calls.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def serve_forever(self):
            calls.append("serve")

    monkeypatch.setattr(app, "WebServer", FakeServer)
    return calls


@pytest.mark.parametrize(
    "args, expected",
    [
        ([], {"config_name": None, "config_path": None, "address": "127.0.0.1", "port": 8765}),
        (["--port", "8766"], {"config_name": None, "config_path": None, "address": "127.0.0.1", "port": 8766}),
        (
            ["--config-name", "unitree_g1"],
            {"config_name": "unitree_g1", "config_path": None, "address": "127.0.0.1", "port": 8765},
        ),
        (
            ["--config-path", "/tmp/robot.yaml"],
            {"config_name": None, "config_path": "/tmp/robot.yaml", "address": "127.0.0.1", "port": 8765},
        ),
    ],
)
def test_web_binds_the_deployment_configuration(args, expected, monkeypatch, capsys, server_calls):
    monkeypatch.setenv(ENDPOINT_ENV, "0123456789abcdef" * 4)
    assert app.main(args) == 0
    assert server_calls == [expected, "serve"]
    output = capsys.readouterr().out
    assert "已绑定部署配置：g1" in output
    assert "页面不发现、不切换机器人" in output
    assert ENDPOINT_ENV not in output


def test_removed_mode_switch_is_not_accepted(monkeypatch, capsys, server_calls):
    with pytest.raises(SystemExit) as error:
        app.main(["--shared-key-env", ENDPOINT_ENV])
    assert error.value.code == 2
    assert not server_calls
    assert not capsys.readouterr().out


def test_config_selectors_are_mutually_exclusive(monkeypatch, server_calls):
    with pytest.raises(SystemExit) as error:
        app.main(["--config-name", "a", "--config-path", "/tmp/b.yaml"])
    assert error.value.code == 2
    assert not server_calls


@pytest.mark.parametrize("args", [["--host", "0.0.0.0"], ["--port", "0"], ["--port", "65536"]])
def test_bind_constraints_are_not_relaxed(args, monkeypatch, server_calls):
    with pytest.raises(SystemExit) as error:
        app.main(args)
    assert error.value.code == 2
    assert not server_calls


def test_help_lists_config_binding(monkeypatch, capsys, server_calls):
    with pytest.raises(SystemExit) as error:
        app.main(["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "--config-name" in output and "--config-path" in output
    assert "--shared-key-env" not in output
    assert not server_calls


def test_unresolvable_deployment_config_fails_fast(monkeypatch, capsys, server_calls):
    class Failing:
        def __init__(self, **_kwargs):
            raise ValueError("Robot configuration name or path is required")

    monkeypatch.setattr(app, "WebServer", Failing)
    assert app.main([]) == 2
    output = capsys.readouterr()
    assert "无法绑定部署配置" in output.err
    assert "ROBOT_CONFIG" in output.err
    assert "未启动新的 Web 服务" in output.err
    assert "Traceback" not in output.err


def test_port_conflict_reports_actionable_message_without_traceback(monkeypatch, capsys, server_calls):
    def fail_bind(self):
        raise OSError(errno.EADDRINUSE, "Address already in use")

    monkeypatch.setattr(app.WebServer, "serve_forever", fail_bind)
    assert app.main(["--port", "8765"]) == 1
    output = capsys.readouterr()
    assert "127.0.0.1:8765" in output.err
    assert "端口被占用" in output.err
    assert "--port" in output.err
    assert "未启动新的 Web 服务" in output.err
    assert "Traceback" not in output.err


def test_unrelated_startup_os_errors_are_not_misreported_as_port_conflicts(monkeypatch, capsys, server_calls):
    def fail_permission(self):
        raise OSError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(app.WebServer, "serve_forever", fail_permission)
    with pytest.raises(OSError) as raised:
        app.main([])
    assert raised.value.errno == errno.EACCES
    assert "端口被占用" not in capsys.readouterr().err
