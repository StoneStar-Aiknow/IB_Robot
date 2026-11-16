"""Online prepare/execute must reach the Gateway without a local catalog.

The repository G1 preset has logical runtime interfaces and no local skill
source. The compatibility CLI used to load the full runtime context before
``validate``/``execute``, so a perfectly reachable Gateway was never contacted:
the command failed locally with ``description_required``. These tests pin the
replacement contract: online prepare/execute bind only the transport projection
and read status/snapshot from the bound Gateway; a missing Gateway is reported
as ``SERVER_UNAVAILABLE``, never as a local description error.
"""

from __future__ import annotations

import json

import pytest

from robot_skill_cli import catalog, cli
from robot_skill_cli.ros_bridge import BridgeError


@pytest.mark.parametrize(
    ("command", "arguments"),
    [
        ("validate", ["wave_hand"]),
        ("execute", ["wave_hand", "--task-id", "task-online-plan"]),
    ],
)
def test_online_prepare_and_execute_bind_the_gateway_without_a_local_catalog(
    monkeypatch, capsys, command, arguments
) -> None:
    captured = {}

    def forbidden(**_kwargs):
        raise AssertionError("online command compiled the local catalog")

    monkeypatch.setattr(catalog, "load_catalog_context", forbidden)
    monkeypatch.setattr(catalog, "compile_local_snapshot", forbidden)

    def create_bridge(transport):
        captured["transport"] = transport
        raise BridgeError(
            "SERVER_UNAVAILABLE",
            "service unavailable: /embodied/get_skill_gateway_status",
            exit_code=4,
        )

    monkeypatch.setattr(cli, "_create_bridge", create_bridge)

    exit_code = cli.main(["--config-name", "unitree_g1_arm_presets", command, *arguments])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 4
    assert payload["error"]["code"] == "SERVER_UNAVAILABLE"
    assert captured["transport"].robot_id == "unitree_g1_arm_presets"
    assert captured["transport"].status_service == "/embodied/get_skill_gateway_status"


def test_online_status_rejects_a_gateway_for_another_robot(monkeypatch, capsys) -> None:
    class Bridge:
        def start(self):
            return True

        def get_status(self, **_kwargs):
            return {"robot_name": "unitree_g1", "timeout_policy": {}}

        def close(self):
            pass

    monkeypatch.setattr(cli, "_create_bridge", lambda _transport: Bridge())

    assert cli.main(["--config-name", "unitree_g1_arm_presets", "status"]) == 3

    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "TARGET_IDENTITY_CHANGED"


def test_managed_direct_execute_rejects_a_moving_gateway_skill(monkeypatch) -> None:
    """The Hermes guard reads ``moves_robot`` from the Gateway, not a local view."""
    monkeypatch.setenv("IBROBOT_HERMES_MANAGED_REQUIRED", "1")
    monkeypatch.setattr(
        cli,
        "_prepare_request",
        lambda *_args, **_kwargs: (
            {},
            "",
            {},
            {},
            {"name": "wave_hand", "moves_robot": True},
        ),
    )

    with pytest.raises(cli._CliArgumentError, match="managed run-workflow"):
        cli._run_execute(
            type("Args", (), {"task_id": "task-online-plan"})(),
            type("Context", (), {"view": {"timeout_policy": {"rpc_timeout_sec": 5.0}}})(),
            object(),
        )
