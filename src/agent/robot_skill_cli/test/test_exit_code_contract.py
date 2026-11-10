"""Exit-code contract for the config-bound CLI (restored coverage).

Semantics recovered from the discovery-era ``test_exit_code_contract.py`` that
was archived for Issue #238 with the removed discovery/online flow. The online
dispatch and compatibility-adapter assertions retired with that flow; their
coverage destination is recorded in the independent removal commit message and
Issue #238, not in this tree.

The shared code-to-exit mapping matrix is owned by ``embodied_common``'s
exit-code test. This file keeps only the CLI-specific behaviour on the
configuration-bound path: the error envelope, empty stderr, and passthrough of
the Gateway-supplied code. It belongs to the separate exit-code change: it
imports ``embodied_common.exit_codes``, which that change introduces, so it is
not part of the frozen removal commit and cannot be collected before it.

``INVALID_REQUEST`` is intentionally a Gateway rejection (3): the same code can
come from the server, and the client must not guess the source. Only codes the
client itself decides locally are mapped to caller input (2).
"""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from embodied_common import exit_codes
from robot_client.errors import ClientError
from robot_skill_cli import catalog, cli, output


def _run_status_with_client_error(monkeypatch, code: str) -> int:
    monkeypatch.setattr(catalog, "load_transport_context", Mock(return_value=(object(), object())))
    monkeypatch.setattr(cli, "_run_runtime_command", Mock(side_effect=ClientError(code, f"failure: {code}")))
    return cli.main(["--config-name", "unitree_g1", "status"])


@pytest.mark.parametrize(
    "code",
    [
        "INVALID_REFERENCE",
        "NOT_CONNECTED",
        "ROS_UNAVAILABLE",
        "DISCOVERY_TIMEOUT",
        "NEW_SERVER_SIDE_REJECTION",
    ],
)
def test_cli_uses_the_shared_exit_code_and_keeps_the_error_envelope(monkeypatch, capsys, code):
    assert _run_status_with_client_error(monkeypatch, code) == exit_codes.client_error_exit_code(code)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope["error"] == {"code": code, "message": f"failure: {code}"}
    assert captured.err == ""


def test_server_side_invalid_request_keeps_its_code_and_rejection(monkeypatch, capsys):
    """The server may reject a request as INVALID_REQUEST; do not turn it into 2."""
    assert "INVALID_REQUEST" not in exit_codes.INVALID_INPUT_ERROR_CODES
    assert _run_status_with_client_error(monkeypatch, "INVALID_REQUEST") == exit_codes.EXIT_GATEWAY_REJECTED
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "INVALID_REQUEST"


def test_output_module_reexports_the_shared_exit_contract():
    assert output.EXIT_SUCCESS == exit_codes.EXIT_SUCCESS
    assert output.EXIT_INVALID_INPUT == exit_codes.EXIT_INVALID_INPUT
    assert output.EXIT_GATEWAY_REJECTED == exit_codes.EXIT_GATEWAY_REJECTED
    assert output.EXIT_ROS_UNAVAILABLE == exit_codes.EXIT_ROS_UNAVAILABLE
    assert output.EXIT_TIMEOUT == exit_codes.EXIT_TIMEOUT
    assert output.EXIT_SIGINT == exit_codes.EXIT_SIGINT
    assert output.EXIT_SIGTERM == exit_codes.EXIT_SIGTERM


def test_stop_contract_reexports_the_shared_exit_contract():
    from robot_client import stop_contract

    assert stop_contract.EXIT_ROS_UNAVAILABLE == exit_codes.EXIT_ROS_UNAVAILABLE
    assert stop_contract.EXIT_TIMEOUT == exit_codes.EXIT_TIMEOUT
