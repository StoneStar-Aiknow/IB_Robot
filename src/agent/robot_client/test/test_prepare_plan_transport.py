"""The prepare endpoint is a first-class transport contract key.

Regression: ``prepare_plan_service`` was declared in ``ENDPOINT_TYPES`` and
``ENDPOINT_CONFIG_KEYS`` but had no :class:`GatewayTransport` field. A
non-default deployment value passed validation and was then silently dropped,
so it was missing from ``endpoints``, ``identity_digest`` and ``bridge_kwargs``,
and the prepare call always went to the default address. These tests pin the
whole projection: robot-config ``embodied`` mapping -> ``GatewayTransport`` ->
``endpoints`` -> identity/target key -> ``bridge_kwargs`` -> RosBridge dispatch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from robot_client.client import RobotClient
from robot_client.connection import bridge_kwargs
from robot_client.transport import ENDPOINT_CONFIG_KEYS, gateway_transport
from robot_client.transports.ros_bridge import RosBridge

PREPARE_DEFAULT = "/embodied/prepare_agent_plan"


def _projection(prepare_endpoint: str = "/deployment/g1/prepare_plan") -> dict[str, str]:
    """Return the robot_config transport projection shape with every key set."""
    values = {key: f"/deployment/g1/{key}" for key in ENDPOINT_CONFIG_KEYS}
    values["prepare_plan_service"] = prepare_endpoint
    return values


def test_prepare_endpoint_survives_the_transport_projection_chain():
    transport = gateway_transport(_projection(), robot_id="g1", robot_name="g1")

    assert transport.prepare_plan_service == "/deployment/g1/prepare_plan"
    assert transport.endpoints["prepare_plan_service"] == "/deployment/g1/prepare_plan"
    assert bridge_kwargs(transport)["prepare_plan_service"] == "/deployment/g1/prepare_plan"


def test_only_the_prepare_endpoint_changes_the_identity_and_target_key():
    first = gateway_transport(_projection("/deployment/g1/prepare-a"), robot_id="g1", robot_name="g1")
    second = gateway_transport(_projection("/deployment/g1/prepare-b"), robot_id="g1", robot_name="g1")

    first_without_prepare = {key: value for key, value in first.endpoints.items() if key != "prepare_plan_service"}
    second_without_prepare = {key: value for key, value in second.endpoints.items() if key != "prepare_plan_service"}
    assert first_without_prepare == second_without_prepare
    assert first.endpoints["prepare_plan_service"] != second.endpoints["prepare_plan_service"]
    assert first.identity_digest() != second.identity_digest()
    assert RobotClient(first).session["target_key"] != RobotClient(second).session["target_key"]


def test_prepare_endpoint_reaches_the_ros_bridge_dispatch(monkeypatch):
    transport = gateway_transport(_projection(), robot_id="g1", robot_name="g1")
    bridge = RosBridge(**bridge_kwargs(transport))
    calls = []

    class FakeRequest:
        pass

    class FakePrepareAgentPlan:
        Request = FakeRequest

    bridge._PrepareAgentPlan = FakePrepareAgentPlan
    bridge._prepare_plan_client = object()

    def fake_call_service(client, request, *, service_name, timeout_sec):
        calls.append((client, service_name, timeout_sec))
        return SimpleNamespace(success=True, allowed=True, plan=None, error_code="", message="", diagnostics=[])

    monkeypatch.setattr(bridge, "_call_service", fake_call_service)
    monkeypatch.setattr(bridge, "_agent_plan_dict", lambda plan: {})
    monkeypatch.setattr(bridge, "_diagnostics", lambda diagnostics: [])

    result = bridge.prepare_agent_plan(
        request_id="request-1",
        raw_command="wave",
        workflow_steps=[],
        timeout_sec=1.0,
    )

    assert result["success"] is True
    assert calls == [(bridge._prepare_plan_client, "/deployment/g1/prepare_plan", 1.0)]


def test_missing_prepare_endpoint_keeps_the_documented_default():
    assert gateway_transport({}).prepare_plan_service == PREPARE_DEFAULT


@pytest.mark.parametrize("value", [None, 7, "", "   ", {"interface": "motion.named"}])
def test_prepare_endpoint_rejects_non_string_or_empty_values(value):
    with pytest.raises(ValueError, match="embodied.prepare_plan_service must be a non-empty endpoint string"):
        gateway_transport({"prepare_plan_service": value})
