"""The neutral transport refuses silently unusable endpoint values."""

from __future__ import annotations

import pytest

from robot_client.transport import ENDPOINT_CONFIG_KEYS, ENDPOINT_TYPES, gateway_transport


def test_string_endpoints_are_projected():
    transport = gateway_transport(
        {
            "skill_gateway_status_service": "/deployment/status",
            "prepare_plan_service": "/deployment/prepare",
            "execute_program_action": "/deployment/program",
        },
        robot_id="g1",
        robot_name="g1",
    )

    assert transport.status_service == "/deployment/status"
    assert transport.prepare_plan_service == "/deployment/prepare"
    assert transport.execute_program_action == "/deployment/program"
    assert len(transport.endpoints) == len(ENDPOINT_TYPES) == 18


def test_every_endpoint_type_has_a_transport_field():
    transport = gateway_transport({})

    assert set(transport.endpoints) == set(ENDPOINT_TYPES)


@pytest.mark.parametrize("value", [{"interface": "motion.named"}, 7, "", "   ", None])
def test_non_string_or_empty_endpoint_is_rejected(value):
    with pytest.raises(ValueError, match="embodied.skill_action_name must be a non-empty endpoint string"):
        gateway_transport({"skill_action_name": value})


def test_every_declared_endpoint_key_is_guarded():
    for key in ENDPOINT_CONFIG_KEYS:
        with pytest.raises(ValueError, match=f"embodied.{key} must be a non-empty endpoint string"):
            gateway_transport({key: {"interface": "motion.named"}})


def test_unknown_keys_are_left_alone():
    transport = gateway_transport({"runtime_named_motion": {"interface": "motion.named"}, "enabled": True})

    assert transport.status_service == "/embodied/get_skill_gateway_status"
