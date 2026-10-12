import pytest

from embodied_common.caller_auth import caller_token_env_name, is_caller_authorized


def test_caller_token_env_name_is_stable():
    assert caller_token_env_name("agent_incubation:local_operator") == (
        "IBROBOT_EXECUTION_TOKEN_AGENT_INCUBATION_LOCAL_OPERATOR"
    )


def test_unconfigured_caller_scope_does_not_bypass_scope_allowlist(monkeypatch):
    monkeypatch.delenv("IBROBOT_EXECUTION_TOKEN_AGENT_INCUBATION_LOCAL_OPERATOR", raising=False)
    assert is_caller_authorized("agent_incubation:local_operator", "", ("robot_skill_cli",)) is False


def test_configured_token_is_required_for_all_gateway_queries(monkeypatch):
    monkeypatch.setenv("IBROBOT_EXECUTION_TOKEN_ROBOT_SKILL_CLI", "secret")
    allowed = ("robot_skill_cli",)
    assert is_caller_authorized("robot_skill_cli", "secret", allowed) is True
    assert is_caller_authorized("robot_skill_cli", "wrong", allowed) is False
    assert is_caller_authorized("robot_skill_cli", "", allowed) is False


@pytest.mark.parametrize("scope", ["", "unknown", "agent_incubation:local_operator"])
def test_unknown_or_empty_scope_is_rejected(scope):
    assert is_caller_authorized(scope, "", ("robot_skill_cli",)) is False
