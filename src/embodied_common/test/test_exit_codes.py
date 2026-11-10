from __future__ import annotations

import pytest

from embodied_common import exit_codes


def test_exit_status_values_stay_stable() -> None:
    assert exit_codes.EXIT_SUCCESS == 0
    assert exit_codes.EXIT_INVALID_INPUT == 2
    assert exit_codes.EXIT_GATEWAY_REJECTED == 3
    assert exit_codes.EXIT_ROS_UNAVAILABLE == 4
    assert exit_codes.EXIT_TIMEOUT == 124
    assert exit_codes.EXIT_SIGINT == 130
    assert exit_codes.EXIT_SIGTERM == 143


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("INVALID_TARGET", exit_codes.EXIT_INVALID_INPUT),
        ("INVALID_REFERENCE", exit_codes.EXIT_INVALID_INPUT),
        ("INVALID_DESCRIPTOR", exit_codes.EXIT_GATEWAY_REJECTED),
        ("INVALID_LOCAL_STATE", exit_codes.EXIT_INVALID_INPUT),
        ("INVALID_TIMEOUT", exit_codes.EXIT_INVALID_INPUT),
        ("NOT_CONNECTED", exit_codes.EXIT_GATEWAY_REJECTED),
        ("TARGET_NOT_FOUND", exit_codes.EXIT_INVALID_INPUT),
        ("AMBIGUOUS_TARGET", exit_codes.EXIT_INVALID_INPUT),
        ("UNKNOWN_SKILL", exit_codes.EXIT_INVALID_INPUT),
        ("UNSUPPORTED_COMMAND", exit_codes.EXIT_INVALID_INPUT),
        ("ROS_UNAVAILABLE", exit_codes.EXIT_ROS_UNAVAILABLE),
        ("TARGET_UNREACHABLE", exit_codes.EXIT_ROS_UNAVAILABLE),
        ("SERVER_UNAVAILABLE", exit_codes.EXIT_ROS_UNAVAILABLE),
        ("DISCOVERY_TIMEOUT", exit_codes.EXIT_TIMEOUT),
        ("READ_ONLY", exit_codes.EXIT_GATEWAY_REJECTED),
        ("INVALID_RESPONSE", exit_codes.EXIT_GATEWAY_REJECTED),
        ("LEGACY_GAME_CATALOG", exit_codes.EXIT_GATEWAY_REJECTED),
    ],
)
def test_client_error_codes_map_to_documented_exit_status(code: str, expected: int) -> None:
    assert exit_codes.client_error_exit_code(code) == expected


def test_unknown_error_codes_fail_closed_as_gateway_rejections() -> None:
    assert exit_codes.client_error_exit_code("NEW_SERVER_SIDE_REJECTION") == exit_codes.EXIT_GATEWAY_REJECTED
    assert exit_codes.client_error_exit_code("PROTOCOL_VIOLATION") == exit_codes.EXIT_GATEWAY_REJECTED


def test_error_code_normalization_is_case_and_space_insensitive() -> None:
    assert exit_codes.client_error_exit_code(" invalid_target ") == exit_codes.EXIT_INVALID_INPUT
    assert exit_codes.client_error_exit_code("ros_unavailable") == exit_codes.EXIT_ROS_UNAVAILABLE


def test_invalid_timeout_is_not_classified_as_an_expired_deadline() -> None:
    assert "INVALID_TIMEOUT" not in exit_codes.TIMEOUT_ERROR_CODES
    assert exit_codes.client_error_exit_code("INVALID_TIMEOUT") != exit_codes.EXIT_TIMEOUT
