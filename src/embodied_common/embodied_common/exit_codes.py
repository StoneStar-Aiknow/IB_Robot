"""Cross-process exit status contract for robot-skill client consumers.

The values are part of the operator-facing process contract documented in
``src/agent/robot_skill_cli/README.md``. They live in ``embodied_common`` so the
CLI output layer and the ROS transport import one definition instead of keeping
per-package copies. Importing modules keep their historical paths as compatible
re-exports.

``ClientError`` codes are mapped only at the CLI edge; the client library itself
never interprets them as process exit codes.
"""

from __future__ import annotations

EXIT_SUCCESS = 0
EXIT_INVALID_INPUT = 2
EXIT_GATEWAY_REJECTED = 3
EXIT_ROS_UNAVAILABLE = 4
EXIT_TIMEOUT = 124
EXIT_SIGINT = 130
EXIT_SIGTERM = 143

# Caller-side argument, configuration, local-state and schema failures. These
# are decided before the Gateway evaluates a request, so they are not Gateway
# rejections. Session preconditions such as NOT_CONNECTED keep their historical
# rejection bucket through the fail-closed default.
#
# ``INVALID_REQUEST`` deliberately stays out of this set: the Gateway may also
# produce that code for a request it rejects, so the client cannot classify it
# as caller input without guessing its origin. It falls through to the generic
# rejection bucket (3), which keeps a server rejection a rejection. Explicitly
# local codes (for example INVALID_REFERENCE or INVALID_TARGET) remain 2.
INVALID_INPUT_ERROR_CODES = frozenset(
    {
        "AMBIGUOUS_TARGET",
        "INVALID_CREDENTIAL_BINDING",
        "INVALID_CURSOR",
        "INVALID_LOCAL_STATE",
        "INVALID_REFERENCE",
        "INVALID_TARGET",
        "INVALID_TIMEOUT",
        "TARGET_NOT_FOUND",
        "UNKNOWN_SKILL",
        "UNSUPPORTED_COMMAND",
    }
)

# Transport or Gateway availability failures.
TRANSPORT_UNAVAILABLE_ERROR_CODES = frozenset(
    {
        "ROS_UNAVAILABLE",
        "SERVER_UNAVAILABLE",
        "TARGET_UNREACHABLE",
    }
)

# Expired deadlines. Parameter validation such as INVALID_TIMEOUT deliberately
# does not belong here.
TIMEOUT_ERROR_CODES = frozenset({"DISCOVERY_TIMEOUT"})


def client_error_exit_code(code: str) -> int:
    """Map one robot client error code to the documented CLI exit status.

    Unknown codes stay Gateway rejections (``3``) so a new server rejection or
    protocol violation fails closed instead of being reported as caller input.
    """
    normalized = code.strip().upper() if isinstance(code, str) else ""
    if normalized in INVALID_INPUT_ERROR_CODES:
        return EXIT_INVALID_INPUT
    if normalized in TRANSPORT_UNAVAILABLE_ERROR_CODES:
        return EXIT_ROS_UNAVAILABLE
    if normalized in TIMEOUT_ERROR_CODES:
        return EXIT_TIMEOUT
    return EXIT_GATEWAY_REJECTED
