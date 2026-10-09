"""Shared caller authentication for Gateway execution contracts."""

from __future__ import annotations

import os


def caller_token_env_name(caller_scope: str) -> str:
    """Return the stable environment variable name for a caller scope."""
    normalized = str(caller_scope).strip().upper().replace(":", "_")
    return f"IBROBOT_EXECUTION_TOKEN_{normalized}"


def is_caller_authorized(
    caller_scope: str,
    caller_token: str,
    allowed_scopes: tuple[str, ...],
    configured_tokens: dict[str, str] | None = None,
) -> bool:
    """Validate scope and its optional configured transport token.

    Tokens authenticate transport requests and are deliberately excluded from
    canonical Program identity and the durable execution payload.
    """
    scope = str(caller_scope).strip()
    if scope not in allowed_scopes:
        return False
    configured_token = (
        configured_tokens.get(scope, "")
        if configured_tokens is not None and scope in configured_tokens
        else os.environ.get(caller_token_env_name(scope), "")
    )
    return not configured_token or str(caller_token) == configured_token


__all__ = ["caller_token_env_name", "is_caller_authorized"]
