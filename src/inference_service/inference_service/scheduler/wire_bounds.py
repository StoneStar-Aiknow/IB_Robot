"""Bounded scheduled-control-plane text helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping

WIRE_ERROR_CODE_BYTES = 64
WIRE_ERROR_STAGE_BYTES = 64


def utf8_size(value: str) -> int:
    return len(value.encode("utf-8"))


def truncate_utf8(value: object, max_bytes: int) -> str:
    if max_bytes < 1:
        return ""
    encoded = str(value).encode("utf-8")
    if len(encoded) <= max_bytes:
        return str(value)
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def set_scheduled_error(
    error: object,
    *,
    code: object,
    message: object = "",
    recoverable: bool = False,
    stage: object = "",
    details: Mapping[str, object] | None = None,
    max_message_bytes: int = 1024,
    max_details_bytes: int = 8192,
) -> None:
    error.code = truncate_utf8(code, WIRE_ERROR_CODE_BYTES)
    error.message = truncate_utf8(message, max_message_bytes)
    error.recoverable = recoverable
    error.stage = truncate_utf8(stage, WIRE_ERROR_STAGE_BYTES)
    details_json = ""
    if details:
        serialized = json.dumps(dict(details), sort_keys=True, separators=(",", ":"), default=str)
        if utf8_size(serialized) <= max_details_bytes:
            details_json = serialized
        else:
            marker = json.dumps({"truncated": True}, separators=(",", ":"))
            if utf8_size(marker) <= max_details_bytes:
                details_json = marker
    error.details_json = details_json
