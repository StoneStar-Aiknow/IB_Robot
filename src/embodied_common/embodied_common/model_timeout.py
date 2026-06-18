"""Shared timeout semantics for external VLM model output."""

from __future__ import annotations

DEFAULT_MODEL_OUTPUT_IDLE_TIMEOUT_SEC = 120.0


def resolve_model_output_idle_timeout(
    configured_timeout_sec: float | None,
    override_timeout_sec: float | None = None,
) -> float:
    """Return the socket idle-timeout used while waiting for model output."""

    if override_timeout_sec is not None and override_timeout_sec > 0.0:
        return float(override_timeout_sec)
    if configured_timeout_sec is not None and configured_timeout_sec > 0.0:
        return float(configured_timeout_sec)
    return DEFAULT_MODEL_OUTPUT_IDLE_TIMEOUT_SEC
