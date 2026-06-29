"""Compatibility exports for shared VLM model timeout helpers."""

from embodied_common.model_timeout import DEFAULT_MODEL_OUTPUT_IDLE_TIMEOUT_SEC, resolve_model_output_idle_timeout

__all__ = ["DEFAULT_MODEL_OUTPUT_IDLE_TIMEOUT_SEC", "resolve_model_output_idle_timeout"]
