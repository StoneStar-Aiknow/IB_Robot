"""VLM API client re-export for the planner package.

The implementation lives in ``perception_service.api_client`` so that the
HTTP/payload/auth/timeout logic is owned by exactly one module.
``vlm_task_planner`` only needs an alias named :meth:`plan` to keep the
planner-side vocabulary ("plan a skill sequence") explicit.
"""

from __future__ import annotations

from typing import Any

from perception_service.api_client import VLMAPIClient as _PerceptionVLMAPIClient


class VLMAPIClient(_PerceptionVLMAPIClient):
    """Thin alias that exposes ``plan`` as a synonym for ``analyze``."""

    def plan(
        self,
        messages: list[dict[str, Any]],
        timeout_sec: float | None = None,
    ) -> tuple[str, dict[str, Any]]:
        return self.analyze(messages, timeout_sec=timeout_sec)


__all__ = ["VLMAPIClient"]
