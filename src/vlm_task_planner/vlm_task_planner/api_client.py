"""VLM API client wrapper for the planner package."""

from __future__ import annotations

from typing import Any

from embodied_common.vlm_api_client import VLMAPIClient as _SharedVLMAPIClient


class VLMAPIClient(_SharedVLMAPIClient):
    """Thin alias that exposes ``plan`` as a synonym for ``analyze``."""

    def plan(
        self,
        messages: list[dict[str, Any]],
        timeout_sec: float | None = None,
    ) -> tuple[str, dict[str, Any]]:
        return self.analyze(messages, timeout_sec=timeout_sec)


__all__ = ["VLMAPIClient"]
