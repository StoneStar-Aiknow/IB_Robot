"""OpenAI-compatible VLM API client."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from perception_service.model_timeout import resolve_model_output_idle_timeout


class VLMAPIClient:
    """Call an OpenAI-compatible chat-completions API."""

    def __init__(
        self,
        provider: str,
        base_url: str,
        api_key_env: str,
        model: str,
        timeout_sec: float,
    ) -> None:
        self._provider = provider
        self._base_url = base_url.rstrip("/")
        self._api_key_env = api_key_env
        self._model = model
        self._timeout_sec = timeout_sec

    def plan(
        self,
        messages: list[dict[str, Any]],
        timeout_sec: float | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if self._provider not in {"kimicode", "openai_compatible"}:
            raise RuntimeError(f"unsupported VLM API provider: {self._provider}")

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.1,
        }
        headers = {
            "Content-Type": "application/json",
        }
        api_key_env = self._api_key_env.strip()
        api_key = ""
        if api_key_env:
            api_key = os.getenv(api_key_env, "").strip()
        if self._provider == "kimicode":
            if not api_key_env:
                raise RuntimeError("kimicode provider requires api_key_env to be configured")
            if not api_key:
                raise RuntimeError(f"missing API key environment variable: {api_key_env}")
            headers["Authorization"] = f"Bearer {api_key}"
            headers["User-Agent"] = "KimiCLI/1.3"
        elif api_key_env:
            if not api_key:
                raise RuntimeError(f"missing API key environment variable: {api_key_env}")
            headers["Authorization"] = f"Bearer {api_key}"

        request = urllib.request.Request(
            url=f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        output_idle_timeout_sec = resolve_model_output_idle_timeout(
            configured_timeout_sec=self._timeout_sec,
            override_timeout_sec=timeout_sec,
        )
        with urllib.request.urlopen(request, timeout=output_idle_timeout_sec) as response:
            body = response.read().decode("utf-8")

        response_json = json.loads(body)
        choices = response_json.get("choices", [])
        if not choices:
            raise RuntimeError("VLM API returned no choices")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            content = "\n".join(text_parts)

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("VLM API returned empty content")

        return content, response_json
