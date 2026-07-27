"""OpenAI-compatible VLM API client shared by embodied runtime packages."""

from __future__ import annotations

import base64
import json
import os
import pathlib
import time
import urllib.error
import urllib.request
from typing import Any

from embodied_common.model_timeout import resolve_model_output_idle_timeout

_YAML_BASENAME = "vlm_models.yaml"


def _default_yaml_path() -> pathlib.Path:
    """Resolve the installed vlm_models.yaml under the package share space.

    Falls back to the in-tree copy when the package is not installed (e.g. running
    straight from a source checkout during development).
    """
    try:
        from ament_index_python.packages import get_package_share_directory

        share = pathlib.Path(get_package_share_directory("embodied_common"))
        installed = share / "config" / _YAML_BASENAME
        if installed.exists():
            return installed
    except Exception:
        pass
    return pathlib.Path(__file__).parent / _YAML_BASENAME


def _parse_response_json(body: str) -> dict[str, Any]:
    """Parse the HTTP response body as JSON with a friendly error on failure."""
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        preview = body.strip()[:200]
        raise RuntimeError(f"VLM API returned invalid JSON: {preview!r}") from exc


class VLMAPIClient:
    """Call an OpenAI-compatible chat-completions API."""

    def __init__(
        self,
        provider: str,
        base_url: str,
        api_key_env: str,
        model: str,
        timeout_sec: float,
        temperature: float = 0.1,
        multimodal: bool = True,
    ) -> None:
        self._provider = provider
        self._base_url = base_url.rstrip("/")
        self._api_key_env = api_key_env
        self._model = model
        self._timeout_sec = timeout_sec
        self._temperature = temperature
        self._multimodal = multimodal

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
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
        return headers

    def _http_post(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout_sec: float,
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            url=f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                error_body = ""
            raise RuntimeError(f"VLM API HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"VLM API network error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError(f"VLM API timeout after {timeout_sec}s") from exc
        return _parse_response_json(body)

    def analyze(
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
        headers = self._build_headers()
        output_idle_timeout_sec = resolve_model_output_idle_timeout(
            configured_timeout_sec=self._timeout_sec,
            override_timeout_sec=timeout_sec,
        )
        response_json = self._http_post(payload, headers, output_idle_timeout_sec)

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

    def complete(
        self,
        messages: list[dict[str, Any]],
        timeout_sec: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        force_json: bool = False,
        enable_thinking: bool = True,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call chat-completions; return status/content/tool_calls/reasoning/usage/timing dict."""
        t_start = time.monotonic()

        def _err(msg: str) -> dict[str, Any]:
            return {
                "status": "error",
                "model": self._model,
                "content": "",
                "tool_calls": [],
                "reasoning": "",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "timing_ms": int((time.monotonic() - t_start) * 1000),
                "error": msg,
            }

        if self._provider not in {"kimicode", "openai_compatible"}:
            return _err(f"unsupported VLM API provider: {self._provider}")

        send_messages = messages
        if not self._multimodal:
            stripped = []
            for msg in send_messages:
                if isinstance(msg.get("content"), list):
                    text = "\n".join(str(p.get("text", "")) for p in msg["content"] if p.get("type") == "text")
                    stripped.append({**msg, "content": text})
                else:
                    stripped.append(msg)
            send_messages = stripped

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": send_messages,
            "temperature": self._temperature,
        }

        if tools:
            payload["tools"] = tools
        elif force_json:
            payload["response_format"] = {"type": "json_object"}

        if not enable_thinking and "dashscope.aliyuncs.com" in self._base_url:
            payload["enable_thinking"] = False

        if extra_body:
            payload.update(extra_body)

        try:
            headers = self._build_headers()
        except RuntimeError as exc:
            return _err(str(exc))

        output_idle_timeout_sec = resolve_model_output_idle_timeout(
            configured_timeout_sec=self._timeout_sec,
            override_timeout_sec=timeout_sec,
        )

        try:
            response_json = self._http_post(payload, headers, output_idle_timeout_sec)
        except RuntimeError as exc:
            return _err(str(exc))

        choices = response_json.get("choices", [])
        if not choices:
            return _err("VLM API returned no choices")
        message = choices[0].get("message", {})

        content = message.get("content") or ""
        if isinstance(content, list):
            content = "\n".join(str(item.get("text", "")) for item in content if item.get("type") == "text")

        tool_calls = message.get("tool_calls") or []
        reasoning = message.get("reasoning_content") or message.get("thinking") or ""

        # A valid response must carry either text content or tool calls; empty both is an error.
        if not content.strip() and not tool_calls:
            return _err("VLM API returned empty content and no tool_calls")

        usage_raw = response_json.get("usage", {})
        usage = {
            "prompt_tokens": usage_raw.get("prompt_tokens", 0),
            "completion_tokens": usage_raw.get("completion_tokens", 0),
            "total_tokens": usage_raw.get("total_tokens", 0),
        }

        return {
            "status": "ok",
            "model": self._model,
            "content": content,
            "tool_calls": tool_calls,
            "reasoning": reasoning,
            "usage": usage,
            "timing_ms": int((time.monotonic() - t_start) * 1000),
            "error": None,
        }


def _load_yaml_config(yaml_path: pathlib.Path) -> dict[str, Any]:
    import yaml

    try:
        with open(yaml_path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise RuntimeError(f"VLM models config not found: {yaml_path}") from None


def _detect_image_mime(image: bytes) -> str:
    """Detect image MIME type from magic bytes; raise on unsupported formats."""
    if image[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image[:4] == b"RIFF" and image[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("unsupported image format (expected PNG / JPEG / WEBP)")


class VLMClient:
    """High-level VLM client: one-line access with auto routing, message building and context."""

    def __init__(self, yaml_path: pathlib.Path | None = None, system: str | None = None) -> None:
        resolved_path = yaml_path or _default_yaml_path()
        self._config = _load_yaml_config(resolved_path)
        self._defaults = self._config.get("defaults", {})
        self._models_cfg = self._config.get("models", {})
        if not self._models_cfg:
            raise RuntimeError(f"no 'models' configured in {resolved_path}")
        self._default_model: str = self._defaults.get("model", next(iter(self._models_cfg)))
        if self._default_model not in self._models_cfg:
            raise RuntimeError(f"defaults.model '{self._default_model}' not found under 'models' in {resolved_path}")
        self._client_cache: dict[str, VLMAPIClient] = {}
        self._history: list[dict[str, Any]] = []
        # A persistent system prompt is prepended to every request but kept OUT of
        # _history, so a long conversation never pushes it out of context and
        # clear_history() never drops it. None means no system message is sent.
        self._system = system

    def _resolve_model(self, model: str | None) -> str:
        if model is None:
            return self._default_model
        if model not in self._models_cfg:
            # An explicitly requested but unknown model is a configuration error, not a
            # runtime condition to paper over. Silently falling back would send the
            # conversation (and its history) to an unintended model/endpoint.
            raise ValueError(f"unknown model '{model}'; available models: {', '.join(self._models_cfg.keys())}")
        return model

    def _get_client(self, model_name: str) -> VLMAPIClient:
        if model_name not in self._client_cache:
            m = self._models_cfg[model_name]
            self._client_cache[model_name] = VLMAPIClient(
                provider=m["provider"],
                base_url=m["base_url"],
                api_key_env=m["api_key_env"],
                model=model_name,
                timeout_sec=float(m.get("timeout_sec", self._defaults.get("timeout_sec", 90))),
                temperature=float(m.get("temperature", self._defaults.get("temperature", 0.1))),
                multimodal=bool(m.get("multimodal", True)),
            )
        return self._client_cache[model_name]

    def _build_user_message(self, text: str, image: bytes | None) -> dict[str, Any]:
        if image is None:
            return {"role": "user", "content": text}
        mime = _detect_image_mime(image)
        b64 = base64.b64encode(image).decode()
        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": text},
            ],
        }

    def chat(
        self,
        text: str,
        image: bytes | None = None,
        model: str | None = None,
        clear_history: bool = False,
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = True,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a message and return the model response. Manages history automatically."""
        if clear_history:
            self._history = []

        model_name = self._resolve_model(model)
        client = self._get_client(model_name)

        # Current turn sends the real image; history stores a text placeholder so
        # past images are not re-sent every turn (token blowup / non-multimodal break).
        current_message = self._build_user_message(text, image)
        history_message = {"role": "user", "content": f"{text}\n[图片已省略]"} if image is not None else current_message

        # System prompt (if any) leads every request but is never stored in _history.
        system_prefix = [{"role": "system", "content": self._system}] if self._system else []
        send_messages = [*system_prefix, *self._history, current_message]

        result = client.complete(
            send_messages,
            tools=tools,
            enable_thinking=enable_thinking,
            extra_body=extra_body,
        )

        # Only commit to history on success; failed requests (network / quota / empty
        # response) must not leave a half-written turn behind. Callers that want to
        # retry or fall back to another model start from a clean, consistent history.
        if result["status"] == "ok":
            self._history.append(history_message)
            assistant_message: dict[str, Any] = {"role": "assistant", "content": result["content"]}
            if result["tool_calls"]:
                assistant_message["tool_calls"] = result["tool_calls"]
            self._history.append(assistant_message)

        return result

    def clear_history(self) -> None:
        self._history = []
