"""Single-call cloud LLM client. Stateless: one call in, one unified JSON out."""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG = Path(__file__).parent / "llm_models.yaml"


def call_llm(
    model: str,
    prompt: str | None = None,
    messages: list[dict] | None = None,
    system: str | None = None,
    image_path: str | None = None,
    tools: list[dict] | None = None,
    force_json: bool = False,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Call a cloud LLM and return a unified response dict.

    Args:
        model:       Named model key from llm_models.yaml (required).
        prompt:      User message to send. Appended as the latest user turn after
                     any messages in the history list.
        messages:    Full conversation history as OpenAI-format message list.
                     When provided, caller is responsible for managing history.
        system:      System prompt, prepended as the first message.
        image_path:  Local image file path. Attached to the last user message.
                     Silently ignored if the model does not support multimodal.
        tools:       OpenAI-format tool/function definitions, passed through as-is.
                     When provided, force_json is ignored.
        force_json:  Ask the model to return JSON only (response_format).
        config_path: Override path to llm_models.yaml.

    Returns:
        dict with keys: status_code, message, model, provider, content,
        tool_calls, reasoning, usage, timing, raw.
    """
    t_start = time.monotonic()

    cfg_path = Path(config_path) if config_path else _DEFAULT_CONFIG
    try:
        with open(cfg_path) as f:
            config = yaml.safe_load(f).get("llm_client", {})
    except Exception as exc:
        return _error(500, f"Failed to load config: {exc}")

    specs = config.get("models", {})
    if model not in specs:
        return _error(404, f"Model '{model}' not found in config", model=model)
    spec = specs[model]

    multimodal = spec.get("multimodal", False)
    image_degraded = bool(image_path and not multimodal)
    effective_image = None if image_degraded else image_path

    try:
        msgs = _build_messages(prompt, messages, system, effective_image)
    except (FileNotFoundError, ValueError) as exc:
        return _error(400, str(exc), model=model)

    if not any(m["role"] != "system" for m in msgs):
        return _error(400, "No user message provided", model=model)

    api_key_env = spec.get("api_key_env", "")
    api_key = os.getenv(api_key_env, "").strip() if api_key_env else ""
    if api_key_env and not api_key:
        return _error(401, f"API key env var not set: {api_key_env}", model=model)

    payload: dict[str, Any] = {
        "model": spec["model_id"],
        "messages": msgs,
        "temperature": config.get("default_temperature", 0.1),
    }
    if tools:
        payload["tools"] = tools
    elif force_json:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    timeout = config.get("default_timeout_sec", 60)
    url = spec["base_url"].rstrip("/") + "/chat/completions"

    t_req = time.monotonic()
    try:
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
    except urllib.error.HTTPError as exc:
        try:
            err_body = exc.read().decode(errors="replace")[:300]
        except Exception:
            err_body = ""
        code = {400: 400, 401: 401, 403: 401, 429: 429}.get(exc.code, 502)
        return _error(code, f"HTTP {exc.code}: {err_body}", model=model)
    except urllib.error.URLError as exc:
        return _error(503, f"Network error: {exc.reason}", model=model)
    except TimeoutError:
        return _error(408, f"Timed out after {timeout}s", model=model)

    timing = {
        "request_ms": round((time.monotonic() - t_req) * 1000),
        "total_ms": round((time.monotonic() - t_start) * 1000),
    }

    try:
        raw = json.loads(body)
    except json.JSONDecodeError:
        return _error(502, f"Invalid JSON: {body[:200]}", model=model, timing=timing)

    choices = raw.get("choices", [])
    if not choices:
        return _error(502, "No choices in response", model=model, timing=timing, raw=raw)

    message = choices[0].get("message", {})
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "\n".join(p.get("text", "") for p in content if p.get("type") == "text")

    usage_raw = raw.get("usage", {})
    return {
        "status_code": 200,
        "message": "OK (image ignored: model is text-only)" if image_degraded else "OK",
        "model": model,
        "provider": "openai_compatible",
        "content": content,
        "tool_calls": message.get("tool_calls") or [],
        "reasoning": message.get("reasoning_content") or message.get("thinking") or "",
        "usage": {
            "prompt_tokens": _get_token(usage_raw, "prompt_tokens", "input_tokens"),
            "completion_tokens": _get_token(usage_raw, "completion_tokens", "output_tokens"),
            "total_tokens": usage_raw.get("total_tokens", 0),
        },
        "timing": timing,
        "raw": raw,
    }


def _build_messages(
    prompt: str | None,
    messages: list[dict] | None,
    system: str | None,
    image_path: str | None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    if system:
        result.append({"role": "system", "content": system})

    if messages:
        result.extend(messages)
    if prompt:
        result.append({"role": "user", "content": prompt})

    if image_path:
        data_url = _encode_image(image_path)
        image_block = {"type": "image_url", "image_url": {"url": data_url}}
        # Attach to last user message, or append a new one
        for i in range(len(result) - 1, -1, -1):
            if result[i]["role"] == "user":
                c = result[i]["content"]
                result[i]["content"] = (
                    [{"type": "text", "text": c}, image_block] if isinstance(c, str) else [*c, image_block]
                )
                break
        else:
            result.append({"role": "user", "content": [image_block]})

    return result


def _encode_image(path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path_str}")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(
        path.suffix.lstrip(".").lower(), "image/jpeg"
    )
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _get_token(usage: dict, primary: str, fallback: str) -> int:
    v = usage.get(primary)
    return v if v is not None else usage.get(fallback, 0)


def _error(
    status_code: int,
    message: str,
    model: str = "",
    timing: dict | None = None,
    raw: dict | None = None,
) -> dict[str, Any]:
    return {
        "status_code": status_code,
        "message": message,
        "model": model,
        "provider": "openai_compatible",
        "content": "",
        "tool_calls": [],
        "reasoning": "",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "timing": timing or {"request_ms": 0, "total_ms": 0},
        "raw": raw or {},
    }
