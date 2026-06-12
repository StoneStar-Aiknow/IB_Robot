"""Shared JSON parsing utilities for embodied pipeline packages."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_blob(raw_text: str, context_name: str = "response") -> str:
    """Extract the first valid JSON object from raw LLM text."""
    text = (raw_text or "").strip()
    if not text:
        raise ValueError(f"{context_name} is empty")

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, index)
            return json.dumps(obj, ensure_ascii=False)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"{context_name} does not contain a JSON object")


def load_json_mapping(raw_value: str, context_name: str = "JSON mapping") -> dict[str, Any]:
    if not raw_value:
        return {}
    try:
        loaded = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {context_name}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{context_name} must decode to a JSON object")
    return loaded


def load_json_list(raw_value: str, context_name: str = "JSON list") -> list[Any]:
    if not raw_value:
        return []
    try:
        loaded = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {context_name}: {exc}") from exc
    if not isinstance(loaded, list):
        raise ValueError(f"{context_name} must decode to a JSON list")
    return loaded


def string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a string list")
    return [str(item).strip() for item in value if str(item).strip()]


def parse_confidence(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        confidence = default
    elif isinstance(value, int | float):
        confidence = float(value)
    elif isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            confidence = default
        else:
            keyword_mapping = {
                "low": 0.25,
                "medium": 0.6,
                "high": 0.85,
                "低": 0.25,
                "中": 0.6,
                "中等": 0.6,
                "高": 0.85,
            }
            if normalized in keyword_mapping:
                confidence = keyword_mapping[normalized]
            elif normalized.endswith("%"):
                confidence = float(normalized[:-1].strip()) / 100.0
            else:
                confidence = float(normalized)
    else:
        raise ValueError("confidence must be numeric or a supported string label")

    if confidence > 1.0 and confidence <= 100.0:
        confidence /= 100.0
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence must be in [0.0, 1.0]")
    return confidence
