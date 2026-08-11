"""Deterministic lossless segmentation for bounded ZipVoice requests."""

from __future__ import annotations

import re
import unicodedata

from voice_tts_service.errors import TTSError

_PRIMARY_BOUNDARIES = frozenset("。！？!?；;")
_SECONDARY_BOUNDARIES = frozenset("，、：:,")


def normalize_text(text: str) -> str:
    """Normalize Unicode width and whitespace while retaining punctuation."""

    normalized = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", normalized).strip()


def _candidate_boundary(text: str, limit: int, boundaries: frozenset[str]) -> int | None:
    for index in range(min(limit, len(text)) - 1, -1, -1):
        if text[index] in boundaries:
            return index + 1
    return None


def _split_bounded(text: str, max_chars: int) -> list[str]:
    result = []
    remaining = text
    while len(remaining) > max_chars:
        boundary = _candidate_boundary(remaining, max_chars, _PRIMARY_BOUNDARIES)
        if boundary is None:
            boundary = _candidate_boundary(remaining, max_chars, _SECONDARY_BOUNDARIES)
        if boundary is None:
            boundary = max_chars
        result.append(remaining[:boundary])
        remaining = remaining[boundary:]
    if remaining:
        result.append(remaining)
    return result


def segment_text(text: str, max_chars: int, max_segments: int) -> tuple[str, list[str]]:
    """Return normalized text and ordered segments whose concatenation is identical."""

    if max_chars <= 0 or max_segments <= 0:
        raise ValueError("segment limits must be positive")
    normalized = normalize_text(text)
    if not normalized:
        raise TTSError("INVALID_TEXT", "text is empty or contains only whitespace")
    segments = _split_bounded(normalized, max_chars)
    if len(segments) > max_segments:
        raise TTSError("REQUEST_TOO_LARGE", f"text requires {len(segments)} segments; limit is {max_segments}")
    if any(not segment or len(segment) > max_chars for segment in segments) or "".join(segments) != normalized:
        raise RuntimeError("internal text segmentation invariant failed")
    return normalized, segments
