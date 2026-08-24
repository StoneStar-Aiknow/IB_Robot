"""Deterministic text normalization and segmentation for ZipVoice requests."""

from __future__ import annotations

import re
import unicodedata

from voice_tts_service.errors import TTSError

_PRIMARY_BOUNDARIES = frozenset("。！？!?；;")
_SECONDARY_BOUNDARIES = frozenset("，、：:,")
_SUPPORTED_PUNCTUATION = _PRIMARY_BOUNDARIES | _SECONDARY_BOUNDARIES | frozenset(".…")
_FORMAT_MARKS = frozenset("#*_`~'\"‘’“”()（）[]【】{}<>《》")


def _canonical_char(char: str) -> str:
    normalized = unicodedata.normalize("NFKC", char)
    return normalized if len(normalized) == 1 else char


def _is_supported_punctuation(char: str) -> bool:
    return _canonical_char(char) in _SUPPORTED_PUNCTUATION


def _is_format_mark(char: str) -> bool:
    return _canonical_char(char) in _FORMAT_MARKS


def _is_special_symbol(char: str) -> bool:
    codepoint = ord(char)
    category = unicodedata.category(char)
    is_variation_selector = 0xFE00 <= codepoint <= 0xFE0F or 0xE0100 <= codepoint <= 0xE01EF
    return (
        category.startswith("S")
        or is_variation_selector
        or char == "\u20e3"
        or (category.startswith("P") and not _is_supported_punctuation(char) and not _is_format_mark(char))
    )


def _nearest_content_char(text: str, start: int, step: int) -> str:
    index = start
    while 0 <= index < len(text):
        char = text[index]
        if not char.isspace() and not _is_format_mark(char) and unicodedata.category(char) != "Cf":
            return char
        index += step
    return ""


def _append_sentence_boundary(result: list[str]) -> None:
    while result and result[-1].isspace():
        result.pop()
    if not result or not _is_supported_punctuation(result[-1]):
        result.append("。")


def _sanitize_symbols(text: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if _is_format_mark(char) or unicodedata.category(char) == "Cf":
            index += 1
            continue
        if char in "\r\n":
            start = index
            while index < len(text) and text[index].isspace():
                index += 1
            before = _nearest_content_char(text, start - 1, -1)
            after = _nearest_content_char(text, index, 1)
            if before and after and not _is_supported_punctuation(before) and not _is_supported_punctuation(after):
                _append_sentence_boundary(result)
            continue
        if _is_special_symbol(char):
            start = index
            while index < len(text) and (
                _is_special_symbol(text[index])
                or _is_format_mark(text[index])
                or unicodedata.category(text[index]) == "Cf"
            ):
                index += 1
            before = _nearest_content_char(text, start - 1, -1)
            after = _nearest_content_char(text, index, 1)
            if not _is_supported_punctuation(before) and not _is_supported_punctuation(after):
                _append_sentence_boundary(result)
            continue
        if unicodedata.category(char).startswith("C"):
            index += 1
            continue
        result.append(char)
        index += 1
    return "".join(result)


def normalize_text(text: str) -> str:
    """Remove unsupported formatting and retain pauses around unspoken symbols."""

    normalized = unicodedata.normalize("NFKC", _sanitize_symbols(text))
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
