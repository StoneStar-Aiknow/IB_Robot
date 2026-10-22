"""Deterministic text normalization and segmentation for ZipVoice requests."""

from __future__ import annotations

import re
import unicodedata

from voice_tts_service.errors import TTSError

_PRIMARY_BOUNDARIES = frozenset("。！？!?；;.")
_SECONDARY_BOUNDARIES = frozenset("，、：:,")
_SUPPORTED_PUNCTUATION = _PRIMARY_BOUNDARIES | _SECONDARY_BOUNDARIES | frozenset("…")
_FORMAT_MARKS = frozenset("#*_`~'\"‘’“”()（）[]【】{}<>《》")
# Symbols the frontends read aloud after a number: cn2an and espeak both verbalize them.
_UNIT_SYMBOLS = frozenset("%‰℃℉°")
# Separators that are spoken inside numbers or words: dates, ranges, fractions, "state-of-the-art".
_NUMBER_SEPARATORS = frozenset("-~/:")
_WORD_SEPARATORS = frozenset("-'/")
# Inline separators that are not spoken still mark only a short clause pause.
_INLINE_SEPARATORS = frozenset("-‐‑‒–—―/\\|~")


def _canonical_char(char: str) -> str:
    normalized = unicodedata.normalize("NFKC", char)
    if char in "‘’":
        return "'"
    return normalized if len(normalized) == 1 else char


def _is_ascii_letter(char: str) -> bool:
    return char.isascii() and char.isalpha()


def _is_spoken_symbol(text: str, index: int) -> bool:
    """Return whether a symbol carries meaning the frontend can verbalize in this context."""

    char = _canonical_char(text[index])
    before = text[index - 1] if index > 0 else ""
    after = text[index + 1] if index + 1 < len(text) else ""
    if char in _UNIT_SYMBOLS:
        return before.isdigit()
    if char in _NUMBER_SEPARATORS and before.isdigit() and after.isdigit():
        return True
    if char in _WORD_SEPARATORS and _is_ascii_letter(before) and _is_ascii_letter(after):
        return True
    # A leading minus sign is read as "负" or "minus"; after a word or digit it is a separator.
    return char == "-" and after.isdigit() and not (before.isascii() and before.isalnum())


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


def _append_sentence_boundary(result: list[str], boundary: str = "。") -> None:
    while result and result[-1].isspace():
        result.pop()
    if not result or not _is_supported_punctuation(result[-1]):
        result.append(boundary)


def _sanitize_symbols(text: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if _is_spoken_symbol(text, index):
            result.append(_canonical_char(char))
            index += 1
            continue
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
                inline = all(_canonical_char(value) in _INLINE_SEPARATORS for value in text[start:index].strip())
                _append_sentence_boundary(result, "，" if inline else "。")
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
    # NFKC expands the temperature signs into "°C"/"°F"; keep them as one unit symbol.
    normalized = normalized.replace("°C", "℃").replace("°F", "℉")
    return re.sub(r"\s+", " ", normalized).strip()


def _is_number_separator(text: str, index: int) -> bool:
    return 0 < index < len(text) - 1 and text[index - 1].isdigit() and text[index + 1].isdigit()


def _is_token_char(text: str, index: int) -> bool:
    char = text[index]
    return (char.isascii() and char.isalnum()) or _is_number_separator(text, index)


def _avoid_splitting_token(text: str, limit: int) -> int:
    """Move a forced character-window split to the start of a number or ASCII word when possible."""

    boundary = limit
    while boundary > 0 and _is_token_char(text, boundary - 1) and _is_token_char(text, boundary):
        boundary -= 1
    return boundary or limit


def _candidate_boundary(text: str, limit: int, boundaries: frozenset[str]) -> int | None:
    for index in range(min(limit, len(text)) - 1, -1, -1):
        if text[index] in boundaries and not _is_number_separator(text, index):
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
            # English text without punctuation still splits between words.
            boundary = _candidate_boundary(remaining, max_chars, frozenset(" "))
        if boundary is None:
            boundary = _avoid_splitting_token(remaining, max_chars)
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
