"""Shared ZipVoice components used by both 310P and ONNX adapters.

This module owns the Chinese tokenizer, prompt profile dataclass, cross-fade
concatenation, and flow-matching timestep schedule.  Both deployment adapters
import from here so the reuse contract is explicit and refactor-safe.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voice_tts_service.errors import BackendInferenceError, BackendLoadError

_PINYIN_TAG = re.compile(r"<([A-Za-z]+[1-5])>")
_ASCII_LETTER = re.compile(r"[A-Za-z]")
_PUNCTUATION = {";", ":", ",", ".", "!", "?", "…"}


@dataclass(frozen=True)
class PromptProfile:
    """Fixed prompt tokens and features for zero-shot voice cloning."""

    tokens: np.ndarray
    features: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.features.shape[1])


class ChineseTokenizer:
    """ZipVoice Emilia-style Chinese frontend using project dependencies."""

    def __init__(self, token_file: Path) -> None:
        try:
            import cn2an
            import jieba
            import pypinyin
            from pypinyin.contrib import tone_convert
        except (ImportError, OSError) as exc:
            raise BackendLoadError(f"ZipVoice Chinese frontend dependency is unavailable: {exc}") from exc
        self._cn2an = cn2an
        self._jieba = jieba
        self._style = pypinyin.Style
        self._lazy_pinyin = pypinyin.lazy_pinyin
        self._to_finals_tone3 = tone_convert.to_finals_tone3
        self._to_initials = tone_convert.to_initials
        self._jieba.default_logger.setLevel(logging.WARNING)
        self._jieba.initialize()
        self._token_to_id: dict[str, int] = {}
        try:
            for line in token_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    token, value = line.split("\t")[:2]
                    self._token_to_id[token] = int(value)
        except (OSError, ValueError) as exc:
            raise BackendLoadError(f"failed to read ZipVoice token table {token_file}: {exc}") from exc
        try:
            self.pad_id = self._token_to_id["_"]
        except KeyError as exc:
            raise BackendLoadError("ZipVoice token table does not define '_' padding") from exc

    @staticmethod
    def _map_punctuation(text: str) -> str:
        replacements = {
            "，": ",",
            "。": ".",
            "！": "!",
            "？": "?",
            "；": ";",
            "：": ":",
            "、": ",",
            "‘": "'",
            "“": '"',
            "”": '"',
            "’": "'",
            "⋯": "…",
            "···": "…",
            "・・・": "…",
            "...": "…",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        return text

    def _split_pinyin(self, syllable: str) -> list[str]:
        initial = self._to_initials(syllable, strict=False)
        final = self._to_finals_tone3(syllable, strict=False, neutral_tone_with_five=True)
        return [value for value in (f"{initial}0" if initial else "", final) if value]

    def text_to_tokens(self, text: str) -> list[str]:
        mapped = self._map_punctuation(text.strip())
        if not mapped:
            raise BackendInferenceError("text must not be empty")
        if mapped[-1] not in _PUNCTUATION:
            mapped += "."
        if _PINYIN_TAG.search(mapped):
            raise BackendInferenceError("the verified 310P frontend does not support inline <pinyin3> tags")
        if _ASCII_LETTER.search(mapped):
            raise BackendInferenceError(
                "the verified 310P frontend supports Chinese, Arabic numbers, and punctuation but not English words"
            )
        normalized = self._cn2an.transform(mapped, "an2cn")
        words = list(self._jieba.cut(normalized))
        syllables = self._lazy_pinyin(
            words,
            style=self._style.TONE3,
            tone_sandhi=True,
            neutral_tone_with_five=True,
        )
        tokens: list[str] = []
        for syllable in syllables:
            if syllable[:-1].isalpha() and syllable[-1:] in "12345":
                tokens.extend(self._split_pinyin(syllable))
            else:
                tokens.append(syllable)
        unknown = list(dict.fromkeys(token for token in tokens if token not in self._token_to_id))
        if unknown:
            raise BackendInferenceError(f"ZipVoice token table does not contain: {unknown}")
        return tokens

    def tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self._token_to_id[token] for token in tokens]

    @staticmethod
    def chunk_tokens(tokens: list[str], max_tokens: int) -> list[list[str]]:
        if max_tokens <= 0:
            raise BackendInferenceError("ZipVoice token capacity must be positive")
        sentences: list[list[str]] = []
        current: list[str] = []
        for token in tokens:
            current.append(token)
            if token in _PUNCTUATION:
                sentences.append(current)
                current = []
        if current:
            sentences.append(current)

        chunks: list[list[str]] = []
        current = []
        for sentence in sentences:
            while len(sentence) > max_tokens:
                if current:
                    chunks.append(current)
                    current = []
                split_at = max_tokens
                if split_at > 1 and sentence[split_at - 1].endswith("0"):
                    split_at -= 1
                chunks.append(sentence[:split_at])
                sentence = sentence[split_at:]
            if len(current) + len(sentence) <= max_tokens:
                current.extend(sentence)
            else:
                if current:
                    chunks.append(current)
                current = list(sentence)
        if current:
            chunks.append(current)
        return chunks


def cross_fade_concat(waves: list[np.ndarray], sample_rate: int, seconds: float) -> np.ndarray:
    """Concatenate wave segments with a linear cross-fade at segment boundaries."""

    result = np.asarray(waves[0], dtype=np.float32)
    nominal = int(round(sample_rate * seconds))
    for wave in waves[1:]:
        wave = np.asarray(wave, dtype=np.float32)
        count = min(nominal, result.size, wave.size)
        if count <= 0:
            result = np.concatenate([result, wave])
            continue
        fade_in = np.linspace(0.0, 1.0, count, endpoint=False, dtype=np.float32)
        merged = result[-count:] * (1.0 - fade_in) + wave[:count] * fade_in
        result = np.concatenate([result[:-count], merged, wave[count:]])
    return result


def timesteps(num_steps: int, t_shift: float) -> np.ndarray:
    """Return the flow-matching timestep schedule following the ZipVoice convention."""

    raw = np.linspace(0.0, 1.0, num_steps + 1, dtype=np.float32)
    return t_shift * raw / (1.0 + (t_shift - 1.0) * raw)


__all__ = ["ChineseTokenizer", "PromptProfile", "cross_fade_concat", "timesteps"]
