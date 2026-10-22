"""Shared ZipVoice components used by both 310P and ONNX adapters.

This module owns the Emilia-style Chinese/English tokenizer, prompt profile
dataclass, duration model, sentence-aware chunk planning, tail sentinel
cropping, pause-aware concatenation, and flow-matching timestep schedule.
Both deployment adapters import from here so the reuse contract is explicit
and refactor-safe.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from voice_tts_service.errors import BackendInferenceError, BackendLoadError

logger = logging.getLogger(__name__)

_PINYIN_TAG = re.compile(r"<([A-Za-z]+[1-5])>")
_PUNCTUATION = frozenset(";:,.!?")
_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_DATE = re.compile(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)")
_TIME = re.compile(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?(?!\d)")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_RANGE = re.compile(r"(\d)\s*[-~]\s*(\d)")
# Units after a number in Chinese text are read in Chinese ("35ms" -> "35毫秒") before the
# model-name rule below would spell them as letters.
_UNIT_WORDS = {
    "ms": "毫秒",
    "s": "秒",
    "min": "分钟",
    "km": "公里",
    "m": "米",
    "cm": "厘米",
    "mm": "毫米",
    "kg": "千克",
    "Hz": "赫兹",
    "kHz": "千赫兹",
    "MHz": "兆赫兹",
    "GHz": "吉赫兹",
    "fps": "帧每秒",
}
_CHINESE_UNIT = re.compile(
    r"(?<=[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff])(\d+(?:\.\d+)?)\s*("
    + "|".join(sorted(_UNIT_WORDS, key=len, reverse=True))
    + r")(?![A-Za-z])"
)
# A number glued to a following Latin letter after Chinese text is a model name such as "昇腾310P";
# it is read digit by digit ("三一零P") instead of as a quantity ("三百一十").
_MODEL_NUMBER = re.compile(r"(?<=[㐀-䶿一-鿿豈-﫿])(\d+)(?=[A-Za-z])")
# Digit readings bypass pypinyin so tone sandhi cannot turn "三一零" into "三亦零".
_DIGIT_PINYIN = ("ling2", "yi1", "er4", "san1", "si4", "wu3", "liu4", "qi1", "ba1", "jiu3")
# Project names whose spoken form differs from the spelling espeak would read.
DEFAULT_LEXICON = {"openEuler": "open欧拉"}
# Pinyin and punctuation are the only zh tokens; anything cn2an leaves behind is unspoken.
_ZH_LEFTOVER = re.compile(r"[^㐀-䶿一-鿿豈-﫿\s]")

# "嗯。" appended after a pause: the model treats the generated span as speech that
# continues past its end, so the real final syllable is kept away from that edge.
TAIL_SENTINEL = (" ", "n0", "n2", ".")
# Natural English needs fewer frames per IPA token than the Chinese default prompt implies.
DEFAULT_ENGLISH_TOKEN_SCALE = 1.8
# The prompt's frames-per-token ratio includes its edge silence; the default prompt speaks at
# 0.98 of it, and a slightly faster rate leaves the model less surplus time to scatter as pauses.
DEFAULT_SPOKEN_TOKEN_SCALE = 0.95
# Pause length of a punctuation mark inside a chunk, in prompt-rate tokens.
_PAUSE_TOKEN_WEIGHTS = {",": 2.0, ":": 2.0, ";": 3.0, ".": 3.0, "!": 3.0, "?": 3.0}
_SPACE_TOKEN_WEIGHT = 0.5
_CLAUSE_PUNCTUATION = frozenset(".!?;,:")
# Clauses shorter than this share a chunk with the next clause; longer ones end their chunk.
DEFAULT_CLAUSE_CHUNK_TOKENS = 16
# Silence after the last chunk lets the playback chain drain before the stream ends.
DEFAULT_TAIL_SILENCE_MS = 100
_CUT_SEARCH_FRAMES = 10
_PAUSE_MS = {".": 250, "!": 250, "?": 250, ";": 200, ":": 120, ",": 120}
_WORD_BREAK_PAUSE_MS = 60
_EDGE_FADE_SEC = 0.005
_TRIM_WINDOW_SEC = 0.01
_TRIM_MARGIN_WINDOWS = 3
_TRIM_THRESHOLD_DB = -40.0


@dataclass(frozen=True)
class PromptProfile:
    """Fixed prompt tokens and features for zero-shot voice cloning."""

    tokens: np.ndarray
    features: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.features.shape[1])


@dataclass(frozen=True)
class TokenGroup:
    """Tokens of one word or punctuation mark; chunks never split inside a group."""

    tokens: tuple[str, ...]
    kind: str  # "zh", "en", "punct", "space", or "sentinel"


@dataclass(frozen=True)
class ChunkPlan:
    """One text-encoder request and the frame layout of its generated span.

    ``frames`` holds the frames of every token.  ``start_offset`` is where the
    first token starts relative to the end of the prompt audio; it is negative
    because the exported encoder layout starts the text inside the prompt's
    trailing audio, and :func:`chunk_conditions` sets it.
    """

    tokens: tuple[str, ...]
    content_tokens: int
    frames: tuple[int, ...]
    pause_after_ms: int
    start_offset: int = 0

    @property
    def has_sentinel(self) -> bool:
        return len(self.tokens) > self.content_tokens

    @property
    def target_frames(self) -> int:
        return sum(self.frames)


class DurationModel:
    """Per-token frame durations derived from the prompt's frames-per-token ratio.

    The exported text encoders (ONNX and the 310P OM) repeat every token's
    embedding for the same number of frames, so the flow decoder gets no hint
    of where pauses go and spreads surplus time as pauses between words.  The
    runtime re-lays the condition with these durations instead: punctuation
    gets pause frames, English IPA tokens get shorter frames, and spoken tokens
    use less than the prompt ratio, which includes the prompt's edge silence.
    ``speed`` scales every generated frame like the PyTorch ZipVoice option.
    """

    def __init__(
        self,
        prompt_frames: int,
        prompt_tokens: int,
        *,
        speed: float = 1.0,
        english_token_scale: float = DEFAULT_ENGLISH_TOKEN_SCALE,
        spoken_token_scale: float = DEFAULT_SPOKEN_TOKEN_SCALE,
    ) -> None:
        if prompt_frames <= 0 or prompt_tokens <= 0:
            raise BackendInferenceError("ZipVoice prompt profile must have frames and tokens")
        if speed <= 0 or english_token_scale <= 0 or spoken_token_scale <= 0:
            raise BackendInferenceError("ZipVoice speed and token scales must be positive")
        self.prompt_frames = prompt_frames
        self.prompt_tokens = prompt_tokens
        self.speed = speed
        self.english_token_scale = english_token_scale
        self.spoken_token_scale = spoken_token_scale
        self.frames_per_token = prompt_frames / prompt_tokens

    def _token_weight(self, kind: str, token: str) -> float:
        if kind == "punct":
            return _PAUSE_TOKEN_WEIGHTS.get(token, 1.0)
        if token == " ":
            return _SPACE_TOKEN_WEIGHT * self.spoken_token_scale
        if kind == "en":
            return self.spoken_token_scale / self.english_token_scale
        return self.spoken_token_scale

    def token_frames(self, groups: Sequence[TokenGroup]) -> list[int]:
        return [
            max(1, round(self._token_weight(group.kind, token) * self.frames_per_token / self.speed))
            for group in groups
            for token in group.tokens
        ]

    def target_frames(self, groups: Sequence[TokenGroup]) -> int:
        return sum(self.token_frames(groups))


def duration_model(prompt: PromptProfile, config: dict) -> DurationModel:
    """Build the duration model from a prompt profile and a deployment runtime config."""

    return DurationModel(
        prompt.frame_count,
        int(prompt.tokens.shape[1]),
        speed=float(config.get("speed", 1.0)),
        english_token_scale=float(config.get("english_token_scale", DEFAULT_ENGLISH_TOKEN_SCALE)),
        spoken_token_scale=float(config.get("spoken_token_scale", DEFAULT_SPOKEN_TOKEN_SCALE)),
    )


class EmiliaTokenizer:
    """ZipVoice Emilia-style frontend: pinyin for Chinese, espeak IPA for English.

    ``lexicon`` maps whole ASCII words to the text that should be read instead,
    for names espeak cannot pronounce the way the project says them.
    """

    def __init__(self, token_file: Path, lexicon: Mapping[str, str] | None = None) -> None:
        if lexicon is not None and not (
            isinstance(lexicon, Mapping)
            and all(isinstance(key, str) and key and isinstance(value, str) for key, value in lexicon.items())
        ):
            raise BackendLoadError("ZipVoice lexicon must map non-empty words to replacement text")
        self._lexicon = {**DEFAULT_LEXICON, **(lexicon or {})}
        self._lexicon_pattern = (
            re.compile(
                r"(?<![A-Za-z])("
                + "|".join(re.escape(word) for word in sorted(self._lexicon, key=len, reverse=True))
                + r")(?![A-Za-z])"
            )
            if self._lexicon
            else None
        )
        try:
            import cn2an
            import jieba
            import pypinyin
            from pypinyin.contrib import tone_convert
        except (ImportError, OSError) as exc:
            raise BackendLoadError(f"ZipVoice Chinese frontend dependency is unavailable: {exc}") from exc
        try:
            from piper_phonemize import phonemize_espeak
        except (ImportError, OSError):
            # English is optional: Chinese requests keep working and English ones fail explicitly.
            phonemize_espeak = None
        self._cn2an = cn2an
        self._jieba = jieba
        self._style = pypinyin.Style
        self._lazy_pinyin = pypinyin.lazy_pinyin
        self._to_finals_tone3 = tone_convert.to_finals_tone3
        self._to_initials = tone_convert.to_initials
        self._phonemize_espeak = phonemize_espeak
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
        self.tail_sentinel = TAIL_SENTINEL if all(token in self._token_to_id for token in TAIL_SENTINEL) else ()

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
            "⋯": ".",
            "…": ".",
            "···": ".",
            "・・・": ".",
            "...": ".",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        return re.sub(r"\.{2,}", ".", text)

    @staticmethod
    def _is_number_separator(text: str, index: int) -> bool:
        return (
            text[index] in ".,:"
            and 0 < index < len(text) - 1
            and text[index - 1].isdigit()
            and text[index + 1].isdigit()
        )

    @classmethod
    def _split_languages(cls, text: str) -> list[tuple[str, str]]:
        """Split text into ("zh"|"en"|"punct", text) runs.

        Digits, spaces, and spoken symbols follow the language of the preceding
        word in the same clause, so "第3号" stays Chinese and "I have 3 apples"
        stays English.  Text without any word reads numbers in Chinese.
        """

        kinds: list[str | None] = []
        for index, char in enumerate(text):
            if char in _PUNCTUATION and not cls._is_number_separator(text, index):
                kinds.append("punct")
            elif _CJK.match(char):
                kinds.append("zh")
            elif char.isascii() and char.isalpha():
                kinds.append("en")
            else:
                kinds.append(None)
        resolved: list[str] = []
        previous: str | None = None
        for index, kind in enumerate(kinds):
            if kind == "punct":
                previous = None
                resolved.append(kind)
                continue
            if kind is None:
                kind = previous
                if kind is None:
                    following = next((value for value in kinds[index + 1 :] if value != "punct" and value), None)
                    kind = following or "zh"
            else:
                previous = kind
            resolved.append(kind)
        runs: list[tuple[str, str]] = []
        for char, kind in zip(text, resolved, strict=True):
            if runs and runs[-1][0] == kind and kind != "punct":
                runs[-1] = (kind, runs[-1][1] + char)
            else:
                runs.append((kind, char))
        return runs

    def _split_pinyin(self, syllable: str) -> list[str]:
        initial = self._to_initials(syllable, strict=False)
        final = self._to_finals_tone3(syllable, strict=False, neutral_tone_with_five=True)
        return [value for value in (f"{initial}0" if initial else "", final) if value]

    @staticmethod
    def _normalize_chinese_numbers(text: str) -> str:
        """Rewrite number formats cn2an misreads (dates as negatives, clock times, thousands)."""

        text = _DATE.sub(r"\1年\2月\3日", text)
        text = _TIME.sub(lambda match: f"{match[1]}点{match[2]}分" + (f"{match[3]}秒" if match[3] else ""), text)
        text = _THOUSANDS.sub("", text)
        return _RANGE.sub(r"\1至\2", text)

    def _chinese_groups(self, text: str) -> list[TokenGroup]:
        normalized = self._cn2an.transform(self._normalize_chinese_numbers(text), "an2cn")
        normalized = re.sub(r"\s+", " ", _ZH_LEFTOVER.sub(" ", normalized)).strip()
        if not normalized:
            return []
        words = list(self._jieba.cut(normalized))
        syllables = self._lazy_pinyin(
            words,
            style=self._style.TONE3,
            tone_sandhi=True,
            neutral_tone_with_five=True,
        )
        # pypinyin returns one syllable per hanzi and one item per non-hanzi word.
        counts = [len(word) if _CJK.match(word[0]) else 1 for word in words]
        if sum(counts) != len(syllables):
            counts = [1] * len(syllables)
        groups: list[TokenGroup] = []
        offset = 0
        for count in counts:
            tokens: list[str] = []
            for syllable in syllables[offset : offset + count]:
                if syllable[:-1].isalpha() and syllable[-1:] in "12345":
                    tokens.extend(self._split_pinyin(syllable))
                else:
                    tokens.append(syllable)
            offset += count
            if tokens == [" "]:
                groups.append(TokenGroup((" ",), "space"))
            elif tokens:
                groups.append(TokenGroup(tuple(tokens), "zh"))
        return groups

    def _english_groups(self, text: str) -> list[TokenGroup]:
        if self._phonemize_espeak is None:
            raise BackendInferenceError("English text requires piper_phonemize, which is not installed")
        phonemes = [token for sentence in self._phonemize_espeak(text.strip(), "en-us") for token in sentence]
        groups: list[TokenGroup] = []
        word: list[str] = []
        for token in [*phonemes, " "]:
            if token != " ":
                word.append(token)
                continue
            if word:
                groups.append(TokenGroup(tuple(word), "en"))
                word = []
            groups.append(TokenGroup((" ",), "space"))
        return groups[:-1]

    def _digit_groups(self, digits: str) -> list[TokenGroup]:
        return [TokenGroup(tuple(self._split_pinyin(_DIGIT_PINYIN[int(digit)])), "zh") for digit in digits]

    def _text_groups(self, text: str) -> list[TokenGroup]:
        groups: list[TokenGroup] = []
        for kind, run in self._split_languages(text):
            if kind == "punct":
                groups.append(TokenGroup((run,), "punct"))
            elif kind == "zh":
                groups.extend(self._chinese_groups(run))
            else:
                groups.extend(self._english_groups(run))
        return groups

    def text_to_groups(self, text: str) -> list[TokenGroup]:
        if self._lexicon_pattern is not None:
            text = self._lexicon_pattern.sub(lambda match: self._lexicon[match[1]], text)
        text = _CHINESE_UNIT.sub(lambda match: match[1] + _UNIT_WORDS[match[2]], text)
        mapped = self._map_punctuation(text.strip())
        if not mapped:
            raise BackendInferenceError("text must not be empty")
        if mapped[-1] not in _PUNCTUATION:
            mapped += "."
        if _PINYIN_TAG.search(mapped):
            raise BackendInferenceError("the ZipVoice frontend does not support inline <pinyin3> tags")
        groups: list[TokenGroup] = []
        position = 0
        for match in _MODEL_NUMBER.finditer(mapped):
            groups.extend(self._text_groups(mapped[position : match.start()]))
            groups.extend(self._digit_groups(match[1]))
            position = match.end()
        groups.extend(self._text_groups(mapped[position:]))
        unknown = list(
            dict.fromkeys(token for group in groups for token in group.tokens if token not in self._token_to_id)
        )
        if unknown:
            raise BackendInferenceError(f"ZipVoice token table does not contain: {unknown}")
        return groups

    def text_to_tokens(self, text: str) -> list[str]:
        return [token for group in self.text_to_groups(text) for token in group.tokens]

    def tokens_to_ids(self, tokens: Sequence[str]) -> list[int]:
        return [self._token_to_id[token] for token in tokens]


def _split_after(groups: Sequence[TokenGroup], punctuation: frozenset[str]) -> list[list[TokenGroup]]:
    parts: list[list[TokenGroup]] = []
    current: list[TokenGroup] = []
    for group in groups:
        current.append(group)
        if group.kind == "punct" and group.tokens[0] in punctuation:
            parts.append(current)
            current = []
    if current:
        parts.append(current)
    return parts


def _strip_spaces(groups: list[TokenGroup]) -> list[TokenGroup]:
    start, end = 0, len(groups)
    while start < end and groups[start].kind == "space":
        start += 1
    while end > start and groups[end - 1].kind == "space":
        end -= 1
    return groups[start:end]


def plan_chunks(
    groups: Sequence[TokenGroup],
    durations: DurationModel,
    *,
    max_tokens: int,
    max_target_frames: float,
    sentinel: Sequence[str] = (),
    clause_chunk_tokens: int | None = None,
) -> list[ChunkPlan]:
    """Pack whole clauses into chunks, falling back to words for an oversized clause.

    Every chunk reserves room for the tail sentinel so the final syllable of each
    chunk is not generated at the edge of the frame budget.  With
    ``clause_chunk_tokens``, a chunk ends after any clause once it holds that
    many tokens, so punctuation pauses come from the exact silence inserted
    between chunks instead of the model's guess, and the model has less room to
    drift from the planned layout.
    """

    sentinel_group = TokenGroup(tuple(sentinel), "sentinel")
    reserve_tokens = len(sentinel)
    reserve_frames = durations.target_frames([sentinel_group]) if sentinel else 0.0
    if max_tokens <= reserve_tokens or max_target_frames <= reserve_frames:
        raise BackendInferenceError("ZipVoice chunk capacity cannot fit the tail sentinel")

    def fits(candidate: Sequence[TokenGroup]) -> bool:
        tokens = sum(len(group.tokens) for group in candidate) + reserve_tokens
        return tokens <= max_tokens and durations.target_frames(candidate) + reserve_frames <= max_target_frames

    chunks: list[list[TokenGroup]] = []
    current: list[TokenGroup] = []

    def flush() -> None:
        nonlocal current
        stripped = _strip_spaces(current)
        if stripped:
            chunks.append(stripped)
        current = []

    def add(parts: Sequence[Sequence[TokenGroup]], fallback) -> None:
        nonlocal current
        for part in parts:
            if fits([*current, *part]):
                current.extend(part)
                continue
            flush()
            if fits(part):
                current = list(part)
            else:
                fallback(part)

    def add_words(clause: Sequence[TokenGroup]) -> None:
        def reject(group: Sequence[TokenGroup]) -> None:
            raise BackendInferenceError("a single ZipVoice word exceeds the chunk capacity")

        add([[group] for group in clause], reject)

    for clause in _split_after(groups, _CLAUSE_PUNCTUATION):
        add([clause], add_words)
        if clause_chunk_tokens is not None and sum(len(group.tokens) for group in current) >= clause_chunk_tokens:
            flush()
    flush()

    plans = []
    for chunk in chunks:
        content = tuple(token for group in chunk for token in group.tokens)
        frames = durations.token_frames(chunk) + (durations.token_frames([sentinel_group]) if sentinel else [])
        last = chunk[-1]
        pause = _PAUSE_MS.get(last.tokens[0], _WORD_BREAK_PAUSE_MS) if last.kind == "punct" else _WORD_BREAK_PAUSE_MS
        plans.append(
            ChunkPlan(
                tokens=content + tuple(sentinel),
                content_tokens=len(content),
                frames=tuple(frames),
                pause_after_ms=pause,
            )
        )
    return plans


def content_frame_count(plan: ChunkPlan, generated: np.ndarray) -> int:
    """Return how many generated frames belong to the request text.

    The cut lands on the quietest frame near the planned content end, which is
    the pause between the final punctuation and the sentinel.
    """

    frame_count = int(generated.shape[0])
    if not plan.has_sentinel or frame_count < 2:
        return frame_count
    content_end = plan.start_offset + sum(plan.frames[: plan.content_tokens])
    low = max(1, content_end - _CUT_SEARCH_FRAMES)
    high = min(frame_count - 1, content_end + _CUT_SEARCH_FRAMES)
    if low >= high:
        return max(1, min(content_end, frame_count))
    energy = np.convolve(generated.mean(axis=1), np.ones(3, dtype=np.float32) / 3.0, mode="same")
    return low + int(np.argmin(energy[low:high]))


def token_embeddings(text_condition: np.ndarray, token_count: int, features_len: int) -> np.ndarray | None:
    """Recover one row per token from the exported encoder's evenly repeated condition.

    Returns ``None`` when the rows are not the expected repeats, so callers keep
    the encoder's own layout instead of guessing.
    """

    token_dur = features_len // token_count if token_count > 0 else 0
    if token_dur < 1 or text_condition.shape[0] < token_count * token_dur:
        return None
    starts = np.arange(token_count) * token_dur
    embeddings = text_condition[starts]
    if not np.array_equal(embeddings, text_condition[starts + token_dur - 1]):
        return None
    return embeddings


def layout_condition(
    embeddings: np.ndarray, prompt_tokens: int, prompt_frames: int, frames: Sequence[int]
) -> np.ndarray:
    """Repeat prompt tokens as the exported encoder does and target tokens by their planned frames.

    Prompt tokens keep the encoder's ``prompt_frames // prompt_tokens`` frames each,
    so the first target token starts inside the prompt's trailing audio exactly as
    the model expects; starting it after the prompt makes the model drop the first
    syllable.
    """

    counts = np.full(prompt_tokens, prompt_frames // prompt_tokens, dtype=np.int64)
    counts = np.concatenate([counts, np.asarray(frames, dtype=np.int64)])
    return np.repeat(embeddings[: prompt_tokens + len(frames)], counts, axis=0)


def chunk_conditions(
    plan: ChunkPlan,
    prompt: PromptProfile,
    text_condition: np.ndarray,
    features_len: int,
) -> tuple[np.ndarray, np.ndarray, ChunkPlan]:
    """Return the text condition, speech condition, and effective plan for one chunk.

    The encoder repeats every token's embedding for the same number of frames,
    which gives the model no hint of pause lengths or of English being faster.
    The target tokens are re-laid with the plan's per-token durations.  If the
    encoder output is not the expected repeat layout, its own condition is used.
    """

    prompt_tokens = int(prompt.tokens.shape[1])
    embeddings = token_embeddings(text_condition, prompt_tokens + len(plan.tokens), features_len)
    if embeddings is None:
        logger.warning("ZipVoice text condition is not token-repeated; keeping the encoder layout")
        token_dur = max(1, features_len // (prompt_tokens + len(plan.tokens)))
        plan = replace(plan, frames=(token_dur,) * len(plan.tokens))
        target_start = prompt_tokens * token_dur
        condition = text_condition[:features_len]
    else:
        target_start = prompt_tokens * (prompt.frame_count // prompt_tokens)
        condition = layout_condition(embeddings, prompt_tokens, prompt.frame_count, plan.frames)
    speech = np.zeros_like(condition)
    speech[: prompt.frame_count] = prompt.features[0, : condition.shape[0]]
    return condition, speech, replace(plan, start_offset=target_start - prompt.frame_count)


def _trim_silence(wave: np.ndarray, sample_rate: int) -> np.ndarray:
    window = max(1, int(sample_rate * _TRIM_WINDOW_SEC))
    count = wave.size // window
    if count == 0:
        return wave
    rms = np.sqrt(np.mean(wave[: count * window].reshape(count, window) ** 2, axis=1))
    peak = float(rms.max())
    if peak <= 0.0:
        return wave
    voiced = np.flatnonzero(rms > peak * 10.0 ** (_TRIM_THRESHOLD_DB / 20.0))
    start = max(0, int(voiced[0]) - _TRIM_MARGIN_WINDOWS) * window
    end = min(wave.size, (int(voiced[-1]) + 1 + _TRIM_MARGIN_WINDOWS) * window)
    return wave[start:end]


def _fade_edges(wave: np.ndarray, sample_rate: int) -> np.ndarray:
    count = min(int(sample_rate * _EDGE_FADE_SEC), wave.size // 2)
    if count <= 0:
        return wave
    wave = wave.copy()
    ramp = np.linspace(0.0, 1.0, count, endpoint=False, dtype=np.float32)
    wave[:count] *= ramp
    wave[-count:] *= ramp[::-1]
    return wave


def join_chunks(
    waves: Sequence[np.ndarray],
    pauses_ms: Sequence[int],
    sample_rate: int,
    *,
    tail_silence_ms: int = 0,
) -> np.ndarray:
    """Trim chunk edge silence and join chunks with punctuation-dependent pauses."""

    pieces: list[np.ndarray] = []
    for index, wave in enumerate(waves):
        trimmed = _fade_edges(_trim_silence(np.asarray(wave, dtype=np.float32), sample_rate), sample_rate)
        pieces.append(trimmed)
        pause_ms = pauses_ms[index] if index < len(waves) - 1 else tail_silence_ms
        pieces.append(np.zeros(int(sample_rate * max(0, pause_ms) / 1000), dtype=np.float32))
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


def timesteps(num_steps: int, t_shift: float) -> np.ndarray:
    """Return the flow-matching timestep schedule following the ZipVoice convention."""

    raw = np.linspace(0.0, 1.0, num_steps + 1, dtype=np.float32)
    return t_shift * raw / (1.0 + (t_shift - 1.0) * raw)


__all__ = [
    "DEFAULT_ENGLISH_TOKEN_SCALE",
    "DEFAULT_LEXICON",
    "DEFAULT_CLAUSE_CHUNK_TOKENS",
    "DEFAULT_SPOKEN_TOKEN_SCALE",
    "DEFAULT_TAIL_SILENCE_MS",
    "ChunkPlan",
    "DurationModel",
    "EmiliaTokenizer",
    "PromptProfile",
    "TAIL_SENTINEL",
    "TokenGroup",
    "chunk_conditions",
    "content_frame_count",
    "duration_model",
    "join_chunks",
    "layout_condition",
    "plan_chunks",
    "timesteps",
    "token_embeddings",
]
