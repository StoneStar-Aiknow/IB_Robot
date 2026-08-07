"""Verified ZipVoice-Distill adapter for the Ascend 310P1 bucket OMs."""

from __future__ import annotations

import importlib
import json
import logging
import re
import sys
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from voice_tts_service.backend import AudioResult
from voice_tts_service.errors import BackendInferenceError, BackendLoadError, TTSError
from voice_tts_service.prompt_audio import PromptAudio

_PINYIN_TAG = re.compile(r"<([A-Za-z]+[1-5])>")
_ASCII_LETTER = re.compile(r"[A-Za-z]")
_PUNCTUATION = {";", ":", ",", ".", "!", "?", "…"}


@contextmanager
def _temporary_sys_path(path: Path):
    """Temporarily prioritize a bundle-vendored import path."""

    entry = str(path)
    sys.path.insert(0, entry)
    try:
        yield
    finally:
        with suppress(ValueError):
            sys.path.remove(entry)


@dataclass(frozen=True)
class _PromptProfile:
    tokens: np.ndarray
    features: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.features.shape[1])


class _ChineseTokenizer:
    """ZipVoice Emilia-style Chinese frontend with bundle-vendored dependencies."""

    def __init__(self, token_file: Path, vendor_python: Path) -> None:
        try:
            import_context = _temporary_sys_path(vendor_python) if vendor_python.is_dir() else nullcontext()
            with import_context:
                self._cn2an = importlib.import_module("cn2an")
                self._jieba = importlib.import_module("jieba")
                pypinyin = importlib.import_module("pypinyin")
                tone_convert = importlib.import_module("pypinyin.contrib.tone_convert")
        except (ImportError, OSError) as exc:
            raise BackendLoadError(f"ZipVoice Chinese frontend dependency is unavailable: {exc}") from exc
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


class ZipVoice310PAdapter:
    """Run pad-aware text OM, iterative flow OM, and CPU Vocos."""

    runtime_version = "ascend-acl"

    def __init__(self, validated_manifest, runtime_options: dict[str, Any]) -> None:
        self._validated = validated_manifest
        self._root = validated_manifest.bundle_root
        self._runtime_options = dict(runtime_options)
        self._config = self._load_json(self._root / "assets" / "zipvoice_310p.json")
        self._text_session = None
        self._flow_session = None
        self._tokenizer = None
        self._prompt = None
        self._vocos = None
        self._torch = None

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BackendLoadError(f"failed to read ZipVoice 310P config {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise BackendLoadError("ZipVoice 310P config must be a JSON object")
        return value

    def _asset(self, field: str) -> Path:
        value = self._config.get(field)
        if not isinstance(value, str) or not value:
            raise BackendLoadError(f"ZipVoice 310P config field {field!r} must be a relative path")
        path = (self._root / value).resolve()
        try:
            path.relative_to(self._root)
        except ValueError as exc:
            raise BackendLoadError(f"ZipVoice asset path escapes bundle root: {value}") from exc
        if not path.is_file() and not path.is_dir():
            raise BackendLoadError(f"ZipVoice asset is unavailable: {path}")
        return path

    def load(self) -> None:
        try:
            from inference_service.model_sessions import AscendOmRoleSession
        except (ImportError, OSError) as exc:
            raise BackendLoadError(f"shared Ascend role session is unavailable: {exc}") from exc
        deployment = self._validated.deployment
        if deployment.target.soc != "Ascend310P1":
            raise BackendLoadError(f"verified ZipVoice OM requires Ascend310P1, got {deployment.target.soc!r}")
        text_role = str(self._config.get("text_role", "text_encoder"))
        flow_role = str(self._config.get("flow_role", "flow_decoder_1537"))
        device_id = int(self._runtime_options.get("device_id", 0))
        acl_config_path = self._runtime_options.get("acl_config_path")
        self._text_session = AscendOmRoleSession(
            self._validated,
            text_role,
            device_id=device_id,
            acl_config_path=acl_config_path,
        )
        self._flow_session = AscendOmRoleSession(
            self._validated,
            flow_role,
            device_id=device_id,
            acl_config_path=acl_config_path,
        )
        try:
            self._text_session.load()
            self._flow_session.load()
            self.runtime_version = self._flow_session.runtime_version or "ascend-acl"
            self._load_assets()
        except Exception:
            self.close()
            raise

    def _load_assets(self) -> None:
        profile_name = str(self._runtime_options.get("prompt_profile", "default"))
        profiles = self._config.get("prompt_profiles", {})
        if not isinstance(profiles, dict) or profile_name not in profiles:
            raise BackendLoadError(f"unknown ZipVoice prompt profile {profile_name!r}")
        profile_relative = profiles[profile_name]
        if not isinstance(profile_relative, str) or not profile_relative:
            raise BackendLoadError(f"ZipVoice prompt profile {profile_name!r} must be a relative path")
        profile_path = (self._root / profile_relative).resolve()
        try:
            profile_path.relative_to(self._root)
        except ValueError as exc:
            raise BackendLoadError(f"ZipVoice prompt profile escapes bundle root: {profile_relative}") from exc
        try:
            with np.load(profile_path, allow_pickle=False) as profile:
                tokens = np.asarray(profile["prompt_tokens"], dtype=np.int64)
                features = np.asarray(profile["prompt_features"], dtype=np.float32)
        except (OSError, KeyError, ValueError) as exc:
            raise BackendLoadError(f"failed to load ZipVoice prompt profile {profile_path}: {exc}") from exc
        if tokens.ndim != 2 or tokens.shape[0] != 1 or features.ndim != 3 or features.shape[0] != 1:
            raise BackendLoadError("ZipVoice prompt profile tensor shapes are invalid")
        if features.shape[2] != 100 or not np.isfinite(features).all():
            raise BackendLoadError("ZipVoice prompt profile features are invalid")
        self._prompt = _PromptProfile(tokens=tokens, features=features)
        self._tokenizer = _ChineseTokenizer(self._asset("tokens_path"), self._asset("vendor_python_path"))
        self._load_vocos()

    def _load_vocos(self) -> None:
        vendor_root = self._asset("vocos_vendor_path").parent
        try:
            with _temporary_sys_path(vendor_root):
                torch = importlib.import_module("torch")
                heads = importlib.import_module("vocos.heads")
                models = importlib.import_module("vocos.models")
        except (ImportError, OSError) as exc:
            raise BackendLoadError(f"ZipVoice Vocos dependency is unavailable: {exc}") from exc

        class VocosDecoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = models.VocosBackbone(
                    input_channels=100,
                    dim=512,
                    intermediate_dim=1536,
                    num_layers=8,
                )
                self.head = heads.ISTFTHead(dim=512, n_fft=1024, hop_length=256, padding="center")

            def forward(self, features):
                return self.head(self.backbone(features))

        vocos = VocosDecoder()
        checkpoint = torch.load(self._asset("vocos_checkpoint_path"), map_location="cpu", weights_only=True)
        incompatible = vocos.load_state_dict(checkpoint, strict=False)
        if incompatible.missing_keys:
            raise BackendLoadError(f"ZipVoice Vocos checkpoint is missing keys: {incompatible.missing_keys}")
        unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("feature_extractor.")]
        if unexpected:
            raise BackendLoadError(f"ZipVoice Vocos checkpoint has unexpected keys: {unexpected}")
        self._torch = torch
        self._vocos = vocos.eval()

    @staticmethod
    def _timesteps(num_steps: int, t_shift: float) -> np.ndarray:
        raw = np.linspace(0.0, 1.0, num_steps + 1, dtype=np.float32)
        return t_shift * raw / (1.0 + (t_shift - 1.0) * raw)

    @staticmethod
    def _max_target_tokens(
        flow_frames: int,
        prompt_frames: int,
        prompt_tokens: int,
        speed: float,
        text_capacity: int,
    ) -> int:
        for target in range(text_capacity, 0, -1):
            frames = int(np.ceil(prompt_frames / prompt_tokens * (prompt_tokens + target) / speed))
            if frames <= flow_frames:
                return target
        raise BackendInferenceError("flow bucket cannot fit one target token")

    @staticmethod
    def _cross_fade_concat(waves: list[np.ndarray], sample_rate: int, seconds: float) -> np.ndarray:
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

    def synthesize(self, text: str, prompt_audio: PromptAudio | None, prompt_text: str) -> AudioResult:
        if prompt_audio is not None or prompt_text.strip():
            raise TTSError(
                "UNSUPPORTED_PROMPT",
                "the verified Ascend310P1 deployment uses a fixed prompt profile and does not support request prompts",
            )
        if any(value is None for value in (self._text_session, self._flow_session, self._tokenizer, self._prompt)):
            raise BackendInferenceError("ZipVoice 310P adapter is not loaded")
        config = self._config
        text_capacity = int(config.get("text_capacity", 256))
        flow_frames = int(config.get("flow_frames", 1537))
        speed = float(config.get("speed", 1.0))
        num_steps = int(config.get("num_steps", 4))
        t_shift = float(config.get("t_shift", 0.5))
        sample_rate = int(config.get("sample_rate", 24000))
        prompt = self._prompt
        token_strings = self._tokenizer.text_to_tokens(text)
        chunk_capacity = self._max_target_tokens(
            flow_frames,
            prompt.frame_count,
            int(prompt.tokens.shape[1]),
            speed,
            text_capacity,
        )
        chunks = self._tokenizer.chunk_tokens(token_strings, chunk_capacity)
        timesteps = self._timesteps(num_steps, t_shift)
        rng = np.random.default_rng(int(config.get("seed", 42)))
        waves = [self._synthesize_chunk(chunk, rng, timesteps) for chunk in chunks]
        if not waves:
            raise BackendInferenceError("ZipVoice frontend produced no token chunks")
        wave = self._cross_fade_concat(waves, sample_rate, float(config.get("cross_fade_sec", 0.1)))
        if wave.size == 0 or not np.isfinite(wave).all():
            raise BackendInferenceError("ZipVoice produced invalid audio")
        return AudioResult(samples=np.clip(wave, -1.0, 1.0), sample_rate=sample_rate)

    def _synthesize_chunk(self, tokens: list[str], rng, timesteps: np.ndarray) -> np.ndarray:
        config = self._config
        prompt = self._prompt
        text_capacity = int(config.get("text_capacity", 256))
        flow_frames = int(config.get("flow_frames", 1537))
        ids = self._tokenizer.tokens_to_ids(tokens)
        padded = np.full((1, text_capacity), self._tokenizer.pad_id, dtype=np.int64)
        padded[0, : len(ids)] = ids
        text_outputs = self._text_session.infer(
            {
                "tts.tokens": padded,
                "tts.tokens_len": np.asarray(len(ids), dtype=np.int64),
                "tts.prompt_tokens": prompt.tokens,
                "tts.prompt_features_len": np.asarray(prompt.frame_count, dtype=np.int64),
                "tts.speed": np.asarray(float(config.get("speed", 1.0)), dtype=np.float32),
            }
        )
        text_full = np.asarray(text_outputs["internal.text_condition"], dtype=np.float32).reshape(1, -1, 100)
        features_len = int(np.asarray(text_outputs["internal.features_len"]).reshape(()))
        mask_full = np.asarray(text_outputs["internal.padding_mask"], dtype=np.bool_).reshape(1, -1)
        if not prompt.frame_count < features_len <= flow_frames:
            raise BackendInferenceError(
                f"ZipVoice chunk requires {features_len} frames outside supported range "
                f"({prompt.frame_count + 1}..{flow_frames})"
            )
        text_condition = np.zeros((1, flow_frames, 100), dtype=np.float32)
        padding_mask = np.ones((1, flow_frames), dtype=np.bool_)
        copy_frames = min(flow_frames, text_full.shape[1])
        text_condition[:, :copy_frames] = text_full[:, :copy_frames]
        padding_mask[:, :copy_frames] = mask_full[:, :copy_frames]
        text_condition[:, features_len:] = 0.0
        padding_mask[:, features_len:] = True
        speech_condition = np.zeros((1, flow_frames, 100), dtype=np.float32)
        speech_condition[:, : prompt.frame_count] = prompt.features
        x = rng.standard_normal((1, flow_frames, 100), dtype=np.float32)
        for step in range(len(timesteps) - 1):
            outputs = self._flow_session.infer(
                {
                    "tts.t": np.asarray(timesteps[step], dtype=np.float32),
                    "tts.flow_x": x,
                    "tts.flow_text_condition": text_condition,
                    "tts.speech_condition": speech_condition,
                    "tts.flow_padding_mask": padding_mask,
                    "tts.guidance_scale": np.asarray(float(config.get("guidance_scale", 3.0)), dtype=np.float32),
                }
            )
            velocity = np.asarray(outputs["tts.velocity"], dtype=np.float32).reshape(x.shape)
            if not np.isfinite(velocity).all():
                raise BackendInferenceError("ZipVoice flow decoder produced NaN or Inf")
            x += velocity * np.float32(timesteps[step + 1] - timesteps[step])
        generated = x[:, prompt.frame_count : features_len, :]
        features = np.transpose(generated, (0, 2, 1)) / np.float32(config.get("feature_scale", 0.1))
        with self._torch.inference_mode():
            return self._vocos(self._torch.from_numpy(features)).cpu().numpy()[0]

    def close(self) -> None:
        errors: list[Exception] = []
        for session in (self._flow_session, self._text_session):
            if session is not None:
                try:
                    session.close()
                except Exception as exc:
                    errors.append(exc)
        self._flow_session = None
        self._text_session = None
        self._vocos = None
        self._torch = None
        self._prompt = None
        self._tokenizer = None
        if errors:
            raise RuntimeError("; ".join(str(error) for error in errors))


def create_ascend_backend(*, validated_manifest, runtime_options: dict[str, Any]) -> ZipVoice310PAdapter:
    """Bundle adapter factory referenced by ``assets/adapter.json``."""

    return ZipVoice310PAdapter(validated_manifest, runtime_options)
