"""ZipVoice-Distill ONNX adapter for Ubuntu/CUDA hosts.

Reuses the 310P adapter's Chinese tokenizer, prompt profile, cross-fade
concatenation, and Vocos vocoder. Replaces the Ascend OM bucket inference
with onnxruntime dynamic-shape inference from the k2-fsa/ZipVoice upstream.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from inference_manifest import TorchDeployment
from inference_service.backends import BackendCapabilities, RuntimeContext
from inference_service.backends.errors import BackendInferenceError as SessionInferenceError
from inference_service.backends.errors import BackendLoadError as SessionLoadError
from inference_service.backends.lifecycle import PartialLoadRollback
from inference_service.generic_runtime import NamedTensorRequest
from inference_service.model_sessions.base import ModelSession

from .errors import BackendLoadError
from .zipvoice_310p_adapter import ZipVoiceAscendSession, _ChineseTokenizer, _PromptProfile

_cross_fade_concat = ZipVoiceAscendSession._cross_fade_concat
_timesteps = ZipVoiceAscendSession._timesteps

logger = logging.getLogger(__name__)


class ZipVoiceOnnxSession(ModelSession):
    """Run the ZipVoice-Distill ONNX pipeline with onnxruntime on Ubuntu."""

    def __init__(self, device_id: int = 0, *, prompt_profile: str = "default", **kwargs) -> None:
        super().__init__(
            "zipvoice-onnx",
            BackendCapabilities(
                supports_cancellation=False,
                thread_safe=False,
            ),
        )
        self._root: Path | None = None
        self._prompt_profile_name = prompt_profile
        self._config: dict[str, Any] = {}
        self._tokenizer: _ChineseTokenizer | None = None
        self._prompt: _PromptProfile | None = None
        self._vocos = None
        self._torch = None
        self._text_encoder = None
        self._fm_decoder = None
        self._feat_dim = 100

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BackendLoadError(f"failed to read ZipVoice ONNX config {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise BackendLoadError("ZipVoice ONNX config must be a JSON object")
        return value

    def _asset(self, field: str) -> Path:
        value = self._config.get(field)
        if not isinstance(value, str) or not value:
            raise BackendLoadError(f"ZipVoice ONNX config field {field!r} must be a relative path")
        path = (self._root / value).resolve()
        try:
            path.relative_to(self._root)
        except ValueError as exc:
            raise BackendLoadError(f"ZipVoice asset path escapes bundle root: {value}") from exc
        if not path.is_file() and not path.is_dir():
            raise BackendLoadError(f"ZipVoice asset is unavailable: {path}")
        return path

    def _load(self, context: RuntimeContext, rollback: PartialLoadRollback) -> None:
        deployment = context.deployment
        model = context.validated_manifest.manifest.model
        if model.kind != "generic" or model.family != "zipvoice":
            raise SessionLoadError("ZipVoice session requires model.kind=generic and model.family=zipvoice")
        if not isinstance(deployment, TorchDeployment) or deployment.backend != "torch":
            raise SessionLoadError("ZipVoice ONNX session requires a compiled torch deployment")
        self._root = context.validated_manifest.bundle_root
        self._config = self._load_json(self._root / "assets" / "zipvoice_onnx.json")
        rollback.defer(self._release_assets)

        self._load_onnx_models()
        self._load_assets()

    def _load_onnx_models(self) -> None:
        try:
            import onnxruntime as ort
        except (ImportError, OSError) as exc:
            raise BackendLoadError(f"onnxruntime is unavailable: {exc}") from exc

        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = 4
        sess_opts.intra_op_num_threads = 4

        text_encoder_path = self._asset("text_encoder_path")
        fm_decoder_path = self._asset("fm_decoder_path")

        providers = ["CPUExecutionProvider"]
        self._text_encoder = ort.InferenceSession(str(text_encoder_path), sess_options=sess_opts, providers=providers)
        self._fm_decoder = ort.InferenceSession(str(fm_decoder_path), sess_options=sess_opts, providers=providers)
        meta = self._fm_decoder.get_modelmeta().custom_metadata_map
        self._feat_dim = int(meta.get("feat_dim", 100))
        logger.info("ZipVoice ONNX loaded: feat_dim=%d", self._feat_dim)

    def _load_assets(self) -> None:
        profile_name = self._prompt_profile_name
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
        if features.shape[2] != self._feat_dim or not np.isfinite(features).all():
            raise BackendLoadError("ZipVoice prompt profile features are invalid")
        self._prompt = _PromptProfile(tokens=tokens, features=features)
        self._tokenizer = _ChineseTokenizer(self._asset("tokens_path"))
        self._load_vocos()

    def _load_vocos(self) -> None:
        try:
            import torch

            from voice_tts_service.vocos_backend import ZipVoiceVocos
        except (ImportError, OSError) as exc:
            raise BackendLoadError(f"ZipVoice Vocos dependency is unavailable: {exc}") from exc
        vocos = ZipVoiceVocos()
        checkpoint = torch.load(self._asset("vocos_checkpoint_path"), map_location="cpu", weights_only=True)
        incompatible = vocos.load_state_dict(checkpoint, strict=False)
        if incompatible.missing_keys:
            raise BackendLoadError(f"ZipVoice Vocos checkpoint is missing keys: {incompatible.missing_keys}")
        unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("feature_extractor.")]
        if unexpected:
            raise BackendLoadError(f"ZipVoice Vocos checkpoint has unexpected keys: {unexpected}")
        self._torch = torch
        self._vocos = vocos.eval()

    def _execute(self, request: NamedTensorRequest) -> Mapping[str, object]:
        if any(
            value is None
            for value in (self._tokenizer, self._prompt, self._torch, self._vocos, self._text_encoder, self._fm_decoder)
        ):
            raise SessionInferenceError("ZipVoice ONNX session assets are not loaded")
        try:
            text = np.asarray(request.inputs["tts.text"], dtype=np.uint8).tobytes().decode("utf-8")
        except (KeyError, UnicodeDecodeError, ValueError) as exc:
            raise SessionInferenceError("ZipVoice request text is not valid UTF-8", code="invalid_text") from exc
        if (
            np.asarray(request.inputs.get("tts.prompt_audio", ())).size
            or np.asarray(request.inputs.get("tts.prompt_text", ()), dtype=np.uint8).size
        ):
            raise SessionInferenceError(
                "the ONNX deployment uses a fixed prompt profile and does not support request prompts",
                code="unsupported_prompt",
            )

        config = self._config
        text_capacity = int(config.get("text_capacity", 256))
        num_steps = int(config.get("num_steps", 8))
        t_shift = float(config.get("t_shift", 0.5))
        sample_rate = int(config.get("sample_rate", 24000))
        cross_fade_sec = float(config.get("cross_fade_sec", 0.1))

        token_strings = self._tokenizer.text_to_tokens(text)
        chunks = self._tokenizer.chunk_tokens(token_strings, text_capacity)
        timesteps = _timesteps(num_steps, t_shift)
        rng = np.random.default_rng(int(config.get("seed", 42)))
        waves = [self._synthesize_chunk(chunk, rng, timesteps) for chunk in chunks]
        if not waves:
            raise SessionInferenceError("ZipVoice frontend produced no token chunks")
        wave = _cross_fade_concat(waves, sample_rate, cross_fade_sec)
        if wave.size == 0 or not np.isfinite(wave).all():
            raise SessionInferenceError("ZipVoice produced invalid audio")
        return {"tts.audio": np.ascontiguousarray(np.clip(wave, -1.0, 1.0), dtype=np.float32)}

    def _synthesize_chunk(self, tokens: list[str], rng, timesteps: np.ndarray) -> np.ndarray:
        config = self._config
        speed = float(config.get("speed", 1.0))
        guidance_scale = float(config.get("guidance_scale", 3.0))
        feature_scale = float(config.get("feature_scale", 0.1))
        prompt = self._prompt
        feat_dim = self._feat_dim

        ids = self._tokenizer.tokens_to_ids(tokens)
        padded = np.full((1, len(ids)), self._tokenizer.pad_id, dtype=np.int64)
        padded[0, : len(ids)] = ids

        tokens_arr = padded
        prompt_tokens = prompt.tokens
        prompt_features_len = np.asarray(prompt.frame_count, dtype=np.int64)
        speed_arr = np.asarray(speed, dtype=np.float32)

        text_condition = self._run_text_encoder(tokens_arr, prompt_tokens, prompt_features_len, speed_arr)
        text_condition = np.asarray(text_condition, dtype=np.float32)
        if text_condition.ndim != 3:
            raise SessionInferenceError(f"text_encoder output has unexpected shape: {text_condition.shape}")

        num_frames = text_condition.shape[1]
        x = rng.standard_normal((1, num_frames, feat_dim), dtype=np.float32)

        speech_condition = np.zeros((1, num_frames, feat_dim), dtype=np.float32)
        copy_frames = min(prompt.frame_count, num_frames)
        speech_condition[:, :copy_frames, :] = prompt.features[:, :copy_frames, :]

        for step in range(len(timesteps) - 1):
            velocity = self._run_fm_decoder(
                np.asarray(timesteps[step], dtype=np.float32),
                x,
                text_condition,
                speech_condition,
                np.asarray(guidance_scale, dtype=np.float32),
            )
            velocity = np.asarray(velocity, dtype=np.float32).reshape(x.shape)
            if not np.isfinite(velocity).all():
                raise SessionInferenceError("ZipVoice flow decoder produced NaN or Inf")
            x += velocity * np.float32(timesteps[step + 1] - timesteps[step])

        generated = x[:, prompt.frame_count :, :]
        features = np.transpose(generated, (0, 2, 1)) / np.float32(feature_scale)
        with self._torch.inference_mode():
            return self._vocos(self._torch.from_numpy(features)).cpu().numpy()[0]

    def _run_text_encoder(self, tokens, prompt_tokens, prompt_features_len, speed):
        te = self._text_encoder
        out = te.run(
            [te.get_outputs()[0].name],
            {
                te.get_inputs()[0].name: tokens,
                te.get_inputs()[1].name: prompt_tokens,
                te.get_inputs()[2].name: prompt_features_len,
                te.get_inputs()[3].name: speed,
            },
        )
        return out[0]

    def _run_fm_decoder(self, t, x, text_condition, speech_condition, guidance_scale):
        fd = self._fm_decoder
        out = fd.run(
            [fd.get_outputs()[0].name],
            {
                fd.get_inputs()[0].name: t,
                fd.get_inputs()[1].name: x,
                fd.get_inputs()[2].name: text_condition,
                fd.get_inputs()[3].name: speech_condition,
                fd.get_inputs()[4].name: guidance_scale,
            },
        )
        return out[0]

    def _release_assets(self) -> None:
        self._vocos = None
        self._torch = None
        self._prompt = None
        self._tokenizer = None
        self._text_encoder = None
        self._fm_decoder = None

    def _close(self) -> None:
        self._release_assets()
        self._root = None
        self._config = {}

    @property
    def runtime_version(self) -> str:
        try:
            import onnxruntime as ort

            return f"onnxruntime-{ort.__version__}"
        except Exception:
            return "onnxruntime"


__all__ = ["ZipVoiceOnnxSession"]
