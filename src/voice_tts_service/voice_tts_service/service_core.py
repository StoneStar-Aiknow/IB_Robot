"""Pure request validation and synthesis orchestration, independent of ROS."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from inference_service.generic_runtime import NamedTensorRequest
from voice_tts_service.audio_utils import float_pcm_to_wav
from voice_tts_service.errors import TTSError
from voice_tts_service.prompt_audio import PromptAudio, decode_prompt
from voice_tts_service.text_segmenter import segment_text


@dataclass(frozen=True)
class TTSLimits:
    segment_max_chars: int = 200
    segment_pause_ms: int = 150
    max_request_chars: int = 4000
    max_prompt_audio_bytes: int = 10 * 1024 * 1024
    max_prompt_duration_sec: float = 30.0
    max_segments: int = 32
    max_response_audio_bytes: int = 64 * 1024 * 1024


@dataclass(frozen=True)
class SynthesizedSegment:
    index: int
    text: str
    wav_data: bytes
    sample_rate: int
    duration_sec: float
    inference_time_ms: float
    pause_after_ms: int


@dataclass(frozen=True)
class SynthesisOutput:
    normalized_text: str
    segments: tuple[SynthesizedSegment, ...]
    total_duration_sec: float
    total_inference_time_ms: float


@dataclass(frozen=True)
class PreparedSynthesisRequest:
    """Validated request data that is safe to execute after model loading."""

    normalized_text: str
    segments: tuple[str, ...]
    prompt: PromptAudio | None
    prompt_text: str


class TTSServiceCore:
    """Validate TTS requests and adapt them to the shared model-session contract."""

    def __init__(
        self,
        infer: Callable[[NamedTensorRequest], object] | None,
        limits: TTSLimits,
        *,
        sample_rate: int = 24000,
    ) -> None:
        self.infer = infer
        self.limits = limits
        self.sample_rate = sample_rate

    def prepare_request(
        self,
        text: str,
        prompt_audio_bytes: bytes = b"",
        prompt_audio_format: str = "",
        prompt_text: str = "",
    ) -> PreparedSynthesisRequest:
        """Validate and decode a request without requiring a loaded backend."""

        if len(text) > self.limits.max_request_chars:
            raise TTSError("REQUEST_TOO_LARGE", f"text exceeds {self.limits.max_request_chars} characters")
        normalized_text, segments = segment_text(
            text,
            self.limits.segment_max_chars,
            self.limits.max_segments,
        )
        if len(prompt_audio_bytes) > self.limits.max_prompt_audio_bytes:
            raise TTSError(
                "PROMPT_TOO_LARGE",
                f"prompt audio exceeds {self.limits.max_prompt_audio_bytes} bytes",
            )
        prompt = decode_prompt(
            prompt_audio_bytes,
            prompt_audio_format,
            prompt_text,
            max_bytes=self.limits.max_prompt_audio_bytes,
            max_duration_sec=self.limits.max_prompt_duration_sec,
        )
        return PreparedSynthesisRequest(
            normalized_text=normalized_text,
            segments=tuple(segments),
            prompt=prompt,
            prompt_text=prompt_text,
        )

    def synthesize(
        self,
        text: str,
        prompt_audio_bytes: bytes = b"",
        prompt_audio_format: str = "",
        prompt_text: str = "",
    ) -> SynthesisOutput:
        prepared = self.prepare_request(text, prompt_audio_bytes, prompt_audio_format, prompt_text)
        return self.synthesize_prepared(prepared)

    def synthesize_prepared(self, prepared: PreparedSynthesisRequest) -> SynthesisOutput:
        """Execute a previously validated request against the current backend."""

        if self.infer is None:
            raise TTSError("MODEL_NOT_READY", "Voice TTS model session is not ready")

        started = time.perf_counter()
        output_segments = []
        total_audio_bytes = 0
        for index, segment_text_value in enumerate(prepared.segments):
            segment_started = time.perf_counter()
            request = self._model_request(segment_text_value, prepared.prompt, prepared.prompt_text, index)
            result = self.infer(request)
            try:
                samples = np.asarray(result.outputs["tts.audio"], dtype=np.float32).reshape(-1)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise TTSError("INFERENCE_FAILED", "model session returned an invalid TTS result") from exc
            sample_rate = self.sample_rate
            wav_data, duration_sec = float_pcm_to_wav(samples, sample_rate)
            total_audio_bytes += len(wav_data)
            if total_audio_bytes > self.limits.max_response_audio_bytes:
                raise TTSError(
                    "RESPONSE_TOO_LARGE",
                    f"synthesized audio exceeds {self.limits.max_response_audio_bytes} bytes",
                )
            output_segments.append(
                SynthesizedSegment(
                    index=index,
                    text=segment_text_value,
                    wav_data=wav_data,
                    sample_rate=sample_rate,
                    duration_sec=duration_sec,
                    inference_time_ms=(time.perf_counter() - segment_started) * 1000.0,
                    pause_after_ms=self.limits.segment_pause_ms if index < len(prepared.segments) - 1 else 0,
                )
            )
        return SynthesisOutput(
            normalized_text=prepared.normalized_text,
            segments=tuple(output_segments),
            total_duration_sec=sum(segment.duration_sec for segment in output_segments),
            total_inference_time_ms=(time.perf_counter() - started) * 1000.0,
        )

    @staticmethod
    def _model_request(
        text: str,
        prompt: PromptAudio | None,
        prompt_text: str,
        segment_index: int,
    ) -> NamedTensorRequest:
        prompt_samples = () if prompt is None else prompt.samples
        prompt_rate = 0 if prompt is None else prompt.sample_rate
        return NamedTensorRequest(
            request_id=f"voice-tts-{segment_index}-{time.monotonic_ns()}",
            inputs={
                "tts.text": np.frombuffer(text.encode("utf-8"), dtype=np.uint8),
                "tts.prompt_audio": np.asarray(prompt_samples, dtype=np.float32),
                "tts.prompt_sample_rate": np.asarray(prompt_rate, dtype=np.int64),
                "tts.prompt_text": np.frombuffer(prompt_text.encode("utf-8"), dtype=np.uint8),
            },
            metadata={"service_type": "ibrobot_msgs/srv/SynthesizeSpeech"},
        )


@dataclass(frozen=True)
class AudioResult:
    """Logical mono float PCM returned by one session inference."""

    samples: Sequence[float]
    sample_rate: int
