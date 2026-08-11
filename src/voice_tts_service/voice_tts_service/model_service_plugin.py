"""Typed ZipVoice plugin for the shared model-service host."""

from __future__ import annotations

from array import array

from inference_service.backends import BackendError, RuntimeContext
from inference_service.model_service_plugin import ModelServiceError, ModelServicePlugin, PluginRuntimeStatus
from inference_service.model_sessions import MODEL_SESSION_BUILDER_REGISTRY, ModelSession
from inference_service.pipeline import GenericModelPipeline, ModelResultAdapter, ModelStage, SequentialModelExecutor

from .defaults import VOICE_TTS_DEFAULTS
from .errors import TTSError
from .model_session_builders import register_zipvoice_session_builder
from .service_core import TTSLimits, TTSServiceCore


class ZipVoiceSynthesizePlugin(ModelServicePlugin):
    """Expose ZipVoice through the family-neutral typed model-service host."""

    service_type = "ibrobot_msgs/srv/SynthesizeSpeech"

    def __init__(self, _host, validated, options) -> None:
        register_zipvoice_session_builder()
        model = validated.manifest.model
        if model.kind != "generic" or model.family != "zipvoice":
            raise ValueError("ZipVoice plugin requires model.kind=generic and model.family=zipvoice")

        allowed = {
            "acl_config_path",
            "device_id",
            "prompt_profile",
            "segment_max_chars",
            "segment_pause_ms",
            "max_request_chars",
            "max_prompt_audio_bytes",
            "max_prompt_duration_sec",
            "max_segments",
            "max_response_audio_bytes",
        }
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ValueError(f"unknown ZipVoice runtime options: {unknown}")

        context = RuntimeContext(validated_manifest=validated, runtime_options=options)
        self._session: ModelSession = MODEL_SESSION_BUILDER_REGISTRY.create(context)
        session_options = {name: options[name] for name in ("acl_config_path", "device_id") if name in options}
        session_context = RuntimeContext(validated_manifest=validated, runtime_options=session_options)
        self._pipeline = GenericModelPipeline(
            "voice-tts-zipvoice",
            context,
            SequentialModelExecutor(
                (ModelStage("model", self._session),),
                ModelResultAdapter(),
                components=(self._session,),
                component_contexts={id(self._session): session_context},
            ),
            supports_cancellation=self._session.capabilities.supports_cancellation,
        )
        self._core = TTSServiceCore(
            self._pipeline.execute,
            TTSLimits(
                **{
                    name: options.get(name, VOICE_TTS_DEFAULTS[name])
                    for name in TTSLimits.__dataclass_fields__
                    if name != "sample_rate"
                }
            ),
        )
        self._closed = False
        try:
            self._pipeline.load()
        except Exception:
            self._pipeline.close()
            self._closed = True
            raise

    def handle(self, request, response) -> str:
        try:
            output = self._core.synthesize(
                request.text,
                bytes(request.prompt_audio),
                request.prompt_audio_format,
                request.prompt_text,
            )
        except Exception as exc:
            code = getattr(exc, "code", "")
            if isinstance(exc, TTSError):
                code = exc.code
            elif isinstance(exc, BackendError) and code == "unsupported_prompt":
                code = "UNSUPPORTED_PROMPT"
            elif isinstance(exc, BackendError):
                code = "INFERENCE_FAILED"
            else:
                code = "INTERNAL_ERROR"
            raise ModelServiceError(
                str(exc),
                error_code=code,
                audio_segments=[],
                total_duration_sec=0.0,
                total_inference_time_ms=0.0,
            ) from exc

        from ibrobot_msgs.msg import SynthesizedAudio

        response.audio_segments = [
            SynthesizedAudio(
                index=segment.index,
                text=segment.text,
                audio_data=array("B", segment.wav_data),
                audio_format="wav_pcm_s16le",
                sample_rate=segment.sample_rate,
                channels=1,
                duration_sec=segment.duration_sec,
                inference_time_ms=segment.inference_time_ms,
                pause_after_ms=segment.pause_after_ms,
            )
            for segment in output.segments
        ]
        response.total_duration_sec = output.total_duration_sec
        response.total_inference_time_ms = output.total_inference_time_ms
        response.error_code = ""
        return f"synthesized {len(output.segments)} audio segment(s)"

    def runtime_status(self) -> PluginRuntimeStatus:
        health = self._pipeline.diagnostics().executor_health
        state = health.state.value if hasattr(health.state, "value") else str(health.state)
        runtime_version = self._session.runtime_version
        ready = health.ready and bool(runtime_version)
        return PluginRuntimeStatus(
            state=state,
            ready=ready,
            failure_reason=""
            if ready
            else (
                "loaded model runtime did not expose a version"
                if health.ready
                else (health.message or health.reason_code or f"model runtime is {state}")
            ),
            metadata={"runtime_version": runtime_version} if runtime_version else {},
        )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._pipeline.close()


__all__ = ["ZipVoiceSynthesizePlugin"]
