"""Typed ZipVoice plugin for the shared model-service host."""

from __future__ import annotations

from array import array

from inference_manifest import CompiledDeployment
from inference_service.backends import BackendError, RuntimeContext
from inference_service.model_service_plugin import ModelServiceError, ModelServicePlugin, PluginRuntimeStatus
from inference_service.model_sessions import ModelSession

from .defaults import VOICE_TTS_DEFAULTS
from .errors import TTSError
from .service_core import TTSLimits, TTSServiceCore
from .zipvoice_310p_adapter import ZipVoiceAscendSession


class ZipVoiceSynthesizePlugin(ModelServicePlugin):
    """Expose ZipVoice through the family-neutral typed model-service host."""

    service_type = "ibrobot_msgs/srv/SynthesizeSpeech"

    def __init__(self, _host, validated, options) -> None:
        model = validated.manifest.model
        if model.kind != "generic" or model.family != "zipvoice":
            raise ValueError("ZipVoice plugin requires model.kind=generic and model.family=zipvoice")
        deployment = validated.deployment
        if not isinstance(deployment, CompiledDeployment) or deployment.backend != "ascend":
            raise ValueError("ZipVoice plugin requires a compiled Ascend deployment")

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

        device_id = int(options.get("device_id", VOICE_TTS_DEFAULTS["device_id"]))
        self._session: ModelSession = ZipVoiceAscendSession(
            device_id=device_id,
            prompt_profile=str(options.get("prompt_profile", VOICE_TTS_DEFAULTS["prompt_profile"])),
        )
        self._core = TTSServiceCore(
            self._session.infer,
            TTSLimits(
                **{
                    name: options.get(name, VOICE_TTS_DEFAULTS[name])
                    for name in TTSLimits.__dataclass_fields__
                    if name != "sample_rate"
                }
            ),
        )
        self._closed = False
        session_options = {name: options[name] for name in ("acl_config_path", "device_id") if name in options}
        try:
            self._session.load(RuntimeContext(validated_manifest=validated, runtime_options=session_options))
        except Exception:
            self._session.close()
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
        health = self._session.health()
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
            self._session.close()


__all__ = ["ZipVoiceSynthesizePlugin"]
