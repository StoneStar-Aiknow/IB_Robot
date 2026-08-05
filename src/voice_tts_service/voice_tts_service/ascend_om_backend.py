"""Bundle-driven Ascend 310P ZipVoice backend."""

from __future__ import annotations

from typing import Any

from voice_tts_service.backend import AudioResult
from voice_tts_service.errors import BackendInferenceError, BackendLoadError, TTSError
from voice_tts_service.model_manager import TTSBundle, load_factory
from voice_tts_service.prompt_audio import PromptAudio


class AscendOmBackend:
    """Delegate ZipVoice OM pipeline scheduling to the verified bundle adapter."""

    runtime_version = ""

    def __init__(self, bundle: TTSBundle, runtime_options: dict[str, Any] | None = None) -> None:
        self._bundle = bundle
        self._runtime_options = dict(runtime_options or {})
        self._adapter = None

    def load(self) -> None:
        factory_spec = self._bundle.adapter.get("om_backend_factory")
        if not factory_spec:
            raise BackendLoadError(
                "ZipVoice bundle does not declare om_backend_factory; "
                "310P OM ABI verification is required before enabling this deployment"
            )
        factory = load_factory(factory_spec, field="om_backend_factory")
        try:
            self._adapter = factory(validated_manifest=self._bundle.validated, runtime_options=self._runtime_options)
            if hasattr(self._adapter, "load"):
                self._adapter.load()
        except Exception as exc:
            raise BackendLoadError(f"failed to load ZipVoice Ascend adapter: {exc}") from exc
        self.runtime_version = str(getattr(self._adapter, "runtime_version", "ascend_acl"))

    def synthesize(self, text: str, prompt_audio: PromptAudio | None, prompt_text: str) -> AudioResult:
        if self._adapter is None:
            raise BackendInferenceError("ZipVoice Ascend backend is not ready")
        try:
            result = self._adapter.synthesize(text, prompt_audio, prompt_text)
        except TTSError:
            raise
        except Exception as exc:
            raise BackendInferenceError(f"ZipVoice Ascend synthesis failed: {exc}") from exc
        if isinstance(result, AudioResult):
            return result
        try:
            samples, sample_rate = result
            return AudioResult(samples=samples, sample_rate=int(sample_rate))
        except (TypeError, ValueError) as exc:
            raise BackendInferenceError("ZipVoice Ascend adapter returned an invalid audio result") from exc

    def close(self) -> None:
        if self._adapter is not None and hasattr(self._adapter, "close"):
            self._adapter.close()
        self._adapter = None
