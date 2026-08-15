"""ZipVoice session builders registered with the shared model-session registry."""

from __future__ import annotations

from inference_manifest import TorchDeployment
from inference_service.backends import RuntimeContext
from inference_service.model_sessions import MODEL_SESSION_BUILDER_REGISTRY, ModelSession

from .defaults import VOICE_TTS_DEFAULTS
from .zipvoice_310p_adapter import ZipVoiceAscendSession
from .zipvoice_onnx_adapter import ZipVoiceOnnxSession


def build_zipvoice_session(context: RuntimeContext) -> ModelSession:
    """Construct the manifest-selected ZipVoice session without loading it."""

    deployment = context.deployment
    options = context.runtime_options
    if deployment.backend == "ascend":
        return ZipVoiceAscendSession(
            device_id=int(options.get("device_id", VOICE_TTS_DEFAULTS["device_id"])),
            prompt_profile=str(options.get("prompt_profile", VOICE_TTS_DEFAULTS["prompt_profile"])),
        )
    if isinstance(deployment, TorchDeployment):
        return ZipVoiceOnnxSession(
            prompt_profile=str(options.get("prompt_profile", VOICE_TTS_DEFAULTS["prompt_profile"])),
        )
    raise ValueError(f"ZipVoice plugin does not support backend: {deployment.backend}")


def register_zipvoice_session_builder() -> None:
    """Register ZipVoice once for the canonical generic model identity."""

    if MODEL_SESSION_BUILDER_REGISTRY.get("generic", "zipvoice", "", "ascend") is None:
        MODEL_SESSION_BUILDER_REGISTRY.register(
            "generic",
            "zipvoice",
            "",
            "ascend",
            build_zipvoice_session,
        )
    if MODEL_SESSION_BUILDER_REGISTRY.get("generic", "zipvoice", "", "torch") is None:
        MODEL_SESSION_BUILDER_REGISTRY.register(
            "generic",
            "zipvoice",
            "",
            "torch",
            build_zipvoice_session,
        )


register_zipvoice_session_builder()


__all__ = ["build_zipvoice_session", "register_zipvoice_session_builder"]
