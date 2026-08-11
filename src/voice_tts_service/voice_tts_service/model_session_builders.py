"""ZipVoice session builders registered with the shared model-session registry."""

from __future__ import annotations

from inference_manifest import CompiledDeployment
from inference_service.backends import RuntimeContext
from inference_service.model_sessions import MODEL_SESSION_BUILDER_REGISTRY, ModelSession

from .defaults import VOICE_TTS_DEFAULTS
from .zipvoice_310p_adapter import ZipVoiceAscendSession


def build_zipvoice_session(context: RuntimeContext) -> ModelSession:
    """Construct the manifest-selected ZipVoice session without loading it."""

    deployment = context.deployment
    if not isinstance(deployment, CompiledDeployment) or deployment.backend != "ascend":
        raise ValueError("ZipVoice plugin requires a compiled Ascend deployment")
    options = context.runtime_options
    return ZipVoiceAscendSession(
        device_id=int(options.get("device_id", VOICE_TTS_DEFAULTS["device_id"])),
        prompt_profile=str(options.get("prompt_profile", VOICE_TTS_DEFAULTS["prompt_profile"])),
    )


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


register_zipvoice_session_builder()


__all__ = ["build_zipvoice_session", "register_zipvoice_session_builder"]
