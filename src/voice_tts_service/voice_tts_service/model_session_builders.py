"""ZipVoice session builders registered with the shared model-session registry."""

from __future__ import annotations

from inference_manifest import TorchRuntimeProfile
from inference_service.backends import RuntimeContext
from inference_service.unified_runtime import RuntimeDependencyError, RuntimeProviders, SessionBuilderRegistry

from .defaults import VOICE_TTS_DEFAULTS
from .zipvoice_310p_adapter import ZipVoiceAscendSession
from .zipvoice_onnx_adapter import ZipVoiceOnnxSession


def build_zipvoice_session(
    context: RuntimeContext,
    *,
    providers: RuntimeProviders | None = None,
):
    """Construct the manifest-selected ZipVoice session without loading it."""

    options = context.runtime_options
    if context.backend == "ascend":
        if context.target_runtime != "acl":
            raise ValueError("ZipVoice Ascend requires target.runtime='acl'")
        device_id = context.device_id
        if device_id is None:
            device_id = int(options.get("device_id", VOICE_TTS_DEFAULTS["device_id"]))
        elif "device_id" in options and options["device_id"] != device_id:
            raise ValueError("ZipVoice device_id option does not match the typed runtime profile")
        return ZipVoiceAscendSession(
            device_id=int(device_id),
            prompt_profile=str(options.get("prompt_profile", VOICE_TTS_DEFAULTS["prompt_profile"])),
            runtime_manager=(getattr(providers, "acl_runtime_provider", None) if providers is not None else None),
        )
    if context.backend == "torch":
        if not isinstance(context.backend_profile, TorchRuntimeProfile) or context.backend_profile.device != "cpu":
            raise ValueError("ZipVoice ONNX requires a typed cpu Torch profile")
        return ZipVoiceOnnxSession(
            prompt_profile=str(options.get("prompt_profile", VOICE_TTS_DEFAULTS["prompt_profile"])),
        )
    raise ValueError(f"ZipVoice plugin does not support backend: {context.backend}")


def register_zipvoice_session_builder(registry: SessionBuilderRegistry | None = None) -> None:
    """Register ZipVoice once for the canonical tensor-model identity."""

    if registry is None:
        raise RuntimeDependencyError(
            "register_zipvoice_session_builder requires an explicit session registry",
            code="session_builder_registry_required",
        )
    if registry.get("tensor_model", "zipvoice", "synthesize", "ascend") is None:
        registry.register(
            "tensor_model",
            "zipvoice",
            "synthesize",
            "ascend",
            build_zipvoice_session,
        )
    if registry.get("tensor_model", "zipvoice", "synthesize", "torch") is None:
        registry.register(
            "tensor_model",
            "zipvoice",
            "synthesize",
            "torch",
            build_zipvoice_session,
        )


__all__ = ["build_zipvoice_session", "register_zipvoice_session_builder"]
