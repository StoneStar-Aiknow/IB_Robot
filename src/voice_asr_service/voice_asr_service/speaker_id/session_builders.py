"""Session builders for the speaker-embedding deployments.

Registered explicitly, like ``model_session_builders`` does for
speech_direction: the registry is composed per service, so a package states
which backends it is prepared to run rather than relying on a global default.
The Ascend deployment runs a trunk cut before the statistics are concatenated,
because the whole graph compiled as one OM corrupts that concatenation; the
host joins the halves and applies the head. See ``package_campplus_bundle``.
"""

from __future__ import annotations

from inference_service.model_sessions import build_ascend_model_session, build_onnx_model_session
from inference_service.unified_runtime import RuntimeDependencyError, SessionBuilderKey

SPEAKER_EMBEDDING_IDENTITY = ("tensor_model", "campplus", "embed")
SPEAKER_EMBEDDING_BACKENDS = ("onnx", "ascend")

_BUILDERS = {"onnx": build_onnx_model_session, "ascend": build_ascend_model_session}


def register_speaker_embedding_session_builders(registry=None) -> None:
    if registry is None:
        raise RuntimeDependencyError(
            "register_speaker_embedding_session_builders requires an explicit session registry",
            code="session_builder_registry_required",
        )
    interface, model_type, operation = SPEAKER_EMBEDDING_IDENTITY
    for backend, builder in _BUILDERS.items():
        key = SessionBuilderKey(interface, model_type, operation, backend)
        if registry.get(key) is None:
            registry.register(key, builder)


__all__ = [
    "SPEAKER_EMBEDDING_BACKENDS",
    "SPEAKER_EMBEDDING_IDENTITY",
    "register_speaker_embedding_session_builders",
]
