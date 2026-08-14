"""Voice ASR model-session builders."""

from __future__ import annotations

from inference_service.model_sessions import MODEL_SESSION_BUILDER_REGISTRY, build_ascend_model_session


def register_speech_direction_session_builder() -> None:
    key = ("perception", "fullsubnet_cumulative_stateful", "", "ascend")
    if MODEL_SESSION_BUILDER_REGISTRY.get(*key) is None:
        MODEL_SESSION_BUILDER_REGISTRY.register(*key, build_ascend_model_session)


register_speech_direction_session_builder()

__all__ = ["register_speech_direction_session_builder"]
