"""Stable Voice TTS service errors."""


class TTSError(RuntimeError):
    """An expected failure that maps to a stable public error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class BackendLoadError(TTSError):
    """Backend bundle, dependency, or ABI loading failed."""

    def __init__(self, message: str) -> None:
        super().__init__("MODEL_NOT_READY", message)


class BackendInferenceError(TTSError):
    """Backend inference failed for one segment."""

    def __init__(self, message: str) -> None:
        super().__init__("INFERENCE_FAILED", message)
