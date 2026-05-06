"""Voice ASR package exports."""

from importlib import import_module

_EXPORTS = {
    "AudioCaptureModule": (".audio_capture_module", "AudioCaptureModule"),
    "AudioConfig": (".audio_capture_module", "AudioConfig"),
    "CaptureState": (".audio_capture_module", "CaptureState"),
    "RingBuffer": (".audio_capture_module", "RingBuffer"),
    "FileInputModule": (".file_input_module", "FileInputModule"),
    "FileResult": (".file_input_module", "FileResult"),
    "FileState": (".file_input_module", "FileState"),
    "FileError": (".file_input_module", "FileError"),
    "ASRInferenceModule": (".asr_inference_module", "ASRInferenceModule"),
    "ASRResult": (".asr_inference_module", "ASRResult"),
    "ASRState": (".asr_inference_module", "ASRState"),
    "VADModule": (".vad_module", "VADModule"),
    "VADConfig": (".vad_module", "VADConfig"),
    "VADState": (".vad_module", "VADState"),
    "VADResult": (".vad_module", "VADResult"),
    "StateMachine": (".state_machine", "StateMachine"),
    "NodeState": (".state_machine", "NodeState"),
    "ActiveMode": (".state_machine", "ActiveMode"),
    "ModelBundle": (".model_manager", "ModelBundle"),
    "ResolvedModelAssets": (".model_manager", "ResolvedModelAssets"),
    "STREAMING_ZH_BUNDLE": (".model_manager", "STREAMING_ZH_BUNDLE"),
    "OFFLINE_ZH_BUNDLE": (".model_manager", "OFFLINE_ZH_BUNDLE"),
    "default_model_root": (".model_manager", "default_model_root"),
    "infer_model_bundle": (".model_manager", "infer_model_bundle"),
    "download_model_bundle": (".model_manager", "download_model_bundle"),
    "resolve_model_assets": (".model_manager", "resolve_model_assets"),
}


def __getattr__(name):
    """Load package exports lazily so light-weight submodules stay importable."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "AudioCaptureModule",
    "AudioConfig",
    "CaptureState",
    "RingBuffer",
    "FileInputModule",
    "FileResult",
    "FileState",
    "FileError",
    "ASRInferenceModule",
    "ASRResult",
    "ASRState",
    "VADModule",
    "VADConfig",
    "VADState",
    "VADResult",
    "StateMachine",
    "NodeState",
    "ActiveMode",
    "ModelBundle",
    "ResolvedModelAssets",
    "STREAMING_ZH_BUNDLE",
    "OFFLINE_ZH_BUNDLE",
    "default_model_root",
    "infer_model_bundle",
    "download_model_bundle",
    "resolve_model_assets",
]
