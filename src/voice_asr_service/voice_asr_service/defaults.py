"""Shared default values for Voice ASR configuration."""

VOICE_ASR_DEFAULTS = {
    "enabled": False,
    "active_mode": "continuous",
    "language": "zh",
    "bundle_path": "models/voice_asr/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-int8-2026-06-05",
    "deployment": "torch_cpu",
    "max_recording_duration": 10.0,
    "vad_sensitivity": 0.6,
    "vad_bundle_path": "models/silero-vad",
    "vad_deployment": "torch_cpu",
    "realtime_pre_roll_seconds": 0.5,
    # Silence appended before the final decode so the streaming encoder's last
    # chunk (and right context) is flushed; 480 ms-chunk models need >= 1.5 s.
    "final_tail_padding_seconds": 1.5,
    "publish_partial": True,
    "output_topic": "/voice_command",
    "sample_rate": 16000,
    "chunk_size": 512,
    "buffer_seconds": 5.0,
    "audio_input_channel": 1,
    "exit_on_init_failure": True,
    "input_mode": "shared_audio",
    "chunk_topic": "/audio/speech_chunk",
    "external_min_samples": 3200,
    "external_idle_timeout": 2.0,
}
