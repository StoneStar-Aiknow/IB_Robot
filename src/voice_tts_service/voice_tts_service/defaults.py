"""Shared Voice TTS runtime defaults."""

VOICE_TTS_DEFAULTS = {
    "enabled": False,
    "bundle_path": "models/voice_tts/zipvoice",
    "deployment": "",
    "service_name": "/voice_tts/synthesize",
    "load_service_name": "/voice_tts/load",
    "unload_service_name": "/voice_tts/unload",
    "load_on_startup": False,
    "prompt_profile": "default",
    "segment_max_chars": 200,
    "segment_pause_ms": 150,
    "max_request_chars": 4000,
    "max_prompt_audio_bytes": 10 * 1024 * 1024,
    "max_prompt_duration_sec": 30.0,
    "max_segments": 32,
    "max_response_audio_bytes": 64 * 1024 * 1024,
    "device_id": 0,
    "exit_on_init_failure": True,
}
