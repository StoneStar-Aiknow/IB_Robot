"""Cross-package contract: robot_config's typed VoiceTTS defaults track the shared dict.

robot_config treats voice_tts_service as an undeclared runtime plugin (its launch
builder imports it lazily with an explicit install hint), so the defaults-sync
assertions live here with the defaults owner instead of in robot_config's tests.
"""

from robot_config.config import VoiceTTSConfig
from voice_tts_service.defaults import VOICE_TTS_DEFAULTS


def test_robot_config_typed_defaults_match_shared_voice_tts_defaults():
    defaults = VoiceTTSConfig()

    assert defaults.bundle_path == VOICE_TTS_DEFAULTS["bundle_path"]
    assert defaults.deployment == VOICE_TTS_DEFAULTS["deployment"]
    assert defaults.service_name == VOICE_TTS_DEFAULTS["service_name"]
    assert defaults.playback_service_name == VOICE_TTS_DEFAULTS["playback_service_name"]
    assert defaults.playback_timeout_sec == VOICE_TTS_DEFAULTS["playback_timeout_sec"]
    assert defaults.prompt_profile == VOICE_TTS_DEFAULTS["prompt_profile"]
    assert defaults.max_response_audio_bytes == VOICE_TTS_DEFAULTS["max_response_audio_bytes"]
