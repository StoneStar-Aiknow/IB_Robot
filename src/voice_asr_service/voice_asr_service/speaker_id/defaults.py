"""Shared defaults for the speaker-embedding node, importable without ROS."""

from robot_config.voice_contract import SPEAKER_EMBEDDING_IDLE_TIMEOUT_SEC, SPEAKER_EMBEDDING_OUTPUT_TOPIC
from voice_asr_service.speaker_id.embedder import MIN_SPEECH_SECONDS

SPEAKER_ID_DEFAULTS = {
    "chunk_topic": "/audio/speech_chunk",
    "output_topic": SPEAKER_EMBEDDING_OUTPUT_TOPIC,
    #: Schema-v3 bundle and the deployment inside it, externally provisioned
    #: like the ASR bundle. Empty: a deployment has to be chosen, never guessed.
    "bundle_path": "",
    "deployment": "",
    "sample_rate": 16000,
    "min_speech_seconds": MIN_SPEECH_SECONDS,
    "max_speech_seconds": 6.0,
    # Per-window voiceprints, for telling a segment that holds two voices from one voice:
    # a 2 s window every window_hop_seconds over the first window_max_seconds of the
    # segment (the ASR's own max_recording_duration). 0, the default, turns them off;
    # they cost one more model call per hop of speech.
    "window_hop_seconds": 0.0,
    "window_max_seconds": 10.0,
    # Below the ASR's external_idle_timeout (2.0), so an END-less utterance's
    # voiceprint is published before its transcript; see robot_config.voice_contract.
    "idle_timeout_sec": SPEAKER_EMBEDDING_IDLE_TIMEOUT_SEC,
    "exit_on_init_failure": True,
}
