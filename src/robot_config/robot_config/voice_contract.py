"""Voice pipeline defaults that more than one package needs.

robot_config validates them, voice_asr_service's nodes fall back to them when
launched without a value, and the embodied launch derives the voice bridge's
subscriptions from them; each is written here once and imported everywhere
else. Kept free of imports so a node can read it without loading the loader.
"""

#: Where ``voice_asr_node`` publishes each final transcript tagged with its
#: speech segment (``ibrobot_msgs/SpeechTranscript``), under external_vad input.
VOICE_TRANSCRIPT_TOPIC = "/voice/transcript"

#: Where ``speaker_id_node`` publishes its voiceprints
#: (``voice_asr.speaker_embedding.output_topic``).
SPEAKER_EMBEDDING_OUTPUT_TOPIC = "/voice/speaker_embedding"

#: ``speaker_id_node``'s idle timeout for a segment the source never ends. It
#: must stay below ``VoiceASRConfig.external_idle_timeout`` so the voiceprint
#: lands before its transcript; the loader enforces that.
SPEAKER_EMBEDDING_IDLE_TIMEOUT_SEC = 1.2
