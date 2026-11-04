"""Per-utterance speaker embeddings (voiceprints) for the addressee gate.

The agent's voice bridge opens a short window after its wake phrase, and every
utterance in the room reaches it while that window is open. A voiceprint of the
utterance that said the wake phrase lets the bridge keep the window for the
person who opened it. This package only produces the measurement; who is
addressed is decided in ``ibrobot_agent.voice_bridge``.

It is a node of its own rather than a step inside ``voice_asr_node`` because of
ordering: the embedding has to reach the bridge *before* the transcript it
belongs to. Extracting in parallel publishes it at the end of speech, while the
transcript still waits for VAD silence plus the tail padding.
"""
