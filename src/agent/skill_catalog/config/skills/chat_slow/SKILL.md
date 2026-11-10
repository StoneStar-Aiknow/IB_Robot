# chat_slow

Apologise with a recorded reply for reacting slowly.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `chat_slow` (recordings `S03` in `commissioning/audio/summit_clips.yaml`).
- recordings: 抱歉抱歉，我想了一下下～
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 你怎么反应这么慢 / 你好慢 / 反应太慢了.
