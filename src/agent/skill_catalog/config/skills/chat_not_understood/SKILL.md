# chat_not_understood

Say with a recorded reply that the robot did not understand.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `chat_not_understood` (recordings `S07` in `commissioning/audio/summit_clips.yaml`).
- recordings: 唔，这个我没听懂呢！ / 你能换个说法吗？
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 你听懂了吗 / 你明白吗.
