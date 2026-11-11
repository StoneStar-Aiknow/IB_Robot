# chat_tired

Say with a recorded reply that the robot is not tired.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `chat_tired` (recordings `S04` in `commissioning/audio/summit_clips.yaml`).
- recordings: 不累不累，我是机器人呀！
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 你累不累 / 你累吗 / 你辛苦吗.
