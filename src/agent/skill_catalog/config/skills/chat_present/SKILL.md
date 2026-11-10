# chat_present

Say with a recorded reply that the robot is here.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `chat_present` (recordings `S06` in `commissioning/audio/summit_clips.yaml`).
- recordings: 我在呢我在呢！ / 嗯嗯，一直都在哦～
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 你还在吗 / 你在吗 / 你好你还在吗 / 有人吗.
