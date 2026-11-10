# answer_language

Say with a recorded reply that the robot mainly speaks Chinese.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `language` (recordings `Q34` in `commissioning/audio/summit_clips.yaml`).
- recordings: 我现在主要说中文哦～ / 嘿嘿，我主要说中文呢！
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 你能说英语吗 / 你会说英语吗 / 你会英文吗 / can you speak english.
