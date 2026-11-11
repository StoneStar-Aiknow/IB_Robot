# answer_safety

Ask with a recorded reply to interact as the on-site instructions say.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `safety_interact` (recordings `Q35` in `commissioning/audio/summit_clips.yaml`).
- recordings: 请按现场提示和我互动哦。 / 跟着现场提示和我玩就好哦。
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 你会伤害人吗 / 离你近危险吗 / 你安全吗 / 你会打人吗.
