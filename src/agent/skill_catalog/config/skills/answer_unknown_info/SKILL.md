# answer_unknown_info

Say with a recorded reply that the robot cannot look this up for now.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `cannot_lookup` (recordings `Q36` in `commissioning/audio/summit_clips.yaml`).
- recordings: 唔，这个我暂时查不到哦～ / 哎呀，这个我暂时查不到呢！
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 今天几号 / 天气怎么样 / 现在几点 / 今天星期几 / 明天会下雨吗.
