# explain_fruit_prop

Say with a recorded reply that the fruits are props and cannot be eaten.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `fruit_prop` (recordings `Q08` in `commissioning/audio/summit_clips.yaml`).
- recordings: 这是水果道具，不能吃哦！ / 嘿嘿，是道具水果，可不能吃哦～
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 这个水果能吃吗 / 是真的水果吗 / 水果能吃吗 / 这是真水果吗.
