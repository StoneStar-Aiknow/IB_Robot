# explain_pickup

Say with a recorded reply that the fruit slides from the bowl into the pickup box.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `gift_where` (recordings `Q05` in `commissioning/audio/summit_clips.yaml`).
- recordings: 我放进碗里，它会滑到领取盒！ / 放进碗里滑下去，你去领取盒拿哦！
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 你怎么送给我 / 我去哪里拿 / 在哪里领 / 怎么领取 / 我去哪拿礼物.
