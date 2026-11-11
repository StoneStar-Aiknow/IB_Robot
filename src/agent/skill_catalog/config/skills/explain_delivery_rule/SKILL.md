# explain_delivery_rule

Say with a recorded reply that the robot cannot throw or hand over the fruit; it goes into the pickup box.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `delivery_unsupported` (recordings `F04B` in `commissioning/audio/summit_clips.yaml`).
- recordings: 不能扔也不能递哦，我放领取盒里～ / 哎呀，递不了，我放领取盒，你自己拿哦！ / 我只放进领取盒，你去拿哦～
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 把苹果扔给我 / 扔给我 / 直接递给我 / 放到我手里 / 把苹果递给我 / 直接给我.
