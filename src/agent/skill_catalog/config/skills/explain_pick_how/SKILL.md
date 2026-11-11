# explain_pick_how

Explain with a recorded reply how the robot picks a fruit and delivers it.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `intro_pick_how` (recordings `I04` in `commissioning/audio/summit_clips.yaml`).
- recordings: 先听你要哪个水果，再找到它在哪，抓起来送到领取盒哦！ / 嗯嗯，看准位置再伸手，抓稳了就送到领取盒～
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 你怎么抓水果 / 你是怎么抓东西的 / 抓取的原理 / 你怎么拿起来的.
