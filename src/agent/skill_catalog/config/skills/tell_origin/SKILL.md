# tell_origin

Say with a recorded reply that the software shown is developed by the openEuler open source community.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `who_made` (recordings `Q12` in `commissioning/audio/summit_clips.yaml`).
- recordings: 嘿嘿，这是 openEuler 开源社区开发的软件，能跑在很多机器人上哦！ / 我们展示的是 openEuler 开源社区的软件，很多机器人都能用呀～
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 是谁把你做出来的 / 你背后的团队是谁 / 谁开发的你 / 你是哪个公司的 / 谁做的你.
