# tell_venue

Name the venue, 北京中关村国际创新中心, with a recorded reply.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `venue` (recordings `Q16` in `commissioning/audio/summit_clips.yaml`).
- recordings: 我在北京中关村国际创新中心哦！ / 嘿嘿，在北京中关村国际创新中心呀～
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 你在哪儿 / 你在哪个城市 / 这是哪里 / 你现在在哪.
