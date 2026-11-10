# fruit_not_available

Say with a recorded reply that this fruit is not available, only lemon, apple and orange.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `fruit_not_available` (recordings `F03` in `commissioning/audio/summit_clips.yaml`).
- recordings: 哎呀，这个没有呢，换一个吧～ / 唔，今天只有柠檬苹果橙子哦！ / 这个真没有，挑别的吧～ / 我只抓台上的水果哦～
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 我要西瓜 / 我选菠萝 / 可以选香蕉吗 / 有没有葡萄 / 我想要草莓 / 给我一个梨.
