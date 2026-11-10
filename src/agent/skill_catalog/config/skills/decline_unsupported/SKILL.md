# decline_unsupported

Say with a recorded reply that the robot cannot do this request yet.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `cannot_do` (recordings `W09` in `commissioning/audio/summit_clips.yaml`).
- recordings: 哎呀，这个我还不会呢～ / 呜，这个我还没学会哦！ / 嘿嘿，这个难倒我啦！
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 唱首歌 / 讲个故事 / 作首诗 / 写首诗 / 拥抱一下 / 抱抱我 / 模仿青蛙 / 模仿小动物 / 看小红书.
