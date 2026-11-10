# list_capabilities

List with a recorded reply what the robot can do: greet, move, mirror and give fruit.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `capabilities` (recordings `Q30` in `commissioning/audio/summit_clips.yaml`).
- recordings: 我会打招呼、做动作，还能送礼物哦！ / 嘿嘿，能陪你照镜子，还能送水果！
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 你会做什么 / 你能干什么 / 你有什么本事 / 你除了抓东西还能干什么 / 你能执行什么任务.
