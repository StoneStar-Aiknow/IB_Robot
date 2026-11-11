# chat_food

Say with a recorded reply that the robot does not eat; it charges.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `chat_food` (recordings `S05` in `commissioning/audio/summit_clips.yaml`).
- recordings: 我不用吃饭哦，充电就行！
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 机器人吃什么 / 你吃饭吗 / 你要吃东西吗 / 你饿不饿.
