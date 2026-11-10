# explain_ears

Say with a recorded reply that the robot hears through its microphone.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `robot_ears` (recordings `Q22` in `commissioning/audio/summit_clips.yaml`).
- recordings: 我有麦克风呀，耳朵灵着呢！
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 你为什么能听见我说话 / 你怎么听见的 / 你有耳朵吗 / 你用什么听.
