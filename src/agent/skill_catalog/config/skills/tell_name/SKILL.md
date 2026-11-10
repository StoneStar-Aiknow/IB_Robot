# tell_name

Tell the robot's name, 欧拉, with a recorded reply.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `robot_name` (recordings `Q10` in `commissioning/audio/summit_clips.yaml`).
- recordings: 嗯哼，我叫欧拉呀！ / 嘿嘿，我就是欧拉～ / 我叫欧拉，和 openEuler 同名哦！
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 你叫什么名字 / 你是不是欧拉 / 你的名字是什么 / 你叫什么.
