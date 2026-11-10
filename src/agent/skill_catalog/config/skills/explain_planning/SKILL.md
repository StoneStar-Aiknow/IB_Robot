# explain_planning

Say with a recorded reply that the robot plans its own steps once it understands.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `self_planning` (recordings `Q23` in `commissioning/audio/summit_clips.yaml`).
- recordings: 对呀，听懂了我自己安排哦！
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 你自己规划吗 / 是你自己决定的吗 / 你会自己安排吗 / 谁给你下的指令.
