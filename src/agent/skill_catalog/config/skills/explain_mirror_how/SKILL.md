# explain_mirror_how

Explain with a recorded reply how the robot follows the visitor's arm movements.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `intro_mirror_how` (recordings `I03` in `commissioning/audio/summit_clips.yaml`).
- recordings: 我用摄像头看你手臂的姿势，再变成我的关节动作，跟着你动哦！ / 嘿嘿，眼睛盯着你的手臂，你怎么动，我就怎么学～
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 你怎么跟着我动 / 你是怎么模仿的 / 你怎么学我的动作 / 照镜子的原理 / 你怎么知道我在做什么动作.
