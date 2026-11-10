# explain_arms_only

Say with a recorded reply that the robot can only imitate arm movements, not legs.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `arms_only` (recordings `Q32` in `commissioning/audio/summit_clips.yaml`).
- recordings: 我只能学手臂动作哦～ / 腿学不了，手臂可以呀！
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 你能学我蹲下吗 / 你能跳起来吗 / 你会跳舞吗 / 你能走路吗 / 你能踢腿吗.
