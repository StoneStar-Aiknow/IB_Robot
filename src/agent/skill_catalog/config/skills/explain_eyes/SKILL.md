# explain_eyes

Say with a recorded reply that the robot sees through the camera on its head.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `robot_eyes` (recordings `Q21` in `commissioning/audio/summit_clips.yaml`).
- recordings: 我头上有摄像头哦，看得见你！ / 嘿嘿，靠头上的摄像头看你呀～
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 你有眼睛吗 / 你通过哪个摄像头看我 / 你能看见我吗 / 你的眼睛在哪.
