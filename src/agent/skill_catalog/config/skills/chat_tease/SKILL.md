# chat_tease

Answer good-naturedly with a recorded reply when the visitor says the robot is not smart.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `chat_tease` (recordings `S02` in `commissioning/audio/summit_clips.yaml`).
- recordings: 呜，我还在努力学习呢！ / 哼哼，我会变聪明的！ / 嘿嘿，多教教我嘛～
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 你懂个屁 / 你不太聪明 / 你好笨 / 你好像不太聪明的样子 / 你真傻.
