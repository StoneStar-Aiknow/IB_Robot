# chat_dont_know

Say with a recorded reply that the robot does not know.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `chat_dont_know` (recordings `S08` in `commissioning/audio/summit_clips.yaml`).
- recordings: 哎呀，这个我不知道呢～
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 你知道吗 / 这个你知道吗.
