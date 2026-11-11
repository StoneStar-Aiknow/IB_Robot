# explain_photo

Say with a recorded reply that the experience keeps pictures automatically and the visitor chooses them afterwards.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `photo` (recordings `Q33` in `commissioning/audio/summit_clips.yaml`).
- recordings: 体验中会自动留图，结束后自己选哦！ / 嘿嘿，玩的时候自动留图，玩完自己挑～
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 你能给我拍照吗 / 你会拍照吗 / 给我拍张照 / 能拍照吗.
