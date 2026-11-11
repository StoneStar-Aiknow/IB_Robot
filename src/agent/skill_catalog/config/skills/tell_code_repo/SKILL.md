# tell_code_repo

Say with a recorded reply that the QR code beside the robot leads to the code repository.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `code_repo` (recordings `Q14` in `commissioning/audio/summit_clips.yaml`).
- recordings: 扫旁边二维码，就能看到代码仓库哦！ / 欢迎参与呀，扫码进代码仓库看看～
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 你的代码在哪里 / 我能参与开发吗 / 代码仓库在哪 / 怎么参与开发 / 在哪下载.
