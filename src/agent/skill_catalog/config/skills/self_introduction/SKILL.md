# self_introduction

The robot plays its recorded self-introduction clip (fixed audio, pre-recorded by the team).

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name.
- the deployment profile resolves it: `embodied.runtime_play_audio.clips.self_intro`.
- playback goes through `/voice_tts/play`; the ASR is muted (`/voice_control` stop/start) while it plays.
- aliases: 自我介绍 / 介绍一下你自己 / 你是谁.
