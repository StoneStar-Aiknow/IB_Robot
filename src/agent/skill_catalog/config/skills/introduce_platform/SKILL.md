# introduce_platform

The robot plays the recorded answer about the chip, the operating system and the framework.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `intro_platform`.
- the deployment profile resolves it: `embodied.runtime_play_audio.clips.intro_platform`.
- playback goes through `/voice_tts/play`; ASR stays active. Cancellation uses `/voice_tts/stop` and waits for the admitted playback response. Already buffered hardware audio may still play.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 用什么芯片 / 什么芯片 / 什么系统 / 你的系统是什么 / 你用的是什么系统.
