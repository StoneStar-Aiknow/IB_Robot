# mimic_not_ready

The robot plays the recorded reply that the mimic game is not ready and the fruit gift is available.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `mimic_unavailable`.
- the deployment profile resolves it: `embodied.runtime_play_audio.clips.mimic_unavailable`.
- playback goes through `/voice_tts/play`; ASR stays active. Cancellation uses `/voice_tts/stop` and waits for the admitted playback response. Already buffered hardware audio may still play.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 动作模仿 / 照镜子 / 跟着我做 / 学我的动作.
