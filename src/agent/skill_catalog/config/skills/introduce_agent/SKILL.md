# introduce_agent

The robot plays the recorded answer about what the Agent does.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `intro_agent`.
- the deployment profile resolves it: `embodied.runtime_play_audio.clips.intro_agent`.
- playback goes through `/voice_tts/play`; ASR stays active. Cancellation uses `/voice_tts/stop` and waits for the admitted playback response. Already buffered hardware audio may still play.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: Agent有什么用 / 你聪明在哪 / 你的智能体是干什么的.
