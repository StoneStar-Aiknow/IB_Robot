# introduce_in_detail

The robot plays the 22 s introduction of the platform and the Agent. It joins two recorded parts, because the third part describes the placement step this release does not have.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `intro_detail`.
- the deployment profile resolves it: `embodied.runtime_play_audio.clips.intro_detail`.
- playback goes through `/voice_tts/play`; ASR stays active. Cancellation uses `/voice_tts/stop` and waits for the admitted playback response. Already buffered hardware audio may still play.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 详细介绍一下 / 详细介绍一下你自己 / 详细讲一下原理.
