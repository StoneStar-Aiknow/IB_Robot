# greet_visitor

The robot plays the opening line and waves one hand while it plays. The wave is the one `wave_hand` replays.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `greeting`.
- the deployment profile resolves it: `embodied.runtime_play_audio.clips.greeting`.
- playback goes through `/voice_tts/play`; ASR stays active. Cancellation uses `/voice_tts/stop` and waits for the admitted playback response. Already buffered hardware audio may still play.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 你好 / 你好呀 / 打个招呼 / 跟大家打招呼 / 欢迎一下.
