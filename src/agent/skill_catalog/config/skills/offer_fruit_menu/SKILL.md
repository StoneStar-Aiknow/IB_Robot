# offer_fruit_menu

The robot plays the recorded question listing lemon, apple and orange.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `fruit_menu`.
- the deployment profile resolves it: `embodied.runtime_play_audio.clips.fruit_menu`.
- playback goes through `/voice_tts/play`; ASR stays active. Cancellation uses `/voice_tts/stop` and waits for the admitted playback response. Already buffered hardware audio may still play.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 选水果 / 水果礼物 / 我要水果 / 有什么水果.
