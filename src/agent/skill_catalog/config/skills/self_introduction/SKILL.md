# self_introduction

The robot plays its recorded self-introduction clip (fixed audio, pre-recorded by the team) and waves
one hand while it plays.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name and the runtime
  motion `wave` (the same recorded wave `wave_hand` replays).
- the deployment profile resolves the clip: `embodied.runtime_play_audio.clips.self_intro`.
- every check for both the clip and the motion runs before anything is sent; the playback request goes
  out first and the wave right after it, without waiting for either to be accepted.
- a playback request that cannot be sent sends no wave; a playback failure while the wave runs cancels
  the wave and waits for it to stop.
- the skill succeeds only when the playback call and the wave both succeed.
- playback goes through `/voice_tts/play`; the ASR is muted (`/voice_control` stop/start) until both end.
- the playback service has no cancel: a skill cancel or timeout stops the wave and stops waiting for the
  clip, which may keep playing. Playback success means the service finished its request, not that the
  speaker finished.
- aliases: 自我介绍 / 介绍一下你自己 / 你是谁.
