# self_introduction

The robot plays its recorded self-introduction clip with the motion selected by its implementation.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name and the runtime
  motion `wave` for `x2_runtime_v1`, or the original `self_intro` delivery recording for
  `aimdk_x2_summit`. The summit recording includes both arms and radian hand targets.
- the deployment profile resolves the clip: `embodied.runtime_play_audio.clips.self_intro`.
- every check for both the clip and the motion runs before anything is sent; the playback request goes
  out first and the bound motion right after it, without waiting for either to be accepted.
- a playback request that cannot be sent sends no motion; a playback failure while the motion runs cancels
  the motion and waits for it to stop.
- the skill succeeds only when the playback call and the motion both succeed.
- playback goes through `/voice_tts/play`; ASR stays active during playback.
- cancellation requests `/voice_tts/stop` and cancels the bound motion. The skill waits for playback,
  stop and motion results before reporting safe cancellation. The runtime owns the motion's exit policy.
- unconfirmed cleanup is reported as unknown, not safe cancellation. Already buffered hardware audio
  may still play; a playback response does not prove that the speaker is silent.
- aliases: 自我介绍 / 介绍一下你自己 / 你是谁.

The summit implementation uses a 90 s timeout. It returns to this execution’s measured arm/hand origin
on ordinary cancellation and waits for fresh feedback. The supplied assets remain draft; local Mock
results do not certify robot hardware, pronunciation, or missing permission/license metadata.
