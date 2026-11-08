"""X2 mirror voice: a short phrase spoken as the session changes state.

Modules:

``core``  which session transitions are spoken, each at most once; the queue
          of phrases waiting to be spoken; the phrase chosen for an event;
          when the missing TTS service is reported.
``speaker``  the order of a phrase: queue, WAV cache or synthesis, playback
             segment by segment; precaching, call timeouts, failures and
             ``~/status``. With prerecorded clips: no synthesis, event and
             standby choice, queue priority.
``asr_mute``  when ASR is stopped before a clip and started again after it.

The ``voice`` section of the config, its defaults and its checks are in
``embodied_common.mirror_voice_contract``, which the session's launch builder
shares. Nothing in this package makes a ROS call, so all of it can be
exercised with plain dicts and a fake clock. ``mirror_voice_node`` adapts it
to ROS.
"""
