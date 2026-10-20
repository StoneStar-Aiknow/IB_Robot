"""X2 mirror voice: a short phrase spoken as the session changes state.

Modules:

``core``  which session transitions are spoken, each at most once; the queue
          of phrases waiting to be spoken; the phrase chosen for an event;
          when the missing TTS service is reported.

The ``voice`` section of the config, its defaults and its checks are in
``embodied_common.mirror_voice_contract``, which the session's launch builder
shares. Nothing in this package makes a ROS call, so all of it can be
exercised with plain dicts and a fake clock. ``mirror_voice_node`` adapts it
to ROS.
"""
