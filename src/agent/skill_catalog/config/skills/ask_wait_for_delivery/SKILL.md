# ask_wait_for_delivery

Ask with a recorded reply to wait until the gift has arrived before taking it.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `gift_wait` (recordings `Q07` in `commissioning/audio/summit_clips.yaml`).
- recordings: 先别取，等确认送到再拿哦～ / 还没好呢，送到了再拿哦！
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 可以拿了吗 / 我现在能拿吗 / 能拿了吗 / 到了吗.
