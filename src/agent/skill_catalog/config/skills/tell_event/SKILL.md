# tell_event

Name the event, OpenEuler Summit, with a recorded reply.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `summit_name` (recordings `Q15` in `commissioning/audio/summit_clips.yaml`).
- recordings: 这里是 OpenEuler Summit 操作系统大会哦！ / 嘿嘿，是 OpenEuler Summit 操作系统大会呀～
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 这是什么大会 / 你参加的是什么活动 / 这是什么会 / 什么展会.
