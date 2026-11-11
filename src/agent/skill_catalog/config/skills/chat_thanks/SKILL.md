# chat_thanks

Thank the visitor for a compliment with a recorded reply.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `chat_thanks` (recordings `S01` in `commissioning/audio/summit_clips.yaml`).
- recordings: 嘿嘿，谢谢夸奖！ / 哇，你真会说话！ / 嘻嘻，被你发现啦～
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 你真厉害 / 你好棒 / 你做的挺像的 / 你好可爱 / 真聪明.
