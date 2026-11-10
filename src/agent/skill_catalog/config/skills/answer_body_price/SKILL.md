# answer_body_price

Answer the price of the robot body with a recorded reply: a community partner provides it and we do not know its price.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `price_body` (recordings `Q17B` in `commissioning/audio/summit_clips.yaml`).
- recordings: 本体是社区伙伴提供的，价格我们也不清楚哦～ / 嗯，本体来自社区伙伴，具体多少钱我们也不知道呢～
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 本体多少钱 / 这台机器人多少钱 / 硬件多少钱 / 机器人本身多少钱.
