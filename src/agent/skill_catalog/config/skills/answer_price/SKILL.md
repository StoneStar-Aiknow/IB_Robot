# answer_price

Answer a price or purchase question with a recorded reply: it is open source software; scan the repository QR code.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `price_first` (recordings `Q17A` in `commissioning/audio/summit_clips.yaml`).
- recordings: 嘿嘿，我们展示的是开源软件，能下载用在很多机器人上哦！ / 扫旁边的代码仓库二维码看看哦～
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 你多少钱一台 / 能买吗 / 怎么卖 / 你多少钱 / 在哪里买.
