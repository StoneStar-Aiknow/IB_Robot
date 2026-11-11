# decline_personal_item

Say with a recorded reply that the robot only picks the fruit gifts, not the visitor's belongings.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `item_unsupported` (recordings `F04A` in `commissioning/audio/summit_clips.yaml`).
- recordings: 你的东西我不碰哦，收好啦～ / 嘿嘿，我只拿水果礼物哦！ / 唔，这个不能抓，我只拿水果哦～
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 帮我拿手机 / 把我的包拿过来 / 抓我的手机 / 帮我拿一下杯子 / 把那个瓶子拿给我.
