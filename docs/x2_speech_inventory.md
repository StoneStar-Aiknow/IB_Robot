# X2 展会部署的语音输出清单

本文列出 `aimdk_x2_summit` 部署中机器人说出的每一种语音，以及它的播放方式。录音清单是 `commissioning/audio/summit_clips.yaml`，播放规则见 `src/embodied_common/embodied_common/recorded_clips.py`。

## 1. 播放方式

| 方式 | 说明 |
|---|---|
| 录音技能 | `runtime_play_audio` 技能播放片段，planner 选中或直达命令命中时执行；同一片段有多个文件时随机选一个，不重复上一次 |
| 语音桥录音 | 语音桥要说的文字与录音清单中某一条相同（忽略标点和大小写）时，播放该编号的录音（随机选一个文件），不再合成 |
| 静态 TTS | 代码或配置中的固定文字，没有录音，实时合成 |
| 动态 TTS | planner 的文字回答和带变量的播报，实时合成；英文先按 `tts_pronunciations` 换成中文读法 |
| 录音缺失时 | 语音桥录音的文件不存在时改为实时合成并记录告警；录音技能的文件都不存在时技能失败（`PLAY_AUDIO_CLIP_MISSING`） |

## 2. 录音技能

| 编号 | 技能 | 片段名 | 文件数 |
|---|---|---|---|
| W01 | `greet_visitor` | `greeting` | 3 |
| W04 | `invite_visitor` | `invitation` | 2 |
| W06 | `one_game_at_a_time` | `one_game_at_a_time` | 2 |
| W09 | `decline_unsupported` | `cannot_do` | 3 |
| I01 | `self_introduction` | `self_intro` | 3 |
| I02 | `introduce_platform` | `intro_platform` | 2 |
| I03 | `explain_mirror_how` | `intro_mirror_how` | 2 |
| I04 | `explain_pick_how` | `intro_pick_how` | 2 |
| I05 | `introduce_agent` | `intro_agent` | 3 |
| I06 | `introduce_in_detail` | `intro_detail` | 2 |
| I10 | `explain_pick_scope` | `pick_scope` | 2 |
| F01 | `offer_fruit_menu` | `fruit_menu` | 3 |
| F03 | `fruit_not_available` | `fruit_not_available` | 4 |
| F04A | `decline_personal_item` | `item_unsupported` | 3 |
| F04B | `explain_delivery_rule` | `delivery_unsupported` | 3 |
| Q04 | `explain_gift` | `gift_what` | 4 |
| Q05 | `explain_pickup` | `gift_where` | 2 |
| Q07 | `ask_wait_for_delivery` | `gift_wait` | 2 |
| Q08 | `explain_fruit_prop` | `fruit_prop` | 2 |
| Q10 | `tell_name` | `robot_name` | 3 |
| Q11 | `explain_ib_robot` | `ib_robot_meaning` | 2 |
| Q12 | `tell_origin` | `who_made` | 2 |
| Q13 | `explain_open_source` | `open_source` | 2 |
| Q14 | `tell_code_repo` | `code_repo` | 2 |
| Q15 | `tell_event` | `summit_name` | 2 |
| Q16 | `tell_venue` | `venue` | 2 |
| Q17A | `answer_price` | `price_first` | 1（两段顺序播放） |
| Q17B | `answer_body_price` | `price_body` | 2 |
| Q20 | `answer_linux` | `linux_based` | 2 |
| Q21 | `explain_eyes` | `robot_eyes` | 2 |
| Q22 | `explain_ears` | `robot_ears` | 1 |
| Q23 | `explain_planning` | `self_planning` | 1 |
| Q30 | `list_capabilities` | `capabilities` | 2 |
| Q32 | `explain_arms_only` | `arms_only` | 2 |
| Q33 | `explain_photo` | `photo` | 2 |
| Q34 | `answer_language` | `language` | 2 |
| Q35 | `answer_safety` | `safety_interact` | 2 |
| Q36 | `answer_unknown_info` | `cannot_lookup` | 2 |
| Q37 | `explain_mirror_game` | `mirror_howto` | 2 |
| S01 | `chat_thanks` | `chat_thanks` | 3 |
| S02 | `chat_tease` | `chat_tease` | 3 |
| S03 | `chat_slow` | `chat_slow` | 1 |
| S04 | `chat_tired` | `chat_tired` | 1 |
| S05 | `chat_food` | `chat_food` | 1 |
| S06 | `chat_present` | `chat_present` | 2 |
| S07 | `chat_not_understood` | `chat_not_understood` | 2 |
| S08 | `chat_dont_know` | `chat_dont_know` | 1 |

## 3. 语音桥录音

| 编号 | 文字（第一条） | 来源 |
|---|---|---|
| W03 | 啊？没听清，玩模仿还是选水果呀？ | planner 澄清 `clarification_lines.not_heard` |
| F02 | 要柠檬、苹果还是橙子呀？ | planner 澄清 `fruit_name`；`speech_messages.reason_INVALID_TARGET` |
| F05A | 好嘞，给你拿柠檬！ | `gift_messages.F05`（柠檬） |
| F05B | 好嘞，给你拿苹果！ | `gift_messages.F05`（苹果） |
| F05C | 好嘞，给你拿橙子！ | `gift_messages.F05`（橙子） |
| F06 | 我在找它呢，别动桌子哦～ | 无：录音含停止词“别动”，不接入 |
| F07 | 水果的位置变了，我重新确认一下。 | `speech_messages.retry_TARGET_MOVED`（只用于不带放置的单独抓取） |
| F08 | 拿到了，我把它送到领取盒。 | `gift_messages.F08` |
| F09 | 耶，礼物到啦！去领取盒拿吧～ | `gift_messages.placed` |
| F10 | 礼物还没到，先别取，工作人员来帮忙。 | `gift_messages.place_ended`（放置结束但没有确认松手，或放置未成功时） |
| F12 | 哎呀，这次没成功，我再试一下。 | `gift_messages.F12`、`speech_messages.retry*` |
| F14 | 已经在拿啦，这次改不了哦～ | `spoken_lines.busy_pick` |
| F16 | 还是没成功，请工作人员帮我一下。 | `gift_messages.pick_failed/place_failed`、`speech_messages.failed/reason_*` |
| F17 | 我再确认一下，稍等。 | `gift_messages.verifying`（只在检查碗内时出现；展会 `verify_placement: false`，现在不会触发） |
| F18 | 好啦，这次任务完成啦！ | `gift_messages.completed`（任务确认成功，最后一步为归位或其他非抓放步骤） |
| F19 | 任务已结束，请工作人员确认现场状态。 | `gift_messages.ended`（通用结束提示，未确认任务成功） |
| E02 | 已经停下了。请先走出体验区。 | `gift_messages.cancelled`、`speech_messages.canceled` |
| E06 | 请先别靠近，工作人员正在处理。 | `gift_messages.E06`、`speech_messages.unknown/recovery_failed`（恢复失败或状态不明时） |
| Q01 | 嘿嘿，这轮只能选一份哦～ | `answer_rules.lines.only_one` |
| Q02 | 这轮已经送过礼物啦，重新叫欧拉欧拉可以开始新一轮。 | `answer_rules.lines.already_given`；3 条录音随机播放，不连续重复 |
| Q03A | 每个都有礼物哦，要我帮你挑吗？ | `answer_rules.lines.recommend`、`clarification_lines.recommend` |
| Q03B | 每个都有礼物，我帮你挑一个！ | `answer_rules.lines.random_pick`（代选送水果前先说） |
| Q06 | 可以呀，现在就给你拿吗？ | `answer_rules.lines.confirm_pick`、`clarification_lines.confirm_pick` |
| X07A | 出于安全考虑，我暂时不支持这类肢体接触哦。 | `answer_rules.contact_replies`（拥抱、触摸、亲吻、握手等接触请求；三句轮换，不执行动作） |
| X07B | 为了安全，这类身体接触的互动暂时还不支持哦。 | `answer_rules.contact_replies` |
| X07C | 这类接触互动暂时不支持，我们可以聊聊天。 | `answer_rules.contact_replies` |
| F04C | 不能直接递给你哦，我放进碗里，滑到领取盒你自己拿，可以吗？ | `answer_rules.lines.confirm_delivery`（要求扔、递、直接给时先问；回答“好的”才抓放） |
| X01 | 好呀，看我的！ | `spoken_lines.gesture_start` |
| X02 | 搞定啦！ | `spoken_lines.gesture_done` |
| X03 | 等一下下，我还在忙呢～ | `spoken_lines.busy` |
| X04 | 好嘞～ | `spoken_lines.speech_stopped` |
| X05 | 我在呢！ | `spoken_lines.wake_ack` |
| X06 | 要找我，先叫欧拉欧拉哦～ | `spoken_lines.wake_hint` |
| T01 | 模仿功能还在运行或状态待确认，请稍等。 | 代码中的固定文字（310P 合成超过 3 s，改为录音） |
| T02 | 好呀，我在准备了，稍等一下。 | 代码中的固定文字（310P 合成超过 3 s，改为录音） |
| T03 | 停止状态尚未确认，请工作人员检查机器人和物体。 | 代码中的固定文字（310P 合成超过 3 s，改为录音） |
| T04 | 要我做动作的话，请直接说出动作，比如“挥挥手”。 | 代码中的固定文字（310P 合成超过 3 s，改为录音） |
| T05 | 澄清上下文已过期或不属于当前会话，请重新说明完整请求。 | 代码中的固定文字（310P 合成超过 3 s，改为录音） |
| T06 | 上一次任务的持物状态尚未确认，需要先恢复后才能执行动作。 | 代码中的固定文字（310P 合成超过 3 s，改为录音） |
| T07 | 这次没能完成放置，已完成收尾，可以重新下达指令。 | 代码中的固定文字（310P 合成超过 3 s，改为录音） |
| T08 | 物体已经松开，但还没确认放置成功，请工作人员检查。 | 代码中的固定文字（310P 合成超过 3 s，改为录音） |
| T09 | 物体已经松开，动作还未全部完成，请先别靠近。 | 代码中的固定文字（310P 合成超过 3 s，改为录音） |
| T10 | 我正在确认水果的位置，请先别移动桌上的物品。 | 代码中的固定文字（310P 合成超过 3 s，改为录音） |
| T11 | 我没看到这个水果，请工作人员帮我看看。 | 代码中的固定文字（310P 合成超过 3 s，改为录音） |
| T12 | 照镜子正在进行，本次动作未执行，也不会排队。请结束照镜子后重新下达指令。 | 代码中的固定文字（310P 合成超过 3 s，改为录音） |

## 4. 只做文件级验证的录音

现在没有流程触发这些录音（`consumer: file_only`）：W02、W05、C01、I07、I08、I09、E04、E08、E09、E11。

## 5. 仍然实时合成的语音

### 5.1 静态 TTS（固定文字，310P 合成不超过 3 s）

| 来源 | 文字 |
|---|---|
| 语音桥 continuation | 从当前位置继续准备。 |
| 语音桥请求超时 | 任务超时，正在中止。 |
| 执行反馈 | 已停止。 |
| 执行反馈 | 任务已取消，尚未开始执行。 |
| 执行反馈 | 执行结果未知，请操作员确认机器人状态。 |
| 水果交付 recovery_failed | 自动收尾未完成，请工作人员检查机器人和物体。 |
| 单独抓取 start | 好，我这就去拿，请稍等。 |
| 单独抓取 success | 成功拿到啦！小礼物请找旁边的工作人员领取。 |
| 单独抓取 not_started | 好的，这次先不拿。 |
| 单独抓取 manual_check | 这次没有完成，请工作人员帮我看看。 |
| voice_confirm 模式（展会不用） | 确认超时，正在取消。 / 请说确认或取消。 |

310P（`ascend_310p`）实测：53 句静态和动态样本合成时延中位数 2.1 s，最长 4.6 s；超过 3 s 的 12 句静态文字已改为录音（第 3 节 T01–T12）。

### 5.2 动态 TTS

- planner 的文字回答（人设、笑话、部署事实没有覆盖的问题）：提示词要求不超过 15 个汉字、不用英文；英文仍出现时按 `tts_pronunciations` 读成中文，其余英文删除。
- 带变量的播报：执行失败的阶段说明（“抓取在{阶段}阶段失败。”）、`准备执行：{摘要}。`（旧版 Agent Plan 才会出现）。
- 录音缺失时的兜底（第 1 节）。

## 6. 准备位和展示反馈录音（2026-11-10）

新增 P01–P04，每组 3 条自然文案，运行时随机播放且避免连续重复。

| 编号 | 用途 | 首条文案 |
|---|---|---|
| P01 | 显式回准备位开始；不插入抓取工作流旁白 | 好呀，我先准备一下。 |
| P02 | `return_to_task_pose` 确认成功 | 准备好啦！ |
| P03 | `free_pose` 正常结束；不声称精确到位 | 好啦，动作做完啦！ |
| P04 | `TASK_POSE_NOT_REACHED`，仍保持失败 | 这次没做好，请工作人员看看。 |

候选由本机 ZipVoice `ubuntu_onnx` 和原模型默认提示生成，每条文案生成 3 份，共 36 份。
择优后的 12 份放入 `commissioning/audio/summit/`，SHA256 记录在录音清单中。
候选和 ASR 检查位于 `/mnt/data/lwh/x2_agent_interaction/prepare_feedback_20261110/`。
“就位啦”的 ASR 将“位”识为同音字“为”，单独记录，不把这条当作完全逐字识别通过。
音频自动检查不代替人工试听。

Summit 默认不再启用握手技能；保留底层资源和其他部署，避免影响后续人工启用。

## 7. 重新唤醒的领取提示（2026-11-11）

Q02 替换为 3 条支持重新唤醒的文案。原句保持为匹配入口。
运行时随机播放一条录音，避免连续重复；缺失时沿用实时合成。

| 文件 | 文案 |
|------|------|
| `Q02_wake_01.wav` | 这轮已经送过礼物啦，重新叫欧拉欧拉可以开始新一轮。 |
| `Q02_wake_02.wav` | 嘿嘿，这轮已经送过啦！再叫声欧拉欧拉，咱们就能再选一份哦。 |
| `Q02_wake_03.wav` | 还想要呀？这轮已经送过啦，再喊欧拉欧拉，咱们再来一轮吧！ |

音频使用默认 310P（`10.42.0.251`）的 ZipVoice `ascend_310p` 节点生成。
每条文案调用 3 次；默认模型同一文案的输出一致，保留其中 1 份。
3 份音频均为 24 kHz、单声道、16 位 WAV，SHA256 记录在录音清单中。
本机 ASR 回检与文案逐字一致（忽略标点）。生成记录和回检结果位于
`log/gift_audio_310p_20261111/`。音频文件位于 `commissioning/audio/summit/Q02_wake_*.wav`。

## 8. 任务收尾录音（2026-11-11）

F18 和 F19 各有 3 条录音。运行时随机播放，避免连续重复。
Gateway 确认整个任务成功，且最后一步是归位或其他非抓放步骤时，语音桥使用 F18。
F18 只说明任务完成，不补充物体放置或领取结论。
放置缺少确认、恢复失败和执行状态未知时，语音桥保留各自的检查提示。
通用结束分支使用 F19，保留人工确认的含义。

| 文件 | 文案 |
|------|------|
| `F18_01.wav` | 好啦，这次任务完成啦！ |
| `F18_02.wav` | 搞定啦，我准备好陪你继续玩啦。 |
| `F18_03.wav` | 这次动作做完啦，咱们接着聊吧。 |
| `F19_01.wav` | 任务已结束，请工作人员确认现场状态。 |
| `F19_02.wav` | 这次任务结束啦，请工作人员帮我确认一下现场。 |
| `F19_03.wav` | 我这边结束啦，现场情况请工作人员再检查一下。 |

音频使用默认 310P（`10.42.0.251`）的 ZipVoice `ascend_310p` 节点生成。
文件格式为 24 kHz、单声道、16 位 WAV，SHA256 记录在录音清单中。
生成记录和 ASR 回检结果位于 `log/gift_completion_audio_310p_20261111/`。
