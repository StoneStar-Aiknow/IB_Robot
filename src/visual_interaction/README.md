# visual_interaction

`visual_interaction` 是 X2 的**主动视觉互动**会话层：机器人自己观察当前访客，发现值得回应的
新内容时**动态生成**一句中文短评，没有新内容就安静。它与现有「视觉游戏」链路的区别是请求方向
和文本来源——视觉游戏由访客触发、从固定文案表取句；本包常驻观察、句子由模型生成。

设计与阶段零计划见 [`docs/x2_visual_interaction_design_zh.md`](../../docs/x2_visual_interaction_design_zh.md)。

> **当前状态：骨架 + 假 VL。**
> 真 VL 模型未接入（`vl.implementation` 只接受 `fake`），**不会发出任何声音**，
> 只把要说的话发到 topic 和日志。真实发声依赖尚不存在的全局发声协调器，见
> [发声与协调器 TODO](#发声与协调器-todo)。

## 为什么要新建消息和消费节点

现有视觉游戏链路的契约是**封闭枚举**，动态文本走不通：

- `VisualGameEvent.msg` 的 11 个字段里没有 `text`；要说的话不在消息里，由播报节点拿 `handler`
  去查 `visual_game_contracts._HANDLERS`（`deep_freeze` 冻结的表）。
- 三层校验都是枚举形状：`perception_service_node` 按 `response_contract` 校验（超出
  `allowed_values` 报 `INVALID_RESPONSE_CONTRACT`）→ 网关 `validate_visual_game_result`
  → 播报节点 `announcement_text()` 查表，不在表里返回 `None` 静默丢弃。

把自由文本塞进 `scene_summary` 会在第 1 层被拒、第 3 层被丢。所以本包有自己的
`VisualInteractionUtterance.msg`（带 `text`）和自己的会话节点。

## 分层

照 mirror-session 的做法：**ROS-free 核心 + 薄 ROS 适配**。`session/` 下所有模块都不 import
rclpy，因此整个决策逻辑可以用假时钟、假模型在本机全量测试。

```
visual_interaction/
├── session/                   # ROS-free 核心
│   ├── core.py                # 一轮的检查顺序与发声准入
│   ├── semantics.py           # 封闭语义解析 + 内容白名单
│   ├── topics.py              # 话题键、覆盖状态、两轮确认
│   ├── validity.py            # 有效期、actor_epoch、会话边界
│   ├── hooks.py               # 扩展点（VL / 发声 / 变化门 / 时钟），默认全保守
│   ├── endpoints.py           # 按接口 ID 解析端点 + 无运动门禁
│   └── fake_vl.py             # 脚本化假 VL，测试用
├── visual_interaction_node.py # ROS 适配
├── launch_builders/           # BuilderSpec + entry point 注册
└── phase0/                    # 阶段零测量工具，不参与互动运行时
```

一轮的检查顺序本身是设计决策（详见 `session/core.py` 模块文档）：先确认目标→再过变化门→
调模型→解析答案→覆盖与确认→**发声前再查一次有效性**。最后一步用的是模型跑完之后的时间，
不是本轮开始的时间；假模型瞬间返回，正因如此时钟是一个 port 而不是捕获值。

## 公开接口

| 接口 | 类型 | 说明 |
| --- | --- | --- |
| `/visual_interaction/utterance` | `ibrobot_msgs/msg/VisualInteractionUtterance` | 要说的话及其出处；首版只发不播 |
| `/visual_interaction/status` | `std_msgs/msg/String`（JSON） | 诊断：轮次、帧数、会话、各类结果计数 |
| `/visual_interaction/set_enabled` | `std_srvs/srv/SetBool` | `data=false` 静音、`true` 恢复 |

`set_enabled` 不是可选附件：访客自己碰不到 ROS 服务，所以工作人员控制台或按钮是功能的一部分；
**静音只能由显式 `resume` 解除**，不会被任何定时器清掉。

`VisualInteractionUtterance` 的 `header.stamp` 是**所依据画面中最新一帧的时间**，不是发布时间；
`evidence_start_stamp` 是窗口最老一帧；`latest_start_at` 是最晚可以开始发声的时刻
（由证据时效预算推出），**不是播放结束的截止时间**，安静轮为 0。

## 配置

由 robot YAML 的 `visual_interaction` 块启用，参考
[`aimdk_x2_visual_interaction.yaml`](../robot_config/config/robots/aimdk_x2_visual_interaction.yaml)。
**装包只注册 builder，不启用功能**；缺 `enabled` 键等于关闭。

| 键 | 取值 | 说明 |
| --- | --- | --- |
| `enabled` | `true` | 必须是布尔；缺键=关闭 |
| `frame_source.interface` | `camera.stereo_head_front_left.color_compressed` | 接口 **ID**，不是 topic 名；端点由 `runtime.interface_description` 解析 |
| `vl.implementation` | `fake` | 目前只接受 `fake`；真模型等阶段零 |
| `categories` | `[toy, sign, hat, badge]` | 只能是契约白名单的**子集**，可收窄不可放宽；空列表被拒 |
| `session.confirmation_count` | `2` | 静态话题要在两个不同采样窗口被看到才可说 |
| `session.max_evidence_age_sec` | `8.0` | 证据到发声时允许的最大年龄 |
| `session.min_gap_sec` | `10.0` | 从上一句**估算结束**起算的安静间隔 |
| `session.absent_reset_sec` | `30.0` | 确认离开后重置会话 |
| `session.max_session_sec` | `120.0` | 到期**不重置**，转 `WAITING_NEXT_VISITOR` |

**相机复用 mirror 已在用的那一路压缩流**（endpoint
`/aima/hal/sensor/stereo_head_front_left/rgb_image/compressed`，30 Hz，`CompressedImage`）。
不用 VL demo 实测的 `rgb_head_front_center/rgb_image_half`：厂商 v0.8.1 已把它从公开接口集撤回，
且 `aimdk_robot` 的 profile 契约测试断言它不得出现在任何 endpoint。

所有时序默认值都是**提案，不是测量结果**，阶段零测完才冻结。

## 无运动保证

本功能只观察，不申请运动控制权、不发关节命令。这由两道互补的检查共同保证：

1. **节点侧**（`session/endpoints.py`）：`FORBIDDEN_PREFIXES` 覆盖 `joint.`、`runtime.set_mode`、
   `runtime.control`、`motion.`、`base.`、`arm.`、`hand.`、`gripper.`、`nav.`；
   `assert_observation_only` 拒绝解析任何触碰运动接口的描述集。
2. **launch 侧**（`launch_builders/visual_interaction.py`）：拒绝任何 control mode 里存在
   **使能的** `inference` 或 `executor`。判定与 `robot_config` 一致——
   `execution._executor_enabled` 读的是 `executor.get("enabled", True)`，所以**缺 `enabled` 键
   视为开启**；否则 `aimdk_x2_skills.yaml` 继承进来的 teleop 和 navigation 会被漏判。

节点在一个进程里看不到同一 launch 里的别人，所以两道都需要。

## 诊断：一次安静轮是正常结果

每轮必定落到恰好一个结果。`suppress_reason` 取自契约的封闭集合，读日志不需要解析自然语言：

| suppress_reason | 含义 | 是否异常 |
| --- | --- | --- |
| `NO_NEW_TOPIC` | 模型答「没有新内容」，或变化门没放行 | 否，功能正常 |
| `UNCERTAIN` | 模型答「看不清」；**会打断确认链** | 否，但连续出现要查画面 |
| `DUPLICATE` | 该话题本会话已说过 | 否 |
| `UNCONFIRMED` | 还没攒够两个窗口的确认 | 否 |
| `PRIVACY` | 类别不在白名单，或回复内容越界 | 否，边界在起作用 |
| `SCHEMA_INVALID` | 模型输出不是合法封闭 JSON，或长度越界 | 看频率；高频说明提示词与宿主不一致 |
| `AMBIGUOUS` | 画面里分不清是哪个人 | 否 |
| `STALE` | 目标跟丢，或证据过期 | 否 |
| `COOLDOWN` | 还在上一句之后的安静间隔里 | 否 |
| `MUTED` | 被显式静音 | 否 |
| `IN_FLIGHT` | 该话题正在发声中 | 否 |
| `MODEL_NOT_READY` | 模型不可达 | **是** |
| `WAITING_NEXT_VISITOR` | 会话到最大时长，等工作人员交接 | 否，但需要人介入 |

`NO_NEW_TOPIC` 和 `UNCERTAIN` 故意分开计数：一个是「正常地没话说」，一个是「看不清」，
长时间沉默要能区分这两种。

## 发声与协调器 TODO

**本包目前不发声。** 现状是三个互不知情的发声源直接调 `/voice_tts/synthesize` 和
`/voice_tts/play`——`visual_game_announcer_node.py`、`ibrobot_agent/voice_bridge.py`、
`robot_skill_cli/hermes_tts_hook.py`——之间只有进程内锁，播放节点 FIFO 串行、无优先级，
输的一方最多阻塞到自己超时。全局层面**没有**跨进程发声锁、`is_speaking` 状态、优先级抢占、
频率预算，`PlayAudioFile` 也**没有请求级取消**。另外播放服务返回成功只表示 PCM 已发布并 ack，
**不等于扬声器播完**。

因此 `session/hooks.py` 的 `SpeechPort` 默认实现只发消息、不出声；
`Delivery.estimated_end` 是**估算**，不是硬件确认。协调器（同事的节点）到位后实现同一个
port 即可，会话逻辑不动，消息定义也不用改。

注意：现有测试 `test_new_game_does_not_interrupt_previous_round_playback`
**断言播放中不可打断**，与将来的 barge-in 需求冲突。本包不碰那条测试。

## 阶段零测量工具（`phase0/`）

`phase0/` 不参与互动运行时，是阶段零的测量工具，手动运行。每个工具写一份
`measurement.json`，带上 commit、依赖/资产摘要和运行条件——测量只对它跑过的那套代码、
模型、预处理和 ABI 有效，任何一项变了证据即失效（设计 §11）。

| 工具 | 跑在哪 | 作用 |
| --- | --- | --- |
| `token_budget` | 任意主机 | 按固定 OM 形状核对本功能提示词的 token 预算；只读 tokenizer 数据，不加载模型 |
| `vl_segments` | 板端（读日志） | 从 demo 自己的 `talk.jsonl` 汇总分段耗时与重试分布 |
| `golden_compare` | 任意主机 | 在**事先冻结**的容差下比对两份输出归档 |
| `tts_timing` | 板端 | 在标注的负载条件下测 `/voice_tts/synthesize`；只合成，不播放 |
| `pcm_intervals` | 板端 | 监听 `/audio/play`，报发布间隔与相对采样时钟的滞后；只订阅，不发布 |

```bash
source .shrc_local && python3 -m visual_interaction.phase0.token_budget \
    --tokenizer-dir <含 vocab.json/merges.txt/added_tokens.json 的目录> \
    --output measurement.json
```

**这些工具都不 import 板上 VL demo 的代码。** demo 没有 LICENSE、且自称提示词与逻辑照搬另一个
本机脚本，来源确认前它的预处理和生成循环不往本仓搬：`vl_segments` 读它自己写的日志，
`token_budget` 只读 tokenizer 的**数据文件**（`vocab.json` / `merges.txt` / `added_tokens.json`）。

`phase0/tokenizer.py` 不是自己实现 BPE，而是用 `tokenizers` 库按 **Qwen2 公开的分词规范**
加载那三个数据文件：NFC 规范化 → 按 added token 切开 → Qwen2 预切分正则 → byte-level BPE。
板上 `qwen_tok.py` 自己的注释也写着「对齐 HF 慢速 Qwen2Tokenizer」，两边依据的是同一份规范。

**一个实现细节容易错**：`added_tokens.json` 里带着每个特殊 token 的 id，必须把它们
**先并进词表再建模型**。用 `add_tokens()` 事后追加会让库重新编号（`<img>` 本该 151665，
会变成 151672），token 数就全错了。

#### 上板时要做的一次性对照（阶段零 0C 清单项）

本机验证只覆盖了三种输入（中文短句、图像块、对话模板标记），都与板上输出 id 全等。
但**没有覆盖全部字符组合**——标点、emoji、中英混排的边界都没测过。所以上板后要做一次对照，
确认本机计数可以继续信任：

```bash
# 板上，用 demo 自己的分词器对同一段提示词计数
cd /root/vlx_internvl_310p && python3 -c "
import sys; sys.path.insert(0, 'scripts')
from qwen_tok import Tokenizer
print(len(Tokenizer('tokenizer').encode(open('/tmp/prompt.txt', encoding='utf-8').read())))
"
```

把 `token_budget` 用的同一段提示词落到 `/tmp/prompt.txt`，两边数字应当相同。
对上之后这个工具就可以放心在本机用；对不上就说明有字符走了不同的路径，先定位再继续，
**不要直接改容差或改数字**。

### 已测出的结论：提示词预算比设计假设紧得多

按板上 tokenizer 数据、对当前 OM 导出（P=1280、L=1344、4 帧 × 256 image token = 1024 个位置）
实测，**图像之后只剩 256 个位置留给全部文本**：

| 场景 | prompt token | 余量 |
| --- | --- | --- |
| 最小规则（中文） | 1238 | +42 |
| 最小规则（英文） | 1227 | +53 |
| 完整规则 | 1284 | **−4** |
| 完整 A/B/C 记忆（20 条） | 1321 | **−41** |
| 8 个候选 | 1266 | +14 |
| 最坏组合 | 1395 | **−115** |

封闭 schema 自身（schema 77 + 最小规则 82 + 模板 27）就吃掉 186/256。解码侧不紧
（L=1344 下最小规则仍允许 107 个新 token）。所以**完整规则放不下、满额 A/B/C 记忆放不下**，
阶段零必须在「减帧 / 减每帧 image token / 按更大 P 重新导出 / 把记忆移出提示词」里做取舍。
`token_budget` 对放不下的场景返回退出码 1，这个边界因此是 fail-fast 的，不会留到板上某一轮才炸。

## 测试

```bash
source .shrc_local && ./scripts/build.sh --packages-select visual_interaction
source .shrc_local && colcon test --merge-install --packages-select visual_interaction
source .shrc_local && colcon test-result --verbose --test-result-base build/visual_interaction
```

正式结论只认 `colcon test`；裸 `pytest` 仅用于定向排查。`setup.py` 里的
`extras_require={"test": ["pytest"]}` 是**承重**的——没有它 colcon 会静默回退 unittest、
一个都不收集却报绿。

`session/` 全部 ROS-free，配合 `fake_vl.ScriptedVision` 和显式传入的时间，
测试可以在毫秒内跑完若干分钟的互动且结果确定。`phase0/` 的算术部分
（`stats`、`budget`、`prompts`、`pcm`、`golden`、`report`）同样不需要 ROS、模型或 NPU。

### 七动作回放：先优化说话内容

`action_probe` 默认使用 `natural_reply` 提示词，分类字段与解析方式不变，
但本轮重点是话术：少猜动作目的、明显互动手势直接回应、普通姿势允许安静。
不提供可被模型照抄的示例句，不再强制每句带「你」。安静使用 `reply: ""`，
包括识别出了动作却不值得主动评论的情况；不生成「保持安静」等可被误播的文字。

```bash
# 板端使用既有环境入口；以下参数路径指向自己的回放片段和 demo。
python3 -m visual_interaction.phase0.action_probe \
    --demo-root <demo目录> --frames-root <标注片段目录> --output replies.json

# 原来的四种分类问法保持原文，显式选择即可复现对照。
python3 -m visual_interaction.phase0.action_probe \
    --demo-root <demo目录> --frames-root <标注片段目录> --output baseline.json \
    --variants bare_list glossed glossed_examples two_step_category
```

人工对照原来的七段画面与报告中的 `records[].reply`，重点看：是否编造、是否像面对面回应、
是否复读套话、是否应该安静。分类准确率不代表话术质量。提示词里「同一持续动作不重复说」
只是模型要求，**本次未实现跨轮记忆、宿主去重、冷却或 TTS 播放控制**，也未修改互动运行时。
实际模型效果与固定 OM 的 token 预算仍须在板端回放验证，不能由主机单测代替。
