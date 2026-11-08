# mirror_voice：X2 镜像预录语音与 InternVL 看图（feat/x2-internvl-voice）

开关打开时，镜像相关的话只播预录 wav（走 `/voice_tts/play`），**任何情况下都不合成**：
四个过渡事件（starting / ready / mirroring / goodbye）、STANDBY 里的待机短词，以及镜像中
InternVL 看图后匹配到的一句。配置在 `config/robots/aimdk_x2_voice_mirror_all.yaml` 末尾的
`internvl_voice` 块。

这个目录里只有文本：

| 文件 | 说明 |
|---|---|
| `x2_clips_v1.yaml` | 句表，61 句，`version: 1`。随 `robot_config` 安装，由发布目录的 `clips.tsv` 生成，不要手改 |
| `README.md` | 本文件 |

wav 和模型都不进 git，按下面放到板上。

## 1. 三样东西放在哪

| 东西 | 位置 | 配置键 |
|---|---|---|
| 61 个 wav | `$WORKSPACE/assets/audio/x2_mirror_clips_v1/`，wav 直接放在这一层。在 .5 上是 `/root/x2-feat-internvl-voice-wt/assets/audio/x2_mirror_clips_v1/` | `internvl_voice.clip_dir`（相对 `$WORKSPACE`，也可写绝对路径） |
| InternVL3-1B 310P | `/root/vlx_internvl_310p/`：用到 `om/` 里 3 个 OM、`tokenizer/`、`scripts/`、`live/`、`onnx/embed_tokens_fp32.npy`。部署包 `visual_talk_wrist` 的 `3_vlx.tar` 就解到这里；不挪、不复制 | `internvl_voice.look.demo_root`（绝对路径，或相对 `$WORKSPACE`） |
| 句表 | 本目录的 `x2_clips_v1.yaml`，随 `robot_config` 装到 share 目录 | `internvl_voice.clips_table`，默认 `$(find robot_config)/config/mirror_voice/x2_clips_v1.yaml` |

- wav 是 24000 Hz、单声道、16 bit，文件名 `<id>_v1.wav`，`id` 和句表里的键一一对应。
- `clip_dir` 下找不到 wav 时，还会去默认的 `$WORKSPACE/assets/audio/x2_mirror_clips_v1` 找一次。
- `assets/` 是树里的未跟踪文件，deploy.sh 不会清理；但整棵树删了重建后要重新拷。

## 2. 从哪里拿 wav

发布目录（本机）：`/mnt/oee/WEDDINGPRESENT/VOICE/x2_mirror_clips_v1/`，里面有 61 个 wav、
`clips.tsv`（id、句池、时长、文本）、`SHA256SUMS` 和它自己的 `README.md`。给别人时打成一个包：

    tar czf x2_mirror_clips_v1.tar.gz x2_mirror_clips_v1

拷到板上（约 7 MB）：

    scp -r x2_mirror_clips_v1 root@<板子>:<WORKSPACE>/assets/audio/

## 3. 核对

    cd <WORKSPACE>/assets/audio/x2_mirror_clips_v1 && sha256sum -c SHA256SUMS

应该 61 个都是 OK。启动后看 mirror_voice 的日志，应该打出「用 <目录>，找到 61 / 61」。

- 缺了某个 wav 或格式不对：只有那一句不说，同一事件有别的句子就换一句；**不会改成合成**。
- 一个 wav 都没有：ERROR，镜像照常但没声音。
- 模型目录的 sha256 清单在阶段 0 从 .5 上只读生成后补在这里。

## 4. 配置键

都在 `robot.internvl_voice` 下，默认值见 `aimdk_x2_voice_mirror_all.yaml` 末尾的块：

| 键 | 含义 |
|---|---|
| `enabled` | 开关，见第 5 节 |
| `clips_table` | 句表路径，支持 `$(find robot_config)` |
| `clip_dir` | wav 目录，相对 `$WORKSPACE` |
| `asr` | `mute`（默认：播前停 ASR，播完 `asr_resume_delay_sec` 秒后再开）、`off`、`keep` |
| `asr_resume_delay_sec`、`asr_control_topic` | mute 的延迟和 `/voice_control` 话题 |
| `standby.first_after_sec`、`standby.gap_sec` | 待机词：ready 播完后第一个的等待区间、之后每个的间隔区间（秒） |
| `look.*` | InternVL 看图：`enabled`、`demo_root`、节奏（`round_period_sec`、`window_sec`、`first_after_sec`、`gap_sec`、`max_per_round` 等）、`nice`、`crop_actor`、`refine_hand`、`verify`、`dump_dir`。默认 `round_period_sec: 0`（上一次看完马上看下一次）、`window_sec: 0`（整个 MIRRORING 都看）；嫌负载高就调大这两个 |

某一项写错（类型不对、超范围、未知键）时，只有这一项打 WARNING 并用默认值，其它照常，
不会退回合成，也不会整体关掉。`clips_table` 读不到或不合法时，改用随包安装的句表。

## 5. 怎么关

| 想要 | 改法 |
|---|---|
| 整个特性关掉，回到基线（四句过渡语照旧由 ZipVoice 合成） | 删掉 `internvl_voice` 块，或写 `enabled: false` |
| 只播 wav，不看图（不加载 InternVL） | `look.enabled: false` |
| 看图但不追问 / 不确认 | `look.refine_hand: false` / `look.verify: false` |

注意：只有 `enabled: false` 才算关；`enabled` 写成别的值（比如 `ture`）会打 WARNING 并按开处理。
`visual_interaction.enabled` 也为 true 时，不启动本特性的看图节点。
