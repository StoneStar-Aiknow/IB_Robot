# X2 照镜子接入与验证记录（2026-11-05）

本轮已在 310P 上验证真实相机 → YOLOX → PEAR → 双臂预览与实体机械臂跟随。
新增可选镜像配置，并修复镜像会话与手势、抓取共用 URS 时的持续占用检查。
现场操作者已确认动作效果正常。两轮实体测试均已通过 Gateway 停止并返回 idle。
第二轮末尾出现手部反馈超时并保护性退出，不能据此宣称稳定性验收完成。
随后按用户要求启动包含镜像、Agent、语音和抓取的全量配置，继续准备真人语音测试。

## 环境与模型

- 开发副本：`/mnt/data/lwh/IB_Robot_hz_deploy_20261104`。
- 310P：`10.42.0.251`，工作区 `/root/lwh/IB_Robot_hz_20261104`。
- 基线提交：`e07e7661`；本轮改动在其之后，尚未提交。
- 实测 DDS 域：真实相机用 0；正式定向测试分别用 98、99，与机器人域隔离。

以下模型已经存在，`ascend_310p` manifest 及按需文件摘要校验均通过，无需再次拷贝：

| Bundle | 310P 部署产物 |
| --- | --- |
| `models/yolox_x_640` | `artifacts/ascend_310p/yolox_x_640_bs1.om` |
| `models/pear_parameter_network` | `artifacts/ascend_310p/pear_parameter_network_bs1.om` |

这两个 bundle 是指向 `/root/lwh/models/` 下现有模型的目录链接。
双臂映射使用仓库内 `x2_followme` 及其骨架、关节限位资源，本轮未发现需要补充的模型。
本次范围为上半身双臂跟随，不包含手指模仿或视觉聊天。

## 实际启动与探测

本轮只启动两个模型服务和一个视觉流节点，没有启动物理运行时、镜像会话或 Orin 抓取头。
环境入口仍为 `commissioning/env.sh`，内部加载 `.shrc_local`，再设置现场 DDS 配置。
前台复现命令如下，两个终端均在 310P 工作区根目录执行：

```bash
export HZ_TARGET=robot HZ_ROS_DOMAIN_ID=0
source commissioning/env.sh
python3 /opt/ros/humble/bin/ros2 launch "$PWD/commissioning/mirror_vision.launch.py"
```

```bash
export HZ_TARGET=robot HZ_ROS_DOMAIN_ID=0
source commissioning/env.sh
timeout --signal=TERM --kill-after=10s 200s python3 commissioning/probe_mirror_vision.py
```

探测器只订阅 `/motion_imitation_stream/pear_frame` 与状态话题，
对 `source=yolox` 的有效人体帧计算双臂映射，并检查有限值与限位。
无人体时的 `center_fallback` 帧只计数，不作为照镜子成功证据。
探测器没有创建任何机器人指令发布器。

第一次探测收到 1688 帧中心裁剪结果，没有人体帧，结论为 INCOMPLETE。
随后相机曾识别到人体；第二次探测结果如下：

| 项目 | 实测结果 |
| --- | --- |
| 订阅姿态帧总数 | 535 |
| 有效人体帧 | 10 |
| 成功映射并通过限位检查 | 10 |
| 映射错误 | 0 |
| 探测器创建的机器人指令发布器 | 0 |
| 探测结论 | PASS |

相机分辨率为 2688×1944，输入约 30 Hz；稳定姿态输出约 9～10 Hz。
冷启动期间模型服务尚未就绪，产生过等待与失败计数，不能将其解释为全程零错误。
第二轮结束前的视觉状态累计识别到 104 帧人体。
探测结果中的 `status` 是最近一次周期状态，可能早于最后收到的人体帧。

## 代码改动

1. `commissioning/prepare_voice.py` 增加 `hz_site_mirror_preview` 和 `hz_site_mirror_live`。
   两者保留原有 7 个技能，并增加 `mirror_start`、`mirror_stop`；共声明 5 个感知服务。
   手势、镜像和抓取均映射到 `policy_stream`，避免在技能之间切换厂商控制模式。
   preview 关闭 Agent 执行与镜像物理发布；live 显式开启这两项。
2. `commissioning/site.sh` 支持 `mirror_preview`、`mirror_live`，310P 使用 `model_inference`。
   Orin 仍使用原抓取头配置，不承担镜像模型与会话。
   启动脚本仍传 `authorize_motion:=false`，选择 live 配置本身不会赋予动作执行权限。
3. Gateway 订阅镜像会话的可靠、瞬态本地状态心跳。
   其他技能只有在新鲜 WAIT、未占用、未释放中时才允许继续准入；
   STANDBY、MIRRORING、未知或过期状态会拒绝后续技能。
   因此同处 URS 不再等于可以同时向手臂发指令。
4. start 请求发送前同步锁存接管意图，避免服务应答先于状态回调时放行其他动作。
   超时保留锁存；观察到本轮占用、随后 WAIT 与运行时 idle，或 stop 成功并确认 idle 后才解除。
   start、stop 本身始终可进入会话执行路径；未确认启动时，stop 不再直接跳过会话 RPC。
5. 增加只运行视觉的 launch 和只读探测器，补充 20 项持续占用回归测试。
   修复已有视觉流 builder 测试缺少 `resolve_robot_config_path` 导入的问题。

## 检查结果

| 检查 | 结果 |
| --- | --- |
| 两个镜像模型的 manifest 与文件校验 | 通过 |
| 7 个生成配置、技能目录、镜像 launch builder 校验 | 通过 |
| Gateway 定向 `colcon test` | 65 通过，293 未选中 |
| 镜像 builder、视觉流、x2_followme、LivePlayer、限位定向 `colcon test` | 91 通过 |
| `./scripts/build.sh` 构建 `skill_library` | 通过 |
| 本轮 Python 文件 Ruff 检查与格式化、shell 语法检查 | 通过 |
| 启动脚本的 310P preview/live 与 Orin live 命令生成 | 通过，未启动物理运行时 |

Gateway 测试覆盖共享 URS 下阻挡手势、抓取和 primitive，
以及过期、未来、无效心跳、释放中状态、启动应答竞争、启动超时和停止后的准入。
正式结论来自 `colcon test`，进程正常结束，没有以 timeout 124 作为通过。
最初的镜像组件测试为 90 通过、1 失败，补齐上述导入后重跑为 91 通过。
首次单包构建受本部署未构建的 SO101/LeKiwi 包依赖发现影响，
按现场已有的选择性构建方式忽略本次不构建的源码包后，`skill_library` 构建成功。

310P 上的原始日志位于 `commissioning/logs/`：

- `mirror_vision_live_second_20261105.log`
- `mirror_vision_probe_second_20261105.log`、`mirror_vision_probe_20261105.json`
- `mirror_gateway_colcon_final_20261105.log`
- `mirror_colcon_fixed_20261105.log`
- `mirror_gateway_build_fixed_20261105.log`
- `mirror_preflight_final_20261105.log`

本地副本保存了关键结果到 `commissioning/evidence_mirror_20261105/`。

## 现场预览会话启动结果

随后为现场联调生成了 `hz_mirror_bench_preview`、`hz_mirror_bench_live`。
它们只开放 `mirror_start` / `mirror_stop`，关闭语音、抓取与录制手势执行器；
preview 明确关闭物理手臂和手指指令发布器，live 另行选择，不自动启用。

实际启动的是 **preview**：

```bash
export HZ_TARGET=robot HZ_ROS_DOMAIN_ID=0
source commissioning/env.sh
python3 commissioning/prepare_mirror_bench.py
python3 /opt/ros/humble/bin/ros2 launch embodied_bringup embodied_pipeline.launch.py \
  config_path:="$PWD/commissioning/robots/hz_mirror_bench_preview.yaml" \
  control_mode:=model_inference with_moveit:=false use_sim:=false authorize_motion:=true
```

这里的 `authorize_motion:=true` 用于允许 Gateway 调用两个镜像技能；
它不改变配置中的 `mirror_session.publish_enabled=false`。
运行时、镜像会话、重定向、模型服务、Safety Guard、Gateway 和常驻 CLI 桥均已启动。
Gateway 报 READY，两个技能 ready；`describe mirror_start`、`validate mirror_start` 通过。
会话启动检查报告 runtime already idle，视觉已识别到人体。

CLI 应使用本次 launch 生成的、包含实时接口描述的 `/tmp/ibrobot_interfaces_*/robot.yaml`，
路径以本次 `robot-skill-daemon` 启动参数为准，不复用其他 launch 的临时文件。
直接对未绑定接口的原始 YAML 调用 CLI 会得到 `description_required`。

通知现场操作者后，经 Gateway 执行：

```text
execute mirror_start --task-id mirror-preview-start-20261105-01
```

结果 `success=false`，`CONTROL_MODE_MISMATCH`，执行步数为 0：

```text
runtime rejected mode 'policy_stream': platform rejected UPPERBODY_REMOTE_SPLIT:
NO_TRANSITION_PATH (No transition path from PASSIVE_DEFAULT to UPPERBODY_REMOTE_SPLIT)
```

因此镜像会话没有接管、没有进入跟随，也没有发布实体手臂指令。
后续只读 Gateway status 显示无活动任务、busy=false。
只读会话探测结果为 INCOMPLETE：state=WAIT、holding=false、runtime_mode=idle、commands_sent=0。
DDS 节点发布端点也确认镜像会话只有预览、状态及 ROS 通用话题，没有实体手臂／手指命令发布器。
这不是缺少模型；当时需要现场通过厂商正常控制流程离开零力模式，准备到稳定站立待命或 URS。
当前 CLI 技能目录没有这项站立准备操作，本轮没有绕过 Gateway 调用厂商控制服务。
相关证据为 `mirror_bench_preview_20261105.log` 和 `mirror_bench_start_20261105.jsonl`。

## 进入 URS 后的预览与实体测试

操作者确认已进入 URS 后，重新校验并执行 `mirror_start`，成功接管 `policy_stream`。
状态依次为 WAIT → STANDBY/approach → STANDBY/ready → MIRRORING/running。
预览探测记录 30 次连续变化，无无效数据；发布端点仍只有预览与状态。
随后 `mirror_stop` 成功，厂商确认 STAND_DEFAULT，运行时确认 idle。

在停止确认、旧进程组退出后，启动 `hz_mirror_bench_live.yaml`。
入口与 preview 相同，仅选择 live YAML；该配置的 `publish_enabled=true`。
Gateway 状态 READY，目录仍仅有两个镜像技能，开始与停止校验均通过。

| 项目 | 第一轮实体测试 | 操作者返回后的第二轮 |
| --- | --- | --- |
| start 任务 | `mirror-live-start-20261105-01` | `mirror-live-start-20261105-02` |
| start 结果 | 成功，接管 policy_stream | 成功，接管 policy_stream |
| 20 s 探测内的手臂指令 | 981 | 822 |
| 探测内实际关节反馈 | 982 | 850 |
| 最大反馈变化范围 | 0.3553 rad | 1.0247 rad |
| 无效数值 | 0 | 0 |
| 收尾 | Gateway stop 成功，idle / STAND_DEFAULT | 故障保护退出后 Gateway stop 成功，idle / STAND_DEFAULT |

第一轮停止后的独立检查收到 152 帧关节反馈，未收到新的手臂指令，
WAIT、holding=false、runtime_mode=idle、releasing=false；
实际关节相对准备姿态的最大误差约 0.039 rad，小于 0.1 rad 的检查阈值。

第二轮末尾，运行时报告 `DEGRADED: joint feedback stale: hand`，
会话退出到 WAIT，关闭输出门；随后还出现过相机数据超时告警。
用户确认“已经看到正常了”，证明现场观察到正常跟随，但不代表连续运行稳定性已验收。
本轮没有放宽反馈新鲜度阈值，也没有屏蔽该保护。
第二轮的原始探测器仅按指令和反馈变化判定，输出了 PASS，判定过宽；
现已补充最终必须仍处于 MIRRORING、holding=true 且没有 wait_reason 才可 PASS。
保留原始证据，第二轮按“跟随已验证、发生保护中断”记录。

`mirror-live-stop-20261105-02` 最终成功：约 0.93 s 内确认 STAND_DEFAULT，返回 WAIT / idle。
随后停止 bench 流水线，切换全量配置。

实体证据位于 `commissioning/logs/`：`mirror_live_*20261105.jsonl`、
`mirror_live_first_probe_20261105.json`、`mirror_live_second_probe_20261105.json`、
`mirror_idle_probe_20261105.json` 和 `mirror_bench_live_20261105.log`。

## 后续现场验收

生成配置不会自动启动机器人。`site.sh plan` 可查看实际启动命令：

```bash
source commissioning/env.sh
bash commissioning/site.sh plan 310p mirror_preview
bash commissioning/site.sh plan 310p mirror_live
bash commissioning/site.sh plan orin mirror_live
```

后续仍需验证：

1. 人体丢失处置、连续运行稳定性及手部反馈超时的环境原因。
2. 全量流水线中镜像活动期间拒绝手势/抓取，停止并释放后再允许它们。
3. 真人 Agent 启停的长期重复性，以及同时有其他技能请求时的交互体验。

上述项目尚未完成，不能用单元测试或内存预览代替真机验收。
全量流水线已启动，5 个感知服务、TTS、ASR 与 Agent 同时常驻；未执行长时间资源稳定性测试。
两次单独视觉探测和 bench 进程组均已关闭。
全量配置使用 `hz_site_mirror_live.yaml`、`control_mode:=model_inference`、
`authorize_motion:=true`；Orin 抓取头继续使用 `hz_site_live.yaml`。
两侧均使用 ROS_DOMAIN_ID=0，PID 身份分别记录在 `commissioning/site_process_310p`、
`commissioning/site_process_orin`。

## 全量流水线就绪检查

310P 使用 `hz_site_mirror_live.yaml`，不是仅开放两个技能的 bench 配置。
Gateway 目录包含挥手、握手、举手、飞吻、鼓掌、自我介绍、抓取、开始照镜子、停止照镜子，
9 项均保留；`motion_authorized=true`、control_plane_state=READY。
Agent ready 服务通过，实时运行时为 ACTIVE / idle、stop_latched=false、faults=[]。
`site_ready.py --mirror` 同时确认两个镜像服务、三个抓取模型服务、Orin capture、TTS 和会话服务已发现。
服务发现仅代表端点存在，不代替每项模型推理或硬件前提的验收。

ASR 使用真实 `external_vad + audio.speech`，已捕获现场语音，Agent 已产生回复。
现场语音“做一个挥手的动作”被规划为 `wave_hand`，日志给出 exact plan、执行完成反馈；
当时镜像会话未占用手臂。
随后现场完成真人 Agent 启停，结果见下一节。
当前链路日志显示 `direct_execute_async_presentation`，会先播报计划再执行；
不要仅凭配置中的 `voice_confirm` 字段推断一定会停下来等待二次确认。

Orin 抓取头保持开启，但当前预检曾报告左手未张开（约 1.57 rad 误差），
也曾报告手部反馈与图像不同步。因此不能宣称抓取已经就绪或完成真机验收。
本轮未绕过该预检，也未为了镜像测试额外操纵手指准备抓取。
全量日志为 `site_310p_mirror_live_full_20261105.log` 和
`site_orin_mirror_live_full_20261105.log`，最终就绪快照为
`site_mirror_full_readiness_final_20261105.json`。

## 全量配置的真人 Agent 镜像启停：通过

现场实际说“开始照镜子”，ASR 发布该文本，Agent 请求 `voice-479195db322e`
规划为单步 `mirror_start`，计划／任务 ID 为 `f1179bdebd024f948a923e0129aae6e5`。
播报“正在执行：照镜子”后，Gateway 调用会话服务成功接管 policy_stream，
状态进入 STANDBY，随后满足人体条件进入 MIRRORING/running。

现场随后说“停止照镜子”，Agent 请求 `voice-e7642d08612a`
规划为单步 `mirror_stop`，计划／任务 ID 为 `84e1399d04264893801276557ac2b3b3`。
播报计划后，会话从 MIRRORING/running 进入 wrap_up，再到 WAIT。
服务返回 success=true，约 0.96 s 内确认 STAND_DEFAULT，运行时返回 idle，
收尾最后一条指令距准备姿态 0.00 deg，语音桥随后反馈“执行完成”。

该轮证明真实麦克风 → ASR → Agent 规划／计划呈现 → Gateway → 镜像会话 →
实体跟随 → 真人停止语音 → 收尾释放的链路已运行。
这是现场语音触发，未通过后台发布文本或 CLI 代替用户说话。
完整主日志包含相应请求、exact plan、会话状态转移与结果。
全量流水线保持运行，镜像在本轮结束后为 WAIT，供现场继续测试。
