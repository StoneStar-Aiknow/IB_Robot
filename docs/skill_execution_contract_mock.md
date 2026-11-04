# 抓取执行契约与 Mock 验证

当前基线为 `upstream/oee_summit_2026` 的 `edf611fea82229f1716ea452a6583433b4fc82b6`。
本次使用独立 worktree 和独立构建目录，复用主仓库的 Python 依赖。

## 1 契约边界

公共接口使用 `ExecutionDetail` 的 8 个字段。
定义、枚举及兼容规则见 [实施计划](agent_grasp_execution_contract_bf56fbed.md)。
`state` 表示整个任务的状态，`phase_state` 表示当前阶段的状态。
恢复成功时，整个抓取仍可能失败。新尝试重置当前失败链，历史事件保留原原因。

| 模块 | 职责 |
| --- | --- |
| Skill | 输出阶段、失败原因、恢复结果和已经开始的重试次数；内部重试有次数和预算上限。 |
| Gateway | 校验身份与授权，保存当前详情和事件快照，提供结果与查询；未知状态隔离后续任务。 |
| Agent | 保留权威详情，按原因生成中文反馈。失败后等待新用户指令，不自动重新提交整个 Skill。 |
| TTS | 消费指定渠道、身份和机器人范围的反馈；按执行 ID 和序号去重，终态清理待播阶段。 |

`suggested_action` 仅用于交互提示，不授予动作权限。
`message` 整体最多 300 字符；底层异常和重试记录保留在抓取报告中。
版本 `0`、未知版本或无效展示详情使用通用文案，不改写已知终态。
版本 `1` 缺字段时也会降级，避免把不完整的详情播报为“抓取已验证”。

新的长程 Skill 需要对齐自己的阶段表、原因表、成功条件和恢复证据。
它可复用通用接口、存储、查询和去重机制。公共接口没有苹果、容器或放回字段。

## 2 文本到执行的验证链

测试向 `/agent/request` 发布文本 `帮我抓取苹果`，使用实际运行节点：

`ibrobot_agent → ExecuteProgram → skill_executor → PickObject → grasp_pipeline`

Agent 使用真实模型客户端访问本地 HTTP Mock，获得包含 `pick_object` 的结构化计划。
Gateway 使用真实目录、授权、模型身份校验和 SQLite 记录。
抓取管线使用 Mock 的观测、规划、运动和手爪后端。
TTS 桥接调用 Mock 的合成和播放服务，不使用音箱。
每个测试进程由根 `conftest.py` 分配独立 DDS 域，并限制为 localhost。

测试同时检查 Agent 终态、Gateway 查询、历史事件和播报文本。
模型调用次数和抓取次数用于验证没有重复提交。
失败注入仅在测试后端中存在，不进入生产执行配置。

## 3 覆盖范围

| 场景 | 断言 |
| --- | --- |
| 成功与多步骤 | 文本触发实际调用链；结果标明当前 Skill 和从 0 开始的步骤编号。 |
| 观测、识别、规划失败 | 阶段与细分原因保留；未看到目标、不可达、规划超时有不同中文提示。 |
| 服务异常和内部异常 | RPC 失败、超时、未就绪、服务不可用、意外异常保持不同原因。 |
| 运动与持握失败 | 运动失败、手爪失败、监视失败、空抓、滑落和持握不确定有结构化结果。 |
| 重试与恢复 | 重试次数只统计已开始的尝试；恢复完成保留原始失败，最终成功清空失败字段。 |
| 恢复失败 | 保留原始失败阶段，提示人工检查，不重新抓取。 |
| 取消与超时 | 确认底层停止结果并保留详情；停止结果未知时隔离新请求。 |
| 查询收敛 | Action 通道异常时读取已保存的终态；保留事件序号，取消与执行成功的竞态不误报为未知。 |
| 准入异常 | 执行中返回忙；服务不可用或拒绝目标时，管线不开始执行。 |
| 幂等与回放 | 相同请求 ID 不重复执行；旧数据库能迁移，历史详情不被后续步骤覆盖。 |
| 展示容错 | 长消息不改变已知终态；旧版本、重复事件、乱序事件和终态后的进度能降级处理。 |
| TTS 异常 | 合成失败和播报阻塞不改变动作结果，不触发重复抓取。 |
| 远端规划 | 细分原因经过 `PlanGraspCandidates` 的服务端和客户端后保留。 |

分类映射测试遍历 `outcome_message.FAILURES`。
端到端测试使用故障注入验证当前可枚举的业务错误与执行异常。
它不证明真实模型的识别率、真机抓取成功率或物理停止效果。

## 4 复现

在本 worktree 根目录执行。接口升级需要先重新构建通信节点。

```bash
source .shrc_local
./scripts/build.sh -- --packages-up-to aimdk_grasp ibrobot_agent skill_library embodied_bringup --parallel-workers 4
unset ROS_DOMAIN_ID
colcon test --merge-install --packages-select embodied_bringup --return-code-on-test-failure \
  --pytest-args -q test/test_grasp_execution_contract.py
colcon test-result --test-result-base build/embodied_bringup --verbose
```

单独观察一次文本抓取链路，可运行定向诊断：

```bash
source .shrc_local
unset ROS_DOMAIN_ID
python3 -m pytest -s -q src/agent/embodied_bringup/test/test_grasp_execution_contract.py::test_text_reaches_real_skill_and_success_is_spoken
```

只验证指定错误时，可选择对应场景。每个场景仍从文本请求开始：

```bash
source .shrc_local
unset ROS_DOMAIN_ID
colcon test --merge-install --packages-select embodied_bringup --return-code-on-test-failure \
  --pytest-args -q test/test_grasp_execution_contract.py -k 'PLAN_TIMEOUT or EMPTY_GRASP or unknown_stop'
```

测试使用工作树自己的 `venv`、`build` 和 `install`。
本机补装了仓库已固定版本的 `warp-lang==1.11.1`，未更改主仓库依赖。
模型身份测试使用元数据 fixture，不下载权重，不调用真实推理设备。

X2 文本 Agent 配置已启用 TTS 和 `voice_bridge.feedback_only`。
该入口只消费反馈，不接收语音命令，也不发送执行确认。
真机部署仍需已有 ZipVoice 模型、音频输出绑定和原有动作授权。
本次不启动真机 pipeline。

## 5 验收记录

2026-11-03，在 `bf56fbede40f667f2f934c409f3a70527576a269` 基线完成构建和正式 `colcon test` 验收。
合计 **2585 项通过、54 项跳过，0 失败、0 错误**；另有 1 项明确排除。
各次正式验收进程均正常退出，没有超时退出。

| 包 | 范围 | 通过 | 跳过 |
| --- | --- | ---: | ---: |
| `embodied_common` | 全包，含公共契约与降级转换 | 320 | 0 |
| `ibrobot_agent` | 全包，含查询收敛、中文反馈与 TTS | 395 | 0 |
| `skill_library` | 全包，含存储迁移、终态不变和旧步骤隔离 | 337 | 0 |
| `robot_skill_cli` | 全包，含 ROS Bridge | 472 | 8 |
| `aimdk_grasp` | 全包，含远端规划原因传递 | 402 | 31 |
| `manipulation_execution` | 除下述照镜子资产测试外 | 511 | 2 |
| `embodied_bringup` | 新契约链与现有启动构建测试 | 98 | 6 |
| `robot_config` | X2 抓取、语音及音频绑定相关的 5 个测试文件 | 50 | 7 |

其中 `test_grasp_execution_contract.py` 的 **35 个文本端到端场景全部通过，无跳过**。
每个场景运行实际 Agent、Gateway 和抓取管线节点。
独立 DDS 域仅使用 localhost；运动、感知、模型回答和音频后端均为 Mock。

跳过项需要本机未提供的模型、CUDA、AimDK overlay 或独立 Hermes 环境。
明确排除的是 `test_stream_services_match_the_launched_model_service_endpoints`。
它属于照镜子配置测试，读取缺失的 `yolox_x_640` 模型目录，不属于本轮抓取契约范围。

额外尝试的完整 `robot_config` 回归未通过，涉及缺失模型、未构建导航包和 LeKiwi 启动失败。
长时间运行的仿真测试已中止，未计入通过结果。本次结论只覆盖上表范围。
端到端回归中出现过一次 Agent 启动就绪超时；测试夹具现使用独立的 30 秒启动预算。
动作执行时限保持原值，完整端到端复验已通过。

修改的 37 个 Python 文件通过 Ruff 检查与格式校验。
两份契约文档通过中文规则检查，`git diff --check` 通过。
`libs/lerobot` 的工作树差异来自官方环境初始化的补丁栈，不属于本次代码变更范围。
构建和测试不使用真实机械臂，不能替代真机验收。

复验本次范围时，先执行上面的构建步骤，再依次执行以下命令：

```bash
source .shrc_local
unset ROS_DOMAIN_ID
colcon test --merge-install --return-code-on-test-failure \
  --packages-select embodied_common ibrobot_agent skill_library robot_skill_cli aimdk_grasp manipulation_execution \
  --pytest-args -q -k 'not test_stream_services_match_the_launched_model_service_endpoints'
colcon test --merge-install --return-code-on-test-failure --packages-select embodied_bringup \
  --pytest-args -q test/test_grasp_execution_contract.py test/test_embodied_launch_builder.py
colcon test --merge-install --return-code-on-test-failure --packages-select robot_config \
  --pytest-args -q test/test_aimdk_x2_grasp_config.py test/test_aimdk_x2_voice_agent_config.py \
  test/test_voice_bridge_segment_inputs.py test/test_audio_interface_binding.py test/test_audio_io_launch_builder.py
```

## 6 Rebase 记录

2026-11-04，分支 `feat/skill-execution-contract` 升级到 `47f807726` 基线。
远端分支为 `upstream/oee_summit_2026`，新增 10 个提交。
原有 54 个改动文件均已恢复，仍保持未提交状态。
备份位于 `log/rebase_oee_20261104/`，包含文件归档、哈希清单和原始补丁。
Git 备份引用保留原始基线和 stash。

两处文本冲突均已解决：

- `grasp_pipeline_node.py` 同时保留上游终态日志和执行历史。
- `grasp_request.py` 按所选机械臂报告不可达，并保留 `OUT_OF_REACH` 原因。

上游新增的自动选臂逻辑原本丢失 `reason_code`。
现在，单臂失败或双臂原因相同时，系统保留具体原因。
双臂原因不同时，系统保留通用原因，并记录各臂原因。
预算耗尽时，系统报告 `PLAN_TIMEOUT`。
回归测试覆盖左右臂不可达、双臂原因汇总和预算耗尽。

本次构建使用 `./scripts/build.sh`，结果通过。
以下结果来自本次正式 `colcon test`：

| 包 | 范围 | 通过 | 跳过 | 错误 |
| --- | --- | ---: | ---: | ---: |
| `aimdk_grasp` | 全包 | 460 | 34 | 0 |
| `manipulation_execution` | 排除第 5 节指定的照镜子资产测试 | 513 | 2 | 0 |
| `robot_config` | 第 5 节指定的 5 个文件 | 50 | 7 | 0 |
| `embodied_bringup` | 契约链与启动构建测试，整组复验 | 97 | 6 | 1 |
| `embodied_bringup` | 启动超时用例单独复验 | 1 | 0 | 0 |

首轮端到端运行发生原生段错误。
错误日志包含 ROS 客户端创建处的调用栈。
测试没有生成完整结果。

整组复验没有再次发生段错误，但 `MOTION_FAILED` 场景启动超时。
测试等待 Agent 连接 Gateway，超过 30 秒启动预算。
同一用例单独通过正式复验，动作断言也通过。
端到端批量运行仍存在未定位的启动不稳定性。
两轮异常日志和单独复验结果均保存在备份目录。

本次修改的 5 个 Python 文件通过 Ruff 检查与格式校验。
工作树没有未解决冲突，`git diff --check` 通过。
本次验证使用 Mock 后端。
[真机记录](skill_execution_contract_live_20261104.md) 对应 rebase 前的版本。

2026-11-04，远端整理提交历史后，分支再次 rebase 到 `edf611fea`。
旧基线的 10 个提交合并为 4 个提交。
新旧基线的代码树相同，恢复后的 56 个改动文件也与备份一致。
本次仅更新本文的基线记录，复用第 6 节的构建与测试记录。
