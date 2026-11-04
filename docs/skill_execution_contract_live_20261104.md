# 2026-11-04 抓取执行契约真机验证

状态：反馈链路、内部重试和完整抓取均已验证。
测试平台：灵犀 X2、310P 和 Orin。
时间采用北京时间。
软件验收见 [Mock 验证说明](skill_execution_contract_mock.md)。

## 验证范围

本轮复用机器上的虚拟环境，没有重装依赖。
310P 运行 Agent、技能网关（Gateway）、感知服务和抓取管线。
Orin 运行相机和抓取规划节点。
测试入口为 Agent 文本请求，目标为 `an apple`。
Agent 通过 Gateway 提交 `pick_object`。

## 文本请求与反馈

2026-11-04 13:09:42，测试程序提交一次“帮我抓取苹果”。
请求标识为 `contract-live-20261104-codex-001`。
执行标识为 `bc490c03-dac3-4926-b23c-4fd88f08c9b4`。

| 检查项 | 实测结果 |
| --- | --- |
| 请求次数 | 1 次，没有自动重发 |
| Agent 计划 | 单步 `pick_object(target_name="an apple")` |
| 观察与感知 | 完成，生成 400 个抓取候选 |
| 规划 | 失败，`INTERNAL_ERROR` |
| 动作阶段 | 未进入机械臂接近、闭合或抬升阶段 |
| Gateway 事件 | 序号 1 至 11 连续 |
| 查询 | `GetExecution` 与事件终态一致 |
| 执行详情 | `failed_phase=plan`，`retry_count=0` |
| 恢复与建议 | `NOT_NEEDED`，`MANUAL_CHECK` |
| Agent 终态 | `FAILED`，没有误报抓取成功 |

语音合成（TTS）播报了两段反馈：

- “正在观察目标。”
- “抓取在规划阶段失败。本次任务未完成，请操作员检查机器人状态。”

音频输出共 40 个分块，248492 字节。
用户回复“都听到了”，确认两段反馈均可听见。
终态到达后，语音桥接丢弃了尚未播放的旧阶段反馈。

## 部署错误与修复

抓取结果保留了底层错误：

```text
TypeError: GraspFailure.__init__() got an unexpected keyword argument 'reason_code'
```

Orin 加载了本次部署的 `aimdk_grasp`。
但 `GraspFailure` 来自旧部署的 `manipulation_execution`。
旧构造函数没有 `reason_code` 参数。
这种混合加载遮住了规划器原本的错误。

本轮停止 Orin 抓取规划节点后，补建了两个包：

- `embodied_common`
- `manipulation_execution`

构建使用 `./scripts/build.sh`，没有修改控制参数。
补建后，三个模块均解析到本次部署目录：

- `aimdk_grasp.grasp_request`
- `manipulation_execution.grasp_pipeline.contracts`
- `embodied_common.execution_detail`

`GraspFailure` 的 `reason_code` 构造检查通过。
Orin 上的正式定向测试使用 `colcon test`。
`test_grasp_request.py` 和 `test_remote_head.py` 共 38 项通过。
测试结果为 0 失败、0 错误、0 跳过。
Orin 抓取规划节点已重新启动并完成预热。

接口部署必须同步检查消息定义和 Python 包来源。
只构建 `aimdk_grasp` 与消息包，不能保证依赖版本一致。

## 修复后的规划检查

检查程序只调用观察、感知和规划接口。
程序没有创建机械臂执行器，也没有提交 `PickObject` 执行请求。

首次检查返回 `NO_EXECUTABLE_CANDIDATE` / `PLAN_TIMEOUT`。
错误信息为 `TimeoutError: Planning deadline exceeded`。
Orin 到 310P 的 `reason_code` 传递正常。
场景报告记录了网格容量不足，节点日志记录了规划器重建。
首次检查耗时 36.03 s，包含观察、感知和规划。

第二次检查耗时 2.84 s，场景加载成功。
检查返回 `NO_EXECUTABLE_CANDIDATE` / `NO_COMPLETE_PATH`。
错误信息如下：

```text
RuntimeError: No candidate produced a complete three-segment plan
```

两次检查均识别到苹果并生成 400 个候选。
第二次检查仍未得到完整的接近、抓取和抬升路径。
离线复核显示，第二组候选有 59 个通过粗略可达性筛选。
这项筛选不保证完整轨迹可行，不能把结果归为全部不可达。
本轮保留原有碰撞检查、候选筛选和超时参数。
这两次检查验证了规划接口，没有再次提交 Agent 抓取请求。

## 再次授权后的文本验证

2026-11-04 13:29:31，经用户再次授权，提交一次“帮我抓取苹果”。
请求标识为 `contract-live-20261104-codex-002`。
执行标识为 `eb1b577f-8732-40b8-a7f7-ed5ddea0b732`。

执行前确认 Gateway 空闲、运动已授权，`pick_object.ready=true`。
310P 的 53 个部署文件通过校验，契约模块解析到本次工作树。
Orin 的 `GraspFailure` 支持 `reason_code`，规划消息也包含该字段。
现场只有本次 310P 管线和一个 Orin 抓取规划节点，无需停止其他管线。

| 检查项 | 本次结果 |
| --- | --- |
| Agent 调用 | 仅提交一次，计划为 `pick_object(target_name="an apple")` |
| Skill 内部尝试 | 3 次观察、感知与规划，`retry_count=2` |
| 终态与错误 | `FAILED`，`NO_EXECUTABLE_CANDIDATE` / `NO_COMPLETE_PATH` |
| 失败阶段 | `phase=plan`，`failed_phase=plan`，`phase_state=FAILED` |
| 恢复与建议 | `recovery_state=NOT_NEEDED`，`suggested_action=WAIT_USER` |
| 动作阶段 | 未进入接近、闭合、抬升或放回阶段 |
| 事件与查询 | 23 条连续 Gateway 事件；Agent、查询和末条事件的执行详情一致 |
| 耗时 | 文本提交至终态 9.767 s，Skill 执行 6.573 s |

终态反馈为“没有找到可执行的抓取路径，请调整物体位置后重新下达指令”。
日志确认 TTS 消费了该反馈；音频接口累计输出 46 个分块、283990 字节。
本次现场听感仍待用户确认，不能用音频分块计数代替听感确认。
Agent 没有在 Skill 终态失败后重新提交任务。
本轮再次验证了失败契约和有界内部重试，尚未完成抓取动作验收。
当前结果只能证明未找到完整路径，不能据此认定所有候选均超出机械臂可达范围。

## 重新摆放苹果后的抓取验证

用户重新摆放苹果后，要求再次尝试抓取。
2026-11-04 14:30:33，测试程序提交一次文本请求。
请求标识为 `contract-live-20261104-codex-003`。
执行标识为 `28ea2718-0617-4606-8673-99311e08f9d6`。

执行前确认 Gateway 空闲，`pick_object.ready=true`。
计划为左手抓取苹果、抬升验证、放回并退回起始姿态。
本次保留原有配置，没有降低碰撞检查或持握验证要求。

| 检查项 | 本次结果 |
| --- | --- |
| Agent 调用 | 单次 `pick_object(target_name="an apple")` |
| 观察、感知与规划 | 全部成功 |
| 接近、抓取、闭合与抬升 | 全部成功 |
| 抬升后的持握验证 | 成功 |
| 放回、张手与退回 | 成功，最终阶段为 `home` |
| 重试 | `retry_count=0` |
| 终态 | Agent 与 Gateway 均为 `SUCCEEDED` |
| 完成步骤 | `completed_step_count=1` |
| 执行详情 | `failed_phase` 与 `reason_code` 均为空 |
| 恢复与建议 | `NOT_NEEDED`，`NONE` |
| Gateway 事件 | 序号 1 至 38 连续 |
| 查询 | `GetExecution`、事件历史与 Agent 终态一致 |
| 耗时 | 文本提交至 Agent 终态为 35.30 s |

Gateway 确认物体已抓起、抬升并通过验证，放回流程已完成。
Agent 终态反馈为“物体已经抓起并通过验证”。
TTS 日志确认消费了成功反馈。
本次音频接口输出 39 个分块、234156 字节，现场听感未单独确认。
本次验证了完整动作流程，不能据此反推前两次失败的具体几何原因。

## 结束时的服务状态

310P 服务为 `active`，Orin 保留一个抓取规划节点。
Gateway 返回 `busy=false`，活动任务标识为空。
`control_plane_ready=true`，`pick_object.ready=true`。
服务就绪不代表当前场景一定能完成抓取。

## 证据位置

本地原始记录保存在 `log/contract_live_20261104/`。
此目录由 Git 忽略，不随代码提交。

| 文件 | 内容 |
| --- | --- |
| `contract-live-20261104-codex-001/events.jsonl` | Agent、Gateway、查询与音频计数 |
| `contract-live-20261104-codex-001/pick_result.json` | 抓取原始结果与底层错误 |
| `contract-live-20261104-codex-002/events.jsonl` | 再次授权后的请求、23 条事件、查询与音频计数 |
| `contract-live-20261104-codex-002/pipeline_journal.log` | 本次内部重试、终态和 TTS 日志 |
| `contract-live-20261104-codex-002/final_gateway_status.json` | 本次结束后的空闲与就绪状态 |
| `contract-live-20261104-codex-003/events.jsonl` | 重新摆放后的请求、完整动作阶段与成功终态 |
| `contract-live-20261104-codex-003/pipeline_journal.log` | 完整抓取与 TTS 日志 |
| `contract-live-20261104-codex-003/final_gateway_status.json` | 成功后的空闲与就绪状态 |
| `x2_contract_live_20261104_probe.py` | 单次请求与被动取证程序 |
| `x2_contract_plan_diagnostic.py` | 不执行动作的规划检查程序 |
| `plan-diagnostic-codex-001/result.json` | 修复后首次规划检查结果 |
| `plan-diagnostic-codex-002/result.json` | 再次规划检查结果 |
| `plan-diagnostic-codex-*/observation.npz` | 相机图像、目标点云和抓取候选 |
| `plan-diagnostic-codex-*/reach_analysis.json` | 已保存候选的离线可达性筛选 |
| `orin-evidence/` | 构建、测试、模块来源与节点日志 |
| `final_gateway_status.json` | 结束时的 Gateway 状态 |

310P 原始记录位于 `/root/zf/contract_live_20261104/`。
Orin 构建、测试和节点日志位于 `/home/agi/ibrobot_contract_20261104/`。

第三次文本请求完成了接近、闭合、抬升、持握验证与放回。
成功结论来自完整执行终态及阶段记录。

## Rebase 后的左手苹果与橘子验证

2026-11-04 15:31 至 15:32，完成两步真机抓取验证。
代码基线为 `edf611fea82229f1716ea452a6583433b4fc82b6`，包含工作树中的契约改动。
310P 和 Orin 的 100 个部署文件均通过哈希校验。
消息定义与执行模块均从本次部署目录加载。
两端构建通过。

自动选臂启动检查发现右手没有张开。
用户要求先测试左手，因此两端临时固定 `arm_side=left`。
本轮保持 `watch=stop`，成功后放回物体。
机器人保留右手的实测手型。
本轮没有验收右手抓取或自动选臂。

测试通过 `robot-skill run-workflow` 提交一次两步任务。
计划依次执行 `pick_object(target_name="an apple")` 和 `pick_object(target_name="an orange")`。
请求标识为 `contract-rebased-left-fruits-20261104-001`。
执行标识为 `c516f9b4-6035-4cfa-965e-30f0ae513d87`。
本次入口直接经过 Gateway，没有测试自然语言 Agent 与 TTS。

| 检查项 | 苹果 | 橘子 |
| --- | --- | --- |
| 执行机械臂 | 左臂 | 左臂 |
| 抓取结果 | 成功 | 成功 |
| 抬升与持握验证 | 通过 | 通过 |
| 放回与退回 | 完成，最终为 `home` | 完成，最终为 `home` |
| 内部重试次数 | 0 | 0 |
| Skill 耗时 | 29.94 s | 36.88 s |
| 结束时间 | 15:31:31 | 15:32:08 |

CLI 与 Gateway 查询均返回 `SUCCEEDED`，完成步骤数为 2。
查询中的目标顺序为苹果、橘子。
72 条 Gateway 事件的序号连续。
查询与末条事件的执行详情一致，失败原因为空。
两次抓取的视觉监控均没有报告目标移动、监控故障或停止。
最终 Gateway 返回 `busy=false`，活动任务标识为空。

Orin 定向测试首轮为 62 项通过、1 项 RPC 超时。
超时用例为 `test_nothing_is_judged_before_the_approach_or_once_the_close_started`。
该用例单独通过正式 `colcon test` 复验。
本轮保留两次测试结果，没有调整监控超时参数。

本地证据位于 `log/contract_rebased_live_20261104/`。
`workflow_cli.jsonl` 保存计划与终态，`pick_results.json` 保存两个抓取结果。
`execution_query.json` 和 `execution_history.json` 保存查询与事件历史。
`pipeline_journal.log` 保存运行日志，`orin/` 保存板端校验与测试结果。
310P 的 `x2-contract-left-20261104.service` 保持运行。
Orin 保留本次左手规划节点，入口为 `run_left_head.sh`。

### 执行契约核查

执行契约已参与本次抓取的阶段反馈、Gateway 存储和终态查询。
核查使用本次保存的运行证据，没有追加机器人动作。
8 个接口与实现文件的哈希符合部署清单。
核查结果保存在 `log/contract_rebased_live_20261104/contract_audit.json`。

| 核查项 | 运行证据 |
| --- | --- |
| 强类型详情 | 67 条事件携带完整的版本 1 `ExecutionDetail`，字段校验通过 |
| 阶段反馈 | 每步有 32 条反馈，分别记录 16 个阶段的开始和完成 |
| 步骤上下文 | 苹果为 `workflow_step_index=0`，橘子为 `1`，技能均为 `pick_object` |
| 顺序与完整性 | 72 条事件的序号连续，阶段顺序符合底层完成记录 |
| 终态一致性 | CLI、查询和末条事件均为 `SUCCEEDED`，查询与末条事件的详情一致 |
| 成功事实 | 两次均确认持握、放回和退回，终态原因字段为空，重试次数为 0 |
| 摘要长度 | 全部事件与查询的 `message` 均不超过 300 字符 |

5 条版本 0 详情分别对应任务接收、提交、激活和两个步骤的初始化。
这些记录出现时，抓取阶段尚未反馈。
每步第一条阶段反馈起，详情均为版本 1。

核查还发现两项展示语义需要区分：

- `hold_lifted` 对外映射为 `verify`，最终持握检查也使用 `verify`。
  因此每步有两组 `verify` 事件，底层记录保留两个阶段的原名。
- 52 条动作阶段反馈的 `recovery_state` 为 `UNKNOWN`。
  代码在动作序列返回前保留此值，任务状态仍为 `ACTIVE`。
  步骤完成后，`recovery_state` 变为 `NOT_NEEDED`。
  消费端必须区分恢复字段与任务终态。

本轮证明成功路径的契约传递与存储生效。
失败原因、恢复、内部重试、取消和未知终态没有触发。
反馈补齐逻辑存在于代码中，本轮记录不能证明发生过丢反馈后的补齐。
自然语言 Agent 与 TTS 的请求链路也未覆盖。
