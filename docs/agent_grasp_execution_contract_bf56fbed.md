# bf56fbed 抓取与 Agent 契约实施计划

基线：`bf56fbede40f667f2f934c409f3a70527576a269`。
状态：代码已实现，软件验收见 [Mock 验证说明](skill_execution_contract_mock.md)。
真机进展见 [2026-11-04 验证记录](skill_execution_contract_live_20261104.md)。

以 [outcome_message.py](../src/manipulation_execution/manipulation_execution/grasp_pipeline/outcome_message.py) 为业务语义参考。
保留已有错误解释，补充结构化事实和交互建议。
技能网关（Gateway）统一保存并转发结果。
Agent 生成中文反馈，语音合成（TTS）负责播报。

通用层统一执行详情，抓取侧定义阶段、原因和恢复条件。
后续长程技能复用同一结构，提供各自的业务定义。
本轮聚焦抓取，入口沿用 `ibrobot_agent`。
最终抓取成功须包含抬升后的持握验证。

## 1 保留什么

- 保留现有错误码及 `FAILURES` 的原因解释。
- 保留管线内部的有界重试和执行前预算检查。
- 保留执行授权、身份校验、取消确认和未知状态隔离。
- 保留现有执行入口、事件、查询与记录机制。
- 保留详细报告，存放底层异常和重试历史。
- 保留中文目标转换规则，苹果仍映射为 `an apple`。

`outcome_message.py` 继续作为结果摘要生成器。
抓取侧负责内部重试，Agent 在终态失败后等待新指令。
本轮不扩展照镜子、容器放置和断点恢复。
临时自动放回参数不进入通用契约。

## 2 增加什么

### 2.1 一份通用执行详情

新增 `ibrobot_msgs/msg/ExecutionDetail.msg`。
现有 `state`、`error_code`、`message` 继续使用。

| 新增字段 | 含义 |
| --- | --- |
| `schema_version` | 本版为 `1`。`0` 表示没有结构化详情。 |
| `phase` | 当前或最后执行的阶段。阶段名称由技能定义。 |
| `phase_state` | `RUNNING / SUCCEEDED / FAILED / CANCELED / UNKNOWN`。 |
| `failed_phase` | 当前失败链的原始失败阶段，恢复动作不覆盖它。 |
| `reason_code` | 细分原因，如 `PLAN_TIMEOUT`、`OUT_OF_REACH`。 |
| `recovery_state` | `NOT_NEEDED / IN_PROGRESS / COMPLETED / FAILED / UNKNOWN`。 |
| `retry_count` | 已经开始的内部重试次数，不等于实际抓取次数。 |
| `suggested_action` | `NONE / WAIT_SERVICE / WAIT_USER / MANUAL_CHECK`。 |

交互建议不授予动作权限。
恢复状态不等于当前姿态，也不代表下一次抓取已经获准。
新尝试重置当前失败链，历史保留在事件和报告中。
最终成功时清空失败字段。

### 2.2 抓取事实和原因传递

| 修改位置 | 增加的内容 |
| --- | --- |
| `outcome_message.py` | 将 `FAILURES` 改为结构化规则表。保留摘要，新增交互建议枚举。 |
| `grasp_pipeline/contracts.py`、`grasp_request.py` | 从故障源提供细分原因，区分超时、不可达和规划器故障。 |
| `pipeline.py`、`sequence.py` | 记录失败阶段、恢复结果和实际开始的重试。按执行模式判定成功。 |
| `grasp_pipeline_node.py` | 输出阶段开始、成功、失败、取消，以及完整终态详情。 |
| `PlanGraspCandidates.action`、`grasp_head.py`、`remote_head.py` | 新增并传递 `reason_code`，保留 Orin 到 310P 的规划原因。 |

交互建议结合错误规则和真实恢复状态生成。
恢复失败或状态未知时，使用 `MANUAL_CHECK`。
阶段开始不算阶段完成，重试决定不算重试已经执行。
只观测、只规划成功时，禁止声称已抓起物体。

### 2.3 Gateway 到 Agent 和 TTS 的完整链路

给 `PickObject`、`SkillCommand`、`ExecuteProgram` 的结果和反馈增加 `execution_detail`。
`ExecutionEvent` 和 `GetExecution` 使用同一结构。
Gateway 同步保存事件快照和当前详情，补充存储迁移。
事件、结果和查询携带 `current_skill` 与 `workflow_step_index`。
步骤编号从 `0` 开始，无步骤时技能名为空。

`PickObject` 反馈另带 `sequence`，结果带 `last_feedback_sequence`。
结果附带最近 256 条 `execution_history`，使用同一种强类型详情。
Gateway 使用有界队列接收阶段信息，再用结果中的历史补齐未到达的反馈。
超出窗口时记录缺口，已确认的执行结果保持不变。该流程没有额外等待。
`ExecuteProgram` 结果带 `last_event_sequence`，与查询和事件序号一致。

公共校验和转换函数集中放在 `embodied_common`。
完整 `message` 限制为 300 个字符。
Agent 对缺失或未知版本的详情使用通用反馈。
版本 `1` 必须包含全部 8 个字段；缺字段时同样降级。
展示字段异常不能改写已经确认的执行终态。

Agent 保留详情，按原因和交互建议生成中文文案。
例如，目标未检测到时提示摆放苹果，规划超时时提示检查服务。
恢复失败或执行状态未知时，提示操作员检查。
Agent 使用结构化建议，不解析英文 `Next` 来决定重试。

在现有语音桥接中增加文本请求的反馈消费入口。
入口保留渠道、身份和机器人范围校验。
Agent 转发 Gateway 已入库的进度与终态，播报按执行标识和序号去重。
取消流程通过查询取得终态时，同样保留 Gateway 序号和详情。
终态到达后丢弃旧阶段播报，语音故障不影响执行结果。
启用 X2 配置中的 TTS，按需启用反馈消费入口。

## 3 删除或替换什么

| 现有内容 | 删除或替换方式 |
| --- | --- |
| `message` 中拼接 `State / Retries / Next / Detail` | 删除多段拼接，改为短摘要。事实走结构化字段，长内容留在报告。 |
| `FAILURES` 中英文 `Next` 句子承担决策含义 | 替换为交互建议枚举。Agent 根据原因和建议生成中文表达。 |
| `_state()` 根据 `at_rest` 或报告有无推断状态 | 删除无证据的回位、停止和持握结论。摘要使用执行器确认的事实。 |
| `_success_summary()` 对所有成功都声称已抓取 | 改为区分观测、规划和执行模式。 |
| 把重试记录直接写成 `grasped again` | 删除已完成动作的推断，按实际开始和完成状态描述。 |
| Gateway 的空反馈处理及详情丢失 | 替换空实现，补齐阶段转发、成功结果和查询详情。 |
| Agent 因消息超长进入取消查询或未知状态 | 格式问题在转换层处理。真实执行异常继续沿用取消查询机制。 |
| TTS 固定播报“执行失败，详情见日志” | 已知原因使用具体中文反馈，未知原因保留通用提示。 |

## 4 实施顺序与验收

1. 修复摘要长度、成功模式和状态误报。
2. 增加公共消息与统一转换函数。
3. 补齐抓取事实和两端规划原因。
4. 接通 Gateway 记录、事件和查询。
5. 接入 Agent 中文反馈与文本请求播报。

接口升级时，同步构建 Orin、310P 及其他通信节点。
旧二进制的通信兼容性不能依赖新字段默认值。
正式软件验收使用 `colcon test`。

重点验收 6 组行为：

- 超过 300 字符的消息不会改变已知终态。
- 只观测、只规划不会播报抓取成功。
- 未看到目标、不可达和规划超时有不同反馈。
- 重试与恢复信息准确，取消后不开始新尝试。
- 失败原因经过远程接口和 Gateway 后仍完整。
- 文本请求能播报，重复事件和 TTS 故障不触发重复抓取。

已替换原摘要格式测试。新测试保留错误解释覆盖，增加长度、模式和事实约束。
临时放回流程保持现有配置。Mock 不加载真机驱动，也不发送硬件指令。
