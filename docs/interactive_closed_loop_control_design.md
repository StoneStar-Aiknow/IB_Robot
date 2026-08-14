# 交互式闭环控制设计（Catalog 查询 / Workflow 确认 / 别动停止 / 继续 continuation）

## 1. 背景与目标

在同一个 Hermes 进程内，以最小改动实现以下 5 个能力的闭环：

1. 查询当前 runtime Skill catalog，不创建计划或运动；
2. 拒绝 catalog 之外的动作；
3. 在同一个 Hermes 进程中创建、展示并自然语言确认 Workflow；
4. 执行期间输入「别动」，获得确定停止终态；
5. 输入「继续」，基于 fresh state 创建并确认全新 continuation。

目标是在不修改任何 runtime ROS 节点、不新增 ROS 接口、不违背现有硬边界的前提下，
提供一个可单测的参考闭环控制器，并把同一契约固化到 `ibrobot-control/SKILL.md`，
供真实 Hermes agent 与本控制器共用。

## 2. 现状映射（feature → 已有原语 → gap）

| Feature | 已有实现 | 文件 / 符号 | Gap |
|--------|---------|------------|-----|
| 1 查询 catalog | `robot-skill list-skills/describe/status`、`GetSkillSnapshot`、`capability_view_from_snapshot` | `robot_skill_cli/catalog.py:209`、`ros_bridge.py:316` | 无；仅需在闭环中按只读方式串联 |
| 2 拒绝 catalog 外动作 | `agent_plan_node._normalize_steps` 对每步 `skill_name` 校验 `planner_visible_names` 与 `semantic_level` | `embodied_agent/agent_plan_node.py:297` | 无（runtime 已强制）；闭环在 plan 前再做一次客户端预检，提早拒绝并给出可读原因 |
| 3 创建/展示/NL 确认 Workflow | `plan-workflow`→`validate-plan`→`confirm-plan`；`AgentPlanStore` 状态机 `PLANNED→VALIDATED→CONFIRMED` | `cli.py:341/365/372`、`embodied_agent/agent_plan_store.py:103` | 多为独立子进程；缺「同一进程内会话级 pending 绑定 + 封闭语法 NL 确认」的参考实现 |
| 4 「别动」→ 确定终态 | `cancel-plan` + 轮询 `get_agent_plan_result`，仅 `GoalStatus∈{4,5,6}` 视为确定终态；`SKILL_CANCEL_TIMEOUT`=未知 | `cli.py:553`、`ros_bridge.py:595/629` | 缺执行期可中断 + 确定终态门控的闭环封装 |
| 5 「继续」→ fresh state 全新 continuation | **完全不存在**；契约明确「反 continuation / 反自动重试」 | `SKILL.md:74/103` | 全新 gap：需在确定终态后，以新 `request_id`/`task_id`、重查 fresh catalog、重新 plan/confirm/execute |

结论：真正的新增 gap 是 **Feature 5（continuation）** 与把 1–5 串成可测闭环；
Feature 1/2 在 runtime 已实现，闭环侧只做只读串联与客户端预检。

## 3. 设计原则与硬边界合规

- **纯加法、不侵入 runtime**：只新增 `robot_skill_cli` 内的控制器与测试，不改 `skill_executor_node`/`agent_plan_node`/`AgentPlanStore`/`ibrobot_msgs`。
- **不解析自然语言到结构**：控制器只识别**封闭小词表**（确认/别动/继续），与 `SKILL.md` 既有的封闭确认语法一致；结构化 `workflow_steps` 仍由 Hermes（外部）产出，控制器只接收。
- **依赖注入 bridge**：控制器依赖一个 duck-typed `RosBridge`-like 对象，模块本身不 import `rclpy`/`ibrobot_msgs`（`capability_view_from_snapshot` 经 `view_resolver` 惰性注入），因此可无 ROS 单测（仿 `test_cancel_goal_waits_for_terminal_after_rejected_cancel_response` 的 `SimpleNamespace` 假对象范式）。
- **绝不自动重试**：「继续」是**新的用户请求**，不是重试；需要确定终态作为前置门，unknown 终态直接拒绝继续（合规 `SKILL.md:74/103`）。
- **不发明参数**：所有参数来自 `describe`/`workflow_steps`，不做 schema 外推断。
- **token 不外泄**：`plan_token`/`confirmation_token` 只存在会话内，不出现在 NL、日志、argv。

## 4. 闭环状态机

```
        discover()            prepare_workflow()        confirm()
IDLE ─────────────────► DISCOVERED ──────────────► PREPARED ──────────► CONFIRMED
                           │                            │                    │
                           │ reject_out_of_catalog()    │ expire/cancel       │ execute(stop_event=)
                           ▼ (SKILL_REFERENCE_MISSING)   ▼                     ▼
                          IDLE                         IDLE               EXECUTING
                                                                        ┌─────┴─────┐
                                                            stop()      │           │ terminal(GoalStatus∈{4,5,6})
                                                        ┌──────────────► STOPPING ──► STOPPED(terminal)
                                                        │                 │           │
                                                        │                 │ SKILL_CANCEL_TIMEOUT
                                                        │                 ▼           ▼
                                                        │             UNKNOWN ←──── 终态未知
                                                        │                 │  (拒绝 continue)
                                                        │     continue_workflow()  ✗
                                                        │ terminal
                                                        ▼
                                              SUCCEEDED / FAILED (terminal)
                                                        │
                                              continue_workflow()  ✓ (fresh state)
                                                        ▼
                                                    CONFIRMED（全新 plan）
```

关键门控：

- `confirm()` 只绑定会话内唯一未过期 pending（`_pending`），不接受外部 token。
- `stop()` 必须 wait 到 `get_agent_plan_result().status ∈ {4,5,6}` 才置 `STOPPED`；否则置 `UNKNOWN`。
- `continue_workflow()` 前置条件：`state in {STOPPED, SUCCEEDED, FAILED}`（确定终态）；`UNKNOWN` 时拒绝并返回 `SKILL_CANCEL_TIMEOUT`，不发新运动。

## 5. InteractiveController API

模块：`robot_skill_cli.interactive_control`

```python
class InteractiveController:
    def __init__(self, bridge, *, timeout_policy: dict, id_factory=uuid.uuid4,
                 monotonic=time.monotonic, sleep=time.sleep, view_resolver=None): ...
    # view_resolver 默认惰性调用 robot_skill_cli.catalog.capability_view_from_snapshot，
    # 使本模块 import 时不触发 ibrobot_msgs/rclpy，可在无 ROS 环境单测；生产中 discover() 时
    # bridge 已启动（rclpy + ibrobot_msgs 可用），惰性导入正常工作。

    # Feature 1：只读查询当前 runtime catalog
    def discover(self) -> dict
        # bridge.get_status() + bridge.get_skill_snapshot() + capability_view_from_snapshot()
        # 返回 {robot_name, registry_identity, capabilities, skills, capability_digest}

    # Feature 2：客户端预检，拒绝 catalog 外动作（与 runtime _normalize_steps 同语义）
    def reject_out_of_catalog(self, steps: list[dict], view: dict) -> None
        # 任意 step.skill_name 不在 planner_visible_names 或 semantic_level 不在
        # {atomic_operator, skill} → raise OutOfCatalogError("SKILL_REFERENCE_MISSING", ...)

    # Feature 3：创建 + 展示（会话内绑定 pending）
    def prepare_workflow(self, raw_command: str, steps: list[dict]) -> dict
        # reject_out_of_catalog → normalize_workflow_steps → bridge.plan_agent_command(...)
        # 缓存 _pending = {plan_token, plan_digest, steps, raw_command, registry_identity}
        # 返回展示体：{steps, plan_digest, registry_identity, task_id(预生成), confirm_command}

    # Feature 3：自然语言确认（封闭语法）
    def confirm(self, confirmation_text: str) -> dict
        # classify_confirm(confirmation_text) 命中封闭语法 → bridge.confirm_agent_plan(...)
        # 缓存 _confirmed = {confirmation_token, task_id, task_budget_sec}
        # 否则 raise NotConfirmedError

    # Feature 4：执行；执行期可经 stop_event 中断
    def execute(self, *, stop_event: threading.Event | None = None,
                feedback_callback=None) -> dict
        # wait_for_execute_plan_server → send_agent_plan_goal → 轮询 result_future
        # 每轮检查 stop_event：置位则 bridge.cancel_agent_plan + 轮询 get_agent_plan_result
        #   status∈{4,5,6} → state=STOPPED，缓存 _terminal；否则 state=UNKNOWN
        # 正常终态 → state=SUCCEEDED/FAILED，缓存 _terminal

    # Feature 4：外部线程触发停止
    def request_stop(self) -> None
        # set stop_event + bridge.cancel_agent_plan(task_id)

    # Feature 5：fresh state 全新 continuation
    def continue_workflow(self, raw_command: str, steps: list[dict] | None = None, *, resume: bool = False) -> dict
        # resume=True: 切片 prior_steps[completed_step_count:]，只 plan 剩余步（断点续传）
        # resume=False: plan 调用方传入的 steps（全新整盘）
        # 两种模式都用新 request_id/task_id、重新 discover+reject_out_of_catalog
        # 前置门：确定终态；UNKNOWN 拒绝。返回展示体（含 continues_from / resume / resumed_from_step）
```

`id_factory` 注入便于测试确定 task_id/request_id。

## 6. 自然语言封闭语法

与 `SKILL.md` 既有确认语法对齐，新增停止/继续：

| 意图 | 封闭语法（大小写/去空格后匹配前缀或全等） |
|------|------------------------------------------|
| confirm | `确认执行当前计划` `确认` `确认执行` `执行吧` `好` `好的` `可以` `是的` `confirm` `确认然后挥手` `确认一下计划内容` |
| stop | `别动` `停` `停止` `停下` `stop` `halt` `别动了` |
| continue | `继续` `继续吧` `go on` `continue` `接着来` |

- `classify(text)` 返回 `confirm|stop|continue|unknown`；`unknown` 永不触发任何动作。
- 仅 `confirm` 在 `confirm()` 内被消费；`stop`/`continue` 由上层（Hermes）据 `classify` 结果分别调 `request_stop()`/`continue_workflow()`，控制器不自行解析自由文本到动作序列。

## 7. Feature 5 continuation 语义（fresh state，断点续传）

- continuation = **全新 plan/confirm/execute**，基于「重新 discover 得到的 fresh registry identity 与 fresh catalog」。
- **断点续传**：读取上次确定终态 terminal 的 `completed_step_count`，把原有序列切片为 `original_steps[completed_step_count:]`，**只 plan 剩余步**：
  - 已完成的步骤跳过；
  - 被中途取消的那一步**从头重跑**（不从半截续，安全）；
  - 若 `completed_step_count` 已等于全部步数，则计划已完成，不再 plan，请用户发起新请求。
- 必须使用**新的 `request_id` 与 `task_id`**；绝不复用旧 `plan_token`/`confirmation_token`/`task_id`。
- 必须对剩余步**重新做 `reject_out_of_catalog`**（catalog 可能已 reload，旧 step 可能已失效）。
- 前置门：上一轮必须落在**确定终态**（`STOPPED`/`SUCCEEDED`/`FAILED`）；`UNKNOWN` 时拒绝，返回 `SKILL_CANCEL_TIMEOUT`/`robot stop state is unknown`，且**不发任何新运动**（合规 `SKILL.md:64/83`）。
- continuation 不继承旧 plan 的任何预算/期限；`task_budget_sec` 来自 fresh `status`。
- `InteractiveController.continue_workflow(raw_command, steps=None, *, resume=False)`：
  - `resume=False`（默认）：plan 调用方传入的 `steps`（全新整盘，用于"换一个动作继续"场景）；
  - `resume=True`：切片 `prior_steps[completed_step_count:]`，只 plan 剩余步。

## 8. 安全门：unknown 终态拒绝继续

`stop()` 与 `execute()` 在取消后必须把终态二分：

- 确定（`GoalStatus ∈ {4,5,6}`）→ `state = STOPPED`，允许 `continue_workflow()`。
- 未知（cancel 未收敛 / `get_agent_plan_result` 超时 / status ∉ {4,5,6}）→ `state = UNKNOWN`，
  `continue_workflow()` 抛 `UnknownStopError("SKILL_CANCEL_TIMEOUT", "robot stop state is unknown")`，
  且**不发任何新运动**（合规 `SKILL.md:64/83`）。

## 9. 文件清单与改动范围

| 文件 | 类型 | 说明 |
|------|------|------|
| `docs/interactive_closed_loop_control_design.md` | 新增 | 本设计文档 |
| `src/robot_skill_cli/robot_skill_cli/interactive_control.py` | 新增 | `InteractiveController` + 封闭语法 + 状态机 + 异常类 |
| `src/robot_skill_cli/test/test_interactive_control.py` | 新增 | 纯 Python 单测，`FakeBridge` 覆盖 1–5 |
| `src/robot_skill_cli/resource/ibrobot-control/SKILL.md` | 修改 | 新增「交互式闭环：别动 / 继续」契约段 |

不改：runtime 节点、`ibrobot_msgs`、`AgentPlanStore`、既有 CLI 子命令行为。

## 10. 测试计划（无 ROS，FakeBridge）

`FakeBridge` 用 `SimpleNamespace`/轻量类实现 `RosBridge` 的被调方法，返回可控 future/result。

| 用例 | 覆盖 feature | 断言要点 |
|------|------------|---------|
| `test_discover_is_readonly` | 1 | 只调 status+snapshot；不调 plan/confirm/execute |
| `test_reject_out_of_catalog` | 2 | step 含 `planner_visible_names` 外的 skill → `SKILL_REFERENCE_MISSING`，不 plan |
| `test_prepare_then_confirm_presentation` | 3 | 展示体含 steps/digest/registry_identity/task_id；confirm 命中封闭语法后调 `confirm_agent_plan` |
| `test_confirm_rejects_open_grammar` | 3 | 「行不行」→ `NotConfirmedError`，不 confirm |
| `test_execute_stop_definite_terminal` | 4 | 执行中置 stop_event → cancel → `status=5` → `state=STOPPED` |
| `test_execute_stop_unknown_refuses_continue` | 4+5 | cancel 不收敛 → `state=UNKNOWN` → `continue_workflow` 抛 `SKILL_CANCEL_TIMEOUT` |
| `test_continue_uses_fresh_state_and_new_ids` | 5 | continuation 调用新 discover、新 request_id/task_id、新 plan；旧 token 不复用 |
| `test_continue_resume_slices_remaining_steps` | 5 | 4 步计划 stop(completed=2) → resume=True → 只 plan `steps[2:]`（2 步），跳过已完成，新 task_id |
| `test_continue_resume_already_complete_raises` | 5 | terminal completed=4（全完成）→ resume 抛 `PLAN_ALREADY_COMPLETE` |
| `test_continue_resume_requires_definite_terminal` | 5 | `CONFIRMED` 状态下 resume → 拒绝（`SKILL_CANCEL_TIMEOUT`） |
| `test_continue_resume_unknown_refused` | 5 | UNKNOWN 终态下 resume → 拒绝 |
| `test_continue_requires_definite_terminal` | 5 | `CONFIRMED` 状态下 `continue_workflow` → 拒绝 |

## 11. 风险与未覆盖项

- 真实 Hermes agent 的 NL→`workflow_steps` 仍由外部 Hermes 完成；本控制器只消费结构化 steps + 封闭词表。
- 跨进程「别动」需 Hermes 在另一线程调 `request_stop()`；本控制器以 `stop_event`/`request_stop()` 提供线程安全入口。
- 物理停止 ≠ 取消已接受：仍以 `GoalStatus∈{4,5,6}` 为准，与既有契约一致。
- 不引入自动重试；「继续」必须由用户显式发起。
