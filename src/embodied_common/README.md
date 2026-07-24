# embodied_common 架构契约

`embodied_common` 是具身管线的中立共享包，只承载无业务副作用的公共 helper 与默认 fallback 数据。
它用于消除 `embodied_agent`、`perception_service`、`vlm_task_planner`、`safety_guard`、`skill_library` 之间的重复实现和反向依赖。

## 职责边界

允许放入本包的内容：

- 不启动 ROS 节点、不创建 action/service/client 的纯工具函数。
- 被多个具身包共同使用的轻量数据规整逻辑。
- 作为 fallback 的默认 skill / primitive 描述。
- 不依赖具体业务包内部实现的基础节点 helper。

不应放入本包的内容：

- 依赖感知、规划、安全或执行节点运行状态的业务逻辑。
- 机器人型号、相机 topic、命名位姿、工作空间等应由 `robot_config` 管理的 SSOT 配置。
- `ibrobot_msgs` 已经表达的 ROS msg/srv/action 契约。
- 对 `perception_service`、`vlm_task_planner`、`safety_guard`、`skill_library`、`embodied_agent` 的反向 import。

## 依赖方向

`embodied_common` 位于具身业务包下方：

```text
embodied_agent
perception_service
vlm_task_planner
safety_guard
skill_library
        ↓
embodied_common
        ↓
ibrobot_msgs / rclpy
```

本包不得依赖上述业务包，避免重新形成层级倒置或菱形耦合。

## 对外 API

当前稳定 API：

- `embodied_common.base_node.BaseTaskNode`
- `embodied_common.json_utils.extract_json_blob`
- `embodied_common.json_utils.load_json_mapping`
- `embodied_common.json_utils.load_json_list`
- `embodied_common.json_utils.string_list`
- `embodied_common.json_utils.parse_confidence`
- `embodied_common.command_parser.parse_text_command`
- `embodied_common.command_parser.extract_skill_aliases`
- `embodied_common.command_parser.load_skill_aliases`
- `embodied_common.skill_templates.SUPPORTED_PRIMITIVES`
- `embodied_common.skill_templates.DEFAULT_SKILL_TEMPLATES`
- `embodied_common.skill_templates.DEFAULT_ALLOWED_SKILLS`
- `embodied_common.skill_templates.DEFAULT_WAYPOINT_DURATION_SEC`
- `embodied_common.skill_templates.is_skill_disabled`
- `embodied_common.skill_templates.get_skill_templates`
- `embodied_common.rgbd_snapshot.KNOWN_REQUIRED_INPUTS`
- `embodied_common.rgbd_snapshot.RGBDSnapshotBuffer.build_snapshot`
- `embodied_common.vlm_api_client.VLMAPIClient.analyze`（原有底层接口，返回 `(str, dict)`，向后兼容）
- `embodied_common.vlm_api_client.VLMAPIClient.complete`（底层扩展接口，返回结构化 dict）
- `embodied_common.vlm_api_client.VLMClient`（高层客户端，一行式多模型调用，自动路由 / 建图 / 上下文）

## 输入前置条件（required_inputs）

`build_snapshot(required_inputs=...)` 提供通用的输入门控，供上层按每条请求声明"哪些输入缺失才阻塞"：

- 允许键为 `KNOWN_REQUIRED_INPUTS`（`primary_image` / `ee_pose` / `joint_state`）。
- 传入 `None`（默认）、缺失、或畸形值（非列表、空列表、列表内含非字符串项）时，回退到严格默认——
  primary_image + ee_pose + joint_state 全部要求在线。
- 传入合法子集时只门控该子集，未知/畸形键被安全忽略（不抛异常），使纯视觉请求可在 EE pose /
  joint state 离线时成功。

该词表业务中立，不含任何具体游戏/任务特判。

## VLMClient 使用说明

调用方一行创建实例即可随时调用云端模型，无需手动构造 messages 或指定 provider / key：

```python
from embodied_common.vlm_api_client import VLMClient

vlm = VLMClient()                              # 加载 vlm_models.yaml，一次初始化

vlm.chat("做任务规划")                          # 默认模型（defaults.model）
vlm.chat("分析图片", image=frame_bytes)         # 传入图片字节，自动构造多模态 message
vlm.chat("文本任务", model="my-llm")            # 按模型名路由到 vlm_models.yaml 中的条目
vlm.chat("快速回答", enable_thinking=False)     # 关闭思维链（支持的 provider 生效）
vlm.chat("新话题", clear_history=True)          # 清空上下文后再对话
```

同一个 `VLMClient` 实例的多次 `chat()` 自动累积历史（多轮记忆）；需要全新会话时新建实例。

`chat()` / `complete()` 返回结构化 dict：

| 字段 | 含义 |
|------|------|
| `status` | `"ok"` / `"error"` |
| `model` | 实际调用的模型名 |
| `content` | 文本回复 |
| `reasoning` | 思维链内容（无则为空串） |
| `tool_calls` | function call 列表 |
| `usage` | token 用量 |
| `timing_ms` | 端到端耗时（毫秒） |
| `error` | 出错描述，成功时为 `None` |

### 模型路由配置（vlm_models.yaml）

位于 `embodied_common/vlm_models.yaml`。新增模型只需追加几行，`api_key_env` 只存环境变量名，
`multimodal: false` 时自动过滤图片。API key 通过环境变量注入，不写入任何配置文件：

```bash
export ALIYUN_API_KEY=sk-xxxxxx
export MY_LLM_API_KEY=sk-xxxxxx
```

`get_skill_templates` 会先过滤显式设置 `disabled: true` 的模板，再做深拷贝并就地展开
`trajectory_template` 为 `joint_waypoints`（调用 `expand_trajectory_template`）。只有字面量布尔值
`True` 表示禁用；`robot_config.loader` 会拒绝非布尔 `disabled`。因此 loader、规则入口、resolver、
safety guard 和 MCP catalog 共用同一份启用技能与模板展开语义。

## 与 SSOT 的关系

`DEFAULT_SKILL_TEMPLATES` 是最小闭环的默认 fallback，不是最终机器人级 SSOT。
机器人级 skill 名不在本包维护全局常量，而是从当前 `robot_config` YAML 的
`embodied.skill_templates` 键集合派生。

当前 `robot_config` 已提供 `embodied` 配置段，机器人级 skill templates、命名位姿、
命名目标、workspace 边界和感知相机 topic 均由 `robot_config` YAML 作为单一事实来源
管理。运行时由 `embodied_bringup` 从该 YAML 读取并注入到下游具身节点参数：

- skill templates
- named poses
- named targets
- workspace limits
- perception camera topics

`embodied_common` 只提供安全可运行的默认值，保证独立节点调试时不会因为缺少 launch
注入而完全不可用；一旦通过 `embodied_bringup` 启动，YAML 注入值会覆盖默认值。
