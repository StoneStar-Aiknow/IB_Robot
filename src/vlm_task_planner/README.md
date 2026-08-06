# vlm_task_planner 节点说明

`vlm_task_planner` 是当前具身执行链中的 **VLM 任务规划包**。

它位于：

```text
/voice_command
  -> task_entry_node
  -> /embodied/task_command
  -> vlm_task_planner_node
  -> /embodied/planned_task
  -> task_executor_node
```

## 1. 职责

它负责：

1. 从 ROS 获取任务文本、相机图像和机器人状态
2. 先做场景理解，再调用外部大模型 API 做任务分解
3. 把视觉+语言理解结果转换成受控 skill 序列
4. 当任务需要仓内不存在的 skill 时直接报错
5. 当 API 不可用、图像缺失或置信度不足时回退到规则 planner

## 2. 关键原则

- 不直接输出关节角或底层 pose
- 只输出有限 skill 集
- 若大模型判断任务依赖缺失 skill，则拒绝规划而不是伪造 workaround
- 只消费 Gateway 的**已验证 exact catalog 视图**：每条任务到达时读取 `CatalogViewSynchronizer.current`，
  当目录 reload 推出新 identity、对应 snapshot 尚未校验通过时 `current` 为 `None`，planner 立即以
  `SKILL_REGISTRY_NOT_READY` 拒绝任务（可重规划），而不是用过期技能边界继续规划。技能边界和别名
  均取自该已验证视图，而非节点启动时的静态参数。
- 最终执行仍然经过：
  - `task_executor_node`
  - `safety_guard`
  - `skill_library`

## 3. 主要参数

| 参数 | 说明 |
| --- | --- |
| `planner_mode` | `rule` / `vlm_api` / `hybrid` |
| `primary_camera_topic` | 主 USB 摄像头图像 topic |
| `wrist_camera_topic` | wrist 视角图像 topic，可选 |
| `primary_camera_info_topic` | 主视角相机内参 topic，可选 |
| `primary_aligned_depth_topic` | 主视角对齐深度图 topic，可选 |
| `primary_pointcloud_topic` | 主视角点云 topic，可选 |
| `wrist_camera_info_topic` | wrist 视角内参 topic，可选 |
| `wrist_aligned_depth_topic` | wrist 视角对齐深度图 topic，可选 |
| `wrist_pointcloud_topic` | wrist 视角点云 topic，可选 |
| `ee_pose_topic` | 当前末端位姿 topic |
| `joint_state_topic` | 当前关节状态 topic |
| `require_depth` | 为 `true` 时，VLM 规划前必须拿到至少一路有效 depth |
| `require_pointcloud` | 为 `true` 时，VLM 规划前必须拿到至少一路有效 pointcloud |
| `api_provider` | API provider，当前默认 `openai_compatible` |
| `api_base_url` | OpenAI-compatible base URL，当前默认 `http://localhost:8000/v1` |
| `api_key_env` | API key 的环境变量名；本地服务无鉴权时可留空 |
| `api_model` | 模型名，当前默认 `Qwen3.5-9B` |
| `api_timeout_sec` | 大模型输出空闲超时，由 `embodied.timeouts.model_idle_timeout_sec` 统一注入 |
| `fallback_to_rule_planner` | API 失败时是否回退规则 planner |
| `allowed_skills_json` | VLM 响应允许输出的 skill 边界的**初始默认值**；运行时被已验证 catalog 视图刷新 |
| `skill_aliases_json` | 确定性规则 fallback 别名的**初始默认值**；运行时被已验证 catalog 视图刷新 |
| `skill_catalog_snapshot_service` | Gateway exact snapshot 服务名，默认 `/embodied/get_skill_snapshot` |

`allowed_skills_json` 与 `skill_aliases_json` 只是节点启动时的初始默认值；每条任务到达时，planner 会用
已验证 exact catalog 视图中的 `planner_visible_names` 和 `aliases` 覆盖二者。设置 `disabled: true` 的 skill
不进入该边界，响应中出现禁用或其他边界外 skill 都会被拒绝。
`skill_aliases_json` 不参与 VLM 输出校验，只在 API 失败或规则模式下供确定性 fallback 解析，并且运行时只包含
已验证 catalog 视图中启用且显式设置 `description.rule_entry: true` 的 skill，而不是 catalog 的全部别名。

当前默认走本地 OpenAI-compatible 服务；原远端 Kimicode 仍然保留，只需把配置切回：

- `api_provider=kimicode`
- `api_base_url=https://api.kimi.com/coding/v1`
- `api_key_env=KIMICODE_API_KEY`
- `api_model=kimi-for-coding`

## 4. RealSense / RGB-D 规划输入

当前 `vlm_task_planner_node` 已支持 RealSense / RGB-D 作为规划输入。

### 4.1 当前能力

当前不是只把一张主相机 RGB 发给模型，而是可以联合使用：

1. primary 主视角 RGB
2. wrist 末端视角 RGB
3. `camera_info` 摘要
4. aligned depth 的结构化深度摘要
5. pointcloud 元信息摘要

其中 depth / pointcloud 不是原始数据直传，而是本地先转成：

- 深度范围
- 中位深度
- 中心区域深度
- 深度有效比例
- 近距离障碍提示

这些摘要会和图像、末端位姿、关节状态一起进入 prompt。

### 4.2 推荐配置

```yaml
embodied:
  timeouts:
    task_budget_sec: 180.0
    scene_freshness_sec: 0.5
    model_idle_timeout_sec: 120.0
    rpc_timeout_sec: 5.0
    gripper_settle_sec: 1.5
  planner:
    scene_sources:
      primary_camera_topic: /camera/front_camera/color/image_raw
      primary_camera_info_topic: /camera/front_camera/color/camera_info
      primary_aligned_depth_topic: /camera/front_camera/aligned_depth_to_color/image_raw
      primary_pointcloud_topic: /camera/front_camera/depth/color/points
      ee_pose_topic: /robot_status/ee_pose
      joint_state_topic: /joint_states
      require_depth: true
      require_pointcloud: false
```

### 4.3 兼容性说明

- 如果只配置 `primary_camera_topic`，行为仍与旧版本兼容。
- 如果同时配置 `wrist_camera_topic`，模型会走**双视角联合规划**。
- 如果配置了 depth / pointcloud，模型会额外使用空间摘要做可达性、遮挡和碰撞风险判断。
- planner 会保留入口写入的 `timeout_context`，并用剩余预算约束 VLM 调用。

## 5. 当前输出

输出仍然是：

- `/embodied/planned_task`

即继续复用已有 `TaskCommand` / `TaskStatus` 契约。

## 6. RealSense 运行时注意事项

1. RealSense 传感器订阅已切换为**sensor QoS**，否则容易出现 topic 在发但 planner 收不到图像/深度的问题。
2. 直接手动运行 `realsense2_camera_node` 时，设备常见原始 topic 名通常是：

```text
/camera/camera/color/image_raw
/camera/camera/color/camera_info
/camera/camera/aligned_depth_to_color/image_raw
/camera/camera/depth/color/points
```

3. 如果要严格复用仓内设计的 `/camera/front/...` 标准命名，推荐通过 `robot_config` launch / remap 做统一，而不是在上层节点里硬编码厂商原始 topic。
