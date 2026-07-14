# VR Teleoperation Wire Protocol

本文档定义 Unity XR 应用与 IB-Robot `vr_teleop` 节点之间的 TCP 线协议。
它描述的是当前已实现的、**无显式版本字段**的协议；应用端不得自行增加
坐标预转换、事件压缩或请求/响应语义。

## 1. 适用范围

协议发送端是独立维护的 VR 应用，接收端是：

- ROS 2 package：`robot_teleop`
- executable：`vr_teleop`
- 默认监听地址：`0.0.0.0:8889`
- 实现文件：`robot_teleop/vr_teleop.py`

同一输入协议支持两种机器人端输出 profile：

| Profile | 用途 | 协议相关差异 |
|---|---|---|
| `so101` | 单臂 SO-101 Placo Cartesian teleop | 使用 `controller_side` 指定的一只手柄；`enabled` 控制 clutch；`secondaryButton` 上升沿触发 home |
| `humanoid` | 双臂 humanoid teleop | 同时消费左右手柄；当前不使用 home 按键 |

profile、端口和控制参数由机器人仓库配置决定，不由 TCP 数据包选择。

## 2. Transport

| 项目 | 约定 |
|---|---|
| Transport | TCP |
| Direction | VR 应用 → IB-Robot，单向数据流 |
| Encoding | UTF-8 |
| Framing | NDJSON：每行一个完整 JSON object，以 `\n` 结束 |
| Connection | 服务器一次只处理一个客户端；断线后重新等待连接 |
| Server response | 无 ACK、无命令响应、无版本协商 |
| 推荐频率 | 50 Hz |
| 安全网络 | 可信局域网；协议没有认证、授权或加密 |

TCP 的一次 `send()` 不等于一个数据包。应用端必须依赖换行符划分消息，不能依赖
TCP chunk 边界。`\r\n` 可以被当前接收端接受，但规范发送端应使用 `\n`。

应用端应持续发送状态快照，包括手柄未启用时的帧。对于 SO-101，连续缺帧超过
`so101_command_stale_s`（默认 0.2 秒）会触发 deadman stop；恢复发送后，操作者
必须先释放 `enabled`，再重新按下，机械臂才会恢复控制。

## 3. Coordinate and unit contract

应用端发送 **Unity XR tracking/world space 的原始位姿**。同一会话内，控制器和头显
必须使用同一个稳定坐标原点；不要发送 controller-local 位姿，也不要在应用端提前
转换到 ROS 坐标系。

| 数据 | 单位/顺序 |
|---|---|
| Position | 米，`[x, y, z]` |
| Rotation | quaternion `[x, y, z, w]` |
| Timestamp | 浮点数；推荐为应用启动后的单调秒数 |
| Analog controls | 无量纲，规范范围 `[0.0, 1.0]` |

Unity 原始坐标约定：左手系，`+X` 向右、`+Y` 向上、`+Z` 向前。机器人端执行以下
转换，应用端不得重复执行：

```text
position_ros = [-position_unity.x, -position_unity.z, position_unity.y]
quaternion_ros = [quaternion_unity.x,
                  -quaternion_unity.z,
                   quaternion_unity.y,
                   quaternion_unity.w]
```

位置映射是面向操作者的镜像映射；姿态使用独立的 proper rotation basis。不能将位置
矩阵直接用于四元数，否则会产生腕部姿态耦合或轴向错误。

## 4. Top-level packet

每一帧必须是 JSON object。规范发送端应发送如下字段：

| Field | Type | Required by sender | Receiver fallback | Description |
|---|---|---:|---|---|
| `timestamp` | number | 是 | `0.0` | 客户端时间戳；当前不用于 deadman，deadman 使用机器人端接收时间 |
| `left_controller` | controller object 或 `null` | humanoid 必须 | 缺失/非 object 按不存在处理 | 左手柄当前状态 |
| `right_controller` | controller object 或 `null` | SO-101 默认必须；humanoid 必须 | 缺失/非 object 按不存在处理 | 右手柄当前状态 |
| `headset` | pose object | 建议 | 零位置、identity rotation | 头显状态；当前接收并保存，但控制逻辑尚未使用 |
| `config_mode` | boolean | 建议 | `false` | 保留字段；当前控制逻辑尚未使用 |

协议当前没有 `protocol_version`、sequence number 或 message type。未知 top-level 字段
会被忽略，可用于向后兼容地增加可选信息。

## 5. Controller object

控制器对象是一个**完整状态快照**，不是按键事件。规范字段如下：

| Field | Type | Required by sender | Range/default | Current meaning |
|---|---|---:|---|---|
| `position` | number[3] | 是 | 米 | Unity tracking/world-space position |
| `rotation` | number[4] | 是 | `[x,y,z,w]`，非零有限四元数 | Unity tracking/world-space rotation |
| `grip_value` | number | 是 | `[0,1]`，缺失为 `0` | 夹爪模拟量；`0` 为 open，`1` 为 closed |
| `trigger_value` | number | 建议 | `[0,1]`，缺失为 `0` | 原始 trigger 模拟量；当前机器人控制不直接使用 |
| `thumbstick` | number[2] | 建议 | `[x,y]`，缺失为 `[0,0]` | 保留；当前机器人控制不使用 |
| `primaryButton` | boolean | 建议 | 缺失为 `false` | 保留；当前机器人控制不使用 |
| `secondaryButton` | boolean | 是 | 缺失为 `false` | SO-101 pose 模式下，`false → true` 上升沿触发 home |
| `enabled` | boolean | 是 | 缺失为 `false` | 权威 clutch 状态；按住为控制，释放为停止/重新起夹准备 |

### 5.1 Button semantics

- `enabled` 必须每帧发送当前状态。应用端可以根据 trigger 阈值生成它，但阈值属于
  应用实现，不属于线协议。
- `secondaryButton` 必须发送当前按键状态，而不是只发送一个无释放状态的事件。
  持续为 `true` 只触发一次；发送至少一帧 `false` 后再次变为 `true` 才会再次触发。
- SO-101 pose 模式下，`grip_value` 独立于 `enabled`。即使 clutch 释放，也应继续发送
  grip 状态，以允许操作者独立控制夹爪。
- 接收端为旧应用兼容以下 home 按键别名：`buttonB`、`button_b`、
  `secondary_button`、`bButton`。新应用只能发送规范名 `secondaryButton`。

## 6. Headset object

| Field | Type | Required by sender | Description |
|---|---|---:|---|
| `position` | number[3] | 是 | 与控制器相同 tracking/world space 中的位置，单位米 |
| `rotation` | number[4] | 是 | quaternion `[x,y,z,w]` |

当前机器人控制逻辑不消费头显位姿，但应用端应保留该字段，避免未来 humanoid 或
视角相关功能引入另一种不兼容数据包。

## 7. Canonical packet example

以下示例展示完整双手状态。实际发送时 JSON 必须位于单行，并在末尾追加 `\n`。

```json
{
  "timestamp": 12.345,
  "left_controller": {
    "position": [-0.25, 1.20, 0.45],
    "rotation": [0.0, 0.0, 0.0, 1.0],
    "grip_value": 0.0,
    "trigger_value": 0.0,
    "thumbstick": [0.0, 0.0],
    "primaryButton": false,
    "secondaryButton": false,
    "enabled": false
  },
  "right_controller": {
    "position": [0.25, 1.20, 0.45],
    "rotation": [0.0, 0.0, 0.0, 1.0],
    "grip_value": 0.35,
    "trigger_value": 0.85,
    "thumbstick": [0.0, 0.0],
    "primaryButton": false,
    "secondaryButton": false,
    "enabled": true
  },
  "headset": {
    "position": [0.0, 1.65, 0.0],
    "rotation": [0.0, 0.0, 0.0, 1.0]
  },
  "config_mode": false
}
```

SO-101 单臂应用可以将未使用的 `left_controller` 发送为 `null`，但仍建议发送完整
top-level 结构，以便抓包、回放和未来扩展保持一致。

## 8. Validation and error handling

规范发送端必须保证：

1. 每帧是 JSON object，而不是 array、number、string 或 `null`。
2. 所有数值有限，不发送 `NaN`、`Infinity` 或字符串形式的数值。
3. position、rotation、thumbstick 数组长度分别为 3、4、2。
4. quaternion 非零；应用端应在发送前归一化。
5. 每个 JSON object 后立即发送 `\n`。
6. 单帧应远小于 1 MiB；推荐保持在 4 KiB 内。

机器人端对无法解析的 JSON、错误数组长度、无效四元数或非数值字段采取“丢弃该帧
并继续连接”的策略。没有换行的数据累计超过 1 MiB 时，接收缓冲会被清空。

## 9. Connection and safety behavior

- 应用启动后应自动连接，并在连接断开后退避重连。
- 连接建立后应持续发送帧，不要仅在位姿变化时发送。
- 应用进入暂停、后台、tracking lost 或渲染线程异常状态时，不应继续重复发送一个
  伪装成新数据的冻结位姿。可以停止发送或断开连接，让机器人端 deadman 生效。
- 应用恢复 tracking 后，应先发送 `enabled=false` 的实时帧，等待操作者主动重新按下。
- 协议没有认证和 TLS；禁止将监听端口映射到公网、不可信 Wi-Fi 或不可信 VPN。

## 10. Compatibility rules

当前协议为 unversioned wire format，兼容性按以下规则维护：

- 增加接收端可忽略的可选字段：兼容。
- 增加规范发送端字段但为接收端提供默认值：兼容，但需同步本文档。
- 删除、重命名或改变现有字段类型/单位/坐标系：破坏性变更。
- 改变 `enabled`、`grip_value`、`secondaryButton` 语义：破坏性安全变更。
- 引入 `protocol_version`、双向命令或 ACK：需要新的协议修订和双方同步发布。

本文档是应用端与机器人端之间的 wire contract。实现发生上述变化时，必须在同一个
机器人端提交中同步更新本文档；应用端仓库应引用具体 IB-Robot commit 或发布版本，
不能依赖未固定版本的 `master` 文档。
