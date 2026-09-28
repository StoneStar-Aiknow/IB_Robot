# X2 参数化说话与命名动作示例

本示例通过原有统一入口启动 AimDK runtime 和独立交互节点。机器人 YAML 只选择逻辑接口，
具体文本、语言、动作名和目标侧由每次请求传入；启动本身不会说话或运动。

## 架构边界

- `robot_config/config/robots/aimdk_x2_interaction_demo.yaml` 继承 `aimdk_x2.yaml`，只包含：
  `enabled`、业务服务名、超时，以及 `speech.speak` / `motion.named` 逻辑接口及期望协议类型。
- runtime profile 是实际 endpoint 的唯一来源。统一 launch 在 runtime 就绪后读取公开接口描述，
  校验 kind、direction、message type 和 capability requirements，再把解析出的地址注入 demo。
- `robot_interaction_demo` 仅消费注入的 service/action 地址和 `ibrobot_msgs` 类型；不依赖
  `robot_config`、`robot_runtime`、`aimdk_robot` 或 `aimdk_msgs`，也不查询 runtime 状态。
- CLI 只发送本次业务参数，不读取另一份 YAML，不保存固定文案或动作列表。

因此，更换到提供兼容逻辑接口的其他本体时，业务包无需修改；部署配置选择该本体提供的接口。

## 构建与统一启动

Ubuntu/openEuler 原生源码工作区需要 ROS 2 Humble 和与机器人匹配的 AimDK overlay：

```bash
source .shrc_local
source /path/to/aimdk/install/local_setup.sh
./scripts/build.sh -- --packages-up-to robot_config aimdk_robot
```

重新加载环境，参与终端使用相同 `ROS_DOMAIN_ID` 和 `RMW_IMPLEMENTATION`。以下 localhost
和域 42 只适用于本机 vendor mock：

```bash
source .shrc_local
source /path/to/aimdk/install/local_setup.sh
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
ros2 launch robot_config robot.launch.py \
  robot_config:=aimdk_x2_interaction_demo \
  use_sim:=true
```

统一 launch 会启动 vendor mock、AimDK runtime，完成接口描述绑定后再启动 demo 节点。
`aimdk_x2.yaml` 本身不启用 demo；只有派生配置启用。ASR、ZipVoice、共享 ALSA 和 Agent 仍关闭。

## 参数化调用

另一个使用相同 DDS 环境的终端：

```bash
source .shrc_local
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1

# language 省略时传空字符串，由 runtime 选择默认语言。
ros2 run robot_interaction_demo runtime-demo speak \
  --text "大家好，我是灵犀 X2。这是 IB-Robot 的交互接口示范。" \
  --language zh-CN \
  --priority 50

# 动作名和目标侧由调用者提供；每次运动必须显式授权。
ros2 run robot_interaction_demo runtime-demo motion \
  --name wave \
  --target right \
  --allow-motion
```

可选参数：

- `speak`：`--language`（默认空）、`--priority 0..100`（默认 50）、`--interrupt`。
- `motion`：`--target ''|left|right|both`（默认空）、`--interrupt`、`--allow-motion`。
- 通用：`--service`（默认 `/interaction_demo/execute`）、`--timeout`（默认 45 秒）。

业务服务类型为 `ibrobot_msgs/srv/ExecuteInteractionDemo`。程序也可直接构造 SPEAK 或
NAMED_MOTION 请求，而无需依赖 demo 包的 CLI。

## 结果和安全边界

- demo 节点只按注入地址调用中立 `SpeakText` service 或 `ExecuteNamedMotion` action。
  endpoint 不在业务代码或应用 YAML 中硬编码。
- TTS 的 `phase: accepted` 只表示 runtime 接受请求，不证明扬声器出声或语句结束。
- 动作等待 action 终态，返回 action status、runtime error code 和 message。实际允许的动作名、
  模式和稳定站立条件仍由 runtime 决定；未知动作会被 runtime 拒绝。
- `--allow-motion` 是当前请求的显式授权，不来自 YAML，不替代现场安全判断。
- 节点一次执行一个请求；忙碌时拒绝、不排队。不切换模式、不解除停止锁、不重试。
- 服务端和 CLI 超时都不保证已提交请求被取消。超时/Ctrl-C 后先检查机器人状态再发下一请求。
- 这是操作者 demo，不是 Agent 物理执行入口；Agent 继续走共享 primitive/skill Gateway。

## 扩展到其他部署

复制机器人 YAML，仅修改基础配置或逻辑接口选择；不要在 YAML 中加入演示文案或动作清单：

```yaml
robot:
  base_config: another_robot.yaml
  name: another_interaction_demo
  interaction_demo:
    enabled: true
    service_name: /interaction_demo/execute
    timeout_sec: 40.0
    interfaces:
      speech:
        interface: speech.speak
        kind: service
        type: ibrobot_msgs/srv/SpeakText
      named_motion:
        interface: motion.named
        kind: action
        type: ibrobot_msgs/action/ExecuteNamedMotion
```

该本体的 runtime profile 必须在公开接口描述中提供这些逻辑 ID 和兼容协议，否则绑定失败，
demo 不会启动。包外机器人 YAML 继续遵循既有 `config_path` 和同目录 `base_config` 规则。

## 真机验证（后续由操作者执行）

本次未完成真机验收。默认基础配置选择 `x2_ultra` OmniPicker profile；灵巧手部署应使用对应
profile 和能力要求。

1. 按 [aimdk_robot README](../src/robots/aimdk/aimdk_robot/README.md) 完成 SDK、网络和只读检查。
   不在 PC1 运动控制单元运行应用。
2. 使用同一 YAML，将 `use_sim` 改为 `false`；跨设备环境不要使用 mock 的
   `ROS_LOCALHOST_ONLY=1`，不要同时运行 mock 与真机 runtime。
3. 先发送自定义短文本，分别记录服务结果和实际出声情况。
4. 现场确认稳定站立、活动空间及停止手段后，再参数化请求 wave / raise_hand / clap 等动作，
   核对 action 返回和真实动作。
5. 记录拒绝路径、固件、SDK、硬件版本与日志。
