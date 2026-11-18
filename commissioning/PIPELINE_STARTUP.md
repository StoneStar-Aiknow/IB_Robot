# X2 pipeline 启动指导

本文适用于 310P 与 Orin 分机部署。
当前配置仅安装左手，抓取臂为左臂。
机器人操作系统使用 ROS 2 Humble。

## 启动前置条件

- 310P 已部署当前工作区及运行环境。
- Orin 已部署匹配的代码及运行环境。
- 两端复用现场的数据分发服务（DDS）配置。
- 310P 支持免密 SSH 连接 Orin。
- 厂商运动、通信、音频及相机服务已启动。
- 操作员已清空运动区域并准备急停。
- 凭据已存入现场环境指定的位置。

凭据禁止写入源码、命令示例或 Git 提交。
日志中的工作区路径必须指向当前部署。

## 核对部署配置

配置文件为：

```text
src/robot_config/config/robots/aimdk_x2_voice_mirror_grasp.yaml
```

关键字段如下。省略的字段沿用现场配置。

```yaml
robot:
  task_poses:
    hands: installed
  grasp_execution:
    backend_options:
      installed_hands:
      - left
      arm_side: left
  cbf_safety:
    # 与实物一致：只装左手。数据面也认 [true, true] 配空法兰（见下节），
    # 改成 [true, true] 前先在 .5 实测 CBF 耗时。
    hands_installed:
    - true
    - false
  runtime:
    instance_id: x2_robot_03
```

固定姿态库必须匹配安装手配置：

```text
commissioning/fixed_poses_voice_mirror_grasp.json
```

安装手配置变更后必须重新生成匹配的姿态库。
禁止手动修改姿态库摘要来跳过校验。

### 部署顺序：`hands_installed` 与数据面一起发布

出厂 yaml 仍是 `[true, false]`，与只装左手的实物一致。下面是改成
`[true, true]`（右手空法兰或已装）时的要求；改之前先在 .5 实测 CBF 耗时
（x86 上两手比单手 p99 约高 3 倍）。

`cbf_safety.hands_installed: [true, true]` 只能配支持幻影手的 aimdk_dataplane
（feat/x2-cbf-twohands 起）。e965d015 及更早的数据面不认空法兰：平台报
`{1, 0}`（左手装、右侧空）时判为手配置不匹配（`hand_configuration_matches=false`），
enforce 下 runtime 拒绝进入 mirror_mode，照镜子起不来。

- 发布：数据面先于或同时于新 yaml 上板，不能只换 yaml。
- 回退：先回 yaml 到 `[true, false]`，再回数据面。旧 yaml 配新数据面可用，
  但只在平台报 `{1, 0}` 时与 e965d015 的输出相同。左手掉线报 `{0, 0}` 时两者不同：
  e965d015 判手配置不匹配（`hand_configuration_matches=false`），整机 hold
  （照镜子没开时 runtime 不进 mirror_mode；开着时按不匹配 hold，超过
  `hold_release_after_s` 后照原有规则放行、打 `CBF_RELEASE`）；45635b17 起
  新数据面把左手当幻影手，继续照镜子，并打 `CBF_HAND_ABSENT side=left start`。
- 现场核对：数据面日志 `CBF_HAND_ABSENT side=right start` 表示右侧按幻影手算，
  空法兰时正常。出现 `side=left start` 说明左手报了 type 0（如掉线），左手正被
  当作幻影手检查，应停下查左手；恢复时打 `side=left end`。status 的
  `cbf.phantom_hands` 给出同样的状态。

## 启动完整 pipeline

警告：启动脚本传入 `authorize_motion:=true`。
收到有效动作请求后，系统可能执行真机动作。

1. 从新电脑登录 310P。

   ```bash
   ssh 310p
   ```

2. 在 310P 当前工作区启动 pipeline。

   ```bash
   cd /root/lwh/IB_Robot_oee_summit_2026_8ad9364a && \
   bash commissioning/start_interaction.sh
   ```

脚本加载 `commissioning/env.sh`。
现场环境文件不存在时，脚本加载 `.shrc_local`。
脚本也会请求启动 Orin grasp head。
已有且归属校验通过的启动进程会复用。

远端位置使用以下环境变量：

```text
IBROBOT_GRASP_HEAD_HOST
IBROBOT_GRASP_HEAD_WORKSPACE
```

默认配置可能指向旧工作区。
现场环境必须指定当前 Orin 工作区。
环境脚本可能重设这两个变量。

本部署的 Orin 工作区为：

```text
/home/agi/IB_Robot_oee_summit_2026_8ad9364a
```

## 查看启动结果

1. 在 310P 查看 pipeline 日志。

   ```bash
   cd /root/lwh/IB_Robot_oee_summit_2026_8ad9364a && \
   tail -f log/agent_interaction/pipeline_latest.log
   ```

2. 从新电脑查看 Orin 日志。

   ```bash
   ssh -t 310p 'ssh -t agi@10.0.1.41 "tail -f /home/agi/IB_Robot_oee_summit_2026_8ad9364a/log/grasp_head/head_latest.log"'
   ```

`started` 仅表示启动进程已创建。
`warming up` 表示启动准备尚未完成。
`backends prepared` 也不证明模型调用成功。
必须检查 `grasp_warmup` 中的错误及生成结果。

抓取、放置和归位必须分别核验执行终态。
模型生成候选不代表抓取成功。

## 仅重启 310P pipeline

两条命令均设置 `IBROBOT_MANAGE_GRASP_HEAD=0`。
Orin head 在停止及启动期间保持运行。

1. 在 310P 停止当前 pipeline。

   ```bash
   cd /root/lwh/IB_Robot_oee_summit_2026_8ad9364a && \
   IBROBOT_MANAGE_GRASP_HEAD=0 bash commissioning/stop_interaction.sh
   ```

2. 在停止成功后启动 pipeline。

   ```bash
   cd /root/lwh/IB_Robot_oee_summit_2026_8ad9364a && \
   IBROBOT_MANAGE_GRASP_HEAD=0 bash commissioning/start_interaction.sh
   ```

如果停止失败，先检查残留进程归属。
停止结果未知时，禁止继续发送动作。
保留 head 进程不能保证后续无需重建规划器。

## 单独重启 Orin head

1. 从新电脑进入 Orin 当前工作区。

   ```bash
   ssh -t 310p 'ssh -t agi@10.0.1.41'
   ```

2. 在 Orin 停止并启动 head。

   ```bash
   cd /home/agi/IB_Robot_oee_summit_2026_8ad9364a && \
   bash commissioning/stop_interaction.sh --head && \
   bash commissioning/start_interaction.sh --head
   ```

重启前必须确认当前没有抓放任务。

## 停止完整 pipeline

1. 在 310P 加载环境并停止 pipeline。

   ```bash
   cd /root/lwh/IB_Robot_oee_summit_2026_8ad9364a && \
   export VOICE_POSE_TARGET=robot && \
   source commissioning/env.sh >/dev/null && \
   bash commissioning/stop_interaction.sh
   ```

环境入口提供匹配的远端 head 位置。
停止脚本默认同时停止 Orin head。
脚本检查当前工作区拥有的进程。
旧工作区的独立 runtime 可能仍在运行。

禁止使用广泛的 `pkill` 或 `killall`。
确认旧进程的路径、父进程及启动时间后再停止。
保留厂商底层服务。

## 通过文字控制

1. 从新电脑启动文字终端。

   ```bash
   ssh -t 310p 'cd /root/lwh/IB_Robot_oee_summit_2026_8ad9364a && export VOICE_POSE_TARGET=robot && source commissioning/env.sh >/dev/null && ros2 run ibrobot_agent ibrobot_agent_chat --channel-id voice_asr --principal-id local_operator --robot-scope aimdk_x2_voice_mirror_grasp'
   ```

2. 输入指令，例如 `回到抓取准备位`。

3. 输入 `/stop` 取消当前任务。

文字入口无需唤醒词。
动作仍通过 Capability Gateway 执行。

## 启动故障判断

| 日志或现象 | 检查项 |
| --- | --- |
| `generate_grasps is not available` | 检查 310P 模型进程、初始化日志及服务发现。 |
| 手部反馈缺失 | 核对仅左手配置及左手反馈。 |
| RGB-D 与关节时间不匹配 | 检查图像时间、关节反馈时间及 TF 坐标变换。 |
| `No near-field support plane` | 检查桌面是否进入深度相机视野。 |
| `Checkout-owned nodes remain` | 先停止当前工作区残留节点，再启动。 |
| `PID ownership mismatch` | 核实进程归属，禁止按旧 PID 文件直接杀进程。 |

当前仓库入口为 `start_interaction.sh`。
现场的 `start_prepared_pipeline.sh` 包装脚本不属于本文前提。
本指导不依赖未整理的跨工作区清理包装脚本。
