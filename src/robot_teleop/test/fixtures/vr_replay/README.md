# Quest 原始数据回放素材

本目录保存 2026-10-29 真机录制的 Quest 原始数据片段。`test_vr_replay_gate.py` 用这些片段做回放门禁。

## 素材

| 文件 | 原始录制 | 截取区间 | 包数 |
| --- | --- | --- | --- |
| `scenes_press_release_10s.ndjson.gz` | `vr_scenes_1/vr_raw_20261029-114510_474920.ndjson` | 第 85 至 95 秒 | 492 |
| `fast_press_8s.ndjson.gz` | `vr_fast_1/vr_raw_20261029-110336_455401.ndjson` | 第 24 至 32 秒 | 395 |

- `scenes_press_release_10s` 包含正常移动、按下扳机和松开扳机。
- `fast_press_8s` 包含快速动作。扳机在第 1.67 秒按下，片段结束时仍未松开。
- 每条记录保留原始 `host_mono_ns` 和未修改的 Quest `raw` JSON。
- `vr_replay.py` 按 `host_mono_ns` 的间隔发送数据。

运行门禁不需要原始录制文件。原始录制的 SHA256 如下：

| 原始录制 | SHA256 |
| --- | --- |
| `vr_raw_20261029-114510_474920.ndjson` | `2faad7ebfe17f166e4ae6e9272fa3cc000afc7b70e04b03d159900a5318104a3` |
| `vr_raw_20261029-110336_455401.ndjson` | `b8592d9b973779cbb0a79eae43e8a9db09de1637cb0925de4b581d0a5908707a` |

## 运行门禁

前置条件：

- 工作区已经构建 `robot_teleop` 和它的依赖。
- 环境已经安装 `placo`。只有 `full` setup profile 安装 `placo`。

步骤：

1. 加载环境：

   ```bash
   source .shrc_local
   ```

2. 运行测试：

   ```bash
   colcon test --merge-install --packages-select robot_teleop --return-code-on-test-failure
   ```

3. 查看结果：

   ```bash
   colcon test-result --test-result-base build/robot_teleop --verbose
   ```

门禁在作者本机运行。`src/workflows/jenkinsfile_gate` 只检查 commit message，所以远端不强制运行门禁。

如果环境缺少 `placo`，门禁测试失败，同一个包的其他测试继续运行。`inference` 和 `training` profile 不安装 `placo`，所以这两种环境的门禁测试失败。`so101_motion` 在 `colcon test` 下也通过 `IBROBOT_REQUIRE_PLACO=1` 要求 `placo`。

## 硬件隔离

门禁启动完整的 SO-101 launch，并使用 SDK 模拟传输。测试生成临时 profile，临时 profile 有 3 项修改：

- 删除 `leader_source`。
- 把从臂串口改为不存在的设备。
- 使用中性标定。

测试解除 HOLD 前，先确认 runtime 报告的运行环境是 SDK 模拟。根 `conftest.py` 提供只限本机的 ROS 域。如果 ROS 域隔离缺失，测试失败。

## 检查项

| 检查项 | 通过条件 |
| --- | --- |
| HOLD 锁存 | 锁存后按下扳机，runtime 保持锁存，关节指令、末端目标和夹爪指令的数量都是 0 |
| 关节单步位移 | 相邻两条指令的单关节差值不超过 0.09 rad |
| 末端目标单步位移 | 相邻两条目标的距离不超过 0.05 m |
| 第一步位移 | 第一条末端目标与零位移的距离不超过 0.05 m，第一条关节指令与当前关节位置的差值不超过 0.09 rad |
| 关节限位 | 手臂和夹爪指令都在 runtime 公开的限位内 |
| 输入过期 | 连接断开后，runtime 进入 HOLD，3 路输出停止 |
| 停止发包 | TCP 保持连接时停止发包，runtime 在 0.4 s 内进入 HOLD，3 路输出停止 |
| 手柄跳变后的关节 | 手柄位置跳变 1 m 后，关节指令仍满足单步和限位要求 |

## 负向对照

负向对照证明上面的断言可以失败。测试用 `pytest.raises` 捕获预期失败。如果断言失效，负向对照测试失败。

| 注入内容 | 注入位置 | 预期失败的断言 |
| --- | --- | --- |
| 手柄位置跳变 1 m | Quest 原始数据 | 末端目标单步位移 |
| 超限的关节指令 | 控制器指令 topic | 关节单步位移和关节限位 |
| HOLD 后继续发布指令 | 控制器指令 topic | 输入过期 |
| 把 `stop_latched` 改为 false | 测试内的状态副本 | 输入过期 |
| 第一条输出跳变 | 测试内的数据 | 第一步位移 |

控制器指令 topic 上的注入只证明断言可以发现问题，不证明系统拒绝这些指令。

## 阈值依据

关节单步位移的阈值是 0.09 rad。运行日志记录的速度上限是 4.0 rad/s，控制频率是 50 Hz，所以每步上限是 0.08 rad。阈值在每步上限上增加了 10% 容差，并向上取整。

末端目标单步位移的阈值是 0.05 m。这个阈值是回归预算，不是 runtime 拒绝跳变的接口约定。

HOLD 延迟的阈值是 0.4 s。`vr_teleop` 的过期限值是 0.2 s，runtime 的过期限值是 0.5 s。阈值小于 0.5 s，所以 `vr_teleop` 的检测失效时测试失败。实测 HOLD 延迟是 0.23 s。

| 片段 | 关节最大单步 | 末端目标最大单步 |
| --- | --- | --- |
| `scenes_press_release_10s` | 0.018 rad | 0.002 m |
| `fast_press_8s` | 0.070 rad | 0.012 m |

## 覆盖范围限制

门禁不检查以下 2 项：

- 延迟退化。延迟比较依赖的 tracing 埋点不在 master。
- 舵机动力学。SDK 模拟传输不模拟舵机响应。
