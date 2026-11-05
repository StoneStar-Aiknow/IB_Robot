# RT-Infer 多模型周期调度

2026.11.05 candidate10 提供 PFP 状态复用、参数准备优化、PEAR 持久 I/O、全核独占及周期准入。此版本已完成两块开发板的数值与周期运行验证，但仍有超期和后台任务执行不足，不提供硬实时或生产稳定性保证。

## 安装与兼容性

硬件、内核、CANN、SVM 和工具依赖沿用[部署教程](rt_infer_deployment.md)。新增桥接库为 aarch64 ELF，需要匹配的 CANN 和 C++ 运行库。不要覆盖系统 CANN，也不要将新运行库与旧调度器混用。旧版本客户端停止后正常重启，再安装新目录。

```bash
source .shrc_local
BUNDLE="$PWD/third_party/vendor/rt-infer/2026.11.05"
export RTINFER_PREFIX=/opt/rt-infer/2026.11.05
python3 "$BUNDLE/scripts/priorityctl.py" verify-bundle --bundle "$BUNDLE"
python3 "$BUNDLE/scripts/priorityctl.py" install --bundle "$BUNDLE" --prefix "$RTINFER_PREFIX"
python3 "$RTINFER_PREFIX/scripts/priorityctl.py" --prefix "$RTINFER_PREFIX" doctor
ldd "$RTINFER_PREFIX/lib/persistent_acl_bridge.so"
ldd "$RTINFER_PREFIX/lib/periodic_control_bridge.so"
```

若 doctor 报告不兼容或 ldd 出现 `not found`，先解决依赖，不得强行启动。CPU 0、10、11、12、13、15 必须可用。绑核不等于启动隔离；入口保存当前隔离配置，不会刷写系统或自动修改全局依赖。实测 AI Core 为 1080 MHz，其他频率不能直接套用结果。

## 周期并行复现

模型与参考文件仍是外部前提，使用部署教程中的七模型 JSON 映射和参考目录。需要主动安装兼容的 NumPy 与 ais_bench；优先使用现有项目环境，不覆盖其他后端依赖。

```bash
export RTINFER_MODELS=/absolute/path/to/models.json
export RTINFER_REFERENCES=/absolute/path/to/references
python3 scripts/rt_infer/periodic/run.py --output /absolute/path/to/new-periodic-results
```

入口先检查资产、部署包和设备占用，然后运行 PEAR 单独 30 秒及多模型 90 秒，均去除前 5 秒。输出目录必须不存在。它启动并停止本次调度器、在退出时解除周期准入；停止不恢复原生调度，恢复原生仍需正常重启并执行原生数值验证。失败时保留输出和状态，先检查本次进程是否退出，不要并行启动第二套测试。

固定配置：PEAR 18 Hz、55.56 ms 截止期、优先级 0、CPU13；YOLOX 1 Hz/CPU10；TTS 每 7.5 秒执行 1 TE+3 FM/CPU12；VL 每 2.5 秒执行 1 vision+1 prefill+15 decode/CPU11。后台优先级 3，调度器 CPU15。启用 prefetch、pfp-exclusive、pfp-batch-select、local-pool、pfp-min-cache，最高优先级为 0，准入预算 38 ms；启用 `RT_INFER_SAFE_ARGS_BATCH=1` 和 `RT_INFER_PERSISTENT_IO=1`。

周期控制通过独立映射发布，不能仅复制启动参数却不运行发布端。持久 I/O 桥仅用于固定 PEAR 模型、非重入；输出视图在下一次调用后会被覆盖。主循环仍逐次读取当前输入。核心及桥接实现源码不随本仓库发布。

## 已测结果（独立周期入口）

本节记录 `2026.11.05` 独立周期入口的历史数据。
这些数据不代表默认 launch 集成的性能。

| 同事板测试 | 平均时延 ms | 全释放 P99 ms | 超期或丢帧 |
|---|---:|---:|---:|
| 多模型第一轮，85 秒有效窗口 | 36.61 | 45.84 | 10/1530 |
| 多模型第二轮，85 秒有效窗口 | 40.67 | 44.87 | 9/1530 |

平均值只统计完成帧；P99 从计划释放到同步输出返回，nearest-rank，丢帧视为无穷大。两轮共 19/3060 次违约。TTS 完成数分别为 10/11、1/11，VL 为 17/34、19/34；低 P99 不能代替后台服务保证。直连板两轮 P99 为 44.19/44.10 ms 且无 PEAR 违约，但 VL 同样存在丢弃。

原生 C++ 单模型均值/P99 为 21.00/21.10 ms，候选为 32.54/32.85 ms；该统计边界不同于周期响应，不能混算收益。交替三种输入校验和全核独占审计通过；多模型只检查预热、首次、每 100 次及末次输出，不是逐帧数值验证。

原始周期 CSV、计时配置和完成状态保存在 [验证记录](evidence/rt_infer/periodic-20261105/)。可使用 `scripts/rt_infer/periodic/metrics.py` 的 `analyze_case` 重新计算每个 case，丢帧和后台完成数必须一并报告。

`oee_summit_2026` 分支的公开入口实测、迁移修复和原始 CSV 见
[目标分支验收记录](rt_infer_oee_summit_validation.md)。该记录不替代下面的已知限制。

交付中的核心及桥接二进制与板端已测候选逐字节一致。公开入口对原实验脚本进行了路径参数化、预检和格式整理，已完成离线检查，未将其冒充为已重新执行的整套板端验收。CPU 隔离启动持久性尚有问题，换板后的执行尖峰和后台最低服务保障仍待解决。本版本没有新增 PI0.5 时延结论。

## 默认 launch 集成（oee_summit_2026 live 入口）

默认配置选择 `2026.11.06` 控制器修订包。
该包复用 `2026.11.05` 的 7 个二进制文件，SHA256 均相同。
新版本修复可变状态目录的临时文件扫描竞态。
旧版本目录和清单保持不变。

控制器逐层打开状态目录，不跟随符号链接。
控制器校验目录、锁文件的所有者和权限，再取得锁。
控制器只接受明确列出的普通状态文件。
只有状态、sidecar 和客户端 token 的确切临时名允许在扫描时消失。
`state.json` 消失、锁 inode 替换或目录身份变化均导致失败。
不可变部署包继续使用原有的完整校验。

峰会 live 入口（`aimdk_x2_magicmirror_live*`，经 `aimdk_x2_magicmirror_board.yaml` 继承）在
`rt_infer.enable: true` 时由统一 launch 默认启用软调度，无需手动执行任何 CLI：

* **配置**：`rt_infer` 解析为 `robot_config.rt_infer_config.RtInferConfig`。
  `bundle_version` 决定安装前缀 `/opt/rt-infer/<version>`。
  角色档案与本文的候选 profile 一致。
  PEAR 使用 p0/c13，YOLOX/TTS/VL 使用 p3，调度器使用 c15。
  PEAR 周期准入采用 18 Hz/38 ms。
  解析是纯函数：不在配置阶段读文件或探测主机，非 aarch64 主机也能解析配置。
* **launch owner**（`robot_config.launch_builders.rt_infer_scheduler`，builtin 注册，先于
  perception_models/visual_interaction/voice_tts）：校验本次配置的模型角色，取得独占锁。
  owner 校验部署包和 SVM ABI，再安装部署包。安装入口来自源码或已安装 share 中的部署包。
  安装目标为 `/opt/rt-infer/<version>`。父目录缺失时，owner 创建父目录。
  随后执行 doctor → 创建未 arm 的 periodic mapping → vendor `start` → 轮询 vendor 状态确认
  **存活**（pid+start_ticks 对 /proc 校验，死进程的旧 running 文件不算 ready）。
  任何一步失败即 Shutdown 整个 launch，不静默回退 native。
  仿真/benchmark 目标在执行任何 CLI 前失败。
  owner 拒绝接管已经运行的外部 scheduler。启动成功后由
  `rt_infer_scheduler_supervisor` 进程固定校验本次 daemon 的 pid/start_ticks；调度器死亡时
  退出并关闭依赖模型进程。
* **客户端接入**：入口为 `inference_service.rt_infer_client` console script。
  接入的进程包括 PEAR/YOLOX model_service、ZipVoice TTS host 和 InternVL VL host。
  各自的 builder 用 `--role` 包装为 vendor `priorityctl run` 客户端。
  子进程用 importlib 解析真实 node main，并透传全部 ROS 参数。
  无角色映射的 Ascend 模型在 rt_infer 启用时是 launch 错误，不允许
  在被接管的设备上混跑 native。正常 shutdown 时 owner 只对 launch token、pid 和
  start_ticks 均匹配的客户端发信号。关闭任务在线程中运行，launch 继续分发退出事件。
  owner 持有独占锁，直到关闭任务结束。
  owner 先校验本次 mapping 身份，再 disarm 周期预约，让背景任务排空。
  客户端排空后，owner 才调用 vendor `stop`。
  存在外部客户端、身份无法验证或排空超时时，关闭任务返回失败并保留状态。
  清理使用 pidfd，校验和发送信号期间保持文件描述符打开。
  客户端退出后，owner 才删除本次 token 记录。
  owner 也等待客户端控制器清理登记记录并退出。
  vendor `stop` 返回 0 后，owner 仍须确认实际状态为 `stopped`。
  如果 `start` 失败或超时，owner 不推断调度器归属。
  owner 释放自己的锁，保留现场并要求人工检查。
  owner 不据此停止其他调度器或修改共享状态。
* **周期准入**（`inference_service.rt_infer_periodic`）：由真实执行路径驱动——
  owner 为每次 scheduler 会话创建未 arm 的 mapping。
  每个启用角色完成真实暖机后，上报本次进程的 ready。
  Ascend session 要求各已加载模型完成 3 次执行。VL 要求完成 3 次真实观察。
  owner coordinator 只等待本次有效配置中的角色，并校验 token、boot 和进程身份。
  全部角色 ready 后，coordinator arm 一次，epoch 设为当前时刻后 1 个周期。
  初始化阶段仍使用 soft runtime，但没有周期预约，也不等待 arm。
  如果某个启用角色尚未执行，周期预约保持未启用，软件优先级调度继续生效。
  sidecar 记录 boot id、prefix、owner、daemon identity、token、周期和预算。
  PEAR 客户端只 open 和发布，不执行 arm。客户端在设备执行前发布请求元数据中的
  `rt_infer_release_monotonic_ns`；元数据缺失时发布进入执行路径的单调时钟值。客户端不等待
  人工网格、不修改 deadline/cancellation，关闭时只关闭自己的 handle，不 disarm owner 的
  mapping。最终关闭时，owner disarm；成功停止 scheduler 后删除 mapping。
  ABI 接受未来、非网格对齐的单调 release，也接受不同进程顺序发布。
  disarm 不清除 period；同一 mapping 不能再次 arm。
  如果本次 launch 没有本地 PEAR 服务，owner 不创建周期预留。
  live 流的 `max_pear_hz` 与配置的 18 Hz 一致。
  arm 后，PEAR 休眠期间仍保留周期预约，背景任务可能饥饿。
  PEAR 恢复时发布真实请求时间，不伪造空闲周期的完成记录。
  客户端关闭不会 disarm；重新打开客户端仍使用本次 owner 会话。
* **I/O 行为**：默认集成继续使用仓库 `AclModel` 的 dataset/device-buffer 复用；每次请求仍
  执行必要的 H2D/D2H。launch 未接入 vendor `persistent_acl_bridge.so`，也不设置一个没有消费
  方的“持久 I/O”环境开关。
* **验证口径**：默认路径使用同一固定 GO 网格。
  驱动逐个记录 offered release 的 complete/drop 和 release-to-finish 时延。
  驱动还记录后台链完成数和 scheduler 周期计数。旧的单轮
  `pear_rate=0` 结果不满足该口径，不能作为默认集成收益结论。软调度启动过一次的 boot，
  恢复 native 仍需重启。

默认接入的验收驱动是 `scripts/rt_infer/verify_default_integration.py`。
驱动只启动模型，不启动相机、音频播放或机器人控制图。
每个角色使用一个长生命周期进程。PEAR 和 YOLOX 经 `ModelRuntimeHandle` 与 `ModelSession` 执行。
TTS 采用生产 ZipVoice 的角色调用及输出 reshape，按 TE+3FM 顺序执行。
VL 使用现有 ais_bench API，按 vision+prefill+15decode 顺序执行。

这是固定输入模型负载，不包含业务预处理、语音声码器或文本生成质量验收。
每轮记录计划 release、完成/丢弃、服务时间和响应时间。
统计只计入共同窗口内完成的后台链，窗口后的排空不计入后台进展。
所有运行、输入变体、模块来源和数值检查必须与同次 CSV 一起保留。

新版本的默认接入、三轮实测、后台代价和恢复证据见
[默认集成验证记录](rt_infer_default_integration_validation.md)。
