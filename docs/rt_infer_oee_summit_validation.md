# PR #581 在 oee_summit_2026 上的迁移与板端性能验收

本记录验证从 PR #581 迁移至 `oee_summit_2026` 的公开部署和周期测试入口。测试对象是本分支的独立板端 checkout，未调用之前私有目录中的修改版工作负载脚本。结果只覆盖固定合成输入、18 Hz 和正常优先级配置，不构成硬实时或生产稳定性保证。

## 迁移与执行身份

- 目标基线：`e9fc2556757fec59272e5d7f0857493529c22973`。
- 原 PR：[openEuler/IB_Robot#581](https://atomgit.com/openeuler/IB_Robot/pull/581)，原提交 `b1fcf2cc8fd2ac01db37b6d09ba1b9a2d23083ed`、`ed3d000d4ed1f97c3aa97eb5a0350735b81ada71`。
- 板端实际测试 commit：`02141043ec09a046b1c7cfa2720115ca8f60e554`；tree：`79c542004c36769183fc8c0be0be877a1b2195cd`；工作区干净。后续报告提交不改变被测执行代码。脚本 SHA256 见 source-identity.json。
- 环境：Ascend310P1 RC/device0，aarch64，openEuler Embedded，Linux5.10.0-openeuler，CANN8.1.RC1。未修改驱动、锁频、CPU隔离、网络或机器人业务配置。
- 调度器、运行库、桥接库使用本分支 `third_party/vendor/rt-infer/2026.11.05/`，通过包清单和兼容性检查。二进制与此前板端 candidate10 相同，未修复核心算法。
- 模型及参考文件按 `scripts/rt_infer/periodic/assets.json` 校验，vision 为 **b4**。先前 b2 优化结果不计入本次性能结论。

## 迁移时发现并修复的问题

1. `periodic/baseline.py` 只读取模型映射，没有导出 worker 需要的 NumPy 和 InferSession。补齐导入，并在启动调度器前用同一解释器检查 worker 依赖。
2. 控制器有意清理继承环境中的 RTINFER 变量，但公开参数化 worker 依赖 RTINFER_PREFIX/MODELS/REFERENCES。客户端命令现显式传入这三个应用路径，不削弱控制器环境清理规则。

第一次尝试在 worker 初始化时因缺失 RTINFER_PREFIX 退出，未进入正式采样。日志已单独保存；正常停止并重启后从新目录完整重跑，失败尝试不计入下表。旧 PR 文档明确公开入口尚未完整板端验收，本轮补上了目标分支的实际执行证据。

## 方法与结果

原生姿态 30秒/trim5秒、原生七模型90秒/trim5秒各一轮；公开 `run.py --output ...` 连续执行三轮，每轮姿态30秒/trim5秒、多模型90秒/trim5秒。加载和预热不计入窗口。PEAR18Hz（55.56ms），YOLOX1Hz，TTS每7.5秒1TE+3FM，VL每2.5秒vision+prefill+15decode。姿态CPU13、YOLOX CPU10、TTS CPU12、VL CPU11；调度器CPU15。

公开候选按原入口启用 PFP、prefetch、exclusive、batch-select、local-pool、min-cache、38ms周期准入和持久I/O，姿态priority0、后台priority3。原生对照不使用这些软件优化。因此比较代表完整推荐配置的收益，不是单一算法消融。

响应时间从计划释放到同步返回、含主机输出；nearest-rank百分位。单在途latest-frame，丢帧和超期一起计违约；全释放P99将丢帧视为无穷大。后台只统计共同测量窗口内实际完成的整链，不将高优先级停止后的排空冒充并发完成。

|场景|计划帧|丢帧|超期完成|完成帧P99 ms|全释放P99 ms|最大响应ms|违约率|窗口内TTS完成/到达；VL完成/到达|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|native-full|1530|265|285|277.46|无有限值（丢帧进入 P99）|326.57|35.95%|11/11；34/34|
|native-pose|450|0|0|24.05|24.05|27.03|0.00%|—|
|public-r1/alone-periodic|450|0|0|34.22|34.22|44.58|0.00%|—|
|public-r1/full-periodic|1530|1|1|45.09|45.13|120.39|0.13%|11/11；16/34|
|public-r2/alone-periodic|450|0|0|33.77|33.77|36.81|0.00%|—|
|public-r2/full-periodic|1530|0|4|44.57|44.57|73.31|0.26%|1/11；19/34|
|public-r3/alone-periodic|450|0|0|34.46|34.46|42.57|0.00%|—|
|public-r3/full-periodic|1530|0|6|45.37|45.37|84.55|0.39%|1/11；19/34|

每个场景全部客户端退出码0，三个预热、首个/每100次正式完成及最终输出对照通过；FM仅比较有效153700元素。固定输入抽检不代表逐帧/任意输入完整精度验收。额外使用独立程序从整数release/start/finish重算上述统计，并核对每角色实际运行库、优先级与CPU亲和性；原始CSV和摘要见[证据目录](evidence/rt_infer/oee-summit-20261105/independent-audit.json)。

三轮公开多模型测试共 4590 次计划释放，12 次超期或丢帧。PEAR 分位时延改善与后台服务问题同时成立：第二轮 TTS 只在共同窗口内完成 1/11 句，最大无完成间隔约82.6秒。不能因PEAR P99低于55.56ms就宣称所有模型都满足业务需求。

## 如何复现

在本分支加载 `.shrc_local`，使用版本明确的 CANN8.1.RC1 环境和已安装的兼容ais_bench。此板系统Python3.11的ais_bench需要进程内取消PYTHONNOUSERSITE，未安装/覆盖新依赖；先验证import NumPy/InferSession，再启动。

按[周期部署教程](rt_infer_periodic.md)安装本分支的bundle至独立prefix；设置RTINFER_PREFIX、RTINFER_MODELS、RTINFER_REFERENCES后运行：

```bash
python3 scripts/rt_infer/periodic/run.py --output /absolute/path/to/new-results-r1
# 等本轮退出后，以不同output重复；不要并行启动多组。
```

原生对照使用同分支 `latency_periodic10.py --mode native`，必须在本启动周期尚未使用软调度时执行。正常停止后仍需重启才能恢复原生设备状态。不要复制历史`completed.json`或沿用已有output。

## 已知限制与恢复

- candidate10 在先前独立 b2 评估的反向优先级路径触发 `runtime_adaptor.c:76 get_ttwu_pending_stream` 断言；本次正常优先级验证不证明该问题已解决。尚未重新执行反向优先级、20Hz、10分钟持续或其他策略。
- P99只是分位目标，需同时检查超期/丢帧、最大响应和后台完成数。后台未能跟上全部到达时，不能宣称所有模型满足业务速率。
- 本分支未接入ROS launch/ModelSession调度选择，也未切换生产InternVL为b2；不启动机器人、相机或音频播放。
- 两版控制器各27项Linux离线测试通过；本次改动的三个Python文件Ruff与format检查通过。未触及setup、构建系统或ROS依赖，不需要全工作区colcon构建；本轮也未执行双平台Docker。
- 测试后正常stop并重启，从boot `49cab1ff-f85e-40b7-9c58-3c9646ed475e` 恢复至 `03b4a7c7-490f-4c06-a12f-9c04cac8c537`。PEAR、TE、FM、vision各10次原生数值复验通过，无软件模块/IPC/计算客户端，官方CANN runtime指纹未变。PTP和网络状态只读记录在完整归档中；不把服务active当作时钟锁定证明。
- 完整板端证据：`/data/pr581-oee-summit-perf-20261105-r2/`；首次失败：`/data/pr581-oee-summit-perf-20261105/`。原始日志、控制器状态、执行命令、源码身份和恢复记录另行完整归档，仓库保留可复算CSV与关键JSON。

入库 CSV 仅将 CRLF 换行规范为 LF，字段和时间戳原值不变；未经换行处理的原始文件保留在完整板端归档中。
