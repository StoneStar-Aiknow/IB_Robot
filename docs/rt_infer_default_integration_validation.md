# RT-Infer 默认集成验证记录（2026-11-06）

默认 live 配置选择 `2026.11.06` 控制器修订包。
隔离模型图完成 3 轮原生对照和 3 轮软调度。
软调度每轮的 5 个进程均退出 0，vendor 状态均为 `stopped`。
这些结果证明默认接入和退出路径已实际执行。
结果不提供硬实时或后台公平性保证。

PEAR 三轮全释放 P99 为 45.39、46.17、46.94 ms。
三轮没有 PEAR 丢帧，但仍有 3、2、2 次超期。
TTS 和 VL 的后台进展明显下降，详见下表。
本次没有运行机器人全图、动作、相机或音频播放。

## 版本与执行范围

- 工作分支：`port/pr581-oee-summit-2026`。
- 测量基准 HEAD：`790e909540c770789017d5bf650332deb535d08b`，另有本次源码差异。
- 默认配置：`aimdk_x2_magicmirror_board.yaml` 及继承它的峰会 live 配置。
- 控制器：`2026.11.06-candidate10-controlfix`。
- 测量 boot：`8f445c1a-fa03-4c80-b0c5-fc065ac186c4`。
- 环境：Ascend310P1 RC、CANN 8.1.RC1、内核 `5.10.0-openeuler`。
- ROS 隔离：`ROS_DOMAIN_ID=187`、`ROS_LOCALHOST_ONLY=1`。

新包复用旧包的 7 个公开二进制文件，SHA256 均相同。
新包拥有独立清单、校验和及安装前缀。
`2026.11.05` 旧包保持不变。
默认 owner 自动安装首次不存在的 `/opt/rt-infer/2026.11.06`。
源码、installed share 和安装前缀的 15 个文件哈希均一致。
本次没有引入私有核心源码。

PEAR 和 YOLOX 使用真实 `ModelRuntimeHandle` 与 `ModelSession`。
TTS 使用生产 ZipVoice 的角色调用及输出 reshape。
VL 使用生产适配器采用的 ais_bench API。
这是等价的固定输入模型链，不包括业务预处理、声码器或语言质量验收。
模型产物哈希与实际 manifest 解析路径均已保存。

## 测量口径

每轮保留 4 个长生命周期角色进程。
PEAR 为 18 Hz，YOLOX 为 1 Hz。
TTS 每 7.5 s 顺序执行 TE + 3 FM。
VL 每 2.5 s 顺序执行 b4 vision + prefill + 15 decode。
禁止将 b4 数据与 b2 数据混用。

每个模型先完成 3 次真实预热执行。
全部角色 ready 后，owner 才 arm 周期映射。
正式阶段采用共同 GO 和固定绝对 release。
每个角色只保留一个在途请求，错过的 release 记录为 drop。
每轮持续 95 s，前 5 s 不纳入统计，有效窗口为 90 s。
原生对照先于同 boot 的所有软调度轮次。

全释放时延从计划 release 计到同步输出返回。
P99 使用 nearest-rank，drop 和 incomplete 计为无穷大。
PEAR 截止期为一个 18 Hz 周期，约 55.56 ms。
违约数包括超期和丢帧。

后台进展同时报告两种口径：

- **窗内到达且完成**：release 和 finish 都在共同窗口内。
- **窗内全部完成**：finish 在窗口内，包括窗口开始前已在途的请求。

最长无完成间隔按第二种口径计算，并计入窗口两端。
窗口结束后的排空不计入任何窗内完成量。

## PEAR 结果

| 模式 | 轮次 | 计划释放 | 丢帧 | 超期或丢帧 | 全释放 P99 / ms | 完成帧 P99 / ms |
|---|---:|---:|---:|---:|---:|---:|
| 原生全负载 | 1 | 1620 | 266 | 557 | ∞ | 229.822 |
| 原生全负载 | 2 | 1620 | 272 | 559 | ∞ | 229.562 |
| 原生全负载 | 3 | 1620 | 264 | 561 | ∞ | 229.685 |
| 默认软调度 | 1 | 1620 | 0 | 3 | 45.390 | 45.390 |
| 默认软调度 | 2 | 1620 | 0 | 2 | 46.167 | 46.167 |
| 默认软调度 | 3 | 1620 | 0 | 2 | 46.941 | 46.941 |

单独 PEAR 原生基线的 P99 为 23.825 ms。
该基线完成 1620 次释放，没有超期或丢帧。
软调度三轮最大响应分别为 76.253、73.595、85.749 ms。
因此不能将 P99 解读为硬性 45 ms 上限。

## 后台进展

| 角色 | 轮次 | 窗内计划释放 | 窗内到达且完成 | 窗内全部完成 | 最长无完成间隔 / s |
|---|---:|---:|---:|---:|---:|
| TTS | 1 | 12 | 0 | 1 | 88.237 |
| TTS | 2 | 12 | 3 | 4 | 65.627 |
| TTS | 3 | 12 | 8 | 9 | 28.058 |
| VL | 1 | 36 | 17 | 18 | 5.340 |
| VL | 2 | 36 | 16 | 17 | 6.889 |
| VL | 3 | 36 | 14 | 15 | 7.112 |

第一轮 TTS 的 12 个窗内新请求，没有一个在窗内完成。
但窗口内完成了一个窗前已在途的请求。
因此不能写成“整个 90 s 没有整句完成”。
TTS 和 VL 三轮的全释放 P99 均为无穷大。

YOLOX 每轮完成 90/90 个窗内请求。
三轮 P99 分别为 497.402、501.982、499.306 ms。
原生全负载中，TTS 每轮完成 12/12，VL 每轮完成 36/36。
PEAR 优先级收益不能替代后台服务保证。

## 退出与数值验证

每轮 4 个客户端控制器及 supervisor 均退出 0。
每轮均具备模型完成标记和通过的数值检查。
owner 在窗口结束后 disarm，再等待客户端和控制器退出。
owner 最后执行 vendor stop，并核对实际状态。

| 轮次 | shutdown / s | kernel starts | kernel completes | unknown scans |
|---|---:|---:|---:|---:|
| 1 | 7.547 | 493734 | 493734 | 0 |
| 2 | 7.566 | 496818 | 496818 | 0 |
| 3 | 7.667 | 505145 | 505145 | 0 |

这些 kernel 计数覆盖整个会话，包括预热和最终排空。
它们不是共同窗口的后台完成数。
每轮的 release、句柄关闭、客户端返回和 owner 事件均已保存。

数值验证交替使用 3 组输入。
验证覆盖首次、每 100 次及末次输出。
FM 只比较有效的 153700 个 float。
已检查输入变体能够产生不同输出。
已检查 ACL dataset、设备 buffer 和主机输出 buffer 复用。
旧输出不会因下一次执行而改变。

默认集成没有接入 `persistent_acl_bridge.so`。
每次执行仍有 H2D/D2H 和输出复制。
本次数据不替代旧 C++ 独立桥接路径的测量结果。
各项额外开销尚未单独量化。

## 测试与失败记录

最终定向正式入口为 `colcon test`。
`robot_config` 83 项、`inference_service` 31 项、`visual_interaction` 1 项通过。
合计 115 项，XML 与完整命令已归档。
4 个受影响包均通过 `./scripts/build.sh` 构建。
31 个新增或修改的 Python 文件通过 Ruff 检查。

这不是“整个仓库测试全部通过”的结论。
旧全量对照中的失败 ID 与干净 HEAD 一致。
导航共同 fixture 的挂点也在干净 HEAD 复现。
4 个导航参数用例的排除项单独记录。
最终信号、ownership 和状态竞态改动由上述定向正式测试覆盖。

保留的失败记录包括：

- 固定未来 epoch 导致预热等待，后改为真实 readiness 驱动 arm。
- 客户端父进程退出 130，后修复信号处理及控制器排空顺序。
- 未确认归属的 start 失败可能误 stop，后改为保留现场并返回错误。
- `final-v2` 第三轮因 `state.tmp` 扫描竞态失败，没有 GO 或 CSV。

`final-v2` 前两轮成功不能补作新版本的第三轮。
新 `2026.11.06` 修订修复了可变状态扫描。
真实双进程重命名与状态安全负例均有测试。
不可变包校验没有放宽，旧包也没有原地改写。

## 最终恢复

最终恢复 boot 为 `41744d9c-7845-470f-9d30-4297a5294a95`。
PEAR、TE、FM、vision 各 10 次原生数值检查通过。
实际映射只包含官方 `libruntime.so`。
doctor 显示无 RT 模块、无 RT-Infer IPC、无计算客户端。
全局状态为 `not-started-this-boot`。

官方运行库 SHA256 保持为：

```text
94029220e9849e4c479c34f7ebf7441c0b80631dfd9301667c28af7f20979e37
```

本次没有修改网络、PTP、CPU 隔离、频率或功耗设置。
本文不声称 PTP 已锁定。
软调度后的原生恢复仍要求正常重启。
生产退出不会自动重启。

## 证据与限制

驱动为 [verify_default_integration.py](../scripts/rt_infer/verify_default_integration.py)。
默认配置和生命周期说明见 [周期调度文档](rt_infer_periodic.md)。

本次现场证据位置：

```text
板端：/data/rtinfer-default-finish-20261106/final-v3
本地：/tmp/rtinfer-default-finish/board-v3/final-v3
源码清单：/tmp/rtinfer-default-finish/source-v3-complete-sha256.txt
复算：/tmp/rtinfer-default-finish/final-v3-audit.json
退出时间轴：/tmp/rtinfer-default-finish/final-v3-lifecycle.json
最终恢复：/tmp/rtinfer-default-finish/native-recovery-final.json
测试对照：/tmp/rtinfer-default-finish/test-baseline-comparison.json
复现命令：/tmp/rtinfer-default-finish/REPRODUCE.md
```

原始归档 `final-v3-evidence.tgz` 的 SHA256 为：

```text
0e473563a7d79f035a51d31350ba27de043721ed9e943c7febc447e123bd9cde
```

独立复算确认了 release 索引、整数时间戳、P99、丢帧和退出状态。
后台两种完成口径与最长无完成间隔也已交叉核对。
旧版“仅按窗内 release 队列计算 silence”的输出另存，未删除。

本次只验证匹配的 310P1 RC 环境和固定模型负载。
未验证 20 Hz、10 分钟持续运行或反向优先级。
旧反向优先级路径的断言限制仍保留。
详见[目标分支验证记录](rt_infer_oee_summit_validation.md)。
后台饥饿和超期风险仍然存在。

## 提交前对齐

测量后，上游峰会分支新增 4 个业务提交。迁移分支已无冲突对齐目标
`a0ae373ea4391e201e59d7f147f2707d827d4440`，保留这些业务改动。
对齐后，四个受影响包重新构建成功，RT-Infer 正式定向测试仍为
83 + 31 + 1，共 115 项通过。
模型执行、验收驱动及新控制包文件哈希与板端快照一致；loader 的上游新增逻辑
未冒充为新的板端性能验收。具体差异和日志见
[对齐验证](evidence/rt_infer/default-20261106/post-rebase-verification.json)。
