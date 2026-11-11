# RT-Infer 多模型周期调度

2026.11.05 candidate10 提供 PFP 状态复用、参数准备优化、PEAR 持久 I/O、全核独占及周期准入。此版本已完成两块开发板的数值与周期运行验证，但仍有超期和后台任务执行不足，不提供硬实时或生产稳定性保证。

## 安装与兼容性

硬件、内核、CANN、SVM 和工具依赖沿用[部署教程](rt_infer_deployment.md)。新增桥接库为 aarch64 ELF，需要匹配的 CANN 和 C++ 运行库。不要覆盖系统 CANN，也不要将新运行库与旧调度器混用。旧版本客户端停止后正常重启，再安装新目录。

```bash
source .shrc_local
BUNDLE="$PWD/third_party/vendor/rt-infer/2026.11.05"
export RTINFER_PREFIX=/opt/rt-infer/2026.11.11
python3 "$BUNDLE/scripts/priorityctl.py" verify-bundle --bundle "$BUNDLE"
python3 "$BUNDLE/scripts/priorityctl.py" install --bundle "$BUNDLE" --prefix "$RTINFER_PREFIX"
python3 "$RTINFER_PREFIX/scripts/priorityctl.py" --prefix "$RTINFER_PREFIX" doctor
ldd "$RTINFER_PREFIX/lib/persistent_acl_bridge.so"
ldd "$RTINFER_PREFIX/lib/periodic_control_bridge.so"
```

若 doctor 报告不兼容或 ldd 出现 `not found`，先解决依赖，不得强行启动。CPU 0、10、11、12、13、15 必须可用。绑核不等于启动隔离；入口保存当前隔离配置，不会刷写系统或自动修改全局依赖。实测 AI Core 为 1080 MHz，其他频率不能直接套用结果。

周期入口在加载驱动前检查当前 Python 是否能找到 NumPy、ais_bench 和 aclruntime。使用项目虚拟环境安装匹配依赖，不能以交互终端中另一个 Python 能导入作为验收。客户端沿用入口的 Python 可执行文件，并显式传入经过配置的模型路径、参考目录和周期映射；控制器仍会清除其他继承的调试变量。

## PFP 的两种模式

普通 PFP 按优先级调度，不要求应用提供周期。使用下面的入口即可启用完整的 PFP 调度配置：

```bash
python3 "$RTINFER_PREFIX/scripts/priorityctl.py" --prefix "$RTINFER_PREFIX" start --scheduler pfp --cpu 15
python3 "$RTINFER_PREFIX/scripts/priorityctl.py" --prefix "$RTINFER_PREFIX" run --priority 0 --cpu 13 -- python3 /absolute/path/to/client.py
python3 "$RTINFER_PREFIX/scripts/priorityctl.py" --prefix "$RTINFER_PREFIX" status
python3 "$RTINFER_PREFIX/scripts/priorityctl.py" --prefix "$RTINFER_PREFIX" stop
```

周期感知 PFP 在普通 PFP 的基础上增加下一次释放前的后台准入控制。应用必须创建并持续更新周期映射；不能仅指定文件名而不运行发布端。

```bash
python3 "$RTINFER_PREFIX/scripts/priorityctl.py" --prefix "$RTINFER_PREFIX" start --scheduler pfp --cpu 15 --pfp-mode periodic --periodic-control /absolute/path/to/release.bin
```

周期任务默认优先级为 0；确需改变时通过 `--high-priority` 指定，并让客户端优先级一致。普通模式不需要该参数。下面的完整复现入口自动管理映射、周期发布与解除，不需要用户手工组织这些步骤。

PFP 默认启用预取、全核独占、批量选择、本地节点池和优先级缓存。它们不再作为普通帮助中的逐项开关。旧显式参数暂时兼容，供已有脚本和消融对照使用；显式参数优先，周期模式拒绝关闭独占。FIFO/NICE 默认行为不变。`status` 展示最终生效配置及周期映射路径。开启全部优化不代表所有模型截止期均有保证，周期模式仍可能牺牲后台进展。

## 周期并行复现入口

模型与参考文件仍是外部前提，使用部署教程中的七模型 JSON 映射和参考目录。需要主动安装兼容的 NumPy 与 ais_bench；优先使用现有项目环境，不覆盖其他后端依赖。

```bash
export RTINFER_MODELS=/absolute/path/to/models.json
export RTINFER_REFERENCES=/absolute/path/to/references
python3 scripts/rt_infer/periodic/run.py --output /absolute/path/to/new-periodic-results
```

入口先检查资产、部署包和设备占用，然后运行 PEAR 单独 30 秒及多模型 90 秒，均去除前 5 秒。输出目录必须不存在。它启动并停止本次调度器、在退出时解除周期准入；停止不恢复原生调度，恢复原生仍需正常重启并执行原生数值验证。失败时保留输出和状态，先检查本次进程是否退出，不要并行启动第二套测试。

失败恢复：正在运行、正在启动或需要恢复的控制器状态会在创建本次输出目录前被拒绝。测试超时先通知 case 退出，由 case 回收自身客户端，再停止调度器。调度器启动超时也会向本次创建的进程请求退出；若未在等待期限内退出，状态中保留清理错误并要求人工检查，不强制卸载驱动或终止其他用户任务。

固定配置：PEAR 18 Hz、55.56 ms 截止期、优先级 0、CPU13；YOLOX 1 Hz/CPU10；TTS 每 7.5 秒执行 1 TE+3 FM/CPU12；VL 每 2.5 秒执行 1 vision+1 prefill+15 decode/CPU11。后台优先级 3，调度器 CPU15。启用 prefetch、pfp-exclusive、pfp-batch-select、local-pool、pfp-min-cache，最高优先级为 0，准入预算 38 ms；启用 `RT_INFER_SAFE_ARGS_BATCH=1` 和 `RT_INFER_PERSISTENT_IO=1`。

周期控制通过独立映射发布，不能仅复制启动参数却不运行发布端。持久 I/O 桥仅用于固定 PEAR 模型、非重入；输出视图在下一次调用后会被覆盖。主循环仍逐次读取当前输入。核心及桥接实现源码不随本仓库发布。

## 已测结果

| 同事板测试 | 平均时延 ms | 全释放 P99 ms | 超期或丢帧 |
|---|---:|---:|---:|
| 多模型第一轮，85 秒有效窗口 | 36.61 | 45.84 | 10/1530 |
| 多模型第二轮，85 秒有效窗口 | 40.67 | 44.87 | 9/1530 |

平均值只统计完成帧；P99 从计划释放到同步输出返回，nearest-rank，丢帧视为无穷大。两轮共 19/3060 次违约。TTS 完成数分别为 10/11、1/11，VL 为 17/34、19/34；低 P99 不能代替后台服务保证。直连板两轮 P99 为 44.19/44.10 ms 且无 PEAR 违约，但 VL 同样存在丢弃。

原生 C++ 单模型均值/P99 为 21.00/21.10 ms，候选为 32.54/32.85 ms；该统计边界不同于周期响应，不能混算收益。交替三种输入校验和全核独占审计通过；多模型只检查预热、首次、每 100 次及末次输出，不是逐帧数值验证。

原始周期 CSV、计时配置和完成状态保存在 [验证记录](evidence/rt_infer/periodic-20261105/)。可使用 `scripts/rt_infer/periodic/metrics.py` 的 `analyze_case` 重新计算每个 case，丢帧和后台完成数必须一并报告。

交付中的核心及桥接二进制与板端已测候选逐字节一致。公开入口对原实验脚本进行了路径参数化、预检和格式整理，已完成离线检查，未将其冒充为已重新执行的整套板端验收。CPU 隔离启动持久性尚有问题，换板后的执行尖峰和后台最低服务保障仍待解决。本版本没有新增 PI0.5 时延结论，也没有接入 ROS launch 配置开关。
