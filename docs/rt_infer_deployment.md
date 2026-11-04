# RT-Infer 软件调度部署

RT-Infer 为 Ascend 310P1 RC 提供软件控制的 AI Core 任务调度，可在多个推理进程间选择 FIFO、NICE 或 PFP 策略。NICE 优先选择待下发 block；PFP 限制低优先级模型的新 kernel 准入，已经运行的 block 不会被暂停和恢复。

本集成是可选入口，不修改 IB-Robot 的默认推理后端、全局依赖或系统 CANN。核心实现仅以二进制交付。

## 环境与安装

已验证环境：aarch64、Ascend 310P1 RC、openEuler Embedded BUILD_ID 20251020093351、内核 5.10.0-openeuler、CANN 8.1.RC1。其他环境需要独立适配，不应强制加载驱动。

控制工具需要 root、Python 3.10+、Linux pidfd、taskset、modinfo、insmod、ipcrm、ldd 和 libatomic。模型实验另需匹配的 ais_bench 与 NumPy；不通过本入口修改其他后端的依赖版本。

从仓库根目录执行：

```bash
source .shrc_local
BUNDLE="$PWD/third_party/vendor/rt-infer/2026.11.04"
PREFIX=/opt/rt-infer/2026.11.04
python3 "$BUNDLE/svm_bind_probe.py" --pid self --pasid -1
python3 "$BUNDLE/scripts/priorityctl.py" verify-bundle --bundle "$BUNDLE"
python3 "$BUNDLE/scripts/priorityctl.py" install --bundle "$BUNDLE" --prefix "$PREFIX"
CTL="$PREFIX/scripts/priorityctl.py"
python3 "$CTL" --prefix "$PREFIX" doctor
```

SVM 探针必须输出 `valid=true`；仅 ioctl 返回成功或原生推理成功不够。失败时按包内 [SVM 步骤](../third_party/vendor/rt-infer/2026.11.04/SVM.md)核对内核 ABI、备份、替换模块并重启。安装不自动替换基础驱动、固件或系统 CANN。

源码版本 d8768b5 的共享内存布局与早期 PI0.5 包不同，scheduler/runtime 必须成套升级。退出旧客户端并正常停止旧调度器后，重启再启用新版本，不能仅替换运行库。

## 启动、运行与停止

```bash
python3 "$CTL" --prefix "$PREFIX" start --scheduler pfp --prefetch off --cpu 14
python3 "$CTL" --prefix "$PREFIX" status
python3 "$CTL" --prefix "$PREFIX" run --priority 1 --cpu 13 -- python3 /absolute/path/to/client.py
python3 "$CTL" --prefix "$PREFIX" stop
```

策略可选 `fifo`、`nice`、`pfp`，优先级范围 0–7，数值越小优先级越高。进程内的流使用同一覆盖优先级。CPU 编号必须属于当前设备允许的集合。多个客户端分别通过 `run` 接入同一安装版本。

正常停止后可再次启动同版本软调度，亦可切换策略。恢复原生硬件调度必须退出客户端、正常停止、保存记录并重启，再执行原生数值检查。`stop` 不等于恢复硬调度。

## 七模型并行复现

模型资产不随本 PR 发布。准备 JSON 映射，键为 `pear`、`yolox`、`te`、`fm`、`vision`、`prefill`、`decode`，值为对应 OM 的绝对路径。每个模型还需参考目录 `<name>/inputs.npz`（i0、i1…）和 `outputs.npz`（o0、o1…）。模型及参考必须来自同一固定配置，不能以任意同名模型替代。

```bash
export RTINFER_PREFIX="$PREFIX"
export RTINFER_MODELS=/absolute/path/to/models.json
export RTINFER_REFERENCES=/absolute/path/to/references
export RTINFER_RESULTS=/absolute/path/to/new-results
python3 "$CTL" --prefix "$PREFIX" start --scheduler fifo --prefetch off --cpu 14
python3 scripts/rt_infer/benchmark_multimodel.py --strategy fifo --label fifo-01
python3 "$CTL" --prefix "$PREFIX" stop
```

重复 NICE/PFP 时同步修改启动策略、`--strategy` 和唯一 label。当前复现入口固定客户端 CPU 13/12/11、调度器 CPU 14；姿态优先级 1，TTS/VL 优先级 3。模型预热后等待共同释放点再执行组合，逐调用比较原生参考，FM 仅检查有效 153700 个元素。

每轮调用数为 PEAR 36、YOLOX 3、text_encoder 1、FM 3、vision 1、prefill 1、decode 15。三个进程中姿态含两线程。正式计时不包含数值比较，但包含同步完成；固定合成输入不等于真实机器人全链路。

## 验证与边界

2026-11-04 的 d8768b5 重建版本在另一块直连开发板上完成 FIFO/NICE/PFP 各 60 次正式推理，共 180 次，逐调用数值比较全部通过，九个客户端进程退出码均为 0。结束后重启，PEAR 原生 8 次数值检查通过，无软调度模块、共享内存和计算客户端残留。

[汇总](evidence/rt_infer/multimodel-20261104/summary.json)与各策略目录保留逐调用记录。
本次没有验收 18 Hz 截止期、P99 或长时间稳定性，也未重新测量 PI0.5；此前约 179 ms 不属于本版本性能声明。仅 clone 仓库不能取得上述外部模型及参考资产。
