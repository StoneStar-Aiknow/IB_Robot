# dataplane 调度优先级（F3）

第 0 步的数据显示，按住主要发生在 dataplane 主线程排队等 CPU 的时候。这个脚本只改**正在运行的 dataplane 进程里每个线程**的 nice 值，CPU 亲和性可选。不改内核、sysctl 和网络，launch 那一行也不动。

**默认关。** 在 .5 上要不要开，由 zll 决定。

## 接进启动脚本（一行，加在 ros2 launch 之前）

```bash
python3 <树>/scripts/x2_dataplane_sched/x2_dataplane_nice.py hook || true
```

- **没设 `X2_DATAPLANE_NICE=1`**：hook 只做两件事，一是停掉上次留下的监视进程，二是把上次改过、而且还在运行的 dataplane 改回原值。之后的行为和没有这一行完全一样。
- **设了 `X2_DATAPLANE_NICE=1`**：hook 会在后台起一个监视进程。它等 dataplane 起来以后，把它的所有线程改成 `X2_DATAPLANE_NICE_VALUE`（默认 -5）。
  - 后面新起的线程，或者 dataplane 重启，都会照样改。
  - dataplane 退出 30 s 后，监视进程自己结束。
  - 日志在 `/run/x2-dataplane-nice/watch.log`。
- 要同时绑核，就设 `X2_DATAPLANE_CPUS=4-7`（默认不绑）。

## 现场手动 A/B（同一个人、同一组动作，只改 nice）

```bash
python3 x2_dataplane_nice.py status            # 看各线程的 nice/CPU，以及有没有监视进程
python3 x2_dataplane_nice.py apply --nice -5   # 立即生效，所有线程；改之前先保存原值
python3 x2_dataplane_nice.py restore           # 一条命令还原：停监视进程，按保存的原值改回去
```

## 核对是否生效

- 从进程内部看：status 话题里的 `tick_timing.tick_nice`，就是命令线程读回来的 nice 值。
- 从外部看：`status` 子命令。

## 为什么整个进程用同一个 nice

命令 tick 和接收反馈的线程共用同一把锁。如果只给 tick 线程提优先级，它可能反过来在锁上等一个饿着的接收线程。

另外，Linux 的 nice 是按线程算的：`renice -p PID` 只改主线程，所以脚本会逐个改 `/proc/PID/task/*` 下的每个线程。

## 测试

```bash
python3 -m pytest -q scripts/x2_dataplane_sched/test_x2_dataplane_nice.py
```

测试用的是一个假 dataplane：一个带几个线程的 python 进程，argv[0] 用每个测试自己的路径，所以不会碰到本机真正在跑的 dataplane。

普通用户只能把 nice 调高，不能调低。所以测试里用正的 nice；还原 nice 这一步在非 root 下会失败，测试检查的是它明确报错、并且保留原值。
