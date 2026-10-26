# 2026-10-15 JPEG 解码 30 FPS 复核

验证基于主线 `9fd440d4` 加本次共享 JPEG 适配增量；提交和 PR 只携带源码、测试及验证说明，
不提交原始相机图像、模型或凭证。
使用 310P 上的前中相机半分辨率 JPEG，1344×972，实际 YOLOX/PEAR 服务同时运行。
测试仅向 `/decode_rate_verify/` 私有话题输出，不启动运动控制，也不改动机器人原发布者。

## 相同输入的新旧解码对照

旧原型取自 `6dd388e3`，仅作为临时 benchmark fixture，不加入生产依赖或提交历史。
两版均使用当前主线 ACL runtime、同一张保存 JPEG，以旧→新→新→旧交替执行。
每轮预热 10 次、计时 150 次，每个版本共 300 次计时；冷启动不混入稳态平均值。

- 旧版平均：6.133 ms；新版平均：6.386 ms。
- 新版比旧版增加约 0.254 ms（4.14%），不声称单帧耗时完全无回退。
- 两版输出 NV12 字节 SHA256 一致，CPU 回退均为 0。

## 实际链路与受控输入

模型、解码器和业务配置保持不变，按额外相机订阅关闭→开启→关闭做对照。
每个真实相机阶段运行 22 秒，去除头两个状态窗口后，各统计 9 个窗口（约 18 秒）。
另外对同张 JPEG 以新时间戳做受控 30 Hz 回放，测量阶段运行 32 秒，统计 14 个窗口；
回放只证明解码处理能力，不冒充真实相机的新画面，因此单列报告。

| 条件 | 实收 FPS | 解码 FPS | 稳态新增解码前淘汰 | 新增解码失败 |
|---|---:|---:|---:|---:|
| 真实相机，仅业务订阅 | 30.0000 | 30.0000 | 0 | 0 |
| 真实相机，加额外计时订阅 | 27.9444 | 27.9444 | 0 | 0 |
| 真实相机，撤销额外订阅 | 29.9989 | 29.9989 | 0 | 0 |
| 同张 JPEG 受控 30 Hz 回放 | 29.9993 | 29.9993 | 0 | 0 |

所有阶段 CPU 回退为 0；两个实验的三个子进程均正常退出，没有异常退出。
冷启动期间确有压缩队列淘汰，表中是稳态增量，不声称全程零丢帧。
`frames_dropped` 是下游模型队列的淘汰，不能当作解码丢帧；本次看的是 `compressed_dropped` 增量。

## 对之前约 28 FPS 的更正

额外订阅者开启后，输入与解码同步降到约 27.94 FPS，移除后恢复名义 30 FPS。
这证明新增计时订阅会扰动该现场链路的吞吐，不能把带探针的接收速率当作解码器容量。
发送端、网络/DDS 和接收侧的具体负担尚未分别计量，不把影响武断归到某个硬件模块。
后续吞吐验收只用业务节点自身计数；需要额外图像订阅的时延采样应单独运行、注明测量负担。

本次仅确认解码达到名义 30 FPS。PEAR 最终输出仍约 18 FPS，不宣称整条模型链路达到 30 FPS。

原始 JSON、日志和复现脚本在本机及板端 `/tmp/x2-mainline-jpeg-20261015/`：
`decode-compare.json`、`decode-rate-live_aba.json`、`decode-rate-replay30.json`、
`decode_compare.py`、`decode_rate.launch.py`、`decode_rate_probe.py`。
脚本运行前须在板端工作树加载 `.shrc_local`，使用正确的 ROS domain/RMW。

## 架构与历史依赖

JPEG 后端位于 `observation_transport`，实现主线 `VideoDecoder`、`EncodedPacket`、
`VideoFrame`、metrics/lifecycle 契约，由消费者注入主线 `AclRuntimeManager` lease。
注册表新增可选 `codec` 过滤参数，默认仍为 H.264；显式 JPEG 注册不会改变原 H.264 后端选择。
主线已有 H.264 硬解不能接受 JPEG 字节，本次补齐的是 JPEG 后端及 ROS 消息适配。

不包含旧 `inference_service/backends/ascend/jpeg_decoder.py`、半分辨率 relay、relay launch
或派生话题配置。默认直接使用已有前中半流，缺流不自动拉起发送端。
提交直接以主线为父，不以 `6dd388e3` 或 `0d803f0b` 为父。
保留前期 `f0bb133e`（作者 yidahao）和 `0d803f0b` 中适用的队列、NV12 适配思路及测试，
如实注明来源，不把这些贡献改称全新独立实现，也不复制其他人的 DCO sign-off。

## 正式回归与退出验证

在 310P/openEuler 隔离工作树通过项目 `./scripts/build.sh` 构建相关包，正式测试入口为
`python3 -m colcon test --merge-install --return-code-on-test-failure`。
测试使用 ROS domain 220、`ROS_LOCALHOST_ONLY=1`，现场测试单独使用部署 domain 0 和私有输出。

| 包及范围 | 通过 | 失败 | 错误 | 跳过 |
|---|---:|---:|---:|---:|
| observation_transport 全包 | 25 | 0 | 0 | 0 |
| manipulation_execution 全包（含输入代次修复） | 355 | 0 | 0 | 0 |
| robot_config：stream builder | 27 | 0 | 0 | 0 |
| perception_service：NV12、HRI adapters、plugins | 55 | 0 | 0 | 0 |
| inference_service：共享 codec、ACL、routing 定向集 | 55 | 0 | 0 | 3 |
| 合计（不包含以下已知环境失败） | 517 | 0 | 0 | 3 |

扩大 H.264 FFmpeg 回归时发现 6 个失败实例：临时目录中的 executable 探测/codec 探测，
以及显式 prefix 下 `LD_LIBRARY_PATH` 只应包含测试 lib 的环境假设。
用未修改的主线测试独立复测同样得到 6 项失败；板端实查临时文件权限为 0755、
`os.access(path, os.X_OK)` 仍为 False，且板端存在测试未隔离的 CANN/driver 库目录。
以上 6 个实例在定向通过集里显式排除，原测试和主线 FFmpeg 实现均未修改；
不宣称 H.264 全量回归全绿。基线复测的 8 个实例中 2 项通过、6 项失败。

证据：`test-results.json`、各包 `*-results.xml`、`baseline-ffmpeg-results.xml`、
`inference-expanded-results.xml`，位于同一验证目录。
源码按仓库固定 Ruff 0.11.8 pre-commit 与 Gitlint 检查，正文保留真实人工审查状态。

## 当前性能边界

真实模型服务同跑时，当前新后端最后一个 2 秒窗口的平均解码耗时约 9 ms，
比独立 decoder 微基准更高，不把两种条件的耗时直接比较。
单独计时订阅的另一轮实测匹配了 547 个输入 header 与 PEAR 输出，接收到达→PEAR
本地时延 P50 88.87 ms、P95 106.17 ms；额外订阅影响吞吐，因此不能将该轮约 28 FPS
与表中不挂计时订阅的 30 FPS 当作同一条件。
到达→PEAR 使用接收端单调时钟，不包含相机到接收端的网络时延，也不是跨机 RTT。

## 输入恢复与并发审查修复

重新审查发现并复现三个交错场景：发布者冲突已恢复但旧 RPC 响应重新填入缓存；
PEAR 已取出的旧尺寸图像使用新尺寸的检测框；相机重启时间戳归零后仍使用重启前的检测。
现以 `source_generation` 将压缩队列、解码后队列、检测缓存与在途 YOLOX/PEAR 结果关联到同一代次。
检测到输入不可用、超过恢复间隔重新收帧或图像尺寸变化时递增代次并清空等待槽和检测缓存。
结果写回、ActorState/PEAR 发布与代次校验使用同一状态锁；锁不跨 RPC 等待。

新增 13 个参数化回归实例，覆盖缺失/冲突期间的 YOLOX 和 PEAR 响应、尺寸切换及切回原尺寸、
已取出的旧帧、时间戳重启、解码中失效、全部队列清空、限频重置，以及新检测/新帧恢复发布。
正常 steady state 不改变原 latest-only 和 JPEG/NV12 路径；未被 DDS 图或回调观测到的瞬态不作保证。

本地 AI 元数据审计原先不能识别由自身模板生成的 `**模型信息 (Model)**:`，误报模型缺失。
修复标签的 Markdown 加粗与中英文括号解析，同时保留模型缺失、供应商前缀与无版本工具的拒绝。
相关离线工具测试 26 项通过，独立于上述 ROS 包正式 colcon 统计；不以本地审计通过代替平台检查。
修复证据目录为板端及本机 `/tmp/x2-epoch-fix-20261015/`。

最终修复版本再次执行 30 Hz 保存 JPEG 回放，真实 YOLOX/PEAR 同时运行，14 个稳态窗口平均输入和
解码均为 29.9993 FPS，平均解码 8.98 ms，新增解码前淘汰/解码失败/CPU 回退均为 0。
本轮真实相机另测 14 个窗口：实收 28.6793 FPS、解码 28.6436 FPS、平均解码 9.09 ms，
同样无新增解码前淘汰或失败。这是当前现场输入条件下的实测速率，不以历史 30 FPS 代替本轮结果。
本轮代次修复仍有 30 FPS 解码能力，但不能让已有发送端提供多于其实收的独立新帧。
两轮各三个进程均正常退出。结果见 `replay30.json`、`business-only.json`，
这次的现场低输入速率不强行归因于额外订阅，发送端/网络原因未进一步测定。
