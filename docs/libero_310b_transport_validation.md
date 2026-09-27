# LIBERO → openEuler 310B ACT OM 观测传输验证

验证日期：2026-09-27。关联 [PR #451](https://atomgit.com/openeuler/IB_Robot/pull/451)。

## 范围与结论

本机运行真实 LIBERO 环境、执行 action；通过 DDS 请求与结果通道连接 openEuler Embedded
Ascend310B1 板端，板端运行 ACT OM。图像分别使用 DDS 和 H.264/RTP，模型、任务、种子、
初始状态和动作契约保持一致。不是 hardware mock，也不是同机 loopback。

本次小样本中，两种传输的 9 次实际模型调用均没有观测错配；单路目标图片延迟可见时，
生产路径在模型调用前拒绝不完整观测，重试后取齐。任务结果相同，但不能据此推断全套
LIBERO 成功率或原先 0.86 的差距已经解决。

## 环境与模型

- 主机：Ubuntu 22.04 x86_64，ROS 2 Humble，LIBERO 使用 EGL。
- 推理端：openEuler Embedded aarch64，ACL 报告 `Ascend310B1`，设备 0。
- 两端使用 `rmw_fastrtps_cpp`，每轮独立 ROS domain；RTP 使用软件编码/解码和默认 RFC6184 分包。
- 源码：`11f1c686` 加本次通用 FrameIngress 协议默认值修复；板端在独立目录部署，原工作区未修改。
- 模型：本机已有 `libero_goal_act_verify` ACT checkpoint；两张 256×256 RGB，state 8 维，action 7 维，
  `n_obs_steps=1`，模型 chunk size 100。不是板端原有的 SO-101 6 维香蕉抓取模型。
- 权重 SHA256：`b215e74e88c4ff32b7e3fc70bafb9be6dad981dbdf33dd2ae79556f9f9f8f924`。
- OM：ATC `Ascend310B1`、`precision_mode_v2=fp16`，原生 ACL 读取 ABI 后打包为 `ascend_310b1`。
- 部署指纹：`7f234b4ba859295271642b04800fdc0c2839a0fbc075b798ec95bff19a5b71e9`。
- LIBERO 资产固定 revision `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`，585 个业务文件完整校验。
- `libero_goal` task 8（put the bowl on the plate），seed 10000，init state 0/1，各组两个 episode，
  每 episode 最多 600 个动作。等待反馈调度，未更改模型原生执行策略。

板端编译：112 个 ROS 消息定义、CMake 和 package 文件与已有性能验证目录逐项相同，
复用同一板端已编译消息产物；当前源码 Python 包在独立 overlay 重建。已验证推理、传输、
配置、LeRobot 和 ROS 消息的实际导入路径均属于独立目录。不是一次全量纯净板端构建，
不能替代 WIP 暂缓的双平台 Docker setup/build 验证。

## 模型准备检查

| 输入 | Torch/ONNX 最大绝对误差 | Torch/OM 最大绝对误差 | Torch/OM 平均绝对误差 |
|---|---:|---:|---:|
| 固定随机 smoke 输入 | 9.54e-7 | 0.003460 | 0.001245 |
| 真实 task8 初始观测 | 1.00e-5 | 0.008163 | 0.001510 |

真实观测 Torch/OM 余弦相似度为 0.99999785。固定输入纯 OM 执行 20 次，中位 12.98 ms、
P95 13.09 ms，不包含预处理、DDS、RTP 或仿真耗时。以上仅是数值 smoke/单观测检查，
不是完整模型转换精度验收。DDS/RTP 对照使用同一个 OM，避免比较不同推理后端。

FP32 `origin` 编译曾因 310B MaxPoolV3 不支持 FP32 失败，改为 FP16 后编译通过。
ABI 检查与推理分别在独立进程执行，避免 ACL finalize 后重新初始化设备的限制。

## 真实闭环结果

| 组别 | 源观测数（含两次 reset） | 实际 OM 调用 | 成功结果返回 | readiness 重试 | 输入错配 |
|---|---:|---:|---:|---:|---:|
| DDS | 673 | 9 | 9 | 0 | 0 |
| H.264/RTP | 673 | 9 | 9 | 0 | 0 |
| RTP：目标图片延迟可见 | 673 | 9 | 9 | 50 | 0 |

三组任务结果均为：init state 0 在 71 步成功；init state 1 达到 600 步上限，任务未成功。
不能把这两个 episode 的结果当成 benchmark 整体成功率。

审计方式：临时 wrapper 在端侧 observation sink 的编码/发布入口记录共同源时间、
episode/sequence、state 数值和哈希；在板端实际 `runtime.infer` 前记录输入、图片选择时间、
state 历史时间及匹配差值，不改变生产选择或动作输出。

- DDS：全部 9 次模型输入的两张图片和 state 的 float32 内容哈希与同一次源观测一致。
- RTP：全部 9 次满足 `image_ts == image2_ts == request_ts == state_history_ts`，state 差值为 0，
  state 内容与源数据逐位一致。H.264 图片不要求像素无损，未使用像素哈希相等作为通过标准。
- 延迟组：前三个目标时间各将 image2 的已解码目标条目暂时移出缓存，800 ms 后恢复原内容与
  原始时间戳。是受控缓存可见性注入，不是物理网络延迟或 UDP 丢包实验。
- 延迟组 50 次失败尝试均返回 `observation_not_ready`；目标条目恢复前模型调用次数为 0。
- 延迟组记录 1,348 个解码输出，两路各 674（含重复恢复帧）；每个原始时间戳均与相应相机的
  源观测精确对应，没有无法关联的时间戳。重复解码不代表新增仿真 step。

## 真实链路发现与修复

benchmark 通过 `ManagedFrameIngress` 调用通用 `create_frame_ingress()`，该入口和
`NativeFrameIngress` 的默认协议仍是 5；普通 device 路径显式传版本，因此原有测试没有覆盖
默认值。云侧 v7 拒绝描述符，导致生产环境等待接收端超时。

本次将版本常量置于底层 `observation_transport.frame_ingress`，通用/native 入口和
`inference_service.distributed` 共用值 7，不新增协议版本。新增两个行为测试，验证不显式
传版本时入口生成的 descriptor/status 与当前协议一致。

修复后定向 `colcon test` 共 234 项通过，0 错误、0 失败、0 跳过，进程正常退出。
另行扩测旧 `test_device_video_streams.py` 时有 11 项因调用已不存在的 `flush()`/`_streams`
而失败；生产类和该测试在本次修复前即存在此 API 不一致。本次未宣称该包全量测试通过，
也未为适配这些旧测试恢复已移除 API。

## 未计入通过结果的尝试

1. 直接启动远端云节点时，临时 YAML 未投影 benchmark 的 RTP 配置，导致 unexpected descriptor。
   使用主机同一编译函数展开契约并确认指纹相同后重跑；未进入推理，不计为 episode 对照。
2. 默认协议 5/7 不一致造成启动超时，修复后在新 ROS domain 重跑。
3. 对第一帧解码直接注入 800 ms 延迟时，prepare 阶段遇到 reset service 发现竞态，
   `prepare_failed`、0 动作、0 模型调用。这一失败未被隐藏，也未计入重试通过结果；
   成功的延迟验证明确在 reset 后的目标观测取数阶段注入。

## 证据与复现

本机 worktree 的忽略目录 `tmp/verification/libero-310b-451/` 保存临时启动器、审计脚本、
两端日志、生成配置、canonical episode 结果、测试日志和 `evidence-sha256.json`。
模型与转换中间产物分别位于 `models/libero_goal_act_verify/` 和
`models/_work/libero_goal_act_verify/ascend/`，均不提交。

| 审计文件 | SHA256 |
|---|---|
| `audit-dds.json` | `f4bbfa86b0843d7d08c49434ab952b7541f248875f262ef98e0ffc55f0dacdf9` |
| `audit-rtp.json` | `33f7dc284e5da1495b62a942c865cb860af5247822f23cadee8d9e4516567bc5` |
| `audit-rtp-late.json` | `8149f08e11617cee5016900439d025f9b9c751ee4bc75ff2dbc00c582cc3f39e` |

生产启动器目前默认 loopback 云角色。临时 `host.launch.py` 复用生产 launch，仅移除本机
`pure_inference_node`，板端单独启动同一节点；两端使用相同模型 manifest，RTP 接收地址指向板端。
端侧审计在 sink prepare 后写入，可能增加少量发布开销，不能用本次日志宣称原生端到端延迟。

复核审计可运行：

```bash
source .shrc_local
python3 tmp/verification/libero-310b-451/audit_report.py \
  tmp/verification/libero-310b-451/source-rtp-v7.jsonl \
  tmp/verification/libero-310b-451/cloud-rtp-v7.jsonl
```

验证进程已在结束后关闭；模型和隔离验证目录保留用于复现。WIP Docker 门禁仍暂缓。
