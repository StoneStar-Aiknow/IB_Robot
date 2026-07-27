# OpenHarmony EmbodiedAI 1.0.1 RKNN NPU 推理指南

本文档说明如何将 ACT 策略打包为包含 RKNN deployment 的统一 policy bundle，并在
OpenHarmony EmbodiedAI 1.0.1 的 RK3588 板卡上运行单体或边云分布式推理。

> 板端约束：RoboOH 使用 aarch64/musl、toybox、只读 rootfs 且没有 systemd。板端命令使用
> POSIX `sh`，环境脚本用 `. /data/roboframe/scripts/robooh_1.0.1.env` 加载。发布包的
> `install.sh` 应已完成 RoboFrame、Python pysite 和 RKNNLite 部署。

## 1. Bundle Contract

RKNN 运行时只接受 bundle 根目录下的 `inference_manifest.json`。目录示例：

```text
policy_bundle/
├── config.json
├── policy_preprocessor.json
├── policy_preprocessor_step_*.safetensors
├── policy_postprocessor.json
├── policy_postprocessor_step_*.safetensors
├── artifacts/
│   └── rknn/
│       └── rk3588/
│           └── policy-<sha-prefix>.rknn
└── inference_manifest.json
```

Manifest 中的命名 deployment 决定 backend、target、artifact、execution 和 bindings。运行时
不会扫描目录寻找 `*.rknn`，不会猜测输入顺序，也不会使用 launch `device` 参数选择后端。

LeRobot `config.json` 和 processor 文件保持只读。不要向 `config.json` 写入 IB-Robot backend
字段。

## 2. 主机侧 RKNN 转换

### 2.1 环境

RKNN Toolkit 与主项目依赖可能冲突，使用独立环境：

```bash
python3 -m venv .venv-rknn
. .venv-rknn/bin/activate
python -m pip install rknn-toolkit2 onnx onnxruntime
```

板端 RKNN runtime 版本和主机 toolkit 版本应匹配。版本不一致可能导致模型加载失败或数值
差异，应在目标板上完成最终验证。

### 2.2 从 Policy Checkpoint 导出

回到仓库环境后执行 exporter。Exporter 会：

1. 从 LeRobot checkpoint 导出并简化 ACT ONNX。
2. 调用独立 RKNN Python 环境转换模型。
3. 生成 compiler-resolved runtime ABI JSON。
4. 将 RKNN artifact 复制到 bundle 的 `artifacts/rknn/<deployment>/`。
5. 生成完整 bindings、SHA-256、bundle digest，并更新 `inference_manifest.json`。
6. 使用生产 strict loader 重新验证 deployment。

```bash
source .shrc_local

python3 src/model_utils/model_utils/export_onnx_rknn.py \
    --policy_path "$WORKSPACE/models/502000/pretrained_model" \
    --convert_rknn \
    --rknn_output /tmp/act_policy.rknn \
    --rknn_abi_output /tmp/act_policy.rknn.abi.json \
    --rknn_mode float16 \
    --rknn_venv_python "$WORKSPACE/.venv-rknn/bin/python" \
    --deployment rknn
```

`rknn` 是示例 deployment 名称，可以改为 `rk3588`、`robopi` 等。启动时必须使用同一个
名称。

### 2.3 从已有 ONNX 转换

```bash
source .shrc_local

python3 src/model_utils/model_utils/export_onnx_rknn.py \
    --onnx /path/to/act_policy.onnx \
    --bundle_root "$WORKSPACE/models/502000/pretrained_model" \
    --convert_rknn \
    --rknn_output /tmp/act_policy.rknn \
    --rknn_abi_output /tmp/act_policy.rknn.abi.json \
    --rknn_mode float16 \
    --rknn_venv_python "$WORKSPACE/.venv-rknn/bin/python" \
    --deployment rknn
```

`--onnx --convert_rknn` 必须同时提供 `--bundle_root`，因为 exporter 需要读取 LeRobot
metadata，并验证 runtime ABI 是否与模型 semantic features 一致。

### 2.4 验证 Manifest

```bash
source .shrc_local
PYTHONPATH=src/inference_manifest \
python3 -c "from inference_manifest import load_inference_manifest; print(load_inference_manifest('$WORKSPACE/models/502000/pretrained_model', 'rknn').fingerprint)"
```

不要手工修改 artifact path、binding、UUID/revision 或 bundle digest。模型或 processor 文件变化后，
重新运行 exporter。

## 3. Robot YAML

推理配置位于 `control_modes.<mode>.inference.pipelines`：

```yaml
control_modes:
  model_inference:
    inference:
      enabled: true
      pipelines:
        policy:
          model_path: models/502000/pretrained_model
          deployment: rknn
          execution_mode: monolithic
          request_timeout: 10.0
          default_task: ""
          runtime_options: {}
    executor:
      type: topic
      mode: model_inference
      inference_pipeline: policy
      queue_size: 100
      watermark_threshold: 50
      control_frequency: 20.0
```

相对 `model_path` 只相对于绝对路径环境变量 `WORKSPACE` 解析。板端发布环境若没有合适的
`WORKSPACE`，使用绝对路径，例如 `/data/models/502000/pretrained_model`。

默认 pipeline endpoints：

| 接口 | `policy` pipeline 默认值 |
| --- | --- |
| Action | `/inference/policy/dispatch` |
| Reset service | `/inference/policy/reset` |
| Health | `/inference/policy/health` |
| Action output | `/actions/policy` |
| 分布式 request | `/inference/policy/request` |
| 分布式 result | `/inference/policy/result` |
| 分布式 heartbeat | `/inference/policy/heartbeat` |

## 4. 板端独立验证

板端启动前确认 bundle 已部署到 `/data` 可写区域：

```sh
. /data/roboframe/scripts/robooh_1.0.1.env
export ROS_DOMAIN_ID=51
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

test -f /data/models/502000/pretrained_model/inference_manifest.json
python3 -c "from rknnlite.api import RKNNLite; print('RKNNLite OK')"
```

使用最小化 launch 验证一个 deployment：

```sh
ros2 launch hardware_mock hardware_mock.launch.py robot_config:=so101_single_arm &

ros2 launch inference_service eval_inference.launch.py \
    robot_config_path:=/data/roboframe/install/robot_config/share/robot_config/config/robots/so101_single_arm.yaml \
    model_path:=/data/models/502000/pretrained_model \
    deployment:=rknn \
    pipeline_id:=policy
```

另一个终端触发推理：

```sh
. /data/roboframe/scripts/robooh_1.0.1.env
export ROS_DOMAIN_ID=51
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

ros2 action send_goal /inference/policy/dispatch \
    ibrobot_msgs/action/DispatchInfer \
    "{obs_timestamp: {sec: 0, nanosec: 0}, prompt: '', inference_id: 'rknn-test-001', deadline: {sec: 0, nanosec: 0}}"
```

结果应包含 `success: true`、实际 `chunk_size`、`pipeline_id: policy`、非空
`deployment_fingerprint` 和 `backend_latency_ms`。

## 5. 单板真实硬件闭环

Robot YAML 中的 `policy` pipeline 必须选择 RKNN deployment 且
`execution_mode: monolithic`。然后启动完整系统：

```sh
. /data/roboframe/scripts/robooh_1.0.1.env
export ROS_DOMAIN_ID=51
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

ros2 launch robot_config robot.launch.py \
    config_path:=/data/roboframe/install/robot_config/share/robot_config/config/robots/so101_single_arm.yaml \
    control_mode:=model_inference \
    use_sim:=false
```

典型进程：

```text
pipeline_policy_node        RKNN pipeline、processor、Action server
action_dispatcher_node      action chunk 队列与控制频率
usb_cam_node_exe            top / wrist cameras
so101_hardware              ros2_control hardware interface
```

真机需要内核启用 `CONFIG_USB_ACM=y`，并完成机械臂与相机配置。

## 6. 边云分布式 RKNN

Execution mode 属于 YAML，不是 launch override。Edge 使用的 robot YAML：

```yaml
pipelines:
  policy:
    model_path: /absolute/edge/path/to/policy_bundle
    deployment: rknn
    execution_mode: distributed
    request_timeout: 10.0
```

Edge bundle 可以不包含 cloud-only compiled artifact，但必须有匹配的
`inference_manifest.json` 和本地 processor/tokenizer 文件。Edge 与 cloud 的 bundle digest、
deployment name 和 deployment fingerprint 必须一致。

Ubuntu edge：

```bash
source .shrc_local
export ROS_DOMAIN_ID=51
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

ros2 launch robot_config robot.launch.py \
    config_path:=/absolute/path/to/so101_single_arm_distributed.yaml \
    control_mode:=model_inference \
    use_sim:=true
```

OpenHarmony cloud：

```sh
. /data/roboframe/scripts/robooh_1.0.1.env
export ROS_DOMAIN_ID=51
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

ros2 launch inference_service cloud_inference.launch.py \
    pipeline_id:=policy \
    model_path:=/data/models/502000/pretrained_model \
    deployment:=rknn
```

握手成功前 edge 不发送 inference tensors。Cloud 重启、heartbeat 超时、fingerprint 改变或
backend 离开 `READY` 会使 session 失效；恢复后必须重新握手。

## 7. 机械臂校准

首次使用前在板端执行：

```sh
. /data/roboframe/scripts/robooh_1.0.1.env
ros2 run so101_hardware calibrate_arm --arm follower --port /dev/ttyACM0
```

校准 JSON 通常保存在 `$HOME/.calibrate/so101_follower_calibrate.json`。SSH 和 HDC 的
`HOME` 可能不同，robot YAML 中的校准路径必须解析到实际文件。

## 8. Runtime ABI

RKNN 编译器可能重排输入，因此 manifest bindings 同时记录 semantic、runtime name/index、
dtype、shape 和图像 layout。Runtime 按 binding 映射输入，不依赖 `config.json` 插入顺序或
固定 output index。

图像只有在 binding 声明 `layout: NHWC` 时才执行 NCHW-to-NHWC 转换。状态、token、mask、
action 或其他非图像 tensor 不会因为是 4-D tensor 就自动转置。

## 9. 排障

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `Deployment 'rknn' is not present` | 启动名称与 manifest 不一致 | 查看 `deployments` keys，并使用正确名称或重新导出 |
| artifact load failure | artifact 损坏或 ABI 不兼容 | 重新运行 owning exporter 并发布新 revision |
| `Bundle digest mismatch` | `bundle.files` 与声明 digest 不一致 | 重新生成完整 manifest |
| binding shape/name mismatch | compiler ABI 与 LeRobot feature contract 不一致 | 用同一模型重新转换并保留 ABI JSON |
| RKNN dependency unavailable | RKNNLite 未部署或环境未加载 | `. robooh_1.0.1.env`，检查 `from rknnlite.api import RKNNLite` |
| runtime/toolkit version mismatch | 主机 toolkit 与板端 runtime 不兼容 | 使用与板端 runtime 匹配的 toolkit 重建模型 |
| `/dev/ttyACM0` 不存在 | 内核缺少 USB ACM | 启用 `CONFIG_USB_ACM=y` 并刷入内核 |
| Calibration file not found | `HOME` 或 YAML 路径不一致 | 使用绝对路径或修正校准文件位置 |
| 推理进程 SIGSEGV | 板端环境或 native library 不完整 | 确认已加载 RoboOH 环境并检查 RKNN runtime library |

日志中应出现类似：

```text
Unified pipeline started: id=policy, mode=monolithic, deployment=rknn, backend=rknn
```

具体 latency 取决于模型、RKNN toolkit/runtime 版本、NPU 负载、图像尺寸和 processor 开销，
不应把单次测量值作为固定保证。
