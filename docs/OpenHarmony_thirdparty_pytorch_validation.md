# OpenHarmony `thirdparty_pytorch` 运行时接入与验证记录

本文档面向 **BQ3588HM + OpenHarmony 5.1** 场景，说明三件事：

1. 如何获取 `thirdparty_pytorch` 提供的板端 PyTorch runtime
2. 如何把它和 IB_Robot 的 OpenHarmony 交叉编译产物组合起来
3. 当前真实验证进度到哪里、遇到了什么问题、分别怎么处理

> 当前状态先说清楚：**BQ3588HM 板端 CPU 推理已经验证通过**。  
> 现在已经能在板端通过 `ros2 launch` 真正拉起 `pure_inference_node`，
> 成功加载 `/data/ibrobot/models/502000/pretrained_model` 并完成板端 CPU 推理。  
> **NPU 推理链路正在继续打通中。**

## 1. `thirdparty_pytorch` 到底提供了什么

仓库地址：

- <https://gitcode.com/openharmony-robot/thirdparty_pytorch>

它不是只有 README 的说明仓，仓库本体包含完整的 OpenHarmony Python / ML runtime
构建内容，例如：

- `pytorch-2.10.0/`
- `torchvision-0.25.0/`
- `torchaudio-2.10.0/`
- `scipy-1.16.1/`
- `safetensors-0.7.0/`
- `imageio-2.37.2/`
- `pillow-11.1.0/`
- `protobuf-3.13.0/`
- `pyav-16.1.0/`
- `decord-0.6.0/`
- `python-3.12.7/`
- `dists/`
- `test/`

其中和 IB_Robot 当前接入最相关的是两个大文件：

| 路径 | 作用 |
| --- | --- |
| `test/skh-run.tar.gz` | 板端可直接部署的 Python + torch runtime |
| `dists/skh-sdk-base.tar.xz` | 主机侧交叉编译 SDK 基础层 |

本次验证中，这两个文件都已经实际确认存在。

## 2. 主机侧获取方法

### 2.1 拉取仓库本体

```bash
git clone https://gitcode.com/openharmony-robot/thirdparty_pytorch /tmp/thirdparty_pytorch
cd /tmp/thirdparty_pytorch
```

### 2.2 拉取 LFS 大文件

如果普通 clone 后只拿到了指针文件，需要继续：

```bash
git lfs pull --include='test/skh-run.tar.gz,dists/skh-sdk-base.tar.xz'
```

本次实际验证中，最终确认拿到了：

```text
/tmp/thirdparty_pytorch/test/skh-run.tar.gz
/tmp/thirdparty_pytorch/dists/skh-sdk-base.tar.xz
```

## 3. 板端 PyTorch runtime 的来源与部署方法

`thirdparty_pytorch/test/push.sh` 和 `test/script/run_test.sh` 说明了官方推荐路径：

1. 把 `test/skh-run.tar.gz` 上传到板端 `/data/local/`
2. 解压为 `/data/local/skh-run`
3. 用 `skh-run` 提供的 Python 和动态库启动 torch 生态

### 3.1 上传到板端

可以使用 HDC：

```bash
cd /tmp/thirdparty_pytorch/test
./push.sh
```

或者手动传：

```bash
HDC_BIN=<path-to-hdc>
HDC_TARGET=<board-ip>:8710

"$HDC_BIN" -t "$HDC_TARGET" file send \
  /tmp/thirdparty_pytorch/test/skh-run.tar.gz \
  /data/local/skh-run.tar.gz
```

### 3.2 板端解压

```sh
cd /data/local
tar -zxf skh-run.tar.gz
```

最终目录应为：

```text
/data/local/skh-run
```

### 3.3 板端运行时环境

`run_test.sh` 里给出的核心环境变量是：

```sh
export PYTHONHOME=/data/local/skh-run
export PATH=${PYTHONHOME}/bin:$PATH
export PYTHONPATH=${PYTHONHOME}/lib/python3.12/site-packages
export LD_LIBRARY_PATH=${PYTHONHOME}/lib:${PYTHONHOME}/lib/python3.12/site-packages/torchaudio/lib
export LD_PRELOAD=${PYTHONHOME}/lib/libpython3.12.so.1.0:${PYTHONHOME}/lib/libomp.so
```

## 4. IB_Robot 自身的 OpenHarmony 交叉编译产物如何准备

IB_Robot 板端产物来自：

- `scripts/openharmony/build_ibrobot_oh_custom.sh`

它负责在 Ubuntu 主机上整理 OpenHarmony 交叉编译目录，并产出最终部署到板子的：

```text
$OH_ROOT/custom_build_root/ibrobot_oh_ws/install
```

典型调用方式：

```bash
export OH_ROOT="<your-unified-oh-root>"

./scripts/openharmony/build_ibrobot_oh_custom.sh \
  --oh-root "$OH_ROOT" \
  --image voxelsky/ohos-ros-humble-builder:v0.1.5 \
  --packages ibrobot_msgs,tensormsg,robot_config,inference_service
```

打包部署：

```bash
cd "$OH_ROOT/custom_build_root/ibrobot_oh_ws"
tar -zcpf ibrobot-oh-install.tar.gz install
```

上传到板端后解压到：

```text
/data/ibrobot/install
```

## 5. 板端最终运行链路

IB_Robot 板端运行时涉及两套东西：

1. 官方 ROS for OpenHarmony runtime
2. `thirdparty_pytorch` 提供的 `skh-run`

推荐按下面顺序准备：

### 5.1 先加载官方 ROS runtime

```sh
cd /data
. ./ros2ohos.env
```

### 5.2 再加载部署好的 IB_Robot install

```sh
cd /data/ibrobot
. install/setup.sh
```

### 5.3 启动 Cloud 推理节点

```sh
ros2 launch inference_service cloud_inference.launch.py \
    policy_path:=/data/ibrobot/models/502000/pretrained_model
```

> 本次验证中，模型目录保留并直接使用的是：
>
> ```text
> /data/ibrobot/models/502000/pretrained_model
> ```

## 6. 过程中遇到的问题，以及对应处理方法

### 6.1 `install/setup.sh` 仍然引用容器内前缀

最初板端 source 工作区时会报：

```text
not found: "/mnt/ohos/tmp/install/local_setup.sh"
```

原因：

- OpenHarmony 交叉编译产物里的 `setup.*` 和 `parent_prefix_path` 还保留了容器内的
  `/mnt/ohos/tmp/install`

处理：

- 在 `scripts/openharmony/build_ibrobot_oh_custom.sh` 里增加 runtime postprocess
- 把这些前缀统一改写为板端真实 ROS 前缀 `/data/install`

### 6.2 板端能 source install，但 `lerobot` 导入依赖旧残留

最初板端能导入 `lerobot`，其实依赖的是历史残留目录，不是新部署产物自身完整。

处理：

- 在 `build_ibrobot_oh_custom.sh` 里把 `libs/lerobot/src` 直接随运行时一起带入：
  `install/lerobot/src`
- 再在 `install/setup.*` 中追加 `PYTHONPATH` hook

### 6.3 `ros2 launch` 仍然跑在 `/data/out/bin/python3`

即使板子上已有 `/data/local/skh-run`，最初 `ros2 launch` 拉起的
`pure_inference_node` 仍然因为 shebang 绑定到了：

```text
/data/out/bin/python3
```

结果就是：

```text
ModuleNotFoundError: No module named 'torch'
```

处理：

- 在 `build_ibrobot_oh_custom.sh` 里把 `pure_inference_node` 和
  `lerobot_policy_node` 重写成板端 wrapper
- 只在**节点进程级**切到 `skh-run` runtime，而不是全局污染 `/data/out/bin/ros2`

这样做的原因是：

- `ros2` CLI 仍然需要官方 ROS runtime 保持稳定
- 只有真正依赖 torch 的 inference 节点才应该切到 `skh-run`

### 6.4 `inference_service.core` 没被打进安装产物

最初板端继续往下走时出现：

```text
ModuleNotFoundError: No module named 'inference_service.core'
```

原因：

- `src/inference_service/setup.py` 只打包了顶层 `inference_service`
- 没把 `inference_service.core` 子包带进去

处理：

- 改成 `find_packages(include=['inference_service', 'inference_service.*'])`

### 6.5 不是 torch runtime 坏了，而是 LeRobot lazy-import patch 没进板端包

后续继续启动时曾暴露出：

```text
ModuleNotFoundError: No module named 'datasets'
```

这里最后确认的关键点是：

- 这**不是** `thirdparty_pytorch` 的 torch runtime 本身有问题
- 根因是用于 OpenHarmony 的 LeRobot lazy-import patch  
  `third_party/patches/lerobot/v0.5.1/0004-openharmony-lazy-import-policy-stack.patch`
  **没有真正进入板端 runtime tree**

这个 patch 的目的本来就是：

- 避免 inference-time policy loading 在模块导入阶段把
  `dataset/train/env/processor` 这些训练态依赖一股脑拉进来

问题定位时，板端实际运行到的：

- `lerobot/policies/__init__.py`
- `lerobot/policies/factory.py`
- `lerobot/optim/optimizers.py`

仍然是**未打 patch 的 eager import 版本**，所以又把 `datasets` 链带了进来。

最终处理方式是：

1. 修复 `scripts/openharmony/build_ibrobot_oh_custom.sh`，不再直接复制主机工作树里的
   `libs/lerobot/src`
2. 改为在打包时临时 clone 一份 `libs/lerobot`，并显式应用
   `series.openharmony-5.1.0-musl.txt`
3. 增加 fail-closed 校验，确认以下文件已经变成 lazy import 版本：
   - `lerobot/policies/__init__.py`
   - `lerobot/policies/factory.py`
   - `lerobot/optim/optimizers.py`

修复后重新部署到板端，`datasets` 这个错误已经消失。

### 6.6 policy 会在线拉取 `torchvision` backbone 权重

lazy-import patch 修复后，继续启动时又出现了一个新的独立问题：

```text
urllib.error.URLError: <urlopen error [Errno -3] Try again>
```

进一步检查模型配置可见：

- `vision_backbone: "resnet18"`
- `pretrained_backbone_weights: "ResNet18_Weights.IMAGENET1K_V1"`

也就是说，policy loading 会尝试通过 `torchvision` 在线下载：

```text
https://download.pytorch.org/models/resnet18-f37072fd.pth
```

在当前开发板环境下，这一步因为 DNS / 外网不可用而失败。

处理方式：

1. 在主机上复用已缓存的 checkpoint：

   ```text
   ~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
   ```

2. 复制到板端：

   ```text
   /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
   ```

这样再次启动时，policy loading 就不会再尝试联网下载 backbone 权重。

## 7. 本次实际验证到哪一步

可以明确分成两层：

### 7.1 已确认可用

1. `/data/local/skh-run` 自身能导入：
   - `torch`
   - `torchvision`
   - `torchaudio`
   - `scipy`
   - `safetensors`
2. `skh-run` 和 ROS for OpenHarmony runtime 可以叠加
3. `ros2 launch inference_service cloud_inference.launch.py ...` 已能真正启动
   `pure_inference_node`
4. LeRobot lazy-import patch 已经生效，`datasets` 不再是板端启动 blocker
5. 预置 `resnet18-f37072fd.pth` 后，节点已完成：
   - `policy_path` 解析
   - LeRobot policy / config loading
   - engine 初始化
   - `PureInferenceNode ready`
6. 当前启动日志已经进入：

   ```text
   Waiting for preprocessed batches from edge node...
   ```

### 7.2 当前补充说明

- **板端 CPU 推理：已验证通过**
- **板端 NPU 推理：持续打通中**

## 8. 当前结论

当前可以确认三点：

1. **`thirdparty_pytorch/test/skh-run.tar.gz` 是 OpenHarmony 板端 torch runtime 的正确来源。**
2. **IB_Robot 板端推理链已经解决了 runtime 切换、prefix chain、安装产物不完整，以及 LeRobot OpenHarmony lazy-import patch 未进入打包产物的问题。**
3. **当前 BQ3588HM 板端 CPU 推理已经验证通过，NPU 推理链路正在继续打通。**

## 9. 后续最值得优先处理的点

1. 如果目标环境没有外网，应把 `torchvision` 所需 backbone checkpoint 一并纳入部署流程，
   避免板端首次启动时临时联网下载
2. 继续推进板端 NPU 推理链路，补齐与当前 CPU 路径一致的部署与验证闭环
