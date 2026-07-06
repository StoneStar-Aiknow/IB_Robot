# 模型训练指导文档

本文件记录 IB-Robot 从数据集准备、帧检测预处理到 ACT / Diffusion Policy（DP）模型训练与蒸馏的完整命令流程，覆盖原始规模模型训练、小模型训练和小模型知识蒸馏三条路径。本文为通用训练指南，命令中的数据集与输出路径均为占位符，可按实际任务替换。

前提条件：

- 训练框架：仓库内 `libs/lerobot` 子模块（基于 LeRobot v0.5.1 的定制版本），以 editable 方式安装，源码位于 `libs/lerobot/src`。子模块 HEAD 已内置自适应加权（`ada_weight`/`ada_weight_freq`/`damping_coefficient`）与知识蒸馏（`kd`/`teacher_train_config`/`decoder_out_dim`）支持，无需额外应用 patch。
- 环境加载：训练前必须 `source .shrc_local`。该脚本激活 `venv/`、设置 `PYTHONNOUSERSITE=1` 屏蔽用户目录下可能存在的其他 lerobot 版本，并把 `libs/lerobot/src` 前置到 `PYTHONPATH`，确保 `lerobot-train` 与 `import lerobot` 都解析到子模块源码（详见 [.shrc_local](https://gitcode.com/openeuler/IB_Robot/blob/master/.shrc_local)）。
- 训练入口：`lerobot-train`，由 `libs/lerobot/pyproject.toml` 的 `[project.scripts]` 注册，指向 [lerobot_train.py:main](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/scripts/lerobot_train.py#L701)。
- 帧检测工具：`dataset_tools` 包提供的 ROS 2 节点 `frame_detector`，用于在训练前为数据集帧打上训练权重。入口为 [frame_detector.py:main](https://gitcode.com/openeuler/IB_Robot/blob/master/src/dataset_tools/dataset_tools/frame_detector.py#L804)，注册于 [dataset_tools/setup.py](https://gitcode.com/openeuler/IB_Robot/blob/master/src/dataset_tools/setup.py#L33)。
- 数据集转换工具：`dataset_tools` 包提供的 `bag_to_lerobot`，把 ROS 2 bag 转换为 LeRobot v3 数据集。入口为 [bag_to_lerobot.py:main](https://gitcode.com/openeuler/IB_Robot/blob/master/src/dataset_tools/dataset_tools/bag_to_lerobot.py#L1120)，注册于 [dataset_tools/setup.py](https://gitcode.com/openeuler/IB_Robot/blob/master/src/dataset_tools/setup.py#L30)。
- 设备：训练默认使用 `cuda`，需保证 CUDA toolkit 与 PyTorch 版本匹配；蒸馏还需要能加载 teacher 模型的显存。
- 输出目录：每个训练任务通过 `--output_dir` 指定独立输出路径，避免相互覆盖。
- ROS 2 overlay：`frame_detector`、`bag_to_lerobot` 等 `dataset_tools` 节点通过 `ros2 run` 调用，需先构建工作区并 source overlay（`source install/setup.bash`）才能被找到。
- 所有 `/path/to/xxx` 占位符需替换为环境中的实际绝对路径，详见文末「路径占位符说明」。

训练任务与数据集、模型的对应关系（示例，实际任务可按需调整）：

| 训练任务 | 模型类型 | 是否蒸馏 | Teacher |
|----------|----------|----------|---------|
| ACT 原始规模模型 | ACT (`dim_model=2048`, `n_decoder_layers=7`) | 否 | — |
| ACT 小模型 | ACT (`dim_model=1024`, `n_decoder_layers=2`, `dim_feedforward=1200`) | 否 | — |
| ACT 小模型蒸馏 | ACT (`dim_model=1024`, `n_decoder_layers=2`, `dim_feedforward=1200`) | 是 | ACT 原始规模模型 |
| DP 原始规模模型 | Diffusion (默认 `down_dims=(512,1024,2048)`) | 否 | — |
| DP 小模型蒸馏 | Diffusion (`down_dims=[256,512,1024]`) | 是 | DP 原始规模模型 |

## 1. 数据集准备（bag → LeRobot）

把 ROS 2 bag 转换为 LeRobot v3 数据集，以 `robot_config.yaml` 作为 SSOT 定义关节、相机、契约。

`bag_to_lerobot` 的关键参数（来源：[bag_to_lerobot.py:parse_args](https://gitcode.com/openeuler/IB_Robot/blob/master/src/dataset_tools/dataset_tools/bag_to_lerobot.py#L1068)）：

| 参数 | 含义 | 是否必填 |
|------|------|----------|
| `--bag` / `--bags` / `--bags-dir` | 单个 bag 目录 / 多个 bag / 包含多个 bag 的目录（三选一互斥） | 是 |
| `--robot-config` | robot_config.yaml 路径（SSOT） | 是 |
| `--out` | 输出数据集根目录 | 是 |
| `--repo-id` | 数据集 repo_id 元数据，默认 `rosbag_v30` | 否 |
| `--no-videos` | 存 PNG 图片而非 MP4 视频 | 否 |
| `--image-threads` / `--image-processes` | 图像写入线程数 / 进程数，默认 4 / 0 | 否 |
| `--chunk-size` / `--data-mb` / `--video-mb` | 数据/视频 chunk 大小上限 | 否 |
| `--video-codec` | 视频编码，可选 `auto`/`h264`/`hevc`/`libsvtav1`/`h264_nvenc`/`hevc_nvenc`，默认 `auto` | 否 |
| `--timestamp` | 时间基准，可选 `contract`/`bag`/`header`，默认 `contract`（按契约频率重采样） | 否 |

示例：

```bash
ros2 run dataset_tools bag_to_lerobot \
  --bags-dir /path/to/bags \
  --robot-config /path/to/robot_config.yaml \
  --out /path/to/dataset/my_dataset \
  --repo-id my_dataset
```

转换完成后，`/path/to/dataset/my_dataset` 即为 LeRobot v3 数据集根目录，后续训练命令的 `--dataset.repo_id` 直接填该路径。

## 2. 数据集预处理（帧检测与训练权重）

在训练前对数据集执行帧检测，识别关键帧（critical frame）和冻结帧（freeze frame），并写入逐帧训练权重。关键帧是电流大、速度小的接触/抓取瞬间，权重放大；冻结帧是机器人停滞段，权重置零以抑制无效梯度。

### 2.1 工具入口

帧检测以 ROS 2 节点形式提供，正确调用方式是 `ros2 run dataset_tools frame_detector --ros-args -p <param>:=<value>`，不是 `--<param>=<value>` 形式。节点实现在 [frame_detector.py:FrameDetectorNode](https://gitcode.com/openeuler/IB_Robot/blob/master/src/dataset_tools/dataset_tools/frame_detector.py#L738)。

### 2.2 可用参数

参数通过 `self.declare_parameter(...)` 声明（[frame_detector.py:744-763](https://gitcode.com/openeuler/IB_Robot/blob/master/src/dataset_tools/dataset_tools/frame_detector.py#L744-L763)），完整列表如下：

| ROS 参数名 | 类型 | 默认值 | 含义 |
|---|---|---|---|
| `dataset_path` | string | `""` | LeRobot 数据集根目录（必填） |
| `clip_view` | string_array | `["all"]` | 处理哪些相机视角，`["all"]` 表示全部 |
| `enable_critical_detection` | bool | `True` | 是否检测关键帧 |
| `enable_freeze_detection` | bool | `True` | 是否检测冻结帧 |
| `gripper_pos` | integer_array | `[-1]` | 夹爪索引，`[-1]` 表示取最后一个关节作为夹爪 |
| `gripper_names` | string_array | `["*gripper*"]` | 夹爪关节名匹配模式 |
| `critical_frame_min_current_threshold` | double | `0.5` | 关键帧最小电流阈值 |
| `critical_frame_max_velocity_threshold` | double | `0.01` | 关键帧最大速度阈值 |
| `critical_frame_training_weight` | double | `2.0` | 关键帧训练权重 |
| `critical_frame_min_duration` | int | `3` | 关键帧段最小持续帧数 |
| `n_forward_expansion` | int | `30` | 关键帧向前扩展帧数 |
| `n_backward_expansion` | int | `30` | 关键帧向后扩展帧数 |
| `freeze_head_tail_only` | bool | `False` | 是否只检测首尾冻结段 |
| `freeze_frame_max_velocity` | double | `0.1` | 冻结帧最大速度 |
| `freeze_frame_max_current` | double | `0.2` | 冻结帧最大电流 |
| `freeze_frame_training_weight` | double | `0.0` | 冻结帧训练权重，`0` 表示屏蔽 |
| `freeze_frame_min_duration` | int | `5` | 冻结段最小持续帧数 |

### 2.3 调用示例

```bash
ros2 run dataset_tools frame_detector --ros-args \
  -p dataset_path:=/path/to/dataset/my_dataset \
  -p clip_view:="['all']" \
  -p enable_critical_detection:=true \
  -p critical_frame_min_current_threshold:=0.5 \
  -p critical_frame_max_velocity_threshold:=0.01 \
  -p critical_frame_training_weight:=2.0 \
  -p n_forward_expansion:=30 \
  -p n_backward_expansion:=30 \
  -p enable_freeze_detection:=true \
  -p freeze_frame_max_velocity:=0.1 \
  -p freeze_frame_max_current:=0.2 \
  -p freeze_frame_training_weight:=0.0 \
  -p freeze_head_tail_only:=false \
  -p freeze_frame_min_duration:=5
```

处理完成后，数据集目录下会生成逐帧训练权重文件，并在
`training_weights_distribution.png` 中可视化权重分布。后续 `lerobot-train`
通过 `--ada_weight=true` 读取这些权重做自适应加权。

注意：

- 数组参数（`clip_view`、`gripper_pos`、`gripper_names`）需按 ROS 2 参数语法传入，例如 `-p gripper_pos:="[-1]"` 或 `-p clip_view:="['all']"`，具体格式以 ROS 2 的参数解析器为准。
- 蒸馏任务使用的数据集必须先经过本步骤处理；纯监督训练的数据集若已含权重可跳过，但建议统一执行以保证自适应权重生效。
- 该节点必须先 `source install/setup.bash` 才能被 `ros2 run` 找到。

## 3. ACT 原始规模模型训练

基于 LeRobot ACT 策略训练 banana 场景的原始规模 teacher 模型，作为后续 ACT 蒸馏的 teacher 候选。命令中显式覆盖架构参数，便于复现实验配置；源码默认值见下方参数表。

```bash
lerobot-train \
  --dataset.repo_id=/path/to/dataset/banana_640x480_two_cam \
  --policy.type=act \
  --output_dir=/path/to/output/banana_act \
  --job_name=act_train_banana \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.dim_model=2048 \
  --policy.n_encoder_layers=4 \
  --policy.n_decoder_layers=7 \
  --policy.dim_feedforward=3200 \
  --batch_size=16 \
  --steps=500000 \
  --log_freq=200 \
  --eval_freq=2000 \
  --save_freq=10000 \
  --num_workers=10
```

关键参数（来源：[configuration_act.py:ACTConfig](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/policies/act/configuration_act.py#L25)）：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--policy.dim_model` | Transformer 主隐藏维度；banana 原始规模示例覆盖为 `2048` | `512` |
| `--policy.n_encoder_layers` | 编码器层数 | `4` |
| `--policy.n_decoder_layers` | 解码器层数；banana 原始规模示例覆盖为 `7` | `1` |
| `--policy.dim_feedforward` | 前馈网络扩展维度 | `3200` |
| `--steps` | 总训练步数 | `100000` |
| `--save_freq` | 每多少步保存一次 checkpoint，teacher 模型需保留可用 checkpoint | `20000` |

注意：原始规模模型训练完成后，蒸馏步骤需要 teacher 的 `train_config.json`。Checkpoint 目录结构由 [train_utils.py:save_checkpoint](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/utils/train_utils.py#L67) 生成，`train_config.json` 位于 `<output_dir>/checkpoints/<step>/pretrained_model/train_config.json`；同时存在 `<output_dir>/checkpoints/last` 符号链接指向最新一次 checkpoint（见 [train_utils.py:update_last_checkpoint](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/utils/train_utils.py#L59)，常量定义于 [constants.py:47-48](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/utils/constants.py#L47-L48)）。因此本示例中的 teacher 路径通常写为 `/path/to/output/banana_act/checkpoints/last/pretrained_model/train_config.json`。

## 4. ACT 小模型训练

使用同一份数据集训练小容量 ACT 策略，作为对照基线或轻量部署模型。

```bash
lerobot-train \
  --dataset.repo_id=/path/to/dataset/banana_640x480_two_cam \
  --policy.type=act \
  --output_dir=/path/to/output/banana_act_small \
  --job_name=act_train_banana_small \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.dim_model=1024 \
  --policy.n_encoder_layers=4 \
  --policy.n_decoder_layers=2 \
  --policy.dim_feedforward=1200 \
  --batch_size=16 \
  --steps=500000 \
  --log_freq=200 \
  --eval_freq=2000 \
  --save_freq=10000 \
  --num_workers=10
```

与 banana 原始规模模型的差异在模型容量：`dim_model` 由 `2048` 降至 `1024`，`n_decoder_layers` 由 `7` 降至 `2`，`dim_feedforward` 由 `3200` 降至 `1200`。其余训练超参与原始规模模型一致，便于公平对比。

## 5. ACT 小模型蒸馏

以 ACT 原始规模模型为 teacher 蒸馏小模型。蒸馏时 student 解码器输出维度对齐到 teacher 的 `decoder_out_dim`，并用自适应权重抑制低质量帧。

```bash
lerobot-train \
  --dataset.repo_id=/path/to/dataset/banana_640x480_two_cam \
  --policy.type=act \
  --output_dir=/path/to/output/banana_act_kd \
  --job_name=act_kd_banana \
  --policy.device=cuda \
  --policy.dim_model=1024 \
  --policy.n_encoder_layers=4 \
  --policy.n_decoder_layers=2 \
  --policy.dim_feedforward=1200 \
  --policy.kd=true \
  --policy.teacher_train_config=/path/to/output/banana_act/checkpoints/last/pretrained_model/train_config.json \
  --policy.decoder_out_dim=1024 \
  --policy.push_to_hub=false \
  --ada_weight=true \
  --ada_weight_freq=200 \
  --damping_coefficient=0.7 \
  --batch_size=60 \
  --steps=500000 \
  --log_freq=200 \
  --eval_freq=2000 \
  --save_freq=20000 \
  --num_workers=10
```

蒸馏专属参数（来源：[train.py:TrainPipelineConfig](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/configs/train.py#L37) 与 [configuration_act.py:ACTConfig](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/policies/act/configuration_act.py#L25)）：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--policy.kd` | 开启知识蒸馏 | `False` |
| `--policy.teacher_train_config` | teacher 模型的 `train_config.json` 路径，用于重建 teacher 策略 | `None` |
| `--policy.decoder_out_dim` | student 解码器输出维度，需与 teacher 的解码维度对齐（原始规模 teacher 默认 `1024`） | `1024` |
| `--ada_weight` | 开启自适应样本权重，读取帧检测写入的逐帧权重 | `True` |
| `--ada_weight_freq` | 每多少步更新一次自适应权重 | `200` |
| `--damping_coefficient` | 自适应权重的阻尼系数，平滑权重更新 | `0.7` |

注意：

- `--policy.kd`、`--policy.teacher_train_config`、`--policy.decoder_out_dim` 是**策略配置**（ACT/Diffusion 都有），需通过 `--policy.kd=true` 这种带前缀的形式传入。
- `--ada_weight`、`--ada_weight_freq`、`--damping_coefficient` 是**训练流水线配置**，不带 `policy.` 前缀。
- teacher 加载逻辑在 [lerobot_train.py:257-263](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/scripts/lerobot_train.py#L257-L263)：`if cfg.policy.kd: teacher_policy = get_teacher_model(cfg.policy.teacher_train_config, dataset, cfg.batch_size)`。
- 自适应权重更新逻辑在 [lerobot_train.py:489-536](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/scripts/lerobot_train.py#L489-L536)。
- `--policy.teacher_train_config` 指向的 `train_config.json` 必须能完整重建 teacher（包含 policy type、模型结构、统计量与 dataset contract）。teacher 与 student 必须使用一致的归一化配置和动作维度，否则蒸馏 loss 不可比。
- `--policy.decoder_out_dim` 需与 teacher 的 `decoder_out_dim` 一致；原始规模 ACT teacher 默认 `decoder_out_dim=1024`，若 teacher 改过该值，student 也要同步调整。
- 本步骤 `--save_freq=20000` 比直接训练更稀疏，蒸馏任务通常更关注最终 checkpoint。

## 6. DP 原始规模模型训练

使用 LeRobot 上游默认配置训练 Diffusion Policy（`down_dims=(512,1024,2048)`、`diffusion_step_embed_dim=128`、`spatial_softmax_num_keypoints=32`），作为后续 DP 蒸馏的 teacher 候选。命令中显式列出 `crop_shape`、`horizon` 等关键参数，其余采用默认值。

```bash
lerobot-train \
  --dataset.repo_id=/path/to/dataset/my_dataset \
  --policy.type=diffusion \
  --output_dir=/path/to/output/dp_base \
  --job_name=dp_train_base \
  --policy.device=cuda \
  --policy.crop_shape="[480,640]" \
  --policy.crop_is_random=false \
  --policy.push_to_hub=false \
  --policy.n_obs_steps=6 \
  --policy.horizon=64 \
  --policy.n_action_steps=44 \
  --policy.drop_n_last_frames=15 \
  --ada_weight=true \
  --ada_weight_freq=200 \
  --damping_coefficient=0.7 \
  --batch_size=8 \
  --steps=500000 \
  --log_freq=200 \
  --eval_freq=2000 \
  --save_freq=6000 \
  --num_workers=10
```

DP 原始规模模型关键参数（来源：[configuration_diffusion.py:DiffusionConfig](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/policies/diffusion/configuration_diffusion.py#L25)）：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--policy.crop_shape` | 图像裁剪尺寸 `[H, W]`，原始规模使用 `[480,640]` | `None` |
| `--policy.crop_is_random` | `false` 表示训练时使用中心裁剪而非随机裁剪 | `True` |
| `--policy.n_obs_steps` | 输入观测帧数 | `2` |
| `--policy.horizon` | 扩散模型预测的动作序列长度 | `16` |
| `--policy.n_action_steps` | 单次推理实际执行的动作步数 | `8` |
| `--policy.drop_n_last_frames` | 丢弃末尾帧数，需满足 `drop_n_last_frames = horizon - n_action_steps - n_obs_steps + 1`（`64-44-6+1=15`） | `7` |
| `--policy.down_dims` | U-Net 各下采样阶段的特征维度，默认 `(512,1024,2048)` 即原始规模 | `(512,1024,2048)` |
| `--policy.spatial_softmax_num_keypoints` | SpatialSoftmax 关键点数 | `32` |
| `--policy.diffusion_step_embed_dim` | 扩散时间步嵌入维度 | `128` |

注意：`horizon` 必须是下采样因子 `2 ** len(down_dims)` 的整数倍，否则 diffusion 配置校验会报错（见 [configuration_diffusion.py:213-218](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/policies/diffusion/configuration_diffusion.py#L213-L218)）。当前默认 `down_dims=(512,1024,2048)`，下采样因子为 `8`，`64` 与 `128` 均满足该约束。`--save_freq=6000` 比 ACT 更频繁，DP 训练对 checkpoint 选取更敏感。

## 7. DP 小模型蒸馏

以 DP 原始规模模型为 teacher 蒸馏小模型。相比原始规模模型，蒸馏版使用更长 horizon、更小裁剪尺寸和更小的扩散网络维度。

```bash
lerobot-train \
  --dataset.repo_id=/path/to/dataset/my_dataset \
  --policy.type=diffusion \
  --output_dir=/path/to/output/dp_kd \
  --job_name=dp_kd \
  --policy.device=cuda \
  --policy.crop_shape="[240,320]" \
  --policy.crop_is_random=false \
  --policy.push_to_hub=false \
  --policy.n_obs_steps=12 \
  --policy.horizon=128 \
  --policy.n_action_steps=100 \
  --policy.drop_n_last_frames=17 \
  --ada_weight=true \
  --ada_weight_freq=200 \
  --damping_coefficient=0.7 \
  --batch_size=8 \
  --steps=500000 \
  --log_freq=200 \
  --eval_freq=2000 \
  --save_freq=6000 \
  --num_workers=10 \
  --policy.down_dims="[256,512,1024]" \
  --policy.spatial_softmax_num_keypoints=16 \
  --policy.diffusion_step_embed_dim=64 \
  --policy.kd=true \
  --policy.teacher_train_config=/path/to/output/dp_base/checkpoints/last/pretrained_model/train_config.json \
  --policy.decoder_out_dim=2048
```

DP 蒸馏与原始规模模型差异：

| 维度 | 原始规模模型 | 蒸馏小模型 | 说明 |
|------|--------|------------|------|
| `crop_shape` | `[480,640]` | `[240,320]` | 蒸馏使用更小图像，降低视觉编码器开销 |
| `n_obs_steps` | `6` | `12` | 蒸馏使用更长观测窗口 |
| `horizon` | `64` | `128` | 蒸馏预测更长动作序列 |
| `n_action_steps` | `44` | `100` | 蒸馏单次执行更多步 |
| `drop_n_last_frames` | `15` | `17` | 满足 `horizon - n_action_steps - n_obs_steps + 1`（`128-100-12+1=17`） |
| `down_dims` | 默认 `(512,1024,2048)` | `[256,512,1024]` | 蒸馏使用更小的 U-Net 特征维度 |
| `spatial_softmax_num_keypoints` | 默认 `32` | `16` | 蒸馏关键点数减半 |
| `diffusion_step_embed_dim` | 默认 `128` | `64` | 蒸馏扩散时间步嵌入维度减半 |
| `kd` | 否 | `true` | 蒸馏开启知识蒸馏 |

注意：

- DP 蒸馏的 `--policy.teacher_train_config` 指向 DP 原始规模模型 checkpoint 中的 `train_config.json`，路径形如 `/path/to/output/dp_base/checkpoints/last/pretrained_model/train_config.json`。
- `decoder_out_dim=2048` 需与 teacher 的解码/动作维度对齐（DP 原始规模 teacher 的 `down_dims` 末维为 `2048`）。
- `down_dims` 改为 `[256,512,1024]` 后下采样因子仍为 `8`，`horizon=128` 满足整除约束。
- teacher 与 student 的 `policy.type` 必须一致：ACT 蒸馏用 ACT teacher，DP 蒸馏用 DP teacher，不能交叉。

## 8. 通用训练参数说明

以下参数在多个训练任务中复用，含义一致（来源：[train.py:TrainPipelineConfig](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/configs/train.py#L37)）：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--dataset.repo_id` | LeRobot 数据集路径或 Hub repo id；本地数据集直接填绝对路径 | — |
| `--policy.type` | 策略类型，`act` 或 `diffusion` | — |
| `--output_dir` | 训练输出根目录，同一目录重复运行会被覆盖除非 `--resume=true` | — |
| `--job_name` | 任务名，影响日志/checkpoint 子目录命名 | — |
| `--policy.device` | 训练设备，`cuda` / `cuda:0` / `cpu` / `mps` | — |
| `--policy.push_to_hub` | 是否在训练末尾推送到 HuggingFace Hub。默认 `True`；纯本地训练请显式设 `false` 并省略 `--policy.repo_id`，否则训练结尾会因无 Hub 凭证或 `repo_id` 缺失报错（见 [train.py:147](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/configs/train.py#L147)） | `True` |
| `--batch_size` | 训练批大小，蒸馏任务常显存压力更大，需按 GPU 显存调整 | `8` |
| `--steps` | 总训练步数 | `100000` |
| `--log_freq` | 每多少步打印一次训练日志 | `200` |
| `--eval_freq` | 每多少步执行一次评估 | `20000` |
| `--save_freq` | 每多少步保存一次 checkpoint | `20000` |
| `--num_workers` | DataLoader 工作进程数 | `4` |
| `--resume` | 续训时设为 `true`，需保证 `--output_dir` 指向已有 checkpoint 目录 | `False` |
| `--seed` | 训练与评估随机种子 | `1000` |

Checkpoint 目录结构（由 [train_utils.py:save_checkpoint](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/utils/train_utils.py#L67) 生成）：

```
<output_dir>/
├── checkpoints/
│   ├── <step>/                  # 形如 005000
│   │   ├── pretrained_model/
│   │   │   ├── config.json          # policy config
│   │   │   ├── model.safetensors    # policy weights
│   │   │   ├── train_config.json    # train config（teacher 蒸馏用）
│   │   │   └── processor.json       # processor config
│   │   └── training_state/
│   │       ├── optimizer_state.safetensors
│   │       ├── rng_state.safetensors
│   │       ├── scheduler_state.json
│   │       └── training_step.json
│   └── last -> <step>/          # 指向最新 checkpoint 的符号链接
└── tensorboard/                 # TensorBoard 日志
```

## 9. 路径占位符说明

文中所有 `/path/to/xxx` 需替换为环境中的实际绝对路径。常用占位符对应关系：

| 占位符 | 含义 | 示例 |
|--------|------|------|
| `/path/to/bags` | 待转换的 ROS 2 bag 目录 | `~/datasets/raw_bags` |
| `/path/to/robot_config.yaml` | SSOT 机器人配置文件 | `src/robot_config/robot_config.yaml` |
| `/path/to/dataset/my_dataset` | LeRobot v3 数据集根目录（含帧权重） | `~/datasets/my_dataset` |
| `/path/to/output/...` | 训练输出根目录 | `~/training_outputs/...` |
| `/path/to/output/banana_act/checkpoints/last/pretrained_model/train_config.json` | ACT teacher 的 train_config | `~/training_outputs/banana_act/checkpoints/last/pretrained_model/train_config.json` |
| `/path/to/output/dp_base/checkpoints/last/pretrained_model/train_config.json` | DP teacher 的 train_config | `~/training_outputs/dp_base/checkpoints/last/pretrained_model/train_config.json` |

建议将训练用路径统一放在同一根目录下（如 `~/training_outputs` 与 `~/datasets`），并在蒸馏前确认 teacher checkpoint 路径存在且包含 `train_config.json`。
