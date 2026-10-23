# 模型训练指导文档

本文件记录 IB-Robot 从数据集准备、帧检测预处理到 ACT / Diffusion Policy（DP）模型训练与蒸馏的完整命令流程，覆盖原始规模模型训练、小模型训练和小模型知识蒸馏三条路径；并在第 8 节给出 PI05（π₀.₅）VLA 模型基于预训练权重微调的流程。本文为通用训练指南，命令中的数据集与输出路径均为占位符，可按实际任务替换。

前提条件：

- 训练框架：仓库内 `libs/lerobot` 子模块（当前基于 LeRobot v0.6.0 的定制版本，参见 `third_party/patches/lerobot/INDEX.yaml` 的 `active_tag`），以 editable 方式安装，源码位于 `libs/lerobot/src`。**注意**：v0.5.1 时期的自适应加权参数 `--ada_weight` / `--ada_weight_freq` / `--damping_coefficient` 在 v0.6.0 中已不存在，帧检测权重改由 LeRobot 原生的 `--sample_weighting.type=custom` 读取（ACT、PI05 支持，DP 不支持，见第 2 节）；知识蒸馏（`--policy.kd` / `--policy.teacher_train_config` / `--policy.decoder_out_dim`）已移植到 v0.6.0 训练循环，ACT 与 DP 可用（见第 5、7 节）。
- 环境加载：训练前必须 `source .shrc_local`。该脚本激活 `venv/`、设置 `PYTHONNOUSERSITE=1` 屏蔽用户目录下可能存在的其他 lerobot 版本，并把 `libs/lerobot/src` 前置到 `PYTHONPATH`，确保 `lerobot-train` 与 `import lerobot` 都解析到子模块源码（详见 [.shrc_local](https://gitcode.com/openeuler/IB_Robot/blob/master/.shrc_local)）。
- 训练机环境准备：**`training` profile 仅支持 Ubuntu 22.04**，其他平台在任何依赖安装前会报错退出（不影响 `full` / `inference` 的平台支持）。在 Ubuntu 22.04 上只需要 LeRobot 训练环境（不需要 ROS 2 / rosdep / colcon）时，用 `./scripts/setup.sh --yes --profile training` 初始化。该 profile 只创建 venv、应用 LeRobot 补丁栈并安装 LeRobot 与训练依赖；`source .shrc_local` 后 `lerobot-train` 即可用（详见 [`scripts/setup/README.md`](../scripts/setup/README.md) 的 Setup Profile 一节）。`frame_detector` / `bag_to_lerobot` 等数据集准备工具是 ROS 2 节点，需在带 ROS 2 的机器上完成（或直接拷贝已转换好的数据集到训练机）。
- 训练入口：`lerobot-train`，由 `libs/lerobot/pyproject.toml` 的 `[project.scripts]` 注册，指向 [lerobot_train.py:main](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/scripts/lerobot_train.py#L836)。
- 帧检测工具：`dataset_tools` 包提供的 ROS 2 节点 `frame_detector`，用于在训练前为数据集帧打上训练权重。入口为 [frame_detector.py:main](https://gitcode.com/openeuler/IB_Robot/blob/master/src/dataset_tools/dataset_tools/frame_detector.py#L804)，注册于 [dataset_tools/setup.py](https://gitcode.com/openeuler/IB_Robot/blob/master/src/dataset_tools/setup.py#L34)。
- 数据集转换工具：`dataset_tools` 包提供的 `bag_to_lerobot`，把 ROS 2 bag 转换为 LeRobot v3 数据集。入口为 [bag_to_lerobot.py:main](https://gitcode.com/openeuler/IB_Robot/blob/master/src/dataset_tools/dataset_tools/bag_to_lerobot.py#L1428)，注册于 [dataset_tools/setup.py](https://gitcode.com/openeuler/IB_Robot/blob/master/src/dataset_tools/setup.py#L31)。
- 设备：训练默认使用 `cuda`，需保证 CUDA toolkit 与 PyTorch 版本匹配；蒸馏时 teacher 与 student 同时驻留显存，且共用同一个 `--batch_size`。
- 输出目录：每个训练任务通过 `--output_dir` 指定独立输出路径。非 `--resume` 模式下该目录必须不存在，否则启动即报 `FileExistsError`（[train.py:240-244](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/configs/train.py#L240-L244)），不会覆盖已有结果。
- ROS 2 overlay：`frame_detector`、`bag_to_lerobot` 等 `dataset_tools` 节点通过 `ros2 run` 调用，需先构建工作区并 source overlay（`source install/setup.bash`）才能被找到。`lerobot-train` 只需 `source .shrc_local`，不依赖工作区构建。
- 数据集位置：本地数据集一律显式传 `--dataset.root=<数据集根目录>`；省略 `root` 时 LeRobot 会按 `repo_id` 去 HuggingFace Hub 解析，离线环境会失败。
- 所有 `/path/to/xxx` 占位符需替换为环境中的实际绝对路径，详见文末「路径占位符说明」。

训练任务与数据集、模型的对应关系（示例，实际任务可按需调整）：

| 训练任务 | 模型类型 | 是否蒸馏 | Teacher |
|----------|----------|----------|---------|
| ACT 原始规模模型 | ACT (`dim_model=2048`, `n_decoder_layers=7`) | 否 | — |
| ACT 小模型 | ACT (`dim_model=1024`, `n_decoder_layers=2`, `dim_feedforward=1200`) | 否 | — |
| ACT 小模型蒸馏 | ACT (`dim_model=1024`, `n_decoder_layers=2`, `dim_feedforward=1200`) | 是 | ACT 原始规模模型 |
| DP 原始规模模型 | Diffusion (默认 `down_dims=(512,1024,2048)`) | 否 | — |
| DP 小模型蒸馏 | Diffusion (`down_dims=[256,512,1024]`) | 是 | DP 原始规模模型 |
| PI05 微调（只训 action expert / LoRA） | PI05 (`gemma_2b` VLM + `gemma_300m` action expert，基于 `lerobot/pi05_base`) | 否 | — |

本文各节的训练命令（第 3-7 节、8.3、8.4）均已在单张 RTX 3090（24 GB）+ 当前 v0.6.0 补丁栈上，用 SO-101 单臂双相机数据集（`top` + `wrist` 两路 640×480 视频，6 维 state/action，80 episodes，30 fps，已执行第 2 节帧检测）实测能正常启动、完成训练步与 eval、保存 checkpoint；8.5 的全量微调超出 24 GB，未实测。各节给出的显存为该环境下的实测峰值（训练日志 `mem_gb` 字段），可作为调整 `--batch_size` 的参考。

## 1. 数据集准备（bag → LeRobot）

把 ROS 2 bag 转换为 LeRobot v3 数据集。转换使用的契约（关节、相机、频率）取自**录制时写入的契约快照**，而不是当前的 `robot_config.yaml`：bag 必须位于 `episode_recorder` 生成的录制目录 `<record_root>/episodes/<bag>` 下，`<record_root>/dataset.yaml` 中的 `contract` 快照即为 SSOT（[bag_to_lerobot.py:_dataset_metadata_for_bag](https://gitcode.com/openeuler/IB_Robot/blob/master/src/dataset_tools/dataset_tools/bag_to_lerobot.py#L298)、[_contract_from_dataset_metadata](https://gitcode.com/openeuler/IB_Robot/blob/master/src/dataset_tools/dataset_tools/bag_to_lerobot.py#L687)）。缺少快照的旧录制不再支持；一次转换选中的所有 episode 的契约快照必须完全一致。

`bag_to_lerobot` 的关键参数（来源：[bag_to_lerobot.py:parse_args](https://gitcode.com/openeuler/IB_Robot/blob/master/src/dataset_tools/dataset_tools/bag_to_lerobot.py#L1381)）：

| 参数 | 含义 | 是否必填 |
|------|------|----------|
| `--bag` / `--bags` / `--bags-dir` | 单个 bag 目录 / 多个 bag / 包含多个 bag 的目录（三选一互斥） | 是 |
| `--out` | 输出数据集根目录 | 是 |
| `--repo-id` | 数据集 repo_id 元数据，默认 `rosbag_v30` | 否 |
| `--no-videos` | 存 PNG 图片而非 MP4 视频 | 否 |
| `--image-threads` / `--image-processes` | 图像写入线程数 / 进程数，默认 4 / 0 | 否 |
| `--chunk-size` / `--data-mb` / `--video-mb` | 数据/视频 chunk 大小上限 | 否 |
| `--video-codec` | 视频编码，可选 `auto`/`h264`/`hevc`/`libsvtav1`/`h264_nvenc`/`hevc_nvenc`，默认 `auto` | 否 |
| `--timestamp` | 时间基准，可选 `contract`/`bag`/`header`，默认 `contract`（按契约频率重采样） | 否 |

早期版本的 `--robot-config` 参数已移除，传入会直接报错（[bag_to_lerobot.py:42](https://gitcode.com/openeuler/IB_Robot/blob/master/src/dataset_tools/dataset_tools/bag_to_lerobot.py#L42)）。

示例：

```bash
ros2 run dataset_tools bag_to_lerobot \
  --bags-dir /path/to/record_root/episodes \
  --out /path/to/dataset/my_dataset \
  --repo-id my_dataset
```

转换完成后，`/path/to/dataset/my_dataset` 即为 LeRobot v3 数据集根目录。后续训练命令统一用 `--dataset.repo_id=my_dataset --dataset.root=/path/to/dataset/my_dataset` 指定：`root` 指向数据集根目录，`repo_id` 只作标识。

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

处理完成后，逐帧权重写回数据集 parquet 的 `training_weight` 列（并登记到 `meta/info.json` 的 features），
权重分布可视化保存为数据集根目录下的 `training_weights_distribution.png`。后续 `lerobot-train`
通过 `--sample_weighting.type=custom` 读取该列，对每帧 loss 加权（见
[sample_weighting.py:CustomWeighter](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/utils/sample_weighting.py#L248)）。

注意：

- 数组参数（`clip_view`、`gripper_pos`、`gripper_names`）需按 ROS 2 参数语法传入，例如 `-p gripper_pos:="[-1]"` 或 `-p clip_view:="['all']"`，具体格式以 ROS 2 的参数解析器为准。
- 只有开启 `--sample_weighting.type=custom` 的训练（ACT、PI05）依赖本步骤，否则数据集没有 `training_weight` 列。DP 不支持样本加权，知识蒸馏也不能与样本加权同时开启（见第 5、6 节），这两类训练可以跳过本步骤。
- 该节点必须先 `source install/setup.bash` 才能被 `ros2 run` 找到。

## 3. ACT 原始规模模型训练

基于 LeRobot ACT 策略训练 banana 场景的原始规模 teacher 模型，作为后续 ACT 蒸馏的 teacher 候选。命令中显式覆盖架构参数，便于复现实验配置；源码默认值见下方参数表。

```bash
source .shrc_local
lerobot-train \
  --dataset.repo_id=my_dataset \
  --dataset.root=/path/to/dataset/my_dataset \
  --policy.type=act \
  --output_dir=/path/to/output/banana_act \
  --job_name=act_train_banana \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.dim_model=2048 \
  --policy.n_encoder_layers=4 \
  --policy.n_decoder_layers=7 \
  --policy.dim_feedforward=3200 \
  --sample_weighting.type=custom \
  --dataset.eval_split=0.1 \
  --eval_steps=2000 \
  --batch_size=16 \
  --steps=500000 \
  --log_freq=200 \
  --save_freq=10000 \
  --num_workers=10
```

实测：`batch_size=16` 时峰值显存约 18.9 GB，约 1.3 step/s。`--sample_weighting.type=custom` 读取第 2 节写入的帧权重，数据集未做帧检测时去掉该参数。`--dataset.eval_split=0.1 --eval_steps=2000` 按 episode 留出 10% 作验证集，每 2000 步计算一次 `eval_loss`，不需要时两者一起去掉（`eval_steps>0` 而 `eval_split=0` 会直接报错）。

关键参数（来源：[configuration_act.py:ACTConfig](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/policies/act/configuration_act.py#L24)）：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--policy.dim_model` | Transformer 主隐藏维度；banana 原始规模示例覆盖为 `2048` | `512` |
| `--policy.n_encoder_layers` | 编码器层数 | `4` |
| `--policy.n_decoder_layers` | 解码器层数；banana 原始规模示例覆盖为 `7` | `1` |
| `--policy.dim_feedforward` | 前馈网络扩展维度 | `3200` |
| `--steps` | 总训练步数 | `100000` |
| `--save_freq` | 每多少步保存一次 checkpoint，teacher 模型需保留可用 checkpoint | `20000` |

注意：原始规模模型训练完成后，蒸馏步骤需要 teacher 的 `train_config.json`。Checkpoint 目录结构由 [train_utils.py:save_checkpoint](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/common/train_utils.py#L93) 生成，`train_config.json` 位于 `<output_dir>/checkpoints/<step>/pretrained_model/train_config.json`；同时存在 `<output_dir>/checkpoints/last` 符号链接指向最新一次 checkpoint（见 [train_utils.py:update_last_checkpoint](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/common/train_utils.py#L85)，常量定义于 [constants.py:47-50](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/utils/constants.py#L47-L50)）。因此本示例中的 teacher 路径通常写为 `/path/to/output/banana_act/checkpoints/last/pretrained_model/train_config.json`。

## 4. ACT 小模型训练

使用同一份数据集训练小容量 ACT 策略，作为对照基线或轻量部署模型。

```bash
source .shrc_local
lerobot-train \
  --dataset.repo_id=my_dataset \
  --dataset.root=/path/to/dataset/my_dataset \
  --policy.type=act \
  --output_dir=/path/to/output/banana_act_small \
  --job_name=act_train_banana_small \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.dim_model=1024 \
  --policy.n_encoder_layers=4 \
  --policy.n_decoder_layers=2 \
  --policy.dim_feedforward=1200 \
  --sample_weighting.type=custom \
  --dataset.eval_split=0.1 \
  --eval_steps=2000 \
  --batch_size=16 \
  --steps=500000 \
  --log_freq=200 \
  --save_freq=10000 \
  --num_workers=10
```

与 banana 原始规模模型的差异在模型容量：`dim_model` 由 `2048` 降至 `1024`，`n_decoder_layers` 由 `7` 降至 `2`，`dim_feedforward` 由 `3200` 降至 `1200`。其余训练超参与原始规模模型一致，便于公平对比。实测 `batch_size=16` 时峰值显存约 7.8 GB。

## 5. ACT 小模型蒸馏

以 ACT 原始规模模型为 teacher 蒸馏小模型。开启 `kd` 后，student 在解码器输出后接一层 `Linear(dim_model → decoder_out_dim)` 再接动作头，并以 MSE 对齐冻结 teacher 的解码器输出（[modeling_act.py:compute_kd_loss](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/policies/act/modeling_act.py#L137-L146)、[modeling_act.py:412-413](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/policies/act/modeling_act.py#L412-L413)）。teacher 解码器输出宽度等于 teacher 的 `dim_model`，所以 `decoder_out_dim` 必须填 teacher 的 `dim_model`。

```bash
source .shrc_local
lerobot-train \
  --dataset.repo_id=my_dataset \
  --dataset.root=/path/to/dataset/my_dataset \
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
  --policy.decoder_out_dim=2048 \
  --policy.push_to_hub=false \
  --dataset.eval_split=0.1 \
  --eval_steps=2000 \
  --batch_size=16 \
  --steps=500000 \
  --log_freq=200 \
  --save_freq=20000 \
  --num_workers=10
```

实测：`batch_size=16` 时 teacher + student 峰值显存约 10.9 GB，日志中除监督 loss `loss` 外还会输出实际参与反传的 `total_kd_loss`。显存允许时可调大 `batch_size`（24 GB 卡上 `batch_size=60` 会 OOM）。

蒸馏专属参数（来源：[configuration_act.py:ACTConfig](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/policies/act/configuration_act.py#L118-L120)）：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--policy.kd` | 开启知识蒸馏 | `False` |
| `--policy.teacher_train_config` | teacher 模型的 `train_config.json` 路径，用于重建 teacher 策略并加载其权重 | `None` |
| `--policy.decoder_out_dim` | student 蒸馏投影层的输出维度，**必须等于 teacher 的 `dim_model`**（本例 teacher 为 `2048`） | `1024` |

注意：

- `--policy.kd`、`--policy.teacher_train_config`、`--policy.decoder_out_dim` 是**策略配置**（ACT/Diffusion 都有），需通过 `--policy.kd=true` 这种带前缀的形式传入。
- `decoder_out_dim` 与 teacher `dim_model` 不一致时，第一步就会报 `RuntimeError: The size of tensor a (1024) must match the size of tensor b (2048)`。默认值 `1024` 只适用于 `dim_model=1024` 的 teacher。
- teacher 加载逻辑在 [lerobot_train.py:302-307](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/scripts/lerobot_train.py#L302-L307)：`get_teacher_model(cfg.policy.teacher_train_config, dataset, cfg.batch_size)` 以 `--config_path=<train_config.json> --resume=true` 重建 teacher，但喂给它的是 student 的数据集与 batch。因此 teacher 与 student 必须基于同一数据集（相同的观测 key、动作维度和归一化统计量），`chunk_size` 也要保持一致（默认均为 `100`），否则解码器输出无法逐元素对齐。
- 蒸馏不能与样本加权同时开启：同时传 `--policy.kd=true` 与 `--sample_weighting.*` 会报 `NotImplementedError`（[lerobot_train.py:130](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/scripts/lerobot_train.py#L130)），所以本命令不带 `--sample_weighting.type=custom`。
- 本步骤 `--save_freq=20000` 比直接训练更稀疏，蒸馏任务通常更关注最终 checkpoint。

## 6. DP 原始规模模型训练

使用 LeRobot 上游默认配置训练 Diffusion Policy（`down_dims=(512,1024,2048)`、`diffusion_step_embed_dim=128`、`spatial_softmax_num_keypoints=32`），作为后续 DP 蒸馏的 teacher 候选。命令中显式列出 `crop_shape`、`horizon` 等关键参数，其余采用默认值。

```bash
source .shrc_local
lerobot-train \
  --dataset.repo_id=my_dataset \
  --dataset.root=/path/to/dataset/my_dataset \
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
  --dataset.eval_split=0.1 \
  --eval_steps=2000 \
  --batch_size=8 \
  --steps=500000 \
  --log_freq=200 \
  --save_freq=6000 \
  --num_workers=10
```

实测：`batch_size=8` 时峰值显存约 17.7 GB，约 2.2 step/s。

**DP 不支持样本加权**：`DiffusionPolicy.forward` 没有 `reduction` 参数，追加 `--sample_weighting.type=custom` 会在第一步报 `TypeError: DiffusionPolicy.forward() got an unexpected keyword argument 'reduction'`。因此 DP 命令不带该参数，即使数据集已做帧检测。

DP 原始规模模型关键参数（来源：[configuration_diffusion.py:DiffusionConfig](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/policies/diffusion/configuration_diffusion.py#L25)）：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--policy.crop_shape` | 图像裁剪尺寸 `[H, W]`，原始规模使用 `[480,640]` | `None` |
| `--policy.crop_is_random` | `false` 表示训练时使用中心裁剪而非随机裁剪 | `True` |
| `--policy.n_obs_steps` | 输入观测帧数 | `2` |
| `--policy.horizon` | 扩散模型预测的动作序列长度 | `64` |
| `--policy.n_action_steps` | 单次推理实际执行的动作步数 | `32` |
| `--policy.drop_n_last_frames` | 丢弃末尾帧数，需满足 `drop_n_last_frames = horizon - n_action_steps - n_obs_steps + 1`（`64-44-6+1=15`） | `7` |
| `--policy.down_dims` | U-Net 各下采样阶段的特征维度，默认 `(512,1024,2048)` 即原始规模 | `(512,1024,2048)` |
| `--policy.spatial_softmax_num_keypoints` | SpatialSoftmax 关键点数 | `32` |
| `--policy.diffusion_step_embed_dim` | 扩散时间步嵌入维度 | `128` |

注意：`horizon` 必须是下采样因子 `2 ** len(down_dims)` 的整数倍，否则 diffusion 配置校验会报错（见 [configuration_diffusion.py:209-215](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/policies/diffusion/configuration_diffusion.py#L209-L215)）。当前默认 `down_dims=(512,1024,2048)`，下采样因子为 `8`，`64` 与 `128` 均满足该约束。`--save_freq=6000` 比 ACT 更频繁，DP 训练对 checkpoint 选取更敏感。

## 7. DP 小模型蒸馏

以 DP 原始规模模型为 teacher 蒸馏小模型。student 使用更小的裁剪尺寸和更小的扩散网络维度；U-Net 中间特征经对齐层投影到 teacher 的特征维度后做特征蒸馏。

```bash
source .shrc_local
lerobot-train \
  --dataset.repo_id=my_dataset \
  --dataset.root=/path/to/dataset/my_dataset \
  --policy.type=diffusion \
  --output_dir=/path/to/output/dp_kd \
  --job_name=dp_kd \
  --policy.device=cuda \
  --policy.crop_shape="[240,320]" \
  --policy.crop_is_random=false \
  --policy.push_to_hub=false \
  --policy.n_obs_steps=6 \
  --policy.horizon=64 \
  --policy.n_action_steps=44 \
  --policy.drop_n_last_frames=15 \
  --policy.down_dims="[256,512,1024]" \
  --policy.spatial_softmax_num_keypoints=16 \
  --policy.diffusion_step_embed_dim=64 \
  --policy.kd=true \
  --policy.teacher_train_config=/path/to/output/dp_base/checkpoints/last/pretrained_model/train_config.json \
  --dataset.eval_split=0.1 \
  --eval_steps=2000 \
  --batch_size=8 \
  --steps=500000 \
  --log_freq=200 \
  --save_freq=6000 \
  --num_workers=10
```

实测：`batch_size=8` 时 teacher + student 峰值显存约 6.9 GB，日志输出 `total_kd_loss`。

DP 蒸馏与原始规模模型差异：

| 维度 | 原始规模模型 | 蒸馏小模型 | 说明 |
|------|--------|------------|------|
| `crop_shape` | `[480,640]` | `[240,320]` | 蒸馏使用更小图像，降低视觉编码器开销；teacher 仍按自己的 `crop_shape` 处理同一 batch |
| `n_obs_steps` / `horizon` / `n_action_steps` / `drop_n_last_frames` | `6` / `64` / `44` / `15` | 同左 | **必须与 teacher 完全一致**，原因见下方注意事项 |
| `down_dims` | 默认 `(512,1024,2048)` | `[256,512,1024]` | 蒸馏使用更小的 U-Net 特征维度；末维必须是 `1024`，teacher 末维必须是 `2048` |
| `spatial_softmax_num_keypoints` | 默认 `32` | `16` | 蒸馏关键点数减半 |
| `diffusion_step_embed_dim` | 默认 `128` | `64` | 蒸馏扩散时间步嵌入维度减半 |
| `kd` | 否 | `true` | 蒸馏开启知识蒸馏 |

注意：

- DP 蒸馏的 `--policy.teacher_train_config` 指向 DP 原始规模模型 checkpoint 中的 `train_config.json`，路径形如 `/path/to/output/dp_base/checkpoints/last/pretrained_model/train_config.json`。
- **时间窗口必须与 teacher 一致**：teacher 由自己的 `train_config.json` 重建，但吃的是 student 数据集按 student 的 `n_obs_steps`/`horizon` 采出的 batch，而 teacher 的 `compute_loss` 会断言 `horizon == self.config.horizon`（[modeling_diffusion.py:365](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/policies/diffusion/modeling_diffusion.py#L365)）。student 用 `horizon=128` 而 teacher 为 `64` 时会直接 `AssertionError`。如果确实需要更长的 student 时间窗口，teacher 也要用相同窗口重新训练。
- **特征维度是写死的**：对齐层固定为 `Conv1d(1024, 2048)`、恢复层固定为 `ConvTranspose1d(2048, 1024)`（[modeling_diffusion.py:736-744](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/policies/diffusion/modeling_diffusion.py#L736-L744)），因此 student `down_dims` 末维必须为 `1024`，teacher 末维必须为 `2048`（即默认 `(512,1024,2048)`）。DP 实现不读取 `--policy.decoder_out_dim`，无需传入。
- `down_dims` 改为 `[256,512,1024]` 后下采样因子仍为 `8`，`horizon=64` 满足整除约束。
- 与 ACT 相同，蒸馏不能与 `--sample_weighting` 同时开启；DP 本身也不支持样本加权。
- teacher 与 student 的 `policy.type` 必须一致：ACT 蒸馏用 ACT teacher，DP 蒸馏用 DP teacher，不能交叉。

## 8. PI05（π₀.₅）微调训练

PI05 是 PaliGemma（`gemma_2b`）VLM 加 `gemma_300m` action expert 的 flow-matching VLA 模型，总参数约 4.1B。实现位于 [configuration_pi05.py:PI05Config](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/policies/pi05/configuration_pi05.py#L30) 与 [modeling_pi05.py](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/policies/pi05/modeling_pi05.py)。PI05 不从零训练，而是在预训练权重 `lerobot/pi05_base` 上用自采数据微调。

和 ACT/DP 相比，PI05 训练有以下几处根本差异，后文各步骤都围绕它们展开：

| 维度 | ACT / DP | PI05 |
|------|----------|------|
| 初始化 | 随机初始化 | 必须加载 `lerobot/pi05_base` 预训练权重 |
| 语言输入 | 无 | 数据集每帧的 `task` 文本参与训练，部署时的 prompt 必须与之一致 |
| 状态输入 | 作为张量输入网络 | 归一化后离散成 256 档写进文本 prompt（`Task: …, State: …;\nAction: `），不是网络张量输入，见 [processor_pi05.py:Pi05PrepareStateTokenizerProcessorStep](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/policies/pi05/processor_pi05.py#L53) |
| 归一化 | MEAN_STD / MIN_MAX | STATE、ACTION 用 `QUANTILES`（q01/q99），数据集 stats 必须含分位数 |
| 图像 | 按 `crop_shape` 等配置处理 | 模型内部 resize-with-pad 到 `224×224`，数据集原始分辨率不必是 224 |
| 优化器/调度器 | 各策略预设 | 策略预设 AdamW（`lr=2.5e-5`）+ cosine（warmup 1000 / decay 30000），沿用 openpi |
| 显存 | 单卡消费级 GPU 可全量训练 | 全量微调需要大显存；24 GB 卡只能只训 action expert 或用 LoRA（见 8.3、8.4） |

### 8.1 前置准备

**依赖**：`scripts/setup.sh` 安装 lerobot 时已带上 `pi` extra（transformers、scipy，见 [python_venv.sh](https://gitcode.com/openeuler/IB_Robot/blob/master/scripts/setup/python_venv.sh#L220)），无需额外安装。LoRA 微调用的 `peft` 默认未安装，见 8.4。

**预训练权重**：`lerobot/pi05_base` 的 `model.safetensors` 约 14.5 GB（float32）。建议训练前先下载到 HF 缓存或本地目录，避免训练启动时阻塞下载：

```bash
source .shrc_local
# huggingface.co 不通时改走镜像
export HF_ENDPOINT=https://hf-mirror.com
hf download lerobot/pi05_base
# 或下载到固定目录，之后 --policy.pretrained_path 填该绝对路径
hf download lerobot/pi05_base --local-dir /path/to/pretrained/pi05_base
```

**Tokenizer**：PI05 的 prompt 用 PaliGemma tokenizer `google/paligemma-3b-pt-224` 编码（[processor_pi05.py:152](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/policies/pi05/processor_pi05.py#L152)）。这是 gated 仓库：需要先在 Hugging Face 上同意 PaliGemma 许可，再执行 `hf auth login`（镜像访问同样需要 token），然后预下载：

```bash
source .shrc_local
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224')"
```

注意：微调时 tokenizer 配置继承自 `pi05_base` 的 `policy_preprocessor.json`，并原样写进产出 checkpoint。训练前请确认所用 `pi05_base` 快照的 `policy_preprocessor.json` 中 `tokenizer_processor.tokenizer_name` 为 `google/paligemma-3b-pt-224`。**不要为了绕开 gated 下载把它改成其他 tokenizer**（例如 `bert-base-uncased`）：PaliGemma 的词表与 embedding 是绑定的，换 tokenizer 后训练能照常跑完、loss 也会下降，但语言输入已经是无意义的 token，部署后语言条件失效。

如果手头的 `pi05_base` 快照已被改过（`tokenizer_name` 不是 PaliGemma），不要直接拿它训练，也不要改共享的 HF 缓存，而是在本地目录建一份修正后的副本，再把 `--policy.pretrained_path` 指向该目录：

```bash
source .shrc_local
SRC=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('lerobot/pi05_base'))")
DST=/path/to/pretrained/pi05_base
mkdir -p "$DST"
# 其余文件（含约 14.5 GB 的 model.safetensors）软链接到缓存，只重写 policy_preprocessor.json
for f in "$SRC"/*; do ln -sfn "$(readlink -f "$f")" "$DST/$(basename "$f")"; done
rm "$DST/policy_preprocessor.json"
python3 - "$SRC/policy_preprocessor.json" "$DST/policy_preprocessor.json" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
for step in cfg["steps"]:
    if step["registry_name"] == "tokenizer_processor":
        step["config"]["tokenizer_name"] = "google/paligemma-3b-pt-224"
json.dump(cfg, open(sys.argv[2], "w"), indent=2)
PY
```

训练结束后可用 `grep tokenizer_name <output_dir>/checkpoints/last/pretrained_model/policy_preprocessor.json` 复核产出 checkpoint 用的是 PaliGemma tokenizer。

### 8.2 数据集要求

数据集仍按第 1 节由 `bag_to_lerobot` 生成。PI05 对数据集有以下额外要求，训练前逐项确认：

1. **task 文本必须有意义且与部署一致**。`bag_to_lerobot` 从 bag 元数据 `custom_data["lerobot.operator_prompt"]` 读取 task，缺失时回退到数据集 `dataset.yaml` 的 `default_task`，再缺失则写空字符串（[bag_to_lerobot.py:990-996](https://gitcode.com/openeuler/IB_Robot/blob/master/src/dataset_tools/dataset_tools/bag_to_lerobot.py#L990-L996)）。该 prompt 在录制阶段由 `record_cli` 输入、`episode_recorder` 写入。**注意**：`record_cli` 的初始 prompt 是字面量 `"default_task"`（[record_cli.py:732](https://gitcode.com/openeuler/IB_Robot/blob/master/src/dataset_tools/dataset_tools/record_cli.py#L732)），首次录制直接回车会把 `default_task` 这个字符串当作任务描述录进数据集。PI05 会把它当真实指令学习，因此录制时请输入完整的自然语言任务描述，例如 `Grasp the banana and put it on the plate`。

   已经转换好的数据集不必重录，可直接用 LeRobot 自带的 `lerobot-edit-dataset` 改写 task：

   ```bash
   source .shrc_local
   # modify_tasks 原地修改，先复制一份再改副本
   cp -a /path/to/dataset/my_dataset /path/to/dataset/my_dataset_pi05
   lerobot-edit-dataset --repo_id my_dataset_pi05 --root /path/to/dataset/my_dataset_pi05 \
     --operation.type modify_tasks \
     --operation.new_task "Grasp the banana and put it on the plate"
   ```

   **`modify_tasks` 总是原地修改 `--root` 指向的数据集**：即使传了 `--new_repo_id` / `--new_root` 也会被忽略（[lerobot_edit_dataset.py:558-564](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/scripts/lerobot_edit_dataset.py#L558-L564)），而且会重写 `meta/tasks.parquet`、`meta/episodes/` 与全部 `data/` parquet。所以一定要像上面这样先复制，再对副本操作。不同 episode 需要不同 task 时，改用 `--operation.episode_tasks '{"0": "Task A", "1": "Task B"}'`。
2. **stats 必须包含分位数**。PI05 的 `QUANTILES` 归一化需要 `meta/stats.json` 中 `observation.state` 与 `action` 的 `q01`/`q99`，缺失时训练直接报错（[normalize_processor.py:386-392](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/processor/normalize_processor.py#L386-L392)）。当前 v0.6.0 补丁栈写数据集时默认计算 `q01/q10/q50/q90/q99`，新转换的数据集无需处理；旧数据集需补算：
   ```bash
   source .shrc_local
   python libs/lerobot/src/lerobot/scripts/augment_dataset_quantile_stats.py \
     --repo-id my_dataset --root /path/to/dataset/my_dataset
   ```
3. **观测 key 与部署契约一致**。8.3 的命令会让模型 `config.json` 的 `input_features` 直接取数据集特征。部署时 `inference_service` 要求 robot_config 契约提供 `input_features` 中的每一个 key，缺一个就拒绝启动（[pipeline_policy_node.py:720-728](https://gitcode.com/openeuler/IB_Robot/blob/master/src/inference_service/inference_service/pipeline_policy_node.py#L720-L728)）。因此：
   - 相机 key（如 `observation.images.top`、`observation.images.wrist`）必须与部署用 robot_config 的契约 `observations[].key` 同名。相机数量会固化进导出的 OM/HMM 图，每路相机对应一个静态图像输入。
   - 数据集中 `observation.state` 以外的非图像 `observation.*` 特征（例如为帧检测采集的 `observation.current`）也会进入 `input_features`。PI05 模型本身不使用它们，但部署契约仍须提供这些 key。数据集的契约是录制时写入 `dataset.yaml` 的快照，转换阶段无法再改，因此按下面三种方式之一处理：
     - 部署用的 robot_config 契约同样声明这些观测（例如 `so101_single_arm.yaml` 就带 `observation.current`），数据集无需改动；
     - 部署契约不提供这些观测时（例如 `so101_pi05_ascend_310p_mock.yaml` 只有两路相机和 `observation.state`），先按第 2 节完成帧检测，再用 `lerobot-edit-dataset` 生成去掉这些特征的新数据集用于训练。`remove_feature` 输出到新目录，不改源数据集，并保留 `training_weight` 列：
       ```bash
       source .shrc_local
       lerobot-edit-dataset --repo_id my_dataset --root /path/to/dataset/my_dataset \
         --new_repo_id my_dataset_pi05 --new_root /path/to/dataset/my_dataset_pi05 \
         --operation.type remove_feature --operation.feature_names "['observation.current']"
       ```
     - 不需要帧检测时，可以直接用不含这些观测的 robot_config 契约录制。
   - 契约 `image.resize` 在采集与部署两侧保持一致。不同 resize 虽然都会被 resize-with-pad 到 224，但长宽比或画面裁切不一致会造成分布偏移。
4. **（可选）帧检测权重**。按第 2 节执行 `frame_detector` 后，数据集带 `training_weight` 列，PI05 训练可通过 `--sample_weighting.type=custom` 使用。
5. **task 长度要放得进部署的 prompt 容量**。PI05 会把离散化后的 state 拼进 prompt：`Task: <task>, State: <每维 0-255 的整数>;\nAction: `（[processor_pi05.py:83](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/policies/pi05/processor_pi05.py#L83)），所以 token 数等于固定开销（约 11）、state 部分（每维最多 4 个）与 task 本身之和。Ascend310P Torch-NPU 的 `torch-npu-fast` 部署把语言容量固定为 48 token，超出时直接拒绝请求，不会截断（见 8.8）。实测：6 维 SO-101 单臂在 state 最长时，空 task 已占 35 token，`Grasp the banana and put it on the plate` 合计 45 token，task 只剩约 13 token 的余量；12 维双臂空 task 就要 59 token，只能用容量 200 的 `torch-npu-fidelity`。可用下面的脚本按最坏情况估算：
   ```bash
   source .shrc_local
   python3 - <<'PY'
   from transformers import AutoTokenizer
   tok = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
   task, dims = "Grasp the banana and put it on the plate", 6
   prompt = f"Task: {task}, State: {' '.join(['255'] * dims)};\nAction: "
   print(len(tok(prompt)["input_ids"]))  # torch-npu-fast 要求 <= 48
   PY
   ```

可用下面的脚本快速核对 task 与分位数：

```bash
source .shrc_local
python3 - /path/to/dataset/my_dataset <<'PY'
import json, sys
import pyarrow.parquet as pq
root = sys.argv[1]
print("tasks:", pq.read_table(f"{root}/meta/tasks.parquet").to_pydict())
stats = json.load(open(f"{root}/meta/stats.json"))
for key in ("observation.state", "action"):
    print(key, "has q01/q99:", {"q01", "q99"} <= stats[key].keys())
print("features:", list(json.load(open(f"{root}/meta/info.json"))["features"]))
PY
```

### 8.3 微调命令（24 GB 显存：只训 action expert）

以下命令在 RTX 3090（24 GB）+ 当前 v0.6.0 补丁栈上实测可跑：`batch_size=8` 时峰值显存约 13.3 GB，约 2.0 s/step，可训练参数 693M / 总参数 4.1B。

```bash
source .shrc_local
lerobot-train \
  --dataset.repo_id=my_dataset \
  --dataset.root=/path/to/dataset/my_dataset \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=/path/to/output/pi05_my_task \
  --job_name=pi05_my_task \
  --batch_size=8 \
  --steps=30000 \
  --log_freq=200 \
  --save_freq=5000 \
  --num_workers=8
```

几个容易出错的地方：

- **用 `--policy.type=pi05 --policy.pretrained_path=...`，不要用 `--policy.path=lerobot/pi05_base`**。`--policy.path` 会整份加载 `pi05_base` 的 `config.json`，其中 `input_features` 是预训练数据的相机名（`base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`），不会被数据集覆盖（[factory.py:582](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/policies/factory.py#L582)）。这样训练出的模型会要求部署契约提供这三路相机，和 IB-Robot 契约对不上。用 `--policy.type` 新建配置再以 `pretrained_path` 加载权重，`input_features` 就会取数据集的 key。同理也不建议用 `--rename_map` 把相机映射到 base 的命名上。
- **确认预训练权重确实加载成功**。`PI05Policy.from_pretrained` 在找不到或读不了 `model.safetensors` 时只打印 `Returning model without loading pretrained weights`，然后**用随机初始化继续训练**，不报错（[modeling_pi05.py:1008-1011](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/policies/pi05/modeling_pi05.py#L1008-L1011)）。启动日志中必须出现 `✓ Loaded state dict from model.safetensors` 和 `All keys loaded successfully!`。离线环境可加 `HF_HUB_OFFLINE=1`，并把 `pretrained_path` 指向本地目录。
- **`--dataset.repo_id` + `--dataset.root`**：本地数据集推荐用 `root` 指向数据集根目录，`repo_id` 只作标识。
- **视频解码后端**：若日志出现 `Could not load libtorchcodec`，说明 torchcodec 与本机 FFmpeg/PyTorch 版本不匹配，可加 `--dataset.video_backend=pyav` 绕开。
- **`--output_dir` 不能已存在**：v0.6.0 在非 `--resume` 模式下遇到已存在目录直接报 `FileExistsError`，不会覆盖。

需要帧检测加权时追加 `--sample_weighting.type=custom`。它读取数据集 `training_weight` 列，按帧加权 loss（[sample_weighting.py:CustomWeighter](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/utils/sample_weighting.py#L248)）。PI05 的 `forward` 支持 `reduction="none"`，已实测可用。

需要留出验证集监控 eval loss 时追加 `--dataset.eval_split=0.1 --eval_steps=2000`（按 episode 留出 10%，每 2000 步计算一次）。

### 8.4 LoRA 微调（24 GB 显存）

LoRA 冻结全部预训练权重，只训练注入的低秩 adapter（以及 `full_training_modules` 指定的少量模块），可以在 24 GB 卡上同时微调 VLM 与 action expert。先安装 `peft`（版本约束与 `libs/lerobot/pyproject.toml` 的 `peft-dep` extra 一致）：

```bash
source .shrc_local
pip install "peft>=0.18.0,<1.0.0"
```

推荐对 VLM 语言模型与 action expert 的全部注意力/MLP 投影加 LoRA，并全量训练 action 输入输出投影与时间 MLP：

```bash
source .shrc_local
lerobot-train \
  --dataset.repo_id=my_dataset \
  --dataset.root=/path/to/dataset/my_dataset \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --peft.method_type=LORA \
  --peft.r=16 \
  --peft.lora_alpha=32 \
  --peft.target_modules='.*\.(language_model|gemma_expert)\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)' \
  --peft.full_training_modules='["action_in_proj","action_out_proj","time_mlp_in","time_mlp_out"]' \
  --output_dir=/path/to/output/pi05_my_task_lora \
  --job_name=pi05_my_task_lora \
  --batch_size=8 \
  --steps=30000 \
  --log_freq=200 \
  --save_freq=5000 \
  --num_workers=8
```

实测：`batch_size=8` 时可训练参数约 29M，峰值显存约 11.4 GB，约 2.5 s/step。

说明：

- 不传 `--peft.target_modules` 时使用 PI05 的默认目标（[modeling_pi05.py:_get_default_peft_targets](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/policies/pi05/modeling_pi05.py#L1306)）：只对 action expert 的 `q_proj`/`v_proj` 与 `action_in_proj`/`action_out_proj` 注入 LoRA，默认不全量训练任何模块（`modules_to_save` 为空）。该正则沿用 PI0 的模块名，PI05 的时间 MLP（`time_mlp_in`/`time_mlp_out`）不在其中，默认完全冻结。可训练参数仅约 1.3M（实测 bs=8 约 10.7 GB）。VLM 完全不动，容量比 8.3 的只训 action expert 还小，一般不推荐；上面的推荐命令则通过 `--peft.full_training_modules` 把 action 投影与时间 MLP 设为全量训练。
- 使用 LoRA 时不要再开 `--policy.train_expert_only` / `--policy.freeze_vision_encoder`，冻结范围由 PEFT 决定。`--sample_weighting.type=custom` 与 eval 参数用法与 8.3 相同。
- LoRA 的 checkpoint `pretrained_model/` 只含 `adapter_model.safetensors`、`adapter_config.json` 与 `use_peft=true` 的 `config.json`，**不能直接作为部署 bundle**：OM 导出只在 `config.use_lora` 时重建 LoRA 层，HMM 导出以 `strict=True` 加载完整权重，直接拿 adapter checkpoint 去导出可能得到未加载微调权重的模型。部署前先把 adapter 合并回完整权重：

```bash
source .shrc_local
python3 - /path/to/output/pi05_my_task_lora/checkpoints/last/pretrained_model /path/to/output/pi05_my_task_lora_merged <<'PY'
import shutil, sys
from pathlib import Path

from peft import PeftConfig, PeftModel

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

adapter_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
cfg = PreTrainedConfig.from_pretrained(adapter_dir)
cfg.use_peft = False
base_path = PeftConfig.from_pretrained(adapter_dir).base_model_name_or_path
policy = PI05Policy.from_pretrained(base_path, config=cfg)
merged = PeftModel.from_pretrained(policy, adapter_dir).merge_and_unload()
merged.save_pretrained(out_dir)
for f in adapter_dir.glob("policy_*processor*"):
    shutil.copy2(f, out_dir / f.name)
print(sorted(p.name for p in out_dir.iterdir()))
PY
```

合并后的目录包含 `use_peft=false` 的 `config.json`、完整 `model.safetensors` 与 processor 文件，结构与 8.3 产出的 `pretrained_model/` 相同，之后按 8.8 处理。实测合并结果可被 `PI05Policy.from_pretrained` 以严格模式加载（`All keys loaded successfully!`），并能正常输出 `(1, 50, 6)` 的动作 chunk。

### 8.5 全量 / 部分解冻微调（大显存）

只训 action expert 时 VLM 完全冻结，模型无法针对新场景调整视觉与语言表征；数据量较大或场景与预训练分布差异明显时，建议解冻更多参数：

| 方案 | 参数 | 可训练部分 | 显存需求（估算，batch 较小时） |
|------|------|-----------|------------------------------|
| 只训 action expert | `--policy.train_expert_only=true` | action expert + 投影层（693M） | 实测 bs=8 约 13.3 GB，24 GB 卡可用 |
| LoRA（8.4） | `--peft.*` | VLM + action expert 的 LoRA adapter 与 action 投影（约 29M） | 实测 bs=8 约 11.4 GB，24 GB 卡可用 |
| 冻结视觉编码器 | `--policy.freeze_vision_encoder=true` | Gemma 语言模型 + action expert | bf16 下参数、梯度、Adam 状态合计已超 24 GB，需 40 GB 以上 |
| 全量微调 | 两者均为 `false`（默认） | 全部 4.1B 参数 | 建议 80 GB 卡（A100/H100）或多卡 |

全量微调示例（只需把 8.3 命令中的 `train_expert_only` 去掉，并按显存调大 batch）：

```bash
source .shrc_local
lerobot-train \
  --dataset.repo_id=my_dataset \
  --dataset.root=/path/to/dataset/my_dataset \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=/path/to/output/pi05_my_task_full \
  --job_name=pi05_my_task_full \
  --batch_size=32 \
  --steps=30000 \
  --log_freq=200 \
  --save_freq=5000 \
  --num_workers=8
```

多卡训练可用 `accelerate launch --multi_gpu --num_processes=<N> $(which lerobot-train) ...`，此时有效 batch 为 `batch_size × N`。

### 8.6 关键参数

参数默认值来源：[configuration_pi05.py:PI05Config](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/policies/pi05/configuration_pi05.py#L30)。“部署约束”一列来自 OM/HMM 转换与运行时的实现（见 [om-convert PI05 参考](https://gitcode.com/openeuler/IB_Robot/blob/master/.agents/skills/om-convert/references/pi05.md)、[hmm-convert](https://gitcode.com/openeuler/IB_Robot/blob/master/.agents/skills/hmm-convert/SKILL.md)）。修改这些参数前先确认目标部署后端支持。

| 参数 | 含义 | 默认值 | 部署约束 / 建议 |
|------|------|--------|-----------------|
| `--policy.pretrained_path` | 预训练权重（Hub id 或本地目录） | `None` | 必填，`lerobot/pi05_base` 或其本地副本 |
| `--policy.dtype` | 模型参数精度；`bfloat16` 时视觉塔与 LayerNorm 仍保持 float32 | `float32` | 训练可用 bf16 省显存；导出时统一转 fp16，不受影响 |
| `--policy.train_expert_only` | 冻结整个 VLM，只训 action expert 与投影层 | `False` | 24 GB 卡必须开启 |
| `--policy.freeze_vision_encoder` | 只冻结 SigLIP 视觉编码器 | `False` | — |
| `--policy.gradient_checkpointing` | 梯度检查点，以计算换显存 | `False` | 建议开启 |
| `--policy.chunk_size` | 单次预测的动作步数 | `50` | 固化进 action expert 的噪声输入形状 `(1, chunk_size, max_action_dim)`，改动需重新导出；建议保持 50 |
| `--policy.n_action_steps` | LeRobot 内单次执行步数 | `50` | IB-Robot 编译后运行时返回整个 chunk，实际执行步数由动作执行侧控制 |
| `--policy.num_inference_steps` | flow matching 去噪步数 | `10` | 训练保持 10，不要修改。OM/量化 profile 按 10 步构建；Ascend310P Torch-NPU 的旧版 `torch-npu` 部署（无 `extra_parameters`）要求恰为 10；命名部署在运行时自行覆盖求解器与步数（`torch-npu-fidelity` 为 Euler 10 步，`torch-npu-fast` 为 AB2 3 步），与该字段无关 |
| `--policy.tokenizer_max_length` | prompt token 上限（task + 离散化 state 共用） | `200` | OM 的 VLM 图按 200 导出；不要修改。Ascend310P Torch-NPU 命名部署另有固定的语言容量（`torch-npu-fidelity` 为 200，`torch-npu-fast` 为 48），超出直接拒绝，task 文本要按 8.2 第 5 条控制长度 |
| `--policy.max_state_dim` / `--policy.max_action_dim` | state/action 补零到的维度 | `32` / `32` | 与预训练权重绑定，不要修改 |
| `--policy.image_resolution` | 模型输入图像尺寸 | `(224, 224)` | HMM 按每图 256 token 计算 prefix 长度；不要修改 |
| `--policy.paligemma_variant` / `--policy.action_expert_variant` | 模型规模 | `gemma_2b` / `gemma_300m` | 与预训练权重及 HMM/量化 profile 绑定；不要修改 |
| `--policy.use_relative_actions` | 训练相对动作（`action - state`） | `False` | IB-Robot 部署链路未验证相对动作，建议保持 `False` |
| `--policy.optimizer_lr` | 峰值学习率 | `2.5e-5` | — |
| `--policy.scheduler_warmup_steps` / `--policy.scheduler_decay_steps` | cosine 调度的 warmup / decay 步数 | `1000` / `30000` | `--steps` 小于 decay 步数时自动按比例缩放 |

关于状态 prompt 的一致性：PI05 的状态以离散化数字的形式进入 prompt。当前 v0.6.0 栈的 `Pi05PrepareStateTokenizerProcessorStep` 只离散化真实维度（SO-101 为 6 维），不补零到 `max_state_dim`；早期 v0.5.1 栈的 0015 补丁曾补零到 32 维，该补丁已移除，部署侧也不再兼容按旧 prompt 训练的 checkpoint。因此 PI05 必须用当前补丁栈训练，不要复用 v0.5.1 时期训练的 PI05 checkpoint。

### 8.7 不支持或不建议的用法

- **知识蒸馏**：PI05 虽然有 `--policy.kd` / `--policy.teacher_train_config` 字段，但其 `forward` 收到 teacher 后只是把普通监督 loss 记为 `total_kd_loss`，没有任何 teacher 监督项（[modeling_pi05.py:1256-1304](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/policies/pi05/modeling_pi05.py#L1256-L1304)）。开启后只会额外加载一个 teacher 占用显存，不会产生蒸馏效果。另外 KD 与 `--sample_weighting` 同时开启会直接报 `NotImplementedError`（[lerobot_train.py:129-130](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/scripts/lerobot_train.py#L129-L130)）。
- **直接部署 LoRA checkpoint**：adapter checkpoint 必须先按 8.4 合并回完整权重再进入 8.8 的部署流程。
- **`--policy.path` / `--rename_map` 对齐 base 相机名**：原因见 8.3。

### 8.8 训练产物与部署衔接

PI05 checkpoint 结构与第 9 节通用结构一致，`pretrained_model/` 下包含 `config.json`、`model.safetensors`（bf16 训练约 8.8 GB）、`policy_preprocessor.json`、`policy_postprocessor.json`、归一化统计量 `policy_*_step_*_normalizer_processor.safetensors` 和 `train_config.json`；`training_state/` 只训 action expert 时约 2.1 GB，全量微调时显著更大。

续训：

```bash
source .shrc_local
lerobot-train \
  --resume=true \
  --config_path=/path/to/output/pi05_my_task/checkpoints/last/pretrained_model/train_config.json
```

把 `checkpoints/<step>/pretrained_model` 作为部署 bundle 前，还需要完成以下步骤：

1. **检查 `config.json`**：`input_features` 的相机与状态 key 与部署 robot_config 契约逐一对应（见 8.2 第 3 条），`type` 为 `pi05`，`num_inference_steps`、`chunk_size`、`tokenizer_max_length` 符合 8.6 的部署约束；task 的 token 数在目标部署的语言容量以内（8.2 第 5 条）。
2. **把 tokenizer 放进 bundle**：运行时只接受解析到 bundle 内部的 tokenizer 引用，OM/HMM 导出也只读本地文件，不会在线下载。需要把 `google/paligemma-3b-pt-224` tokenizer `save_pretrained` 到 `<bundle>/tokenizer`，并把 `policy_preprocessor.json` 中的 `tokenizer_name` 改为 bundle 内相对路径。具体流程见 [om-convert PI05 参考](https://gitcode.com/openeuler/IB_Robot/blob/master/.agents/skills/om-convert/references/pi05.md)。
3. **注册 Torch 部署并验证**：本机 CPU/CUDA 用 `ros2 run model_utils package-torch-deployment --bundle-root <bundle> --devices cpu cuda` 注册，然后按 `ibrobot-inference-verify` skill 做端到端推理验证。Ascend310P 上用 Torch-NPU 原生推理时，改用 `ros2 run model_utils package-pi05-ascend-deployments --bundle-root <bundle>`，一次注册 `torch-npu-fidelity`（Euler 10 步，语言容量 200）和 `torch-npu-fast`（AB2 3 步，语言容量 48）两种部署。然后在 robot_config 推理管线中用 `deployment: torch-npu-fast` 或 `deployment: torch-npu-fidelity` 选择其一，并设置 `runtime_options: {model_dtype: fp16}`。参考配置为 `so101_pi05_ascend_310p_mock.yaml`，默认 fast；按 8.2 第 5 条估算的 prompt 超过 48 token 时选 fidelity。详见 [pi05_ascend_310p README](https://gitcode.com/openeuler/IB_Robot/blob/master/src/torch_models/torch_models/pi05_ascend_310p/README.md)。
4. **转换板端模型**：Ascend OM 走 `om-convert` skill（`pi05-export`），后摩 HMM 走 `hmm-convert` skill。两者校准和精度对比用的 observation batch 都应从本次训练数据集生成（`ros2 run model_utils observation-batch dataset --dataset-root <dataset> --policy-path <bundle> ...`），不要用随机输入。
5. **部署 prompt 与训练 task 一致**：robot_config 推理管线的 `default_task`，或下发 `DispatchInfer` 时的 `prompt`，必须与数据集 `meta/tasks.parquet` 中的 task 文本一致（大小写与措辞相同最稳妥）。多任务数据集按需下发对应 task。

## 9. 通用训练参数说明

以下参数在多个训练任务中复用，含义一致（来源：[train.py:TrainPipelineConfig](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/configs/train.py#L78) 与 [default.py:DatasetConfig](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/configs/default.py)）：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--dataset.repo_id` | 数据集标识；本地数据集配合 `--dataset.root` 使用，此时只作标识 | — |
| `--dataset.root` | 本地数据集根目录 | `None` |
| `--dataset.video_backend` | 视频解码后端。日志出现 `Could not load libtorchcodec`（torchcodec 与本机 FFmpeg/PyTorch 不匹配）时设为 `pyav` | 自动选择 |
| `--dataset.eval_split` | 按 episode 留出作验证集的比例，`[0, 1)` | `0.0` |
| `--policy.type` | 策略类型，`act`、`diffusion` 或 `pi05` | — |
| `--output_dir` | 训练输出根目录；非 `--resume` 时必须不存在，否则报 `FileExistsError` | — |
| `--job_name` | 任务名，用于日志与 WandB 运行名，不影响 checkpoint 路径 | 策略类型 |
| `--policy.device` | 训练设备，`cuda` / `cuda:0` / `cpu` / `mps` | — |
| `--policy.push_to_hub` | 是否在训练末尾推送到 HuggingFace Hub。默认 `True`；纯本地训练请显式设 `false`，否则启动时会因 `repo_id` 缺失报错（见 [train.py:263-270](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/configs/train.py#L263-L270)） | `True` |
| `--batch_size` | 训练批大小，蒸馏任务常显存压力更大，需按 GPU 显存调整 | `8` |
| `--steps` | 总训练步数 | `100000` |
| `--log_freq` | 每多少步打印一次训练日志 | `200` |
| `--eval_steps` | 每多少步在验证集上计算一次 `eval_loss`，`0` 表示关闭；大于 0 时要求 `--dataset.eval_split > 0` | `0` |
| `--max_eval_samples` | 每次评估最多使用的验证样本数，`0` 表示全部 | `0` |
| `--save_freq` | 每多少步保存一次 checkpoint（训练结束时总会再保存一次） | `20000` |
| `--sample_weighting.type` | 样本加权策略；`custom` 读取第 2 节写入的 `training_weight` 列。仅 ACT、PI05 支持，且不能与蒸馏同时使用 | 不启用 |
| `--num_workers` | DataLoader 工作进程数 | `4` |
| `--resume` | 续训时设为 `true`，并用 `--config_path` 指向已有 checkpoint 的 `train_config.json` | `False` |
| `--seed` | 训练与评估随机种子 | `1000` |

v0.6.0 的训练配置中已没有 `--eval_freq`（仿真环境评估频率为 `--env_eval_freq`，真机数据集训练用不到），也没有 `--ada_weight` / `--ada_weight_freq` / `--damping_coefficient`；旧命令中的这些参数需按上表替换为 `--dataset.eval_split` + `--eval_steps` 与 `--sample_weighting.type=custom`。

Checkpoint 目录结构（由 [train_utils.py:save_checkpoint](https://gitcode.com/openeuler/IB_Robot/blob/master/libs/lerobot/src/lerobot/common/train_utils.py#L93) 生成）：

```
<output_dir>/
├── checkpoints/
│   ├── <step>/                  # 形如 005000
│   │   ├── pretrained_model/
│   │   │   ├── config.json              # policy config
│   │   │   ├── model.safetensors        # policy weights（LoRA 时为 adapter_model.safetensors + adapter_config.json）
│   │   │   ├── train_config.json        # train config（teacher 蒸馏、续训用）
│   │   │   ├── policy_preprocessor.json # 预处理 pipeline
│   │   │   ├── policy_postprocessor.json
│   │   │   └── policy_*_normalizer_processor.safetensors  # 归一化统计量
│   │   └── training_state/
│   │       ├── optimizer_param_groups.json
│   │       ├── optimizer_state.safetensors
│   │       ├── rng_state.safetensors
│   │       ├── scheduler_state.json     # 仅在使用学习率调度器时存在（如 PI05）
│   │       └── training_step.json
│   └── last -> <step>/          # 指向最新 checkpoint 的符号链接
└── tensorboard/                 # TensorBoard 日志
```

## 10. 路径占位符说明

文中所有 `/path/to/xxx` 需替换为环境中的实际绝对路径。常用占位符对应关系：

| 占位符 | 含义 | 示例 |
|--------|------|------|
| `/path/to/record_root` | `episode_recorder` 录制目录，含 `dataset.yaml` 契约快照与 `episodes/` 下的 bag | `~/datasets/raw/my_task` |
| `/path/to/dataset/my_dataset` | LeRobot v3 数据集根目录（含帧权重） | `~/datasets/my_dataset` |
| `/path/to/output/...` | 训练输出根目录 | `~/training_outputs/...` |
| `/path/to/output/banana_act/checkpoints/last/pretrained_model/train_config.json` | ACT teacher 的 train_config | `~/training_outputs/banana_act/checkpoints/last/pretrained_model/train_config.json` |
| `/path/to/output/dp_base/checkpoints/last/pretrained_model/train_config.json` | DP teacher 的 train_config | `~/training_outputs/dp_base/checkpoints/last/pretrained_model/train_config.json` |
| `/path/to/pretrained/pi05_base` | `lerobot/pi05_base` 预训练权重的本地副本（可直接用 Hub id 代替） | `~/pretrained/pi05_base` |
| `/path/to/output/pi05_my_task` | PI05 微调输出根目录 | `~/training_outputs/pi05_banana` |
| `/path/to/output/pi05_my_task_lora_merged` | LoRA adapter 合并后的完整权重目录 | `~/training_outputs/pi05_banana_lora_merged` |

建议将训练用路径统一放在同一根目录下（如 `~/training_outputs` 与 `~/datasets`），并在蒸馏前确认 teacher checkpoint 路径存在且包含 `train_config.json`。
