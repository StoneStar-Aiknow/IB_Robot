# LeRobot PI0.5 Ascend NPU 推理优化

本目录提供基于官方 LeRobot `0.5.2` 精确基线制作的 PI0.5 NPU 推理优化补丁、更新说明和一个独立时延测试脚本。模型优化与量化工具由 `.patch` 安装；性能测试脚本放在补丁外，避免测试逻辑进入模型代码或补丁源文件。

本目录虽然由 IB_Robot 仓库托管，但它是独立的 LeRobot 补丁分发包。补丁只应用到官方
`huggingface/lerobot` 的 `main` 分支精确提交
`b74a551d38f6cf33ddde8e55b0c6f5a9b0c42e85`，运行和测试均在该独立 LeRobot checkout 中完成。
它不依赖 IB_Robot 的推理 provider、ROS 工作区、`.shrc_local` 或 `libs/lerobot` 子模块，也不能应用到
IB_Robot 当前使用的 LeRobot v0.6 基线。

## 1. 包内容

```text
patch/
├── README.md
├── PATCH_UPDATE.md
├── lerobot_pi05_npu_inference_b74a551.patch
└── test/
    └── pi05_latency.py
```

不包含模型权重、Tokenizer、Git LFS 测试资产或 CANN 编译缓存。

## 2. 官方基线

| 项目 | 值 |
| --- | --- |
| 官方仓库 | `https://github.com/huggingface/lerobot.git` |
| 官方分支 | `main` |
| 精确提交 | `b74a551d38f6cf33ddde8e55b0c6f5a9b0c42e85` |
| LeRobot 版本 | `0.5.2` |
| 补丁文件 | `lerobot_pi05_npu_inference_b74a551.patch` |

补丁只保证应用到该精确提交。拉取基线时显式跳过官方仓库的 Git LFS 大文件；PI0.5 权重按本文单独下载。

## 3. 优化路径

测试入口只暴露四类优化参数：

| 参数 | 默认 | 作用 |
| --- | ---: | --- |
| `--graph-compile` / `--no-graph-compile` | 开 | 启用 TorchAir prefix/denoise 双图及全部已验证正收益优化。 |
| `--quantization` / `--no-quantization` | 关 | 使用 Selective-99 no-smooth INT8 checkpoint；依赖图编译。 |
| `--downsample` / `--no-downsample` | 关 | 使用 AB2/6 替代 Euler/10；依赖图编译。 |
| `--token-length N` | `183` | 设置 E2E Tokenizer padding 或 model 模式合成 token 的静态长度。 |

`--graph-compile --quantization --downsample` 是当前全优化路径，标识为 `pi05_int8_torchair_ab2_6`。量化或降采样与 `--no-graph-compile` 同时使用会在加载模型前被拒绝。

### 3.1 图编译固定优化

- TorchAir fullgraph prefix 图与单张完整 denoise 图；
- Gemma 与 SigLIP QKV 权重融合；
- BF16 Vision PFA 与 shared-prefix FIAS；
- ViT 27 层 FastGELU；
- 静态 mask、position IDs、RoPE 和 denoise 查找表复用；
- QKV 单次 Rotary 与三角函数去重；
- 静态 RMSNorm gamma、NPU RMSNorm/AddRMSNorm；
- AdaRMS modulation 预计算及 shift 到 QKV、MLP、action output bias 的全阶段折叠；
- Prefix 多相机批处理和图安全 Linear/Attention 路径。

### 3.2 Selective-99 no-smooth INT8

量化 ViT `mlp.fc2` 27 层、Prefix LLM `self_attn.o_proj` 18 层和 Prefix `gate/up/down` 54 层，共 99 层；DiT action expert 保持 BF16。Prefix Gate/Up 融合后有 81 个活跃 INT8 投影，并使用 dynamic per-token 激活量化和 hot81 `FRACTAL_NZ` 权重布局。INT8 OProj 通过标准量化 Linear 输出，并继续进入内置 `npu_add_rms_norm` 路径。

### 3.3 图像前处理

PI0.5 NPU rollout 在原始 BHWC `uint8` 阶段执行 H2D，再在设备上转换为 BCHW/FP32 和 `[0,1]`，避免 CPU 侧扩成四倍大小的 FP32 图像后再传输。模型侧保持 BCHW 完成 resize、`[-1,1]` 归一化和 mask 构造，不再执行 layout 往返。非 NPU policy 与需要回传原始 observation 的评测保持原 CPU 行为。

## 4. 环境准备

已验证组合为 Ascend 910B3、Driver `25.5.0`、CANN `9.2.0` 安装树、Python `3.12.13`、PyTorch `2.10.0` 和 Torch-NPU `2.10.0`。其他硬件应按昇腾兼容矩阵选择版本。

```bash
conda create -n lerobot-pi05 python=3.12 -y
conda activate lerobot-pi05

python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0
python -m pip install torch_npu
# python -m pip install /path/to/torch_npu-2.10.0.whl

source /path/to/Ascend/cann/set_env.sh
python -c "import torch, torch_npu, tbe; print(torch.__version__, torch_npu.__version__, torch.npu.is_available(), tbe.__file__)"
```

## 5. 拉取基线并应用补丁

```bash
export IBROBOT_REPO=/path/to/IB_Robot
export WORKSPACE=/path/to/workspace
export LEROBOT_REPO="$WORKSPACE/lerobot"
export PATCH_DIR="$IBROBOT_REPO/src/torch_models/torch_models/pi05_ascend_910b/patch"

test -f "$PATCH_DIR/lerobot_pi05_npu_inference_b74a551.patch"

cd "$WORKSPACE"
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/huggingface/lerobot.git lerobot
cd "$LEROBOT_REPO"
GIT_LFS_SKIP_SMUDGE=1 git checkout --detach b74a551d38f6cf33ddde8e55b0c6f5a9b0c42e85

git apply --check "$PATCH_DIR/lerobot_pi05_npu_inference_b74a551.patch"
git apply --index "$PATCH_DIR/lerobot_pi05_npu_inference_b74a551.patch"
git diff --cached --check
```

上述 `IBROBOT_REPO` 和 `PATCH_DIR` 只用于定位由 IB_Robot 托管的补丁文件；完成应用后，安装、模型下载、
权重量化和时延测试全部在 `LEROBOT_REPO` 指向的独立官方 LeRobot 基线中执行。

`GIT_LFS_SKIP_SMUDGE=1` 必须同时用于 `git clone` 和精确基线的 `git checkout`。该写法只对紧随其后的单条命令生效；如果 checkout 未设置，Git 从最新 `main` 切换到指定提交时仍可能触发 Git LFS 下载。

基线说明不打包进补丁；基线提交由本 README 明确记录。补丁自身由当前仓库的 Git 对象保证完整性，不重复维护 `SHA256SUMS`。

### 5.1 撤销补丁并恢复官方基线

以下命令会丢弃仓库内尚未提交的受跟踪文件修改，执行前先用 `git status --short` 确认没有需要保留的工作。按照本文使用 `git apply --index` 安装补丁时，硬重置即可同时撤销修改和删除补丁新增文件：

```bash
cd "$LEROBOT_REPO"
export LEROBOT_BASE=b74a551d38f6cf33ddde8e55b0c6f5a9b0c42e85

git status --short
git reset --hard "$LEROBOT_BASE"

test "$(git rev-parse HEAD)" = "$LEROBOT_BASE"
git status --short --untracked-files=no
```

如果此前使用的是不带 `--index` 的 `git apply`，新增文件可能保持为未跟踪状态。先预览，再只清理本补丁引入的路径：

```bash
git clean -n -d -- \
  docs/QUANTIZED_CHECKPOINTS.md \
  scripts/npu/README.md \
  scripts/npu/pi05/README.md \
  scripts/npu/pi05/__init__.py \
  scripts/npu/pi05/make_int8_rtn_perchannel_source.py \
  scripts/npu/pi05/make_selective_positive_quant_pi05_checkpoint.py \
  src/lerobot/policies/pi05/vision_siglip_npu.py \
  src/lerobot/quantization/__init__.py \
  src/lerobot/quantization/config.py \
  src/lerobot/quantization/fuse.py \
  src/lerobot/quantization/int8_kernel.py \
  src/lerobot/quantization/linear_int8.py \
  src/lerobot/quantization/replace.py

git clean -f -d -- \
  docs/QUANTIZED_CHECKPOINTS.md \
  scripts/npu/README.md \
  scripts/npu/pi05/README.md \
  scripts/npu/pi05/__init__.py \
  scripts/npu/pi05/make_int8_rtn_perchannel_source.py \
  scripts/npu/pi05/make_selective_positive_quant_pi05_checkpoint.py \
  src/lerobot/policies/pi05/vision_siglip_npu.py \
  src/lerobot/quantization/__init__.py \
  src/lerobot/quantization/config.py \
  src/lerobot/quantization/fuse.py \
  src/lerobot/quantization/int8_kernel.py \
  src/lerobot/quantization/linear_int8.py \
  src/lerobot/quantization/replace.py
```

上述定向清理不会删除仓库根目录下单独保存的模型权重或输出目录。

## 6. 安装到 Conda 环境

```bash
conda activate lerobot-pi05
source /path/to/Ascend/cann/set_env.sh

cd "$LEROBOT_REPO"
python -m pip install -e ".[pi]"
python -c "import lerobot; from lerobot.policies.pi05.modeling_pi05 import PI05Policy; print(lerobot.__version__)"
```

依赖和可编辑源码均安装到当前 Conda 环境，不需要再设置 `PYTHONPATH`。

## 7. 下载模型权重

```bash
export PI05_BF16_MODEL=/path/to/pi05_bf16_checkpoint

hf download lerobot/pi05_libero_finetuned_v044 \
  --revision 6348c67dbbd696bdf89321c07b107434cbee1baf \
  --local-dir "$PI05_BF16_MODEL"

test -f "$PI05_BF16_MODEL/config.json"
test -f "$PI05_BF16_MODEL/model.safetensors"
```

该 checkpoint 的 `model.safetensors` 大小为 `7,473,096,344` 字节，SHA256 为 `877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae`。

## 8. 下载和使用 Tokenizer

`google/paligemma-3b-pt-224` 是受限仓库。先在 Hugging Face 页面接受 Gemma 使用条款，再登录。只下载 Tokenizer 文件，不需要下载 PaliGemma 权重：

```bash
python -m pip install -U "huggingface_hub[cli]"
hf auth login

export PI05_TOKENIZER=/path/to/google-paligemma-3b-pt-224
hf download google/paligemma-3b-pt-224 \
  --include "tokenizer*" \
  --include "special_tokens_map.json" \
  --include "added_tokens.json" \
  --include "config.json" \
  --local-dir "$PI05_TOKENIZER"

test -f "$PI05_TOKENIZER/tokenizer.json"
```

仅 `--test-mode e2e` 必须传入 `--tokenizer-path "$PI05_TOKENIZER"`。该模式使用官方 preprocessor 将任务文本和离散化 state 编码，并按 `--token-length` padding/crop，因此完整 E2E 包含真实 Tokenizer 调用。`--test-mode model` 使用预构造 token tensor，不需要下载 Tokenizer。

## 9. 生成 no-smooth INT8 权重

补丁安装两个量化工具，严格执行两阶段 Selective-99 RTN 流程：

```bash
export RTN_SOURCE=/path/to/pi05_sel99_rtn_source
export PI05_INT8_MODEL=/path/to/pi05_sel99_rtn_runtime

cd "$LEROBOT_REPO"
python scripts/npu/pi05/make_int8_rtn_perchannel_source.py \
  --fp-model-path "$PI05_BF16_MODEL" \
  --output-dir "$RTN_SOURCE"

python scripts/npu/pi05/make_selective_positive_quant_pi05_checkpoint.py \
  --fp-model-path "$PI05_BF16_MODEL" \
  --quant-model-path "$RTN_SOURCE" \
  --source-manifest "$RTN_SOURCE/selection_provenance.csv" \
  --output-dir "$PI05_INT8_MODEL"
```

两个输出目录必须事先不存在。第二阶段会校验 BF16 源哈希、RTN 元数据、99 层计数和分组，并写入 `quantization.smooth=false`。

## 10. 单脚本双模式测试

测试脚本位于补丁文件外：

```text
$PATCH_DIR/test/pi05_latency.py
```

每次执行必须通过 `--test-mode` 选择一种计时模式：

| 模式 | 输出项 | 计时范围 | Tokenizer |
| --- | --- | --- | --- |
| `e2e` | `full_e2e` | 合成 BCHW/FP32 图像、state、task → preprocessor、Tokenizer、归一化、CPU 到 NPU 搬运 → `predict_action_chunk` → postprocessor → NPU 同步。 | 必须提供。 |
| `model` | `model_inference` | 预构造 NPU 图像、image mask、token、token mask 和 noise → `policy.model.sample_actions` → NPU 同步。 | 不需要。 |

`model` 模式在计时前生成已经归一化到 `[-1, 1]` 的 224×224 合成图像、全有效固定 token 和固定噪声，并全部搬到目标 NPU。它不执行 preprocessor、Tokenizer、postprocessor、policy 图像整理或动作维度裁剪，只用于测量模型主推理路径，不代表真实任务语义或 LIBERO 成功率。

权重参数按模式条件必填：`--quantization` 只要求完整的 `--int8-model-path`；`--no-quantization` 只要求 `--bf16-model-path`。原始 BF16 权重仅在生成量化 checkpoint 时需要，不参与完整 INT8 runtime checkpoint 的推理。

全优化完整 E2E：

```bash
cd "$LEROBOT_REPO"
python -u "$PATCH_DIR/test/pi05_latency.py" \
  --test-mode e2e \
  --int8-model-path "$PI05_INT8_MODEL" \
  --tokenizer-path "$PI05_TOKENIZER" \
  --device npu:0 \
  --graph-compile \
  --quantization \
  --downsample \
  --token-length 183 \
  --batch-size 1 \
  --warmup 10 \
  --iterations 100
```

全优化纯模型推理，无需 Tokenizer：

```bash
python -u "$PATCH_DIR/test/pi05_latency.py" \
  --test-mode model \
  --int8-model-path "$PI05_INT8_MODEL" \
  --device npu:0 \
  --graph-compile \
  --quantization \
  --downsample \
  --token-length 183 \
  --batch-size 1 \
  --warmup 10 \
  --iterations 100
```

BF16 图编译 Euler/10：

```bash
python -u "$PATCH_DIR/test/pi05_latency.py" \
  --test-mode e2e \
  --bf16-model-path "$PI05_BF16_MODEL" \
  --tokenizer-path "$PI05_TOKENIZER" \
  --device npu:0 \
  --graph-compile --no-quantization --no-downsample \
  --token-length 183 --warmup 10 --iterations 100
```

无优化 NPU eager：

```bash
python -u "$PATCH_DIR/test/pi05_latency.py" \
  --test-mode e2e \
  --bf16-model-path "$PI05_BF16_MODEL" \
  --tokenizer-path "$PI05_TOKENIZER" \
  --device npu:0 \
  --no-graph-compile --no-quantization --no-downsample \
  --token-length 183 --warmup 10 --iterations 100
```

`--device` 必须显式传入 `npu:<index>`，脚本会调用 `torch.npu.set_device` 并校验当前设备，不能使用隐式默认 NPU。

## 11. 输出文件

默认输出到 LeRobot 根目录：

```text
output/<YYYYMMDD_HHMMSS>_pi05_inference/
├── actions_full_e2e.npy 或 actions_model_inference.npy
├── iteration_records.csv
├── iteration_records.json
├── pi05_latency.log
├── run_metadata.json
├── summary.csv
└── summary.json
```

输出目录必须事先不存在；脚本会拒绝覆盖已有目录。使用 `--output-dir` 时也应为每轮测试指定新的目录，
保证不同配置和重复运行的结果不会混写。

`summary` 每轮只有一行，对应本次选择的 `full_e2e` 或 `model_inference`。`latency_*_ms` 是整批时延，并额外提供均摊单样本 Mean 与吞吐。不同模式、batch 或 Token 长度应在独立进程中分别编译。

两种模式都使用固定种子的 synthetic 输入和真实模型权重；只有 `e2e` 使用真实 Tokenizer 与固定任务文本。结果只代表性能，不代表 LIBERO 任务成功率。

## 12. 常见问题

### 找不到 `tbe`

```bash
source /path/to/Ascend/cann/set_env.sh
python -c "import tbe; print(tbe.__file__)"
```

脚本仅在图编译路径要求 `tbe`；导入失败时会在加载模型前停止。

### Torch-NPU 不提供 `allow_internal_format` 属性

部分 Torch-NPU 版本允许设置、但不允许读取 `torch.npu.config.allow_internal_format`。
当前脚本通过公开的 `torch.npu.set_option({"ALLOW_INTERNAL_FORMAT": ...})` 在 INT8
权重预排布前后显式启用和关闭内部格式，不依赖该私有属性的 getter。若仍出现同名
`AttributeError`，说明执行的是旧版包外测试脚本，应更新本目录的 `test/pi05_latency.py`。

### 日志提示 CUDA 不可用并切换到 CPU

加载 checkpoint 配置时可能先出现以下警告：

```text
No accelerated backend detected. Using default cpu, this will be slow.
Device 'cuda' is not available. Switching to 'cpu'.
```

这是 checkpoint 中默认 `device=cuda` 在准备阶段的配置探测信息，不表示正式计时在 CPU 上执行。测试脚本随后会使用显式传入的 `--device npu:<index>` 调用 `torch.npu.set_device` 并校验当前设备；非 NPU 设备会在加载模型前被拒绝。运行后可检查 `run_metadata.json` 和 `summary.json` 中的 `device` 字段，并通过 `npu-smi info` 确认推理进程位于目标 NPU。

### 首次运行很慢

首次运行包含模型构建、权重加载、INT8 `FRACTAL_NZ` 预排布、GE/TBE 编译和算子 tiling。正式比较只使用 warmup 后的 measured 数据。

### Token 长度如何选择

checkpoint 最大长度为 200；`--token-length` 固定本次静态图长度。不同长度会生成不同 shape 和 tiling，必须独立进程测试。
