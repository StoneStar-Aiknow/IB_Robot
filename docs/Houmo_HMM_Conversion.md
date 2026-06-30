# Houmo HMM 模型转换与板端推理指南

将训练好的策略模型（ACT / PI05 / SmolVLA）转换为 Houmo HMM（`.hmm`）格式，部署到 Houmo LQ50 / M50（xh2）算力卡上推理。

> **板端环境前提**：Houmo 驱动已安装（参考 [`houmo_lq50_driver_install_oee.md`](houmo_lq50_driver_install_oee.md)），`houmo_tcim_runtime_xh2` 已 pip 装入板端 venv，`source scripts/setup/houmo_hmm_env.sh` 已执行。

## 架构差异：ACT vs PI05 vs SmolVLA

| 策略 | 模块数 | 主机侧组件 | 转换工具 | 运行时编排 |
|------|--------|-----------|---------|-----------|
| **ACT** | 1（policy） | 无 | `export_onnx_hmm.py`（仓库内） | 单模块，`HMMRuntimeSession` 直接 run |
| **PI05** | 6 + embedding | action_proj / action_out_fc（CPU Linear） | Houmo 官方 `houmo-examples/pi05`（外部） | 6 模块，host 侧 denoise 循环 |
| **SmolVLA** | 3 + embedding | token embedding（CPU Embedding） | Houmo 官方 `houmo-examples/vla/smolvla`（外部） | 3 模块，host 侧 flow-matching denoise |

运行时统一通过 `HMMRuntimeSession` 按 `policy_type`（act/pi05/smolvla）分发，对上层 launch 流程透明——只需在 robot_config YAML 里配 `device: hmm`。

## 产物目录结构

转换后组装成统一的 `config.hmm.json` 清单 + model 目录，三种策略格式一致：

**ACT**（单模块）：
```
models/<act_policy>/
├── config.hmm.json          # {"policy_type":"act","backend":"hmm","artifacts":{"policy":"model.hmm"},"execution":["policy"]}
├── config.json              # 原策略配置
└── model.hmm                # 编译后的 .hmm
```

**PI05**（6 模块）：
```
models/<pi05_policy>_hmm/
├── config.hmm.json
├── config.json
└── model/
    ├── siglip.hmm                    # vision (SigLIP)
    ├── gemma_2b_prefill.hmm          # Gemma-2B prefill
    ├── gemma_expert_300m_decode.hmm  # Gemma-300M expert decode
    ├── time_mlp.hmm
    ├── action_in_proj.hmm
    ├── action_out_proj.hmm
    └── embedding.pt                  # Gemma token embedding（不在 execution）
```

**SmolVLA**（3 模块，主链 vision→prefill→action；decode 可选）：
```
models/<smolvla_policy>_hmm/
├── config.hmm.json
├── config.json
└── model/
    ├── smolvla_vision.hmm            # SmolVLM2 vision tower + connector
    ├── smolvla_llm_prefill.hmm       # LLM prefill（KV cache 构建）
    ├── smolvla_llm_decode.hmm        # LLM decode（可选，消融用）
    ├── smolvla_action.hmm            # action denoise 分支
    └── token_embedding.pt            # SmolVLM2 text embedding
```

`config.hmm.json` 规则：`artifacts` 列出所有模块（含 embedding）；`execution` 只列参与推理的模块（embedding **不**在 execution）；路径相对清单目录。

## 一、ACT → HMM（仓库脚本，最简单）

ACT 用仓库自带的 `src/model_utils/model_utils/export_onnx_hmm.py`，一步完成 ONNX 导出 + PTQ 量化 + 编译。

### 1.1 准备专用环境

xhquant/tcim 工具链的依赖树与 lerobot 冲突，**必须用独立 venv**（不要装进主 venv）：

```bash
python3 -m venv .venv-hmm
source .venv-hmm/bin/activate
pip install xhquant tcim onnx onnxsim torch   # torch 版本按 xhquant 要求
```

### 1.2 导出 + 量化 + 编译

```bash
source .shrc_local                          # 主环境（导出 ONNX 需要 lerobot）
source .venv-hmm/bin/activate               # 切到 HMM 工具链（量化编译需要 xhquant/tcim）

# 从 policy checkpoint 一步到位
python src/model_utils/model_utils/export_onnx_hmm.py \
    --policy_path models/<act_policy>/pretrained_model \
    --convert_hmm \
    --hmm_target xh2 \
    --hmm_quant_type w8a8h1_sefp \
    --hmm_ncore 2 \
    --hmm_opt_level O2
```

产物：`model.hmm` + `config.hmm.json` + `tcim_work/`（编译中间产物，可删）+ `act_ros2_hmm.onnx`（源 ONNX，保留备重新量化）。

量化类型：`w8a8h1_sefp`（默认，权重8位/激活8位，精度速度均衡）/ `w16a16_sefp`（更高精度，更大）。

## 二、PI05 / SmolVLA → HMM（Houmo 官方脚本）

PI05 / SmolVLA 是多模块 VLA，用 Houmo 官方 `houmo-examples-xh2` 的导出脚本（仓库只提供运行时契约，不内置转换脚本）。

### 2.1 准备转换环境

Houmo 工具链在 Docker 镜像里最省心（避免依赖冲突）：

```bash
# 用 Dadao-xh2-v1.3.0 镜像（24.04 版含完整编译器 xhquant/tcim）
docker run -it --gpus all \
    -v $PWD/models:/work/models \
    harbor.houmo.ai/toolchain/release:Dadao-xh2-v1.3.0-ubuntu24.04-x86.64 bash
```

> ⚠️ **必须用 24.04 镜像**：22.04 镜像只有 runtime（`houmo_tcim_runtime`），没有编译器（`xhquant`/`tcim`），无法量化编译。

### 2.2 安装依赖

镜像预装了大部分依赖，但需补齐 lerobot 示例所需的轻量包 + 修复版本冲突：

```bash
pip install draccus typing_inspect mergedeep orderly_set pyserial deepdiff imageio gymnasium
pip install 'diffusers==0.32.2'          # 镜像自带的 0.36 与 torch 2.8 不兼容
pip install 'transformers@git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi'  # PI05 需要
pip install num2words                     # SmolVLA processor 需要
```

### 2.3 准备模型

VLA 模型的 config 可能有示例 lerobot 不认识的字段（如 `compile_model`、`use_relative_actions`），draccus 严格校验会报 `DecodingError`。需要剥离这些字段——用一个剥离后的 `config.json` 副本（权重从原 `model.safetensors` 读取，不受影响）。

PI05 还需要 `paligemma-3b-pt-224` tokenizer（LLM 导出用），从 modelscope 下载到容器内。

### 2.4 PI05：导出 6 模块（4 export + 4 build 脚本）

```bash
cd houmo-examples-xh2/.../pi05   # 或解压后的 houmo-pi05-20260210/example

# 导出 + PTQ 量化 → hmonnx（用 GPU 加速）
python pi05_export_vision_xh2a_libero.py    --model_path models/<pi05>   # vision (SigLIP)
python pi05_export_llm_xh2a_libero.py       --config config/pi0/llm/pi05_gemma_2b_xh2a_2k_libero_mask.py
python pi05_export_experts_xh2a_libero.py   --config config/pi0/llm/pi05_gemma_expert_300m_xh2a_2k_libero_mask.py
python pi05_export_other_xh2a_libero.py     --model_path models/<pi05>   # time_mlp + action_in/out_proj

# 编译 → .hmm（tcim）
python build_vision.py
python build_gemma_2b_prefill.py
python build_gemma_expert_300m_decode.py
python build_pi05_action_time_linear.py
```

每个 export 脚本默认假设 LIBERO dummy batch，自定义模型可能有 batch key 不匹配（如 banana-pick 用 `top`/`wrist` 而 LIBERO 用 `image`/`image2`）——export 脚本里的 `predict_action_chunk` sanity test 无害但会因 key 不符报错，可跳过（导出只依赖 SigLIP/Gemma 子模块，与 batch key 无关）。

### 2.5 SmolVLA：导出 3 模块（3 export + build）

SmolVLA 主链是 `vision → prefill → action`（decode 可选，消融用）：

```bash
cd houmo-examples-xh2/hmodel/xh2/examples/vla/smolvla

# prefix_length 必须和实际 embed_prefix 输出一致（512x512 + 2 相机 ≈ 177）
python smolvla_export_vision_xh2a.py \
    --model_path models/<smolvla> --lerobot_src ./lerobot/src --device cuda \
    --output_name smolvla_vision --quant_type w8a8h1_sefp

python smolvla_export_llm_kvcache_xh2a.py \
    --model_path models/<smolvla> --lerobot_src ./lerobot/src --device cuda \
    --prefix_length 177 --suffix_length 50 --output_name smolvla_llm

python smolvla_export_action_xh2a.py \
    --model_path models/<smolvla> --lerobot_src ./lerobot/src --device cpu \
    --prefix_length 177 --prefill_meta work_dirs/smolvla_llm_kvcache/meta_info.json \
    --output_name smolvla_action
```

> ⚠️ **action 导出用 CPU + float32**：cuda + fp16 时 denoise 内部 dtype 混用会报 `mat1 and mat2 must have the same dtype`。

用 `tcim.build_from_hmonnx` 把 3 个 hmonnx 编译成 `.hmm`（参考 PI05 的 build 脚本，`llm_opt=True` 给 prefill，`llm_opt=False` 给 vision/action）。

### 2.6 组装产物 + 写清单

把 `.hmm` + `embedding.pt` 组装成上面的产物目录结构，手写 `config.hmm.json`（官方脚本不生成清单）。

## 三、板端部署

### 3.1 装 runtime 到板端 venv

```bash
ssh OPi_20T
# 把 aarch64 Runtime SDK 装进 IB_Robot venv（合法方式，不靠 PYTHONPATH 指向 /root）
/IB_Robot/venv/bin/pip install /root/houmo_tcim_runtime_xh2_linux_aarch64-1.3.0.tar.gz
```

### 3.2 配置板端 runtime 环境

```bash
source scripts/setup/houmo_hmm_env.sh
# 输出: [houmo_hmm_env] LQ50 device_num=1 (TCIM_BACKEND=Xh2HalBackend)
```

> ⚠️ **关键**：必须用 `TCIM_BACKEND=Xh2HalBackend` + `HOUMO_TARGET=xh2`（走 HAL backend，用 `libhal_xh2a.so`）。**不要**用 `HDPL_PLATFORM=ASIC`——那会让 tcim_lite 去找 `libhdplrt_asic.so`，而该库**不在** aarch64 Runtime SDK 里，会导致 `InitDevice` 失败、`device_num=0`。

### 3.3 验证 NPU 可用

```bash
python3 -c "import tcim_lite.runtime as r; print('device_num:', r.get_device_num())"
# 应输出 device_num: 1
```

### 3.4 配置 YAML + 启动

在 robot_config YAML（如 `so101_single_arm.yaml`）加 HMM 模型并选中：

```yaml
models:
  <name>_hmm:
    path: models/<pi05_policy>_hmm     # 指向含 config.hmm.json 的目录
    policy_type: pi05                  # act | pi05 | smolvla
    device: hmm
    lerobot_norm_mode: range_m100_100

control_modes:
  model_inference:
    inference:
      model: <name>_hmm                # 选中上面的 HMM 模型
```

启动（`device: hmm` 由 YAML model 决定，非 CLI 参数）：

```bash
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm \
    control_mode:=model_inference use_sim:=true sim_platform:=mock
```

## 四、已知坑与排查

### 转换侧

| 问题 | 原因 | 解决 |
|------|------|------|
| `DecodingError: fields ... not valid for XxxConfig` | 自定义模型 config 有示例 lerobot 不认识的字段 | 剥离多余字段（如 `compile_model`、`use_relative_actions`），用剥离后的 config.json 副本 |
| `ModuleNotFoundError: num2words` / `draccus` 等 | 容器镜像未预装 | `pip install` 补齐 |
| `diffusers RuntimeError: name 'logger' is not defined` | 镜像 diffusers 0.36 与 torch 2.8 不兼容 | `pip install 'diffusers==0.32.2'` |
| SmolVLA vision `Unsupported ops: ['NonZero']` | SmolVLM2 布尔索引 `position_ids[][mask.view(-1)]` 编译成 NonZero | monkey-patch `SmolVLMVisionEmbeddings.forward`，固定全图输入时直接算 position_ids |
| SmolVLA action `dtype Float and Half` | cuda+fp16 时 denoise 内部 dtype 混用 | action 导出改用 `--device cpu` |
| PI05 `libhdplrt_asic.so not found` | 误设 `HDPL_PLATFORM=ASIC` | 改用 `TCIM_BACKEND=Xh2HalBackend` |

### 运行时侧

| 问题 | 原因 | 解决 |
|------|------|------|
| `device_num: 0` / `InitDevice failed` | 缺 `TCIM_BACKEND=Xh2HalBackend` 或 `libhal_xh2a.so` 不在 LD_LIBRARY_PATH | `source scripts/setup/houmo_hmm_env.sh` |
| `set_input error: Status.UNINITIALIZED` | KV-cache 设备指针共享用错 API（用了 `set_input`/`get_dev_input`） | 用 `set_dev_input(name, prefill.get_dev_input(name))`（PI05）或 `set_dev_input(name, prefill.get_dev_output(name))`（SmolVLA） |
| `Module set_input error`（PI05 prefill） | input 名字写错 | 正确名：`input_1` / `valid_length` / `current_length` / `attention_mask`（非 `inputs_embeds` / `past_seq_length`） |
| PI05 `siglip.hmm` NPU 执行 `ret:110` | PTQ 随机校准数据在 NPU 上产生异常导致挂起 | 用真实数据重新校准量化（模型编译层问题，非 runtime 代码问题） |

## 参考

- Houmo 官方示例：`houmo-examples-xh2`（PI05：`examples/pi05`；SmolVLA：`hmodel/xh2/examples/vla/smolvla`）
- 仓库运行时：`src/inference_service/inference_service/core/hmm/`（pi05/PI05HMMModel.py、smolvla/SmolVLAHMMModel.py、policy_wrapper.py）
- 仓库 ACT 转换：`src/model_utils/model_utils/export_onnx_hmm.py`
- 板端环境：`scripts/setup/houmo_hmm_env.sh`
- 驱动安装：[`houmo_lq50_driver_install_oee.md`](houmo_lq50_driver_install_oee.md)
