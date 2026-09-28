# PI0.5 推理补丁更新说明

本文记录 `lerobot_pi05_npu_inference_b74a551.patch` 的更新范围、参考实现和本轮验证结果。完整安装、Tokenizer 下载、量化和测试命令见同目录 `README.md`。

该补丁虽然由 IB_Robot 仓库托管，但实际目标是独立的官方 LeRobot 源码树。它只支持下表所列
`huggingface/lerobot` 精确基线，不接入 IB_Robot runtime，也不修改或依赖 `libs/lerobot` 子模块。

## 1. 基线

| 项目 | 值 |
| --- | --- |
| 官方仓库 | `https://github.com/huggingface/lerobot.git` |
| 精确提交 | `b74a551d38f6cf33ddde8e55b0c6f5a9b0c42e85` |
| LeRobot 版本 | `0.5.2` |
| 优化参考仓库 | `https://github.com/Launch-pad-Infinity-Edge/lerobot_offical.git` |
| 优化参考提交 | `a04ac0e9ea89d5df2a44a9e1688a4949750b9bb9` |
| 图像前处理参考仓库 | `https://github.com/Launch-pad-Infinity-Edge/lerobot-pi0-new.git` |
| 图像前处理参考提交 | `cc4f26ecf6970526110d77fad36ef69d1409ae71` |
| 更新日期 | `2026-09-28` |

补丁从上述官方精确提交生成，不包含基线说明、模型权重、Git LFS 资产、Tokenizer、编译缓存或性能测试脚本。IB_Robot 仅对分发文件进行版本管理；补丁应用、安装和测试均在独立 LeRobot checkout 中完成。补丁完整性由 IB_Robot 的 Git 对象保证，不额外维护 `SHA256SUMS`。

## 2. 本次接口收敛

- 唯一性能入口收敛为包外 `test/pi05_latency.py`；
- 新增必选 `--test-mode {e2e,model}`，单次执行只统计一种明确口径；
- `e2e` 包含 preprocessor、Tokenizer、设备搬运和 postprocessor，必须提供 Tokenizer；
- `model` 直接计时 `policy.model.sample_actions`，使用预构造 NPU tensor，不需要 Tokenizer；
- 删除补丁内重复的 benchmark、profiler 和共享测试运行时；
- 优化参数只保留图编译、量化、降采样与 Token 长度；
- BF16 与 INT8 权重路径改为按推理模式条件必填，量化推理不再要求传入原始权重；
- `--downsample` 统一映射到固定 AB2/6，替代面向实现细节的 `--ab2`；
- `--device` 必须显式指定 `npu:<index>`，并在模型构建前设置当前 NPU；
- INT8 权重预排布改用公开的 `torch.npu.set_option()` 控制内部格式，兼容不提供可读 `torch.npu.config.allow_internal_format` 属性的 Torch-NPU 版本；
- 图编译启动前检查 `tbe` 导入，避免进入 GE 编译后才暴露环境错误；
- P90 使用 `int((N - 1) * 0.9)` 的离散下界索引，默认 10 次测量不会退化为最大值。

测试脚本放在 `.patch` 外，模型补丁只维护推理运行时和离线量化工具。

## 3. 模型优化内容

包外测试脚本的 `--graph-compile` 会把模型接口收敛为固定的 TorchAir 全优化组合：prefix/denoise 双图、QKV 融合、BF16 Vision PFA、shared-prefix FIAS、ViT FastGELU、静态 mask/RoPE/查找表复用、QKV 单次 Rotary、三角函数去重、静态 RMSNorm gamma、NPU RMSNorm/AddRMSNorm 和 AdaRMS 全阶段 bias folding。直接调用模型 API 时，应按 `README.md` 给出的完整参数组显式开启，不能只根据图编译状态推断其他融合已经生效。

Selective-99 no-smooth INT8 路径额外提供动态 per-token 激活量化、Prefix Gate/Up 融合和 hot81 `FRACTAL_NZ` 权重预排布；DiT 保持 BF16。INT8 OProj 恢复标准量化 Linear 输出及内置 `npu_add_rms_norm` 路径。降采样路径以 AB2/6 替代 Euler/10。

本次删除了 OProj 反量化、残差、RMSNorm 和动态量化的一体化扩展算子，以及对应的模型开关、TorchAir 转换器和测试入口。

在线 NPU rollout 额外增加原始图像 hook：环境产生的 BHWC `uint8` 图像先以紧凑格式搬到 NPU，再完成 BCHW/FP32/`[0,1]` 转换，H2D 字节数是 CPU 先转 FP32 路径的四分之一。模型侧 resize、`[-1,1]` 归一化和 mask 构造全程保持 BCHW，避免原有 BCHW→BHWC→BCHW 往返。非 NPU policy 和需要回传原始 observation 的评测仍使用原 CPU 路径。

## 4. 量化工具

补丁保留：

- `scripts/npu/pi05/make_int8_rtn_perchannel_source.py`；
- `scripts/npu/pi05/make_selective_positive_quant_pi05_checkpoint.py`；
- `docs/QUANTIZED_CHECKPOINTS.md`；
- `src/lerobot/quantization/` 运行时实现。

第一阶段生成 99 层逐输出通道对称 RTN 权重源；第二阶段校验源哈希和层计数，复制未量化参数及 processor sidecar，并生成 `quantization.smooth=false` 的完整 runtime checkpoint。

## 5. 性能验证

此前已从真实 BF16 checkpoint 完成两阶段量化：RTN 权重源 99 层、约 1.9 GiB、27.031 s；完整 runtime checkpoint 包含 198 个量化 tensor、约 5.1 GiB、30.208 s。

本轮在单张 Ascend 910B3 上使用 Driver `25.5.0`、CANN `9.2.0`、Python `3.12.13`、PyTorch/Torch-NPU `2.10.0`、Selective-99 INT8、TorchAir 全图和 AB2/6 完成 10 次 warmup、100 次正式 E2E 测量：mean `48.492 ms`、median `48.817 ms`、P90 `48.944 ms`。与图像优化前同配置 mean `48.533 ms` 基本持平；该合成 E2E 输入已是 BCHW/FP32，主要用于回归检查，不覆盖原始 BHWC/uint8 提前 H2D 的收益。

## 6. 本轮验证结果

- 补丁包含 21 个文件，在 `b74a551d38f6cf33ddde8e55b0c6f5a9b0c42e85` 干净工作树通过 `git apply --check --whitespace=error-all`；
- 应用后的 21 个文件与原运行时及图像前处理参考实现中的可分发内容逐文件一致；
- PI0.5、量化配置和 `Int8W8A8Linear` 导入通过，两个量化工具的 `--help` 可执行；
- 量化与 AB2/6 单元测试共 21 项通过；
- 官方实现通过 Ruff 检查、Ruff 格式检查、`git diff --check`、敏感信息和本地绝对路径扫描；
- 包外 `test/pi05_latency.py` 通过语法检查、`--help` 检查和 IB_Robot pre-commit；
- 源码、测试脚本与文档不再引用已删除的自定义扩展算子；
- 模型源码中不包含 benchmark、latency、profiler 或 synthetic 输入逻辑；
- 图像前处理、观测 hook 和相关量化/降采样定向测试共 54 项通过，另有 1 项按环境预期跳过；

本轮同时完成补丁应用、导入、单元测试、静态质量和第 5 节单卡完整 E2E 回归。
