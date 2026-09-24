# PI0.5 Ascend910B native Torch provider

This package is an independently owned native PyTorch PI0.5 implementation for
Ascend 910B-family NPUs. It is adapted from the current
`lerobot_offical` PI0.5 NPU implementation and keeps the Apache-2.0 notices in
the copied source files.

The package is intentionally self-contained: its local PI Gemma, SigLIP NPU,
and selective INT8 quantization modules do not require edits to
`libs/lerobot`. Imports are adapted to the target LeRobot v0.6 public module
paths. The provider enables NPU fused operations, QKV/Prefix MLP fusion,
post-fusion INT8 `FRACTAL_NZ` prepacking, and the fixed TorchAir inference
graph after strict local checkpoint loading.

Validated source environment: Ascend 910B3, CANN 9.2, PyTorch/Torch-NPU
2.10.0, and Transformers 5.5.x. The provider fails closed unless the physical
SKU contains `Ascend910B` or uses CANN/Torch-NPU's `Ascend910_93*` name for a
910B 93-series product, both Torch packages have base version 2.10.0, and
Transformers is in `[5.4, 5.6)`. The optimized path accepts only `native` or
`bf16` runtime dtype and materializes the model as BF16. The target runtime
must still verify its driver/CANN compatibility before deployment.

Expected bundle contents include `config.json`, `model.safetensors`, local
processor files, and a bundled tokenizer. Quantized bundles may include a
`quantization` object in `config.json`; the provider converts it to the local
`QuantizationConfig` after removing fields unknown to target LeRobot v0.6.

The source implementation was copied from `lerobot_offical` commit `f30268a`
working-tree files. The source working tree was not modified.

For the Selective-99 weight schema, two-stage generation commands, environment
contract, and validation record, see
[`quantization/README.md`](quantization/README.md).
