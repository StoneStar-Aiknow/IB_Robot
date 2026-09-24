#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Selective no-smooth INT8 W8A8 Linear.

State-dict schema (differs from ``nn.Linear`` — the exporter writes these):

- ``qweight``       int8  [out, in]  — weight codes (per-out-channel grid)
- ``weight_scale``  fp32  [out]      — dequant scale per output channel
- ``bias``          fp    [out]      — optional

Activations are dynamically quantized per token with symmetric ``absmax/127``.
The deployed path deliberately does not support SmoothQuant.
"""

import torch
from torch import nn

from .int8_kernel import dynamic_int8_mm_dequant


class Int8W8A8Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        *,
        smooth: bool = False,
    ) -> None:
        super().__init__()
        if smooth:
            raise ValueError("Int8W8A8Linear only supports the no-smooth path")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        # 替换掉的 FP Linear 的接口 dtype——modeling 的 dtype 对齐检查经
        # linear_weight_dtype() 读它(int8 模块没有 fp weight 可查)。
        self.io_dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.register_buffer("qweight", torch.zeros(out_features, in_features, dtype=torch.int8, device=device))
        self.register_buffer("_npu_qweight_prepacked", None, persistent=False)
        nan_fill = {"fill_value": float("nan"), "dtype": torch.float32, "device": device}
        self.register_buffer("weight_scale", torch.full((out_features,), **nan_fill))
        self.smooth_scale = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.bias = None

    def validate_loaded(self, name: str = "") -> None:
        """Raise unless the checkpoint actually provided codes and scales
        (lerobot loads with strict=False; NaN-init catches missing floats,
        all-zero catches a missing int8 tensor)."""
        problems = []
        if not torch.isfinite(self.weight_scale).all():
            problems.append("weight_scale")
        if not self.qweight.any():
            problems.append("qweight (all zeros)")
        if problems:
            raise RuntimeError(
                f"Int8W8A8Linear {name or '(unnamed)'}: not loaded from checkpoint: " + ", ".join(problems)
            )

    def _apply(self, fn):
        result = super()._apply(fn)
        # NPU npu_quant_matmul requires fp32 x2Scale; keep quant metadata in
        # fp32 even when the surrounding model is moved to bf16 inference.
        self.weight_scale.data = self.weight_scale.data.to(dtype=torch.float32)
        return result

    def prepare_npu_qweight_layout(self, layout: str) -> dict[str, object]:
        """在模型加载和投影融合完成后，把 INT8 权重一次性整理成算子亲和布局。"""
        if layout != "nz":
            raise ValueError(f"unsupported NPU qweight layout: {layout!r}")
        if self.qweight.device.type != "npu":
            raise RuntimeError("NPU qweight prepacking requires qweight on NPU")
        import torch_npu

        # torch-npu 2.10 disables internal tensor formats by default.  Without
        # enabling them explicitly npu_format_cast silently leaves the tensor
        # in ND format, so the requested one-time NZ prepack never takes effect.
        torch.npu.config.allow_internal_format = True
        packed = torch_npu.npu_format_cast(self.qweight.t().contiguous(), 29)
        actual_format = torch_npu.get_npu_format(packed)
        # get_npu_format returned the numeric ACL id in older torch-npu
        # releases and the symbolic name in 2.10.
        if actual_format not in (29, "FRACTAL_NZ"):
            raise AssertionError(f"expected FRACTAL_NZ(29), got {actual_format}")
        self._npu_qweight_prepacked = packed
        return {
            "shape": list(packed.shape),
            "format": str(actual_format),
            "nbytes": packed.numel() * packed.element_size(),
        }

    def _kernel_qweight(self) -> torch.Tensor:
        if self._npu_qweight_prepacked is not None:
            return self._npu_qweight_prepacked
        return self.qweight.t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        y = dynamic_int8_mm_dequant(
            x.reshape(-1, self.in_features),
            self._kernel_qweight(),
            self.weight_scale,
            out_dtype=out_dtype,
        )
        y = y.reshape(*x.shape[:-1], self.out_features)
        bias = getattr(self, "_npu_static_bias", self.bias)
        if bias is not None:
            y = y + bias.to(dtype=y.dtype, device=y.device)
        return y

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, a_quant=dynamic_per_token"
        )
