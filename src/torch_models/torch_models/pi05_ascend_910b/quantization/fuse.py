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
"""QKV and Gate/Up fusion for Selective-99 no-smooth INT8 Linears.

A fused QKV projection folds three GEMMs over the SAME input into one. For a
quantized Linear remains exact because dynamic per-token activation
quantization depends only on the shared input. Fusion is a pure concatenation
along the output dimension, and the fused module is an
``Int8W8A8Linear`` (same interface as the FP fused ``nn.Linear``: the NPU PFA
consumer's ``attn.qkv(normed).split(...)`` works unchanged)."""

import torch
from torch import nn

from .config import QuantizationConfig
from .linear_int8 import Int8W8A8Linear


def _fuse_int8_linears(members: "tuple[Int8W8A8Linear, ...]", what: str) -> Int8W8A8Linear:
    """Concatenate Int8W8A8Linear projections sharing one input into a single
    fused module (bit-identical to running them separately).

    Raises TypeError on non-Int8W8A8Linear members and rejects any unexpected
    smoothed module so the deployed path remains strictly no-smooth.
    """
    first = members[0]
    for m in members:
        if not isinstance(m, Int8W8A8Linear):
            raise TypeError(f"{what} expects Int8W8A8Linear members, got {type(m).__name__}")
        if (m.bias is None) != (first.bias is None):
            raise RuntimeError(f"{what}: members mix biased and unbiased projections")
        if m.in_features != first.in_features:
            raise RuntimeError(f"{what}: members must share in_features")
    has_bias = first.bias is not None
    if any(m.smooth_scale is not None for m in members):
        raise RuntimeError(f"{what}: only no-smooth INT8 projections are supported")

    fused = Int8W8A8Linear(
        first.in_features,
        sum(m.out_features for m in members),
        bias=has_bias,
        device=first.qweight.device,
        dtype=first.io_dtype,
        smooth=False,
    )
    with torch.no_grad():
        fused.qweight.copy_(torch.cat([m.qweight for m in members], dim=0))
        fused.weight_scale.copy_(torch.cat([m.weight_scale for m in members], dim=0))
        if has_bias:
            # bias 在 dequant 之后以 fp 相加(见 Int8W8A8Linear.forward),
            # 沿 out 维拼接即可,不影响逐位性(SigLIP ViT 的投影全部带 bias)。
            fused.bias.copy_(torch.cat([m.bias for m in members], dim=0))
    return fused


def fuse_qkv_int8(q: Int8W8A8Linear, k: Int8W8A8Linear, v: Int8W8A8Linear) -> Int8W8A8Linear:
    """Fuse attention q/k/v (shared input; pack built with --fuse-groups qkv)."""
    return _fuse_int8_linears((q, k, v), "fuse_qkv_int8")


class FusedGateUpMLP(nn.Module):
    """GemmaMLP wrapper with gate/up fused into one projection.

    forward = ``down_proj(act_fn(gate) * up)`` where ``gate, up`` come from a
    single ``gate_up`` GEMM split in half — same math as the wrapped MLP
    (bit-identical for INT8 members; fp concat only reorders accumulation).
    Wraps in place of ``layer.mlp``, so every consumer (eager / NPU PFA /
    graph) picks it up with zero call-site changes. ``down_proj`` takes the
    elementwise product as input and therefore cannot join the fusion.
    """

    def __init__(
        self,
        gate_up: nn.Module,
        act_fn: nn.Module,
        down_proj: nn.Module,
        inter: int,
        *,
        gate_proj: nn.Module | None = None,
        up_proj: nn.Module | None = None,
    ):
        super().__init__()
        self.gate_up = gate_up
        self.act_fn = act_fn
        self.down_proj = down_proj
        self.intermediate_size = int(inter)
        # 保留原 gate/up 引用——modeling 里有 layer.mlp.up_proj 的
        # dtype 探测(linear_weight_dtype),包裹后这些属性必须仍然可达。
        # 本包裹在 prepare 时创建、不落 state_dict,引用不会造成重复保存。
        if gate_proj is not None:
            self.gate_proj = gate_proj
        if up_proj is not None:
            self.up_proj = up_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).split([self.intermediate_size, self.intermediate_size], dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


def fuse_mlp_gate_up(mlp: nn.Module) -> FusedGateUpMLP:
    """Wrap a GemmaMLP-shaped module (gate_proj/up_proj/down_proj/act_fn) with
    its gate/up projections fused. INT8 members concatenate codes and scales;
    plain floating-point Linears concatenate weights.
    """
    gate, up = mlp.gate_proj, mlp.up_proj
    if gate.out_features != up.out_features:
        raise RuntimeError("fuse_mlp_gate_up: gate/up out_features differ")
    if isinstance(gate, Int8W8A8Linear):
        fused = _fuse_int8_linears((gate, up), "fuse_mlp_gate_up")
    else:
        weight = torch.cat([gate.weight, up.weight], dim=0).contiguous()
        fused = nn.Linear(
            weight.shape[1],
            weight.shape[0],
            bias=False,
            device=weight.device,
            dtype=weight.dtype,
        )
        with torch.no_grad():
            fused.weight.copy_(weight)
        fused.weight.requires_grad_(False)
    return FusedGateUpMLP(fused, mlp.act_fn, mlp.down_proj, gate.out_features, gate_proj=gate, up_proj=up)


def assert_qkv_fusion_supported(quantization: QuantizationConfig | None) -> None:
    """Whether a quantized checkpoint may enable QKV weight fusion.

    The only supported quantized path is no-smooth ``int8_w8a8``. Graph
    compilation compatibility is validated by ``prepare_inference_optimizations``.
    """
    if quantization is None:
        return
    if quantization.quant_method != "int8_w8a8" or quantization.smooth:
        raise RuntimeError("QKV fusion only supports no-smooth int8_w8a8 checkpoints")
