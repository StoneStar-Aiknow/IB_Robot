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

"""CPU-only contract tests for PI0.5 Ascend 910B quantization helpers."""

import pytest
import torch
from torch import nn

from torch_models.pi05_ascend_910b.modeling_pi05_ascend_910b import (
    _active_int8_linears_for_prepack,
)
from torch_models.pi05_ascend_910b.quantization import (
    FusedGateUpMLP,
    Int8W8A8Linear,
    QuantizationConfig,
    apply_quantization,
    fuse_mlp_gate_up,
    fuse_qkv_int8,
)


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.selected = nn.Linear(8, 6, bias=True)
        self.retained = nn.Linear(8, 6, bias=False)


def _loaded_linear(*, bias: bool = True, seed: int = 7) -> Int8W8A8Linear:
    generator = torch.Generator().manual_seed(seed)
    linear = Int8W8A8Linear(8, 6, bias=bias, dtype=torch.float32, smooth=False)
    with torch.no_grad():
        linear.qweight.copy_(torch.randint(-127, 128, linear.qweight.shape, generator=generator, dtype=torch.int8))
        linear.weight_scale.copy_(torch.rand(6, generator=generator) * 0.01 + 1e-4)
        if linear.bias is not None:
            linear.bias.copy_(torch.randn(6, generator=generator) * 0.01)
    return linear


def test_quantization_config_accepts_only_selective_no_smooth() -> None:
    config = QuantizationConfig(include_regex=r"selected$")
    assert config.quant_method == "int8_w8a8"
    assert config.smooth is False

    with pytest.raises(ValueError, match="int8_w8a8"):
        QuantizationConfig(quant_method="fake_quant")
    with pytest.raises(ValueError, match="no-smooth"):
        QuantizationConfig(smooth=True)
    with pytest.raises(ValueError, match="W8A8"):
        QuantizationConfig(w_bits=4)


def test_apply_quantization_replaces_only_selected_linear() -> None:
    model = ToyModel()
    replaced = apply_quantization(
        model,
        QuantizationConfig(include_regex=r"selected$", exclude_regex=r"$^"),
    )

    assert replaced == 1
    assert isinstance(model.selected, Int8W8A8Linear)
    assert isinstance(model.retained, nn.Linear)
    assert model.selected.smooth_scale is None


def test_no_smooth_linear_forward_and_load_validation() -> None:
    linear = _loaded_linear()
    linear.validate_loaded("selected")
    inputs = torch.randn(2, 3, 8)
    output = linear(inputs)

    assert output.shape == (2, 3, 6)
    assert output.dtype == inputs.dtype
    assert "smooth_scale" not in linear.state_dict()

    unloaded = Int8W8A8Linear(8, 6, smooth=False)
    with pytest.raises(RuntimeError, match="weight_scale|qweight"):
        unloaded.validate_loaded("unloaded")


def test_smooth_linear_is_rejected() -> None:
    with pytest.raises(ValueError, match="no-smooth"):
        Int8W8A8Linear(8, 6, smooth=True)


def _int8(in_features: int, out_features: int, *, seed: int, bias: bool = False) -> Int8W8A8Linear:
    generator = torch.Generator().manual_seed(seed)
    linear = Int8W8A8Linear(
        in_features,
        out_features,
        bias=bias,
        dtype=torch.float32,
        smooth=False,
    )
    with torch.no_grad():
        linear.qweight.copy_(torch.randint(-127, 128, linear.qweight.shape, generator=generator, dtype=torch.int8))
        linear.weight_scale.copy_(torch.rand(out_features, generator=generator) * 0.01 + 1e-4)
        if linear.bias is not None:
            linear.bias.copy_(torch.randn(out_features, generator=generator) * 0.01)
    return linear


def test_qkv_fusion_matches_separate_no_smooth_projections() -> None:
    q_proj = _int8(16, 12, seed=1, bias=True)
    k_proj = _int8(16, 8, seed=2, bias=True)
    v_proj = _int8(16, 8, seed=3, bias=True)
    fused = fuse_qkv_int8(q_proj, k_proj, v_proj)
    inputs = torch.randn(2, 5, 16)

    expected = torch.cat((q_proj(inputs), k_proj(inputs), v_proj(inputs)), dim=-1)
    actual = fused(inputs)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert fused.smooth_scale is None


class ToyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = _int8(16, 24, seed=4)
        self.up_proj = _int8(16, 24, seed=5)
        self.down_proj = _int8(24, 16, seed=6)
        self.act_fn = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(inputs)) * self.up_proj(inputs))


def test_gate_up_fusion_matches_separate_no_smooth_projections() -> None:
    mlp = ToyMLP()
    inputs = torch.randn(2, 5, 16)
    expected = mlp(inputs)
    fused = fuse_mlp_gate_up(mlp)

    assert isinstance(fused, FusedGateUpMLP)
    torch.testing.assert_close(fused(inputs), expected, rtol=1e-6, atol=1e-6)


def test_prepack_selection_excludes_inactive_fusion_aliases() -> None:
    fused_mlp = fuse_mlp_gate_up(ToyMLP())
    attention = nn.Module()
    attention.q_proj = _int8(16, 12, seed=11)
    attention.k_proj = _int8(16, 8, seed=12)
    attention.v_proj = _int8(16, 8, seed=13)
    attention.qkv = fuse_qkv_int8(attention.q_proj, attention.k_proj, attention.v_proj)
    root = nn.ModuleDict({"mlp": fused_mlp, "attention": attention})

    selected = dict(_active_int8_linears_for_prepack(root))

    assert set(selected) == {
        "mlp.gate_up",
        "mlp.down_proj",
        "attention.qkv",
    }
