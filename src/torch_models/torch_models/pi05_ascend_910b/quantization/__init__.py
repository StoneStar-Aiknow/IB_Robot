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
"""PI0.5 Selective-99 no-smooth INT8 checkpoint loading."""

from .config import QuantizationConfig
from .fuse import (
    FusedGateUpMLP,
    assert_qkv_fusion_supported,
    fuse_mlp_gate_up,
    fuse_qkv_int8,
)
from .linear_int8 import Int8W8A8Linear
from .replace import apply_quantization, linear_weight_dtype, validate_quantized

__all__ = [
    "FusedGateUpMLP",
    "Int8W8A8Linear",
    "QuantizationConfig",
    "apply_quantization",
    "assert_qkv_fusion_supported",
    "fuse_mlp_gate_up",
    "fuse_qkv_int8",
    "linear_weight_dtype",
    "validate_quantized",
]
