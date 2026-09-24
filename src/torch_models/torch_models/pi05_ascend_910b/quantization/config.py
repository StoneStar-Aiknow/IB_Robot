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
"""Selective-99 no-smooth INT8 checkpoint configuration."""

from dataclasses import dataclass


@dataclass
class QuantizationConfig:
    """描述 Selective-99 checkpoint 中需要替换为 INT8 Linear 的层。"""

    quant_method: str = "int8_w8a8"
    w_bits: int = 8
    a_bits: int = 8
    w_format: str = "int"
    a_format: str = "int"
    smooth: bool = False
    include_regex: str = r".*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
    exclude_regex: str = r"(?:^|\.)(vision_tower|vision_model|embeddings|embed_tokens|norm|layernorm|lm_head)(?:\.|$)"
    group_size: int = 0

    def __post_init__(self) -> None:
        if self.quant_method != "int8_w8a8":
            raise ValueError("only quant_method='int8_w8a8' is supported")
        if self.w_bits != 8 or self.a_bits != 8:
            raise ValueError("Selective-99 requires W8A8")
        if self.w_format != "int" or self.a_format != "int":
            raise ValueError("Selective-99 requires integer weight and activation formats")
        if self.smooth:
            raise ValueError("only the no-smooth Selective-99 path is supported")
        if self.group_size != 0:
            raise ValueError("Selective-99 uses per-output-channel weights, not group-wise quantization")
