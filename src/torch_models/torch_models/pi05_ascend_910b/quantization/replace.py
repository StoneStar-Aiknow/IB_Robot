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
"""Swap selected ``nn.Linear`` modules for no-smooth INT8 implementations.

Runs on the freshly built (still random-initialized) policy BEFORE the state
dict is loaded, so the quant layers' extra tensors (scales) load from the
checkpoint like any other weight.
"""

import logging
import re

import torch
from torch import nn

from .config import QuantizationConfig
from .linear_int8 import Int8W8A8Linear


def _build_int8_w8a8_linear(base: nn.Linear, qcfg: QuantizationConfig) -> Int8W8A8Linear:
    # 权重码/scale 全部来自 checkpoint(替换发生在加载前),无需从随机初始化的
    # base 拷贝;bias 同样随 state dict 加载。
    return Int8W8A8Linear(
        base.in_features,
        base.out_features,
        bias=base.bias is not None,
        device=base.weight.device,
        dtype=base.weight.dtype,
        smooth=False,
    )


def apply_quantization(model: nn.Module, qcfg: QuantizationConfig) -> int:
    """Replace every Linear matching the config's regexes. Returns the number
    of layers replaced; raises if the config matches nothing (a config/model
    mismatch would otherwise silently evaluate in FP)."""
    if qcfg.quant_method != "int8_w8a8" or qcfg.smooth:
        raise ValueError("only Selective-99 int8_w8a8 with smooth=false is supported")
    include = re.compile(qcfg.include_regex)
    exclude = re.compile(qcfg.exclude_regex)

    targets: list[str] = []
    for name, module in model.named_modules():
        if not include.search(name) or exclude.search(name):
            continue
        if isinstance(module, Int8W8A8Linear):
            raise RuntimeError(f"layer '{name}' is already quantized — apply_quantization ran twice?")
        if not isinstance(module, nn.Linear):
            continue
        targets.append(name)
    if not targets:
        raise RuntimeError(
            f"no Linear layer matches include_regex={qcfg.include_regex!r} — "
            f"quantized checkpoint does not fit this model"
        )

    for name in targets:
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        attr = name.rsplit(".", 1)[-1]
        setattr(parent, attr, _build_int8_w8a8_linear(getattr(parent, attr), qcfg))
    logging.info(f"[quantization] selective no-smooth INT8: replaced {len(targets)} Linear layers")
    return len(targets)


def linear_weight_dtype(module: nn.Module) -> "torch.dtype":
    """The fp dtype a (possibly quantized) Linear expects at its interface.

    The pi05 eager paths align input dtypes via ``some_proj.weight.dtype``;
    quantized Linears (e.g. Int8W8A8Linear) store int codes instead of an fp
    ``weight`` and carry ``io_dtype``."""
    weight = getattr(module, "weight", None)
    if weight is not None:
        return weight.dtype
    return getattr(module, "io_dtype", torch.bfloat16)


def validate_quantized(model: nn.Module) -> None:
    """Post-load check: every quantized Linear must have its tensors actually
    loaded (lerobot loads checkpoints with strict=False, so a missing tensor
    doesn't fail the load itself). Raises RuntimeError naming the first
    offending layer. Any quant-method Linear exposing ``validate_loaded``
    participates."""
    for name, module in model.named_modules():
        if callable(getattr(module, "validate_loaded", None)):
            module.validate_loaded(name)
