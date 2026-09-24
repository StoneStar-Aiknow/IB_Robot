#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

import builtins
import logging
import math
import os
import sys
import types
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from lerobot.utils.import_utils import _transformers_available, require_package
from torch import Tensor, nn
from typing_extensions import Unpack

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers.cache_utils import DynamicCache, DynamicLayer
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma

    from .pi_gemma import (
        PaliGemmaForConditionalGenerationWithPiGemma,
        PiGemmaForCausalLM,
        _gated_residual,
        layernorm_forward,
    )
else:
    CONFIG_MAPPING = None
    DynamicCache = None
    DynamicLayer = None
    modeling_gemma = None
    PiGemmaForCausalLM = None
    _gated_residual = None
    layernorm_forward = None
    PaliGemmaForConditionalGenerationWithPiGemma = None
from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import DEFAULT_IMAGE_SIZE, PI05Config
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OPENPI_ATTENTION_MASK_VALUE,
)

from .quantization import linear_weight_dtype
from .vision_siglip_npu import PI05SiglipVisionModel


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


PI05_GRAPH_DENOISE_STEPS = 10
PI05_AB2_DENOISE_STEPS = 6
PI05_ENABLE_VISION_NPU_PFA_ENV = "LEROBOT_PI05_ENABLE_VISION_NPU_PFA"
PI05_NPU_ATTENTION_BACKEND_ENV = "LEROBOT_PI05_NPU_ATTENTION_BACKEND"
PI05_NPU_ATTENTION_BACKENDS = frozenset({"hybrid", "pfa", "fias", "default"})
PI05_ADARMS_BIAS_FUSION_STAGES = frozenset({"qkv", "qkv_mlp", "mlp_action_out", "all"})
AdaRMSModulation = tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]


def _active_int8_linears_for_prepack(root: nn.Module):
    """Yield active INT8 projections, excluding aliases retained by fused wrappers."""
    from .quantization import FusedGateUpMLP, Int8W8A8Linear

    inactive_ids: set[int] = set()
    for module in root.modules():
        if isinstance(module, FusedGateUpMLP):
            for name in ("gate_proj", "up_proj"):
                alias = getattr(module, name, None)
                if isinstance(alias, Int8W8A8Linear):
                    inactive_ids.add(id(alias))
        fused_qkv = getattr(module, "qkv", None)
        if isinstance(fused_qkv, Int8W8A8Linear):
            for name in ("q_proj", "k_proj", "v_proj"):
                alias = getattr(module, name, None)
                if isinstance(alias, Int8W8A8Linear):
                    inactive_ids.add(id(alias))

    seen_ids: set[int] = set()
    for name, module in root.named_modules():
        module_id = id(module)
        if isinstance(module, Int8W8A8Linear) and module_id not in inactive_ids and module_id not in seen_ids:
            seen_ids.add(module_id)
            yield name, module


def _resolve_npu_attention_backend(value: str | None) -> str:
    """Resolve the experimental attention backend while preserving today's hybrid default."""
    backend = "hybrid" if value is None or not value.strip() else value.strip().lower()
    if backend not in PI05_NPU_ATTENTION_BACKENDS:
        raise ValueError(
            f"{PI05_NPU_ATTENTION_BACKEND_ENV} must be one of {sorted(PI05_NPU_ATTENTION_BACKENDS)}, got {value!r}"
        )
    return backend


def _resolve_adarms_bias_fusion_stage(enabled: bool, stage: str) -> str:
    """Validate an active stage while allowing the explicit disabled fallback."""
    normalized_stage = str(stage).lower()
    if not enabled and normalized_stage == "none":
        return "none"
    if normalized_stage not in PI05_ADARMS_BIAS_FUSION_STAGES:
        raise ValueError(
            "adarms_bias_fusion_stage must be none when disabled or one of "
            "qkv, qkv_mlp, mlp_action_out, all; "
            f"got {stage!r}"
        )
    return normalized_stage


def _is_fp_linear_for_adarms_bias_fusion(module: nn.Module) -> bool:
    """Return whether ``module`` can consume a precomputed AdaRMS shift bias.

    The folding identity needs direct access to a floating-point Linear weight.
    INT8 Linears expose ``qweight`` rather than a floating-point weight, so
    they are not algebraically compatible with this folding target.
    """
    weight = getattr(module, "weight", None)
    return isinstance(module, nn.Linear) and isinstance(weight, torch.Tensor) and weight.dtype.is_floating_point


def _apply_ab2_denoise_update(
    x_t: torch.Tensor,
    v_t: torch.Tensor,
    previous_raw_v_t: torch.Tensor | None,
    dt: float | torch.Tensor,
) -> torch.Tensor:
    """执行 AB2 更新；首步退化为 Euler，历史项始终保存原始模型 velocity。"""
    update_v_t = v_t if previous_raw_v_t is None else 1.5 * v_t - 0.5 * previous_raw_v_t
    return x_t + dt * update_v_t


def _cache_quantized_vision_qkv_biases(vision_tower: nn.Module) -> int:
    """缓存量化 ViT 融合 QKV bias；BF16 tower 不需要也不创建缓存。"""
    cached_qkv_biases = 0
    vision_layers = vision_tower.vision_model.encoder.layers
    quantized_qkv_layers = 0
    for layer in vision_layers:
        qkv = layer.self_attn.qkv
        if qkv is None or not hasattr(qkv, "qweight"):
            continue
        quantized_qkv_layers += 1
        if qkv.bias is None:
            raise AssertionError("quantized ViT QKV projection is missing its bias")
        cached_bias = qkv.bias.detach().to(dtype=torch.bfloat16).contiguous()
        if not torch.equal(cached_bias, qkv.bias.detach().to(dtype=torch.bfloat16)):
            raise AssertionError("ViT QKV bias cache differs from the baseline online cast")
        qkv.register_buffer("_npu_static_bias", cached_bias, persistent=False)
        cached_qkv_biases += 1

    # 静态 bias cache 只服务 Int8W8A8Linear；BF16 nn.Linear
    # 不存在 qweight，合法缓存数为 0。量化 ViT 则要求所有层完整命中。
    if quantized_qkv_layers not in {0, len(vision_layers)}:
        raise AssertionError(
            "expected either zero or all ViT QKV layers to be quantized; "
            f"got {quantized_qkv_layers}/{len(vision_layers)}"
        )
    if cached_qkv_biases != quantized_qkv_layers:
        raise AssertionError(f"expected {quantized_qkv_layers} ViT QKV bias caches, got {cached_qkv_biases}")
    return cached_qkv_biases


def _device_type_from_config(device: str | torch.device | None) -> str:
    """从配置中的 device 字符串或 torch.device 中提取设备类型。"""
    if device is None:
        return ""
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":", maxsplit=1)[0]


def _device_supports_bfloat16(device: torch.device | str | None) -> bool:
    """判断当前推理设备是否可以安全走 bfloat16 计算路径。"""
    device_type = _device_type_from_config(device)
    if device_type == "npu":
        return True
    if device_type == "cuda":
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    return False


def _module_device(module: nn.Module) -> torch.device:
    """返回模块自有参数或 buffer 的设备，兼容没有 fp weight 的量化 Linear。"""
    for param in module.parameters(recurse=False):
        return param.device
    for buffer in module.buffers(recurse=False):
        return buffer.device
    return torch.device("cpu")


def get_safe_dtype(target_dtype, device_type):
    """根据设备能力返回可用 dtype，避免在不支持的设备上使用高精度/低精度类型。"""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "npu" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """把标量 timestep 编码成 flow matching/AdaRMS 使用的 sin-cos 时间特征。"""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    """训练时从 Beta 分布采样 flow matching 时间点。"""
    # Beta sampling uses _sample_dirichlet which isn't implemented for MPS, so sample on CPU
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)


def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """根据 padding mask 和 block/causal 标记构造 transformer 二维 attention mask。

    Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    # padding mask 保持 bool 逻辑与，避免 bool->uint8->bool 的类型往返，
    # 减少 Ascend 图里落到 AICPU 的 Cast。
    pad_2d_masks = pad_masks[:, None, :] & pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def clone_past_key_values(past_key_values):
    """克隆 prefix prefill 产生的 KV cache，避免 denoise step 修改共享缓存。"""
    if isinstance(past_key_values, dict):
        # NPU fused prefix 生成的 dict cache 在 denoise 阶段只读复用，不需要逐层 clone。
        return past_key_values

    cloned_cache = DynamicCache()
    for keys, values, sliding_window in past_key_values:
        layer = DynamicLayer()
        layer.keys = keys.clone()
        layer.values = values.clone()
        layer.is_initialized = True
        if sliding_window is not None:
            layer._sliding_window_tensor = sliding_window
        cloned_cache.layers.append(layer)
    return cloned_cache


def _npu_graph_safe_linear_forward(linear: nn.Linear, input_tensor: torch.Tensor) -> torch.Tensor:
    """NPU 图编译专用 Linear forward，先展平高维输入再恢复形状。"""
    graph_bias = getattr(linear, "_pi05_graph_bias_fp32", None)
    if input_tensor.device.type == "npu" and graph_bias is not None:
        torch_npu = _import_torch_npu()
        if torch_npu is None:
            raise RuntimeError("torch_npu is required for PI0.5 NPU Linear bias Cast elimination")

        if input_tensor.ndim == 1:
            return torch_npu.npu_linear(input_tensor.unsqueeze(0), linear.weight, graph_bias).squeeze(0)

        output_shape = (*input_tensor.shape[:-1], linear.out_features)
        output = torch_npu.npu_linear(
            input_tensor.reshape(-1, input_tensor.shape[-1]),
            linear.weight,
            graph_bias,
        )
        return output.reshape(output_shape)
    # TorchAir/GE 对 rank>2 Linear 的 MatMul 维度推断不稳定；
    # 编译路径统一压平成二维 Linear，再恢复原始 batch/sequence 形状。
    if input_tensor.ndim <= 2 or input_tensor.device.type != "npu":
        return F.linear(input_tensor, linear.weight, linear.bias)

    output_shape = (*input_tensor.shape[:-1], linear.out_features)
    output = F.linear(
        input_tensor.reshape(-1, input_tensor.shape[-1]),
        linear.weight,
        linear.bias,
    )
    return output.reshape(output_shape)


def _npu_graph_safe_linear_with_bias(
    linear: nn.Linear,
    input_tensor: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """通过 NPU 二维 MatMul 路径执行带逐次 FP32 bias 的 Linear。"""
    if input_tensor.device.type != "npu":
        return F.linear(input_tensor, linear.weight, bias.to(dtype=input_tensor.dtype))

    output_shape = (*input_tensor.shape[:-1], linear.out_features)
    torch_npu = _import_torch_npu()
    if torch_npu is None:
        raise RuntimeError("torch_npu is required for PI0.5 NPU Linear bias fusion")
    output = torch_npu.npu_linear(
        input_tensor.reshape(-1, input_tensor.shape[-1]),
        linear.weight,
        bias,
    )
    return output.reshape(output_shape)


def _npu_graph_safe_gemma_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """用 bmm 重写 Gemma eager attention，提升 TorchAir 图捕获稳定性。"""
    # 保持 Gemma eager attention 公式不变，只把四维 matmul
    # 改写为 B*H 维度上的 bmm，规避 TorchAir/GE shape infer 问题。
    key_states = modeling_gemma.repeat_kv(key, module.num_key_value_groups)
    value_states = modeling_gemma.repeat_kv(value, module.num_key_value_groups)

    batch_size, num_heads, query_len, head_dim = query.shape
    key_len = key_states.shape[-2]
    query_3d = query.reshape(batch_size * num_heads, query_len, head_dim)
    key_3d = key_states.reshape(batch_size * num_heads, key_len, head_dim)
    attn_weights = torch.bmm(query_3d, key_3d.transpose(1, 2)) * scaling
    attn_weights = attn_weights.reshape(batch_size, num_heads, query_len, key_len)

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)

    attn_weights_3d = attn_weights.reshape(batch_size * num_heads, query_len, key_len)
    value_3d = value_states.reshape(batch_size * num_heads, key_len, head_dim)
    attn_output = torch.bmm(attn_weights_3d, value_3d)
    attn_output = attn_output.reshape(batch_size, num_heads, query_len, head_dim)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _npu_graph_safe_gemma_attention_module_forward(
    attention: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    past_key_values=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GemmaAttention 的图安全 forward，显式处理 QKV、RoPE、KV cache 和输出投影。"""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attention.head_dim)

    query_states = attention.q_proj(hidden_states).reshape(hidden_shape).transpose(1, 2)
    key_states = attention.k_proj(hidden_states).reshape(hidden_shape).transpose(1, 2)
    value_states = attention.v_proj(hidden_states).reshape(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        while len(past_key_values.layers) <= attention.layer_idx:
            past_key_values.layers.append(DynamicLayer())
        cache_layer = past_key_values.layers[attention.layer_idx]
        if not cache_layer.is_initialized:
            cache_layer.keys = key_states
            cache_layer.values = value_states
            cache_layer.is_initialized = True
        else:
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                attention.layer_idx,
            )

    attn_output, attn_weights = _npu_graph_safe_gemma_attention_forward(
        attention,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling=attention.scaling,
        dropout=0.0 if not attention.training else attention.attention_dropout,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attention.o_proj(attn_output)
    return attn_output, attn_weights


def _npu_graph_safe_siglip_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """用 bmm 重写 SigLIP attention，避免 NPU 编译路径中的四维 matmul 问题。"""
    batch_size, num_heads, query_len, head_dim = query.shape
    key_len = key.shape[-2]
    query_3d = query.reshape(batch_size * num_heads, query_len, head_dim)
    key_3d = key.reshape(batch_size * num_heads, key_len, head_dim)
    attn_weights = torch.bmm(query_3d, key_3d.transpose(1, 2)) * scaling
    attn_weights = attn_weights.reshape(batch_size, num_heads, query_len, key_len)

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)

    value_3d = value.reshape(batch_size * num_heads, key_len, head_dim)
    attn_output = torch.bmm(attn_weights.reshape(batch_size * num_heads, query_len, key_len), value_3d)
    attn_output = attn_output.reshape(batch_size, num_heads, query_len, head_dim)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _npu_graph_safe_siglip_attention_module_forward(
    attention: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """SigLIPAttention 的图安全 forward，用于 prefix 视觉塔编译。"""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attention.head_dim)

    queries = attention.q_proj(hidden_states).reshape(hidden_shape).transpose(1, 2)
    keys = attention.k_proj(hidden_states).reshape(hidden_shape).transpose(1, 2)
    values = attention.v_proj(hidden_states).reshape(hidden_shape).transpose(1, 2)

    attn_output, attn_weights = _npu_graph_safe_siglip_attention_forward(
        attention,
        queries,
        keys,
        values,
        attention_mask,
        scaling=attention.scale,
        dropout=0.0 if not attention.training else attention.dropout,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attention.out_proj(attn_output)
    return attn_output, attn_weights


def _npu_graph_safe_gemma_rotary_forward(
    rotary_emb: nn.Module,
    x: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """图编译路径下显式构造 Gemma RoPE cos/sin，避免动态广播推断不稳。"""
    freqs = position_ids[:, :, None].to(dtype=torch.float32) * rotary_emb.inv_freq[None, None, :].to(
        device=x.device, dtype=torch.float32
    )
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * rotary_emb.attention_scaling
    sin = emb.sin() * rotary_emb.attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def pad_vector(vector, new_dim):
    """把 action/state 最后一维补零到模型内部最大维度。

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """按比例缩放并 padding 图像到目标分辨率，保持 PaliGemma 视觉输入不变形。

    PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else 0.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(inputs_embeds, attention_mask, position_ids, adarms_cond, layers, rotary_emb):
    """训练路径中同时推进 PaliGemma 与 action expert 的一层计算。"""
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = layers[i]
        hidden_states, gate = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond[i])
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)
    batch_size = query_states.shape[0]
    paligemma_layer = layers[0]
    scaling = paligemma_layer.self_attn.scaling
    # Attention computation
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma_layer.self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = paligemma_layer.self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = layers[i]
        end_pos = start_pos + hidden_states.shape[1]
        o_proj_dtype = linear_weight_dtype(layer.self_attn.o_proj)
        if att_output.dtype != o_proj_dtype:
            att_output = att_output.to(o_proj_dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = _gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()
        out_emb, gate = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond[i])
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if linear_weight_dtype(layer.mlp.up_proj) == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = _gated_residual(after_first_residual, out_emb, gate)
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class GemmaConfig:  # see openpi `gemma.py: Config`
    """保存 Gemma 变体的宽度、层数、MLP 和 attention 头配置。"""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        """初始化指定规模 Gemma 结构参数。"""
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """根据 pi0.5 配置名返回 PaliGemma/action expert 使用的 Gemma 规格。"""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


def _import_torch_npu():
    """延迟导入 torch_npu，使非 NPU 环境仍可加载本文件。"""
    try:
        import torch_npu  # type: ignore
    except ModuleNotFoundError:
        return None
    return torch_npu


def _is_npu_tensor_list(inputs_embeds: list[torch.Tensor | None]) -> bool:
    """检查输入 embedding 列表中第一个有效 tensor 是否位于 NPU。"""
    for tensor in inputs_embeds:
        if tensor is not None:
            return tensor.device.type == "npu"
    return False


class PaliGemmaWithExpertModel(
    nn.Module
):  # see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
    """组合 PaliGemma 视觉语言模型和 Gemma action expert 的核心 transformer。"""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        image_size: int = DEFAULT_IMAGE_SIZE,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
    ):
        """按 PI0.5 配置构建 prefix VLM 和 suffix action expert 两套 Gemma。"""
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
        self.gemma_expert = PiGemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)
        self._set_requires_grad()
        self._qkv_weights_fused = False
        self._mlp_weights_fused = False
        self._expert_mlp_weights_fused = False
        self._npu_fused_inference_enabled = False
        self._npu_attention_backend = "hybrid"
        self._shared_prefix_fias_enabled = False

    def to_bfloat16_for_selected_params(
        self,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        *,
        vision_precision: Literal["bfloat16", "float32"] = "float32",
    ):
        """设置推理/训练 dtype，并保留 RMSNorm 等敏感模块为 float32。"""
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
        else:
            raise ValueError(f"Invalid precision: {precision}")
        if vision_precision not in {"bfloat16", "float32"}:
            raise ValueError(f"Invalid vision_precision: {vision_precision}")

        params_to_keep_float32 = [
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]
        if vision_precision == "float32":
            params_to_keep_float32.extend(["vision_tower", "multi_modal_projector"])

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

        if vision_precision == "bfloat16":
            # 推理阶段按 pi0 口径让完整 vision path 降到 bf16；
            # RMSNorm 仍保留 float32，避免归一化数值稳定性变化。
            self.paligemma.model.vision_tower.to(dtype=torch.bfloat16)
            self.paligemma.model.multi_modal_projector.to(dtype=torch.bfloat16)

    def _set_requires_grad(self):
        """根据配置冻结视觉塔或只训练 action expert。"""
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
            for param in self.paligemma.model.vision_tower.parameters():
                param.requires_grad = False
        if self.train_expert_only:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        """切换训练模式时保持已冻结模块处于 eval。"""
        super().train(mode)
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
        if self.train_expert_only:
            self.paligemma.eval()

    @torch.no_grad()
    def fuse_qkv_weights(self):
        """为 Ascend NPU attention 路径预先创建融合 QKV Linear。"""
        if self._qkv_weights_fused:
            return

        # NPU PFA 路径每层只需要一次 QKV 投影。这里在加载权重后
        # 预先拼接 q/k/v 权重，避免推理循环中重复执行三个独立 Linear。
        # Selective-99 INT8 直接拼接 qweight/weight_scale；BF16 路径拼接
        # 浮点权重。两种产物都满足 attn.qkv(normed).split(...) 接口。
        from .quantization import Int8W8A8Linear, fuse_qkv_int8

        for model in (self.paligemma.model.language_model, self.gemma_expert.model):
            for layer in model.layers:
                attn = layer.self_attn
                if isinstance(attn.q_proj, Int8W8A8Linear):
                    attn.qkv = fuse_qkv_int8(attn.q_proj, attn.k_proj, attn.v_proj)
                    continue
                qkv_weight = torch.cat(
                    [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight],
                    dim=0,
                ).contiguous()
                attn.qkv = nn.Linear(
                    qkv_weight.shape[1],
                    qkv_weight.shape[0],
                    bias=False,
                    device=qkv_weight.device,
                    dtype=qkv_weight.dtype,
                )
                attn.qkv.weight.copy_(qkv_weight)
                attn.qkv.weight.requires_grad_(False)

        self._qkv_weights_fused = True

    def fuse_mlp_weights(self, *, include_action_expert: bool = True):
        """把指定 Gemma 主干的 gate/up 投影融合为单次 GEMM(gate_up)。

        FusedGateUpMLP 原地替换 layer.mlp,eager/NPU PFA/图编译
        所有 layer.mlp(x) 调用点零改动生效;down_proj 输入是逐元素积,不参与。
        INT8 成员拼接 qweight/weight_scale，融合与两路独立计算逐位一致；
        BF16 成员直接拼接权重。

        当前正式路径只在 selective INT8 Prefix LLM 上启用；BF16 DiT expert
        的成对消融为负收益，因此必须通过 include_action_expert 显式选择，不能
        再被图编译总开关隐式融合。
        """
        from .quantization import fuse_mlp_gate_up

        if not self._mlp_weights_fused:
            for layer in self.paligemma.model.language_model.layers:
                layer.mlp = fuse_mlp_gate_up(layer.mlp)
            # 兼容历史 metadata：该字段精确表示 Prefix LLM 已融合。
            self._mlp_weights_fused = True

        if include_action_expert and not self._expert_mlp_weights_fused:
            for layer in self.gemma_expert.model.layers:
                layer.mlp = fuse_mlp_gate_up(layer.mlp)
            self._expert_mlp_weights_fused = True

    def mlp_fusion_metadata(self) -> dict[str, Any]:
        """返回当前模型实例真实、持久的 MLP 融合状态。"""
        # 融合会原地替换模块且不可逆；prepare 可能被重复调用，因此
        # metadata 不能只反映本次请求参数，必须从实例实际状态重新生成。
        prefix_fused = bool(self._mlp_weights_fused)
        expert_fused = bool(self._expert_mlp_weights_fused)
        if prefix_fused and expert_fused:
            scope = "all"
        elif prefix_fused:
            scope = "prefix"
        elif expert_fused:
            scope = "expert"
        else:
            scope = "none"
        return {
            "mlp_fusion_scope": scope,
            # 保留历史字段名，用于表示 Prefix LLM 融合状态。
            "mlp_weights_fused": prefix_fused,
            "prefix_mlp_weights_fused": prefix_fused,
            "expert_mlp_weights_fused": expert_fused,
        }

    def enable_npu_fused_inference(self, enabled: bool = True) -> None:
        """启用或关闭 Gemma 主干的 NPU PFA/Rotary/RMSNorm 融合推理路径。"""
        self._npu_fused_inference_enabled = bool(enabled)

    def set_npu_attention_backend(self, backend: str) -> None:
        """Select attention only, leaving QKV/RoPE/RMSNorm/AdaRMS optimizations unchanged."""
        self._npu_attention_backend = _resolve_npu_attention_backend(backend)

    def _prepare_vision_pixels(self, image: torch.Tensor) -> torch.Tensor:
        """把图像 tensor 转成视觉塔当前权重 dtype。"""
        # 图像输入 dtype 跟随 vision tower 权重，避免 prefix
        # 视觉塔内部在 float32/bf16 之间反复 cast。
        vision_dtype = next(self.paligemma.model.vision_tower.parameters()).dtype
        if image.dtype != vision_dtype:
            return image.to(vision_dtype)
        return image

    def embed_image_vision_tower(self, image: torch.Tensor) -> torch.Tensor:
        """执行 SigLIP vision tower，得到每张图像的 patch embedding。"""
        image = self._prepare_vision_pixels(image)
        image_outputs = self.paligemma.model.vision_tower(image)
        return image_outputs.last_hidden_state

    def embed_image_projector(self, image_features: torch.Tensor) -> torch.Tensor:
        """把 SigLIP patch embedding 投影到 PaliGemma/Gemma hidden 维度。"""
        return self.paligemma.model.multi_modal_projector(image_features)

    def embed_image(self, image: torch.Tensor):
        """完成图像从像素到 multimodal embedding 的完整 prefix 编码。"""
        # 本地展开 get_image_features，和 pi0 一样显式控制
        # vision_tower/projector 精度，便于 NPU prefix bf16 推理。
        image_features = self.embed_image_vision_tower(image)
        return self.embed_image_projector(image_features)

    def embed_language_tokens(self, tokens: torch.Tensor):
        """把语言 token id 转换为 PaliGemma 语言 embedding。"""
        return self.paligemma.model.language_model.get_input_embeddings()(tokens)

    def prepare_vision_tower_npu_fused_ops(self, *, enable_qkv_fusion: bool = True) -> dict[str, Any]:
        """把 transformers SigLIP vision tower 替换为本地 NPU PFA/QKV 融合实现。"""
        old_vision_tower = self.paligemma.model.vision_tower
        replaced = False
        if isinstance(old_vision_tower, PI05SiglipVisionModel):
            vision_tower = old_vision_tower
        else:
            first_param = next(old_vision_tower.parameters())
            device = first_param.device
            dtype = first_param.dtype
            training = old_vision_tower.training
            requires_grad_by_name = {name: param.requires_grad for name, param in old_vision_tower.named_parameters()}

            vision_tower = PI05SiglipVisionModel(old_vision_tower.config)
            vision_tower.to(device=device, dtype=dtype)
            # 量化 ViT 先按 FP 结构构建，再把已加载 qweight/weight_scale 的
            # Selective-99 Linear 按同名路径整体移植。
            from .quantization import Int8W8A8Linear

            for mod_name, module in old_vision_tower.named_modules():
                if isinstance(module, Int8W8A8Linear):
                    parent = vision_tower.get_submodule(mod_name.rsplit(".", 1)[0]) if "." in mod_name else vision_tower
                    setattr(parent, mod_name.rsplit(".", 1)[-1], module)
            vision_tower.load_state_dict(old_vision_tower.state_dict(), strict=True)
            for name, param in vision_tower.named_parameters():
                if name in requires_grad_by_name:
                    param.requires_grad_(requires_grad_by_name[name])
            vision_tower.train(training)
            self.paligemma.model.vision_tower = vision_tower
            replaced = True

        if enable_qkv_fusion:
            vision_tower.fuse_qkv_weights()
            cached_qkv_biases = _cache_quantized_vision_qkv_biases(vision_tower)
        vision_backend = "pfa" if self._npu_attention_backend == "hybrid" else self._npu_attention_backend
        vision_tower.set_npu_attention_backend(vision_backend)
        first_vision_attention = vision_tower.vision_model.encoder.layers[0].self_attn
        return {
            "local_siglip_vision_tower": True,
            "replaced_transformers_vision_tower": replaced,
            "attention": {
                "pfa": "torch_npu.npu_prompt_flash_attention",
                "fias": "torch_npu.npu_fused_infer_attention_score",
                "default": "matmul_softmax",
            }[vision_backend],
            "attention_backend": vision_backend,
            "pfa_input_dtype": (
                "float16_compat"
                if vision_backend == "pfa" and first_vision_attention._npu_pfa_force_fp16
                else "native_bfloat16"
                if vision_backend == "pfa"
                else "not_applicable"
            ),
            "qkv_weights_fused": bool(enable_qkv_fusion),
            "cached_int8_qkv_biases": cached_qkv_biases if enable_qkv_fusion else 0,
        }

    def _vision_tower_quant_kind(self) -> str | None:
        """返回 vision tower 是否包含 Selective-99 INT8 Linear。"""
        from .quantization import Int8W8A8Linear

        for m in self.paligemma.model.vision_tower.modules():
            if isinstance(m, Int8W8A8Linear):
                return "int8"
        return None

    def should_enable_vision_tower_npu_fused_ops(self, *, default: bool = True) -> bool:
        """读取环境变量覆盖项；主优化路径默认启用 SigLIP VIT QKV/PFA。"""
        raw_value = os.environ.get(PI05_ENABLE_VISION_NPU_PFA_ENV)
        # int8_w8a8 的 ViT 已支持融合路径——量化 Linear 按同名移植进
        # PI05SiglipVisionModel,QKV 由 fuse_qkv_int8 拼接(含 bias),照常放行。
        if raw_value is None or raw_value.strip() == "":
            return bool(default)
        value = raw_value.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{PI05_ENABLE_VISION_NPU_PFA_ENV} must be a boolean-like value, got {value!r}")

    def enable_shared_prefix_fias(self, enabled: bool = True) -> None:
        """控制 denoise Attention 是否使用 FIAS shared-prefix 输入。"""
        self._shared_prefix_fias_enabled = bool(enabled)

    def _is_npu_modulation_tree(self, modulation) -> bool:
        """检查 AdaRMS 调制是否可用于 NPU 推理热路径。"""
        if isinstance(modulation, torch.Tensor):
            return modulation.device.type == "npu"
        # 固定步骤调制 tuple 由本模块的 non-persistent buffers 构建，设备
        # 迁移后由 _apply() 重建，避免递归遍历产生大量无意义 guards。
        return isinstance(modulation, tuple)

    def _can_use_npu_fused_inference(
        self,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None,
        inputs_embeds: list[torch.FloatTensor] | None,
        adarms_cond: list[torch.Tensor | None] | None,
        adarms_modulations: list[torch.Tensor | None] | None = None,
    ) -> bool:
        """判断当前 forward 是否可走 NPU PFA/Rotary/AdaRMS 融合推理路径。"""
        # QKV 权重融合只是可选投影优化；PFA/Rotary/RMSNorm
        # 融合路径可在 q/k/v 独立投影或量化 Linear 下工作。
        if not self._npu_fused_inference_enabled:
            return False
        if self.training or attention_mask is None or position_ids is None:
            return False
        if inputs_embeds is None or not _is_npu_tensor_list(inputs_embeds):
            return False
        if any(embed is not None and not embed.dtype.is_floating_point for embed in inputs_embeds):
            return False
        if _import_torch_npu() is None:
            return False

        if adarms_cond is None:
            adarms_cond = [None, None]
        if adarms_modulations is None:
            adarms_modulations = [None, None]

        active_model_specs = (
            (inputs_embeds[0], self.paligemma.model.language_model, adarms_cond[0], adarms_modulations[0]),
            (inputs_embeds[1], self.gemma_expert.model, adarms_cond[1], adarms_modulations[1]),
        )
        for embed, model, cond, modulation in active_model_specs:
            if embed is None:
                continue
            if getattr(model.config, "use_adarms", False) and cond is None and modulation is None:
                return False
            if cond is not None and cond.device.type != "npu":
                return False
            if modulation is not None and not self._is_npu_modulation_tree(modulation):
                return False
        return True

    def _npu_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """把 bool/additive attention mask 转换为 NPU PFA 需要的 int8 block mask。"""
        # PFA 使用 int8 阻塞 mask；这里把 bool/additive mask 统一转成
        # torch_npu.npu_prompt_flash_attention 可直接消费的连续格式。
        if attention_mask.dtype == torch.bool:
            blocked_mask = torch.logical_not(attention_mask)
        else:
            blocked_mask = attention_mask < 0
        if blocked_mask.ndim == 3:
            blocked_mask = blocked_mask[:, None, :, :]
        return blocked_mask.to(dtype=torch.int8, memory_format=torch.contiguous_format)

    def _build_npu_rotary_cache(
        self,
        position_ids: torch.LongTensor,
        head_dim: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """为一次 prefix/denoise forward 预先生成所有层复用的 RoPE cos/sin。"""
        # 单次 prefix/denoise forward 内各层 position_ids 相同，
        # 先构造 RoPE cos/sin 后复用，减少每层重复的通用算子开销。
        d_half = head_dim // 2
        freq_exponents = (2.0 / head_dim) * torch.arange(d_half, dtype=torch.float32, device=position_ids.device)
        timescale = 10_000**freq_exponents
        radians = position_ids[..., None].to(torch.float32) / timescale[None, None, :]
        radians = radians[..., None, :]
        # RoPE 两个半维使用同一组角度，三角函数只计算一次再复制结果，
        # 避免 TorchAir 图中生成重复的 Cos/Sin 节点。
        half_cos = torch.cos(radians)
        half_sin = torch.sin(radians)
        cos = torch.cat([half_cos, half_cos], dim=-1)
        sin = torch.cat([half_sin, half_sin], dim=-1)
        return cos.to(dtype=dtype), sin.to(dtype=dtype)

    def _build_npu_qkv_single_rotary_cache(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Q/K 继续使用原 RoPE，V 通过 cos=1、sin=0 恒等透传，
        # 从而让融合 QKV 的 BF16 输出不拆分即可进入一次 RotaryMul。
        rotary_heads = num_heads + num_kv_heads
        qk_cos = cos.expand(-1, -1, rotary_heads, -1)
        qk_sin = sin.expand(-1, -1, rotary_heads, -1)
        identity_cos = torch.ones_like(cos).expand(-1, -1, num_kv_heads, -1)
        identity_sin = torch.zeros_like(sin).expand(-1, -1, num_kv_heads, -1)
        return (
            torch.cat([qk_cos, identity_cos], dim=2).contiguous(),
            torch.cat([qk_sin, identity_sin], dim=2).contiguous(),
        )

    def _prepare_npu_rotary_cache(
        self,
        position_ids: torch.LongTensor,
        attention,
        num_heads: int,
        num_kv_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rotary_cos, rotary_sin = self._build_npu_rotary_cache(
            position_ids,
            attention.head_dim,
            self._attention_projection_dtype(attention),
        )
        qkv_rotary_cos, qkv_rotary_sin = self._build_npu_qkv_single_rotary_cache(
            rotary_cos,
            rotary_sin,
            num_heads,
            num_kv_heads,
        )
        return rotary_cos, rotary_sin, qkv_rotary_cos, qkv_rotary_sin

    def _npu_rotary_emb(
        self,
        torch_npu,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """调用 NPU RotaryMul 融合算子，同时处理 query 和 key 的 RoPE。"""
        # 把 q/k 合并后一次调用 npu_rotary_mul，再按头数切回，
        # 避免 Python 侧重复拆分、拼接 RoPE 中间张量。
        num_q_heads = query_states.shape[2]
        num_kv_heads = key_states.shape[2]
        merged_states = torch.cat([query_states, key_states], dim=2)
        merged_states = torch_npu.npu_rotary_mul(merged_states, cos, sin)
        return merged_states.split([num_q_heads, num_kv_heads], dim=2)

    def _attention_projection_dtype(self, attn: nn.Module) -> torch.dtype:
        """返回 attention q/k/v 投影入口 dtype，兼容可选 fused qkv 与量化 Linear。"""
        qkv = getattr(attn, "qkv", None)
        if qkv is not None:
            return linear_weight_dtype(qkv)
        return linear_weight_dtype(attn.q_proj)

    def _project_attention_qkv(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        q_out: int,
        kv_out: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """执行 attention Q/K/V 投影；qkv 未融合时保留独立投影。"""
        qkv = getattr(attn, "qkv", None)
        if qkv is not None:
            projected = qkv(hidden_states)
            return projected.split([q_out, kv_out, kv_out], dim=-1)
        return (
            attn.q_proj(hidden_states),
            attn.k_proj(hidden_states),
            attn.v_proj(hidden_states),
        )

    @torch.no_grad()
    def prepare_static_norm_gammas(self, *, dtype: torch.dtype) -> dict[str, int]:
        """Precompute `1 + weight` for static layer-local Gemma norms.

        Gemma stores the learnable RMSNorm offset and applies `1 + weight`
        at runtime. The offset is immutable during inference, so materialize
        the effective gamma once after checkpoint loading/device placement.
        AdaRMS layers are intentionally skipped because their gamma depends on
        the timestep condition. Model-final norms are also excluded: the
        prefix graph returns only KV cache, so its final hidden state is dead,
        while the action expert final norm is adaptive.
        """
        buffer_name = "_npu_static_gamma"
        prepared = {
            "input_layernorm": 0,
            "post_attention_layernorm": 0,
            "model_norm": 0,
        }
        for model in (self.paligemma.model.language_model, self.gemma_expert.model):
            for layer in model.layers:
                for norm_name in ("input_layernorm", "post_attention_layernorm"):
                    layernorm = getattr(layer, norm_name)
                    if getattr(layernorm, "dense", None) is not None:
                        continue
                    gamma = layernorm.weight.detach().add(1.0).to(dtype=dtype).contiguous()
                    if buffer_name in layernorm._buffers:
                        layernorm._buffers[buffer_name] = gamma
                    else:
                        layernorm.register_buffer(buffer_name, gamma, persistent=False)
                    prepared[norm_name] += 1
        return prepared

    def _static_norm_gamma(self, layernorm, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return an offline-precomputed static gamma when one is available."""
        norm_weight = getattr(layernorm, "_npu_static_gamma", None)
        if norm_weight is None:
            norm_weight = layernorm.weight.add(1.0)
        if norm_weight.device != hidden_states.device or norm_weight.dtype != hidden_states.dtype:
            norm_weight = norm_weight.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return norm_weight

    def _npu_rms_norm(self, torch_npu, layernorm, hidden_states: torch.Tensor) -> torch.Tensor:
        """用 NPU RMSNorm 融合算子执行普通 Gemma RMSNorm。"""
        # 普通 RMSNorm 使用 NPU 融合算子；Gemma 的 RMSNorm 权重
        # 以零初始化保存，实际计算使用 1 + weight。
        norm_weight = self._static_norm_gamma(layernorm, hidden_states)
        return self._npu_rms_norm_2d(torch_npu, hidden_states, norm_weight, layernorm.eps)

    def _npu_rms_norm_2d(
        self,
        torch_npu,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """把 rank>2 RMSNorm 输入展平到二维后调用 NPU 算子，再恢复原形状。"""
        if hidden_states.ndim <= 2:
            return torch_npu.npu_rms_norm(hidden_states, weight, eps)[0]
        input_shape = hidden_states.shape
        normed = torch_npu.npu_rms_norm(
            hidden_states.reshape(-1, input_shape[-1]),
            weight,
            eps,
        )[0]
        return normed.reshape(input_shape)

    def _npu_adarms_layernorm(
        self,
        torch_npu,
        layernorm,
        hidden_states: torch.Tensor,
        cond: torch.Tensor | None = None,
        modulation: torch.Tensor | AdaRMSModulation | None = None,
        *,
        return_gate: bool = True,
        skip_shift: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """用 NPU RMSNorm 加 AdaRMS 调制完成 action expert 的条件归一化。"""
        # AdaRMS 的耗时拆成两部分：RMS 归一化和 timestep 条件调制。
        # 图编译 denoise 路径优先消费预制表，跳过每层 dense(cond)；
        # 普通 fused eager 路径不传 modulation，仍按原语义实时计算调制量。
        precomputed_modulation = modulation is not None
        if modulation is None:
            if cond is None:
                raise ValueError("AdaRMS fused layernorm requires cond or precomputed modulation")
            if cond.shape[-1] != layernorm.cond_dim:
                raise ValueError(f"Expected cond dim {layernorm.cond_dim}, got {cond.shape[-1]}")

            modulation = layernorm.dense(cond)
            if len(hidden_states.shape) == 3:
                modulation = modulation.unsqueeze(1)
            scale, shift, gate = modulation.chunk(3, dim=-1)
            if not return_gate:
                gate = None
        else:
            # 制表时已按后续消费形态切分：
            # scale_weight 取出为 [hidden]，直接作为 npu_rms_norm gamma；
            # shift/gate 为 [1, 1, hidden]，直接参与逐元素广播。
            if isinstance(modulation, tuple):
                if len(modulation) == 2:
                    # folding 热路径不携带 shift；其影响已经进入下游 Linear bias。
                    scale_weight, gate = modulation
                    shift = None
                else:
                    scale_weight, shift, gate = modulation
                if not return_gate:
                    gate = None
            else:
                scale_weight = modulation[0, 0, 0]
                shift = modulation[1]
                gate = modulation[2] if return_gate else None

        # PI0.5 推理每个 denoise step 对整个 batch
        # 使用同一个 timestep 条件，因此 1+scale 可以作为 npu_rms_norm 的 gamma，
        # 让 RMSNorm 同时完成归一化和缩放；shift 仍保持原 AdaRMS 语义单独相加。
        if not precomputed_modulation:
            scale_weight = 1 + scale.reshape(-1)
        dynamic_weight = scale_weight.to(dtype=hidden_states.dtype).contiguous()
        normed = self._npu_rms_norm_2d(torch_npu, hidden_states, dynamic_weight, layernorm.eps)
        if skip_shift:
            pass
        elif shift is None:
            raise RuntimeError("AdaRMS modulation omitted shift without a folded Linear bias")
        elif shift.dtype == normed.dtype:
            # bf16 制表路径直接在同 dtype 上做偏置相加，减少一次显式 cast。
            normed = normed + shift
        else:
            normed = normed.float() + shift.float()
        if gate is not None and not precomputed_modulation and gate.dtype != hidden_states.dtype:
            # 非制表路径的 AdaRMS dense 可能仍输出 float32；
            # 制表路径已按推理 dtype 保存 gate，避免每层重复搬运。
            gate = gate.to(dtype=hidden_states.dtype)
        return normed.to(hidden_states.dtype), gate

    def _npu_add_adarms_layernorm(
        self,
        torch_npu,
        layernorm,
        residual: torch.Tensor,
        update: torch.Tensor,
        cond: torch.Tensor | None = None,
        modulation: torch.Tensor | AdaRMSModulation | None = None,
        *,
        return_gate: bool = True,
        skip_shift: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """用 npu_add_rms_norm 显式融合 residual add + AdaRMS/RMSNorm。"""
        if getattr(layernorm, "dense", None) is None:
            residual = residual.to(dtype=update.dtype)
            norm_weight = self._static_norm_gamma(layernorm, update)
            normed, _, after_residual = torch_npu.npu_add_rms_norm(
                update,
                residual,
                norm_weight,
                layernorm.eps,
            )
            return normed, None, after_residual

        precomputed_modulation = modulation is not None
        if modulation is None:
            if cond is None:
                raise ValueError("AdaRMS fused add layernorm requires cond or precomputed modulation")
            if cond.shape[-1] != layernorm.cond_dim:
                raise ValueError(f"Expected cond dim {layernorm.cond_dim}, got {cond.shape[-1]}")

            modulation = layernorm.dense(cond)
            if len(residual.shape) == 3:
                modulation = modulation.unsqueeze(1)
            scale, shift, gate = modulation.chunk(3, dim=-1)
            if not return_gate:
                gate = None
        else:
            if isinstance(modulation, tuple):
                if len(modulation) == 2:
                    # folding 热路径只需要 RMSNorm scale 与 residual gate。
                    scale_weight, gate = modulation
                    shift = None
                else:
                    scale_weight, shift, gate = modulation
                if not return_gate:
                    gate = None
            else:
                scale_weight = modulation[0, 0, 0]
                shift = modulation[1]
                gate = modulation[2] if return_gate else None

        if not precomputed_modulation:
            scale_weight = 1 + scale.reshape(-1)
        update = update.to(dtype=residual.dtype)
        dynamic_weight = scale_weight.to(dtype=update.dtype).contiguous()
        normed, _, after_residual = torch_npu.npu_add_rms_norm(
            update,
            residual.to(dtype=update.dtype),
            dynamic_weight,
            layernorm.eps,
        )
        if skip_shift:
            pass
        elif shift is None:
            raise RuntimeError("AdaRMS modulation omitted shift without a folded Linear bias")
        elif shift.dtype == normed.dtype:
            normed = normed + shift
        else:
            normed = normed.float() + shift.float()
        if gate is not None and not precomputed_modulation and gate.dtype != residual.dtype:
            gate = gate.to(dtype=residual.dtype)
        return normed.to(residual.dtype), gate, after_residual

    def _npu_or_adarms_layernorm(
        self,
        torch_npu,
        layernorm,
        hidden_states: torch.Tensor,
        cond: torch.Tensor | None,
        modulation: torch.Tensor | AdaRMSModulation | None = None,
        *,
        return_gate: bool = True,
        skip_shift: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """在普通 RMSNorm、AdaRMS 和原始 fallback layernorm 之间选择执行路径。"""
        if cond is None and getattr(layernorm, "dense", None) is None:
            return self._npu_rms_norm(torch_npu, layernorm, hidden_states), None
        if modulation is not None and getattr(layernorm, "dense", None) is not None:
            return self._npu_adarms_layernorm(
                torch_npu,
                layernorm,
                hidden_states,
                cond=cond,
                modulation=modulation,
                return_gate=return_gate,
                skip_shift=skip_shift,
            )
        if cond is not None and getattr(layernorm, "dense", None) is not None:
            return self._npu_adarms_layernorm(
                torch_npu,
                layernorm,
                hidden_states,
                cond=cond,
                return_gate=return_gate,
                skip_shift=skip_shift,
            )
        return layernorm_forward(layernorm, hidden_states, cond)

    def _cache_layer_tensors(self, past_key_values, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """从 dict cache 或 DynamicCache 中取出指定层的 key/value。"""
        if isinstance(past_key_values, dict):
            cache = past_key_values[layer_idx]
            return cache["key_states"], cache["value_states"]
        keys, values, _sliding_window = list(past_key_values)[layer_idx]
        return keys, values

    def _forward_npu_optimized(
        self,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: dict | None,
        inputs_embeds: list[torch.FloatTensor],
        use_cache: bool | None,
        adarms_cond: list[torch.Tensor | None],
        adarms_modulations: list[torch.Tensor | None] | None = None,
        adarms_bias_plans: list[Any | None] | None = None,
        prepared_npu_attention_mask: torch.Tensor | None = None,
        prepared_npu_rotary_cache: tuple[torch.Tensor, ...] | None = None,
    ):
        """NPU 推理主 forward：按层融合 QKV、RoPE、PFA、RMSNorm/AdaRMS。"""
        # PI0.5 的 denoise 需要 AdaRMS；本路径融合 QKV、RoPE、
        # Prompt Flash Attention，并把 AdaRMS 中的 RMS 归一化替换为
        # npu_rms_norm。图编译 denoise 可传入调制表，跳过每层 dense(cond)。
        torch_npu = _import_torch_npu()
        if torch_npu is None:
            raise RuntimeError("torch_npu is required for PI0.5 NPU fused inference")
        if adarms_modulations is None:
            adarms_modulations = [None, None]
        if adarms_bias_plans is None:
            adarms_bias_plans = [None, None]

        attention_backend = self._npu_attention_backend
        # Default attention consumes the original additive mask. PFA/FIAS use the
        # compact int8 blocked mask, optionally prebuilt outside the denoise loop.
        npu_attention_mask = None
        if attention_backend != "default":
            npu_attention_mask = (
                self._npu_attention_mask(attention_mask)
                if prepared_npu_attention_mask is None
                else prepared_npu_attention_mask
            )
        active_models = [
            (
                0,
                self.paligemma.model.language_model,
                inputs_embeds[0],
                adarms_cond[0],
                adarms_modulations[0],
                adarms_bias_plans[0],
            ),
            (
                1,
                self.gemma_expert.model,
                inputs_embeds[1],
                adarms_cond[1],
                adarms_modulations[1],
                adarms_bias_plans[1],
            ),
        ]
        active_models = [
            (idx, model, embeds, cond, modulation_table, bias_plan)
            for idx, model, embeds, cond, modulation_table, bias_plan in active_models
            if embeds is not None
        ]
        use_shared_prefix_fias = (
            self._shared_prefix_fias_enabled
            and attention_backend in {"hybrid", "fias"}
            and past_key_values is not None
            and len(active_models) == 1
            and active_models[0][0] == 1
            and active_models[0][2].shape[0] == 1
        )
        adarms_modulation_offsets = [0, 0]

        num_layers = active_models[0][1].config.num_hidden_layers
        outputs_by_index: list[torch.Tensor | None] = [inputs_embeds[0], inputs_embeds[1]]
        prefix_past_key_values = past_key_values
        if use_cache and prefix_past_key_values is None:
            prefix_past_key_values = {}

        attention_model = active_models[0][1]
        first_attn = attention_model.layers[0].self_attn
        attention_head_dim = first_attn.head_dim
        attention_num_heads = attention_model.config.num_attention_heads
        attention_num_kv_heads = attention_model.config.num_key_value_heads
        attention_scale_value = 1.0 / math.sqrt(attention_head_dim)
        if prepared_npu_rotary_cache is None:
            rotary_cache = self._prepare_npu_rotary_cache(
                position_ids,
                first_attn,
                attention_num_heads,
                attention_num_kv_heads,
            )
        else:
            rotary_cache = prepared_npu_rotary_cache
        rotary_cos, rotary_sin, qkv_rotary_cos, qkv_rotary_sin = rotary_cache

        for layer_idx in range(num_layers):
            qkv_states_parts = []
            query_states_parts = []
            key_states_parts = []
            value_states_parts = []
            residual_parts = []

            for model_idx, model, hidden_states, cond, modulation_table, bias_plan in active_models:
                layer = model.layers[layer_idx]
                attn = layer.self_attn
                layer_biases = bias_plan[0][layer_idx] if bias_plan is not None else None
                qkv_bias = layer_biases[0] if layer_biases is not None else None
                attn_proj_dtype = self._attention_projection_dtype(attn)
                residual = hidden_states.to(dtype=attn_proj_dtype)
                input_modulation = None
                if modulation_table is not None:
                    input_modulation = modulation_table[adarms_modulation_offsets[model_idx]]
                    adarms_modulation_offsets[model_idx] = adarms_modulation_offsets[model_idx] + 1
                normed, gate = self._npu_or_adarms_layernorm(
                    torch_npu,
                    layer.input_layernorm,
                    residual,
                    cond,
                    input_modulation,
                    skip_shift=qkv_bias is not None,
                )
                if normed.dtype != attn_proj_dtype:
                    normed = normed.to(dtype=attn_proj_dtype)

                num_heads = model.config.num_attention_heads
                num_kv_heads = model.config.num_key_value_heads
                q_out = num_heads * attention_head_dim
                kv_out = num_kv_heads * attention_head_dim
                qkv = getattr(attn, "qkv", None)
                if qkv is not None:
                    if qkv_bias is None:
                        projected_qkv = qkv(normed)
                    else:
                        projected_qkv = _npu_graph_safe_linear_with_bias(qkv, normed, qkv_bias)
                    qkv_states_parts.append(
                        projected_qkv.view(
                            *normed.shape[:-1],
                            num_heads + 2 * num_kv_heads,
                            attention_head_dim,
                        )
                    )
                else:
                    if qkv_bias is not None:
                        raise RuntimeError("AdaRMS QKV bias fusion requires fused QKV weights")
                    query_states, key_states, value_states = self._project_attention_qkv(
                        attn,
                        normed,
                        q_out,
                        kv_out,
                    )
                    query_states_parts.append(query_states.view(*normed.shape[:-1], num_heads, attention_head_dim))
                    key_states_parts.append(key_states.view(*normed.shape[:-1], num_kv_heads, attention_head_dim))
                    value_states_parts.append(value_states.view(*normed.shape[:-1], num_kv_heads, attention_head_dim))
                residual_parts.append((model_idx, residual, gate, cond))

            if qkv_states_parts:
                if query_states_parts:
                    raise RuntimeError("All active PI0.5 attention modules must use the same QKV layout")
                # 融合投影的 head 顺序固定为 [Q, K, V]。Q/K 正常旋转，
                # V 借助恒等 cache 透传，消除 split/cat/RotaryMul/split 的数据编排。
                qkv_states = qkv_states_parts[0] if len(qkv_states_parts) == 1 else torch.cat(qkv_states_parts, dim=1)
                qkv_states = torch_npu.npu_rotary_mul(
                    qkv_states,
                    qkv_rotary_cos,
                    qkv_rotary_sin,
                )
                query_states, key_states, value_states = qkv_states.split(
                    [attention_num_heads, attention_num_kv_heads, attention_num_kv_heads],
                    dim=2,
                )
            else:
                query_states = (
                    query_states_parts[0] if len(query_states_parts) == 1 else torch.cat(query_states_parts, dim=1)
                )
                key_states = key_states_parts[0] if len(key_states_parts) == 1 else torch.cat(key_states_parts, dim=1)
                value_states = (
                    value_states_parts[0] if len(value_states_parts) == 1 else torch.cat(value_states_parts, dim=1)
                )
                query_states, key_states = self._npu_rotary_emb(
                    torch_npu,
                    query_states,
                    key_states,
                    rotary_cos,
                    rotary_sin,
                )

            if use_cache and len(active_models) == 1 and active_models[0][0] == 0:
                # FIAS 要求 shared-prefix KV 连续；在 prefix 图中一次性整理布局，
                # 避免固定 10 步 denoise 的每层循环重复复制整段 prefix cache。
                if self._shared_prefix_fias_enabled and attention_backend in {"hybrid", "fias"}:
                    key_states = key_states.contiguous()
                    value_states = value_states.contiguous()
                prefix_past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            elif past_key_values is not None:
                cached_key_states, cached_value_states = self._cache_layer_tensors(past_key_values, layer_idx)
                if not use_shared_prefix_fias:
                    key_states = torch.cat([cached_key_states, key_states], dim=1)
                    value_states = torch.cat([cached_value_states, value_states], dim=1)

            batch_size = query_states.shape[0]
            if use_shared_prefix_fias:
                # FIAS 按 [shared prefix, action] 解释 KV。prefix 长度取当前静态
                # cache shape，因此 200/132 token 图分别使用各自长度，不硬编码 968。
                att_output = torch_npu.npu_fused_infer_attention_score(
                    query_states,
                    key_states.contiguous(),
                    value_states.contiguous(),
                    key_shared_prefix=cached_key_states,
                    value_shared_prefix=cached_value_states,
                    actual_shared_prefix_len=[cached_key_states.shape[1]],
                    atten_mask=npu_attention_mask,
                    num_heads=attention_num_heads,
                    input_layout="BSND",
                    scale=attention_scale_value,
                    pre_tokens=65535,
                    next_tokens=65535,
                    num_key_value_heads=attention_num_kv_heads,
                    sparse_mode=0,
                    inner_precise=0,
                )[0]
            elif attention_backend == "fias":
                att_output = torch_npu.npu_fused_infer_attention_score(
                    query_states,
                    key_states.contiguous(),
                    value_states.contiguous(),
                    atten_mask=npu_attention_mask,
                    num_heads=attention_num_heads,
                    input_layout="BSND",
                    scale=attention_scale_value,
                    pre_tokens=65535,
                    next_tokens=65535,
                    num_key_value_heads=attention_num_kv_heads,
                    sparse_mode=0,
                    inner_precise=0,
                )[0]
            elif attention_backend == "default":
                # Keep QKV fusion, fused RoPE/RMSNorm and AdaRMS folding enabled;
                # only replace the attention kernel with the graph-safe baseline.
                attn_module = attention_model.layers[layer_idx].self_attn
                att_output, _ = _npu_graph_safe_gemma_attention_forward(
                    attn_module,
                    query_states.transpose(1, 2),
                    key_states.transpose(1, 2),
                    value_states.transpose(1, 2),
                    attention_mask,
                    scaling=attention_scale_value,
                    dropout=0.0,
                )
            else:
                att_output = torch_npu.npu_prompt_flash_attention(
                    query_states,
                    key_states.contiguous(),
                    value_states.contiguous(),
                    num_heads=attention_num_heads,
                    input_layout="BSND",
                    scale_value=attention_scale_value,
                    pre_tokens=65535,
                    next_tokens=65535,
                    atten_mask=npu_attention_mask,
                    num_key_value_heads=attention_num_kv_heads,
                )
            att_output = att_output.reshape(batch_size, -1, attention_num_heads * attention_head_dim)

            next_active_models = []
            start = 0
            for model_idx, model, hidden_states, cond, modulation_table, bias_plan in active_models:
                layer = model.layers[layer_idx]
                layer_biases = bias_plan[0][layer_idx] if bias_plan is not None else None
                gate_bias = layer_biases[1] if layer_biases is not None else None
                up_bias = layer_biases[2] if layer_biases is not None else None
                _residual_model_idx, residual, gate, _cond = residual_parts.pop(0)
                end = start + hidden_states.shape[1]
                o_proj = layer.self_attn.o_proj
                out_emb = o_proj(att_output[:, start:end])

                if cond is None and getattr(layer.post_attention_layernorm, "dense", None) is None:
                    norm_weight = self._static_norm_gamma(layer.post_attention_layernorm, out_emb)
                    out_emb, _, after_first_residual = torch_npu.npu_add_rms_norm(
                        out_emb,
                        residual.to(out_emb.dtype),
                        norm_weight,
                        layer.post_attention_layernorm.eps,
                    )
                    out_emb = layer.mlp(out_emb)
                    out_emb = out_emb + after_first_residual
                else:
                    residual_update = out_emb if gate is None else out_emb * gate
                    post_modulation = None
                    if modulation_table is not None:
                        post_modulation = modulation_table[adarms_modulation_offsets[model_idx]]
                        adarms_modulation_offsets[model_idx] = adarms_modulation_offsets[model_idx] + 1
                    out_emb, gate, after_first_residual = self._npu_add_adarms_layernorm(
                        torch_npu,
                        layer.post_attention_layernorm,
                        residual,
                        residual_update,
                        cond,
                        post_modulation,
                        skip_shift=gate_bias is not None or up_bias is not None,
                    )
                    up_proj_dtype = linear_weight_dtype(layer.mlp.up_proj)
                    if out_emb.dtype != up_proj_dtype:
                        out_emb = out_emb.to(dtype=up_proj_dtype)
                    if gate_bias is None and up_bias is None:
                        out_emb = layer.mlp(out_emb)
                    else:
                        gate_output = (
                            _npu_graph_safe_linear_with_bias(layer.mlp.gate_proj, out_emb, gate_bias)
                            if gate_bias is not None
                            else layer.mlp.gate_proj(out_emb)
                        )
                        up_output = (
                            _npu_graph_safe_linear_with_bias(layer.mlp.up_proj, out_emb, up_bias)
                            if up_bias is not None
                            else layer.mlp.up_proj(out_emb)
                        )
                        out_emb = layer.mlp.down_proj(layer.mlp.act_fn(gate_output) * up_output)
                    out_emb = _gated_residual(after_first_residual, out_emb, gate)

                outputs_by_index[model_idx] = out_emb
                next_active_models.append((model_idx, model, out_emb, cond, modulation_table, bias_plan))
                start = end
            active_models = next_active_models

        for model_idx, model, hidden_states, cond, modulation_table, bias_plan in active_models:
            if cond is None and getattr(model.norm, "dense", None) is None:
                outputs_by_index[model_idx] = self._npu_rms_norm(torch_npu, model.norm, hidden_states)
            else:
                final_modulation = None
                if modulation_table is not None:
                    final_modulation = modulation_table[adarms_modulation_offsets[model_idx]]
                    adarms_modulation_offsets[model_idx] = adarms_modulation_offsets[model_idx] + 1
                outputs_by_index[model_idx], _ = self._npu_or_adarms_layernorm(
                    torch_npu,
                    model.norm,
                    hidden_states,
                    cond,
                    final_modulation,
                    return_gate=False,
                    skip_shift=(bias_plan is not None and bias_plan[1] is not None),
                )

        return outputs_by_index, prefix_past_key_values if use_cache else None

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
        adarms_modulations: list[torch.Tensor | None] | None = None,
        adarms_bias_plans: list[Any | None] | None = None,
        prepared_npu_attention_mask: torch.Tensor | None = None,
        prepared_npu_rotary_cache: tuple[torch.Tensor, ...] | None = None,
    ):
        """统一 transformer forward，按输入形态选择 prefix、suffix 或联合训练路径。"""
        if adarms_cond is None:
            adarms_cond = [None, None]
        if adarms_modulations is None:
            adarms_modulations = [None, None]
        if self._can_use_npu_fused_inference(
            attention_mask, position_ids, inputs_embeds, adarms_cond, adarms_modulations
        ):
            return self._forward_npu_optimized(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                adarms_cond=adarms_cond,
                adarms_modulations=adarms_modulations,
                adarms_bias_plans=adarms_bias_plans,
                prepared_npu_attention_mask=prepared_npu_attention_mask,
                prepared_npu_rotary_cache=prepared_npu_rotary_cache,
            )
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.model.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            paligemma_layers = self.paligemma.model.language_model.layers
            gemma_expert_layers = self.gemma_expert.model.layers
            rotary_emb = self.paligemma.model.language_model.rotary_emb

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            for layers in zip(paligemma_layers, gemma_expert_layers, strict=True):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        layers=layers,
                        rotary_emb=rotary_emb,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        layers=layers,
                        rotary_emb=rotary_emb,
                    )

            # final norm
            final_norms = (
                self.paligemma.model.language_model.norm,
                self.gemma_expert.model.norm,
            )

            def compute_final_norms(inputs_embeds, adarms_cond):
                """训练联合路径末尾分别对 prefix/suffix 执行 final norm。"""
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = layernorm_forward(final_norms[i], hidden_states, adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


class PI05Pytorch(nn.Module):  # see openpi `PI0Pytorch`
    """PI0.5 核心模型，负责训练 loss、推理采样和 NPU/TorchAir 优化路径。"""

    def __init__(self, config: PI05Config, rtc_processor: RTCProcessor | None = None):
        """初始化 PaliGemma+action expert、动作投影头和推理缓存/计时状态。"""
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, invalid resolution: {config.image_resolution}"
            )

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.register_buffer("_action_denoise_dt", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer(
            "_action_denoise_timestep_table",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_action_denoise_adarms_cond_table",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_action_denoise_adarms_scale_weight_table",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_action_denoise_adarms_shift_table",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_action_denoise_adarms_gate_table",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_action_denoise_suffix_position_ids",
            torch.empty(0, dtype=torch.int64),
            persistent=False,
        )
        for name in (
            "_action_denoise_qkv_bias_table",
            "_action_denoise_mlp_gate_bias_table",
            "_action_denoise_mlp_up_bias_table",
            "_action_denoise_action_out_bias_table",
        ):
            self.register_buffer(name, torch.empty(0, dtype=torch.float32), persistent=False)
        self._action_denoise_adarms_modulation_steps: tuple[
            tuple[AdaRMSModulation, ...],
            ...,
        ] = ()
        self._action_denoise_adarms_bias_steps: tuple[
            tuple[
                tuple[tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None], ...],
                torch.Tensor | None,
            ],
            ...,
        ] = ()
        self._adarms_bias_fusion_enabled = False
        self._adarms_bias_fusion_stage = "none"

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False
        self._sample_actions_graph_compile_enabled = False
        self._compiled_action_prefix_forward = None
        self._compiled_action_denoise_10_steps = None
        self._compiled_action_denoise_ab2_6_steps = None
        self._sample_actions_graph_compile_info: dict[str, Any] = {}

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            # Also compile the main forward pass used during training
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def _apply(self, fn):
        """在 module.to()/dtype 转换后重建 AdaRMS per-step buffer 引用。"""
        result = super()._apply(fn)
        self._rebuild_adarms_modulation_step_refs()
        self._rebuild_adarms_bias_step_refs()
        return result

    def _denoise_solver(self) -> str:
        """返回规范化后的去噪求解器名称。"""
        return str(getattr(self.config, "denoise_solver", "euler")).lower()

    def _graph_denoise_steps(self) -> int:
        """返回当前求解器对应的固定图步数。"""
        return PI05_AB2_DENOISE_STEPS if self._denoise_solver() == "ab2" else PI05_GRAPH_DENOISE_STEPS

    def _resolve_denoise_steps(self, num_steps: int | None) -> int:
        """解析实际步数，并固定 AB2 为经过验证的六步时间网格。"""
        solver = self._denoise_solver()
        resolved_steps = (
            PI05_AB2_DENOISE_STEPS
            if num_steps is None and solver == "ab2"
            else self.config.num_inference_steps
            if num_steps is None
            else int(num_steps)
        )
        if resolved_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {resolved_steps}")
        if solver == "ab2" and resolved_steps != PI05_AB2_DENOISE_STEPS:
            raise ValueError(
                f"AB2 inference requires exactly {PI05_AB2_DENOISE_STEPS} denoise steps; got {resolved_steps}"
            )
        return int(resolved_steps)

    def _compiled_action_denoise_for_solver(self):
        """选择当前求解器对应的已编译 denoise 图。"""
        if self._denoise_solver() == "ab2":
            return self._compiled_action_denoise_ab2_6_steps
        return self._compiled_action_denoise_10_steps

    def prepare_inference_optimizations(
        self,
        *,
        enable_npu_fused_ops: bool = False,
        enable_graph_compile: bool | None = None,
        enable_qkv_fusion: bool | None = None,
        enable_mlp_fusion: bool = False,
        mlp_fusion_scope: Literal["all", "prefix"] = "all",
        enable_shared_prefix_fias: bool | None = None,
        enable_adarms_bias_fusion: bool | None = None,
        adarms_bias_fusion_stage: str | None = None,
    ) -> dict[str, Any]:
        """权重加载完成后准备 PI0.5 推理优化，并返回实际启用的优化信息。"""
        self._prepare_action_compute_dtype()
        attention_backend = _resolve_npu_attention_backend(os.environ.get(PI05_NPU_ATTENTION_BACKEND_ENV))
        self.paligemma_with_expert.set_npu_attention_backend(attention_backend)
        enable_graph_compile = (
            bool(getattr(self.config, "compile_inference_graph", False))
            if enable_graph_compile is None
            else bool(enable_graph_compile)
        )
        enable_qkv_fusion = (
            bool(enable_npu_fused_ops or enable_graph_compile) if enable_qkv_fusion is None else bool(enable_qkv_fusion)
        )
        if mlp_fusion_scope not in {"all", "prefix"}:
            raise ValueError(f"mlp_fusion_scope must be 'all' or 'prefix', got {mlp_fusion_scope!r}")
        enable_shared_prefix_fias = (
            bool(getattr(self.config, "shared_prefix_fias", False))
            if enable_shared_prefix_fias is None
            else bool(enable_shared_prefix_fias)
        )
        enable_adarms_bias_fusion = (
            bool(getattr(self.config, "adarms_bias_fusion", False))
            if enable_adarms_bias_fusion is None
            else bool(enable_adarms_bias_fusion)
        )
        adarms_bias_fusion_stage = (
            str(getattr(self.config, "adarms_bias_fusion_stage", "all"))
            if adarms_bias_fusion_stage is None
            else str(adarms_bias_fusion_stage)
        )
        adarms_bias_fusion_stage = _resolve_adarms_bias_fusion_stage(
            enable_adarms_bias_fusion,
            adarms_bias_fusion_stage,
        )
        if enable_adarms_bias_fusion and not enable_graph_compile:
            raise ValueError("AdaRMS bias fusion requires the fixed denoise inference graph")
        if enable_adarms_bias_fusion:
            incompatible_targets = self._adarms_bias_fusion_incompatible_targets(adarms_bias_fusion_stage)
            if incompatible_targets:
                raise ValueError(
                    "AdaRMS bias fusion requires floating-point DiT projections; "
                    "quantization outside the DiT is supported, but these folding targets are "
                    f"incompatible: {', '.join(incompatible_targets)}"
                )
        if enable_shared_prefix_fias and not (enable_npu_fused_ops or enable_graph_compile):
            raise ValueError("shared-prefix FIAS requires NPU fused ops or the inference graph")
        if enable_shared_prefix_fias and attention_backend not in {"hybrid", "fias"}:
            raise ValueError(
                f"shared-prefix FIAS requires attention backend 'hybrid' or 'fias', got {attention_backend!r}"
            )
        self.paligemma_with_expert.enable_shared_prefix_fias(enable_shared_prefix_fias)
        enable_vision_npu_fused_ops = bool(enable_npu_fused_ops or enable_graph_compile)
        static_norm_gamma_counts = (
            self.paligemma_with_expert.prepare_static_norm_gammas(dtype=self._action_compute_dtype())
            if enable_npu_fused_ops
            else {
                "input_layernorm": 0,
                "post_attention_layernorm": 0,
                "model_norm": 0,
            }
        )
        static_norm_gamma_layers = sum(static_norm_gamma_counts.values())
        optimization_info = {
            "denoise_solver": self._denoise_solver(),
            "denoise_steps": self._graph_denoise_steps(),
            "npu_fused_ops_enabled": bool(enable_npu_fused_ops),
            "attention_backend": attention_backend,
            "attention_backend_env": PI05_NPU_ATTENTION_BACKEND_ENV,
            "qkv_fusion_requested": bool(enable_qkv_fusion),
            "qkv_weights_fused": self.paligemma_with_expert._qkv_weights_fused,
            "mlp_fusion_requested": bool(enable_mlp_fusion),
            **self.paligemma_with_expert.mlp_fusion_metadata(),
            "qkv_single_rotary": bool(enable_qkv_fusion),
            "rotary_trig_dedup": bool(enable_qkv_fusion),
            "fixed_denoise_npu_context": bool(enable_graph_compile),
            "shared_prefix_fias_enabled": enable_shared_prefix_fias,
            "denoise_attention": (
                "torch_npu.npu_fused_infer_attention_score_shared_prefix"
                if enable_shared_prefix_fias
                else "torch_npu.npu_fused_infer_attention_score"
                if attention_backend == "fias"
                else "matmul_softmax"
                if attention_backend == "default"
                else "torch_npu.npu_prompt_flash_attention"
            ),
            "action_compute_dtype": str(self._action_compute_dtype()),
            "vision_compute_dtype": str(self._vision_compute_dtype()),
            "adarms_npu_mode": "dynamic_weight",
            "adarms_bias_fusion_enabled": False,
            "adarms_bias_fusion_stage": "none",
            "adarms_bias_fusion_scope": "none",
            "static_norm_gamma_precomputed": bool(static_norm_gamma_layers),
            "static_norm_gamma_layers": static_norm_gamma_layers,
            "static_norm_gamma_counts": dict(static_norm_gamma_counts),
            "static_post_attention_gamma_precomputed": bool(static_norm_gamma_counts["post_attention_layernorm"]),
            "static_post_attention_gamma_layers": static_norm_gamma_counts["post_attention_layernorm"],
            "fixed_denoise_lookup_tables": self._fixed_denoise_lookup_tables_info(),
            "graph_compile": dict(self._sample_actions_graph_compile_info),
            "vision_tower_npu_fused_ops": {
                "local_siglip_vision_tower": isinstance(
                    self.paligemma_with_expert.paligemma.model.vision_tower,
                    PI05SiglipVisionModel,
                ),
                "enabled": False,
                "default_enabled": True,
                "enable_env": PI05_ENABLE_VISION_NPU_PFA_ENV,
                "attention": "transformers_default",
                "attention_backend": "default",
                "qkv_weights_fused": False,
            },
        }
        self.paligemma_with_expert.enable_npu_fused_inference(enable_npu_fused_ops)
        if enable_qkv_fusion:
            # TorchAir 图编译和 eager PFA 路径都复用融合后的 QKV 投影。
            self.paligemma_with_expert.fuse_qkv_weights()
        if enable_mlp_fusion:
            # 正式 selective INT8 路径只融合 Prefix LLM；BF16 DiT
            # expert 的 gate/up 成对消融为负收益。
            self.paligemma_with_expert.fuse_mlp_weights(include_action_expert=mlp_fusion_scope == "all")
            optimization_info.update(self.paligemma_with_expert.mlp_fusion_metadata())
        if enable_vision_npu_fused_ops:
            if self.paligemma_with_expert.should_enable_vision_tower_npu_fused_ops(default=True):
                optimization_info["vision_tower_npu_fused_ops"] = (
                    self.paligemma_with_expert.prepare_vision_tower_npu_fused_ops(enable_qkv_fusion=enable_qkv_fusion)
                )
                optimization_info["vision_tower_npu_fused_ops"]["enabled"] = True
                optimization_info["vision_tower_npu_fused_ops"]["default_enabled"] = True
                optimization_info["vision_tower_npu_fused_ops"]["enable_env"] = PI05_ENABLE_VISION_NPU_PFA_ENV
            optimization_info["qkv_weights_fused"] = self.paligemma_with_expert._qkv_weights_fused
        if enable_npu_fused_ops or enable_graph_compile:
            optimization_info["action_compute_dtype"] = str(self._action_compute_dtype())
            optimization_info["vision_compute_dtype"] = str(self._vision_compute_dtype())
            optimization_info["adarms_npu_mode"] = "dynamic_weight"
        int8_prepack = {"enabled": False, "layers": 0, "nbytes": 0, "format": None}
        if self.config.quantization is not None:
            # Fusion changes the active Linear module set (Selective-99 becomes
            # 81 active projections). Pack only after every QKV/Gate-Up rewrite
            # and before TorchAir captures the fixed inference graphs.
            packed_layers = []
            for name, module in _active_int8_linears_for_prepack(self):
                packed_layers.append((name, module.prepare_npu_qweight_layout("nz")))
            if not packed_layers:
                raise RuntimeError("quantized PI05 Ascend910B model has no active INT8 projections")
            formats = {metadata["format"] for _, metadata in packed_layers}
            int8_prepack = {
                "enabled": True,
                "layers": len(packed_layers),
                "nbytes": sum(int(metadata["nbytes"]) for _, metadata in packed_layers),
                "format": formats.pop() if len(formats) == 1 else "mixed",
            }
        optimization_info["int8_qweight_prepack"] = int8_prepack
        if enable_graph_compile:
            # 先固定 folding stage，再构造热路径调制 buffer。这样被折叠的
            # layernorm 只绑定 (scale, gate)，shift 仅用于一次性生成 bias 表。
            self._adarms_bias_fusion_enabled = enable_adarms_bias_fusion
            self._adarms_bias_fusion_stage = adarms_bias_fusion_stage if self._adarms_bias_fusion_enabled else "none"
            self._refresh_fixed_denoise_lookup_tables()
            if self._adarms_bias_fusion_enabled:
                self._refresh_adarms_bias_tables()
            else:
                self._clear_adarms_bias_step_buffers()
            if (
                not self._sample_actions_graph_compile_enabled
                or self._sample_actions_graph_compile_info.get("denoise_solver") != self._denoise_solver()
            ):
                self.enable_sample_actions_graph_compile()
            optimization_info["fixed_denoise_lookup_tables"] = self._fixed_denoise_lookup_tables_info()
            optimization_info["graph_compile"] = dict(self._sample_actions_graph_compile_info)
            optimization_info["graph_compile"]["shared_prefix_fias"] = enable_shared_prefix_fias
            optimization_info["adarms_bias_fusion_enabled"] = self._adarms_bias_fusion_enabled
            optimization_info["adarms_bias_fusion_stage"] = self._adarms_bias_fusion_stage
            optimization_info["adarms_bias_fusion_scope"] = "dit_only" if self._adarms_bias_fusion_enabled else "none"
        return optimization_info

    def _action_compute_dtype(self, device: torch.device | str | None = None) -> torch.dtype:
        """根据配置和设备能力选择 action expert 推理计算 dtype。"""
        target_device = device
        if target_device is None:
            try:
                target_device = next(self.parameters()).device
            except StopIteration:
                target_device = self.config.device
        requested = getattr(self.config, "dtype", "bfloat16")
        if requested == "bfloat16" and _device_supports_bfloat16(target_device):
            return torch.bfloat16
        return torch.float32

    def _cast_action_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """把动作、噪声等浮点 tensor 转成 action 推理路径使用的 dtype。"""
        target_dtype = self._action_compute_dtype(tensor.device)
        if tensor.dtype.is_floating_point and tensor.dtype != target_dtype:
            return tensor.to(dtype=target_dtype)
        return tensor

    def _vision_compute_dtype(self) -> torch.dtype:
        """返回 prefix 视觉路径的推理 dtype，目前与 action 路径保持一致。"""
        return self._action_compute_dtype()

    def _prepare_action_compute_dtype(self) -> None:
        """统一设置 vision/action/time MLP 的推理 dtype。"""
        target_dtype = self._action_compute_dtype()
        precision: Literal["bfloat16", "float32"] = "bfloat16" if target_dtype == torch.bfloat16 else "float32"
        vision_precision: Literal["bfloat16", "float32"] = (
            "bfloat16" if self._vision_compute_dtype() == torch.bfloat16 else "float32"
        )
        # 与 pi0 保持同一精度口径。权重加载和 model.to(device)
        # 完成后，再统一调整 prefix vision path 与 action expert 的推理 dtype。
        self.paligemma_with_expert.to_bfloat16_for_selected_params(
            precision,
            vision_precision=vision_precision,
        )
        modules = [
            self.action_in_proj,
            self.action_out_proj,
            self.time_mlp_in,
            self.time_mlp_out,
        ]
        for module in modules:
            module.to(dtype=target_dtype)

    def _set_denoise_lookup_buffer(
        self,
        name: str,
        tensor: torch.Tensor,
    ) -> None:
        """注册或更新 denoise 固定查表 buffer，不写入 checkpoint。"""
        if name in self._buffers:
            setattr(self, name, tensor)
        else:
            self.register_buffer(name, tensor, persistent=False)

    def _remove_denoise_lookup_buffers(self, prefixes: tuple[str, ...]) -> None:
        """删除旧的 step 级 buffer，避免切换 folding stage 后残留 shift 输入。"""
        for name in tuple(self._buffers):
            if name.startswith(prefixes):
                delattr(self, name)

    def _fixed_denoise_lookup_tables_info(self) -> dict[str, Any]:
        """导出固定 denoise 查表的形状、dtype 和设备。"""
        scale_table = self._action_denoise_adarms_scale_weight_table
        shift_table = self._action_denoise_adarms_shift_table
        gate_table = self._action_denoise_adarms_gate_table
        qkv_bias_table = self._action_denoise_qkv_bias_table
        mlp_gate_bias_table = self._action_denoise_mlp_gate_bias_table
        mlp_up_bias_table = self._action_denoise_mlp_up_bias_table
        action_out_bias_table = self._action_denoise_action_out_bias_table
        folded_modulation_buffers = sum(
            len(modulation) == 2
            for step_modulations in self._action_denoise_adarms_modulation_steps
            for modulation in step_modulations
        )
        shift_modulation_buffers = sum(
            len(modulation) == 3
            for step_modulations in self._action_denoise_adarms_modulation_steps
            for modulation in step_modulations
        )
        return {
            "denoise_solver": self._denoise_solver(),
            "denoise_steps": self._graph_denoise_steps(),
            "dt_shape": list(self._action_denoise_dt.shape),
            "timestep_table_shape": list(self._action_denoise_timestep_table.shape),
            "adarms_cond_table_shape": list(self._action_denoise_adarms_cond_table.shape),
            "suffix_position_ids_shape": list(self._action_denoise_suffix_position_ids.shape),
            "adarms_modulation_layout": "folding_aware_scale_gate_or_scale_shift_gate_buffers",
            "adarms_scale_weight_table_shape": list(scale_table.shape),
            "adarms_shift_table_shape": list(shift_table.shape),
            "adarms_gate_table_shape": list(gate_table.shape),
            "adarms_modulation_step_buffers": sum(
                len(step_modulations) for step_modulations in self._action_denoise_adarms_modulation_steps
            ),
            "adarms_folded_scale_gate_step_buffers": folded_modulation_buffers,
            "adarms_shift_hot_path_buffers": shift_modulation_buffers,
            "adarms_bias_fusion_enabled": self._adarms_bias_fusion_enabled,
            "adarms_bias_fusion_stage": self._adarms_bias_fusion_stage,
            "qkv_bias_table_shape": list(qkv_bias_table.shape),
            "mlp_gate_bias_table_shape": list(mlp_gate_bias_table.shape),
            "mlp_up_bias_table_shape": list(mlp_up_bias_table.shape),
            "action_out_bias_table_shape": list(action_out_bias_table.shape),
            "adarms_bias_table_dtype": str(qkv_bias_table.dtype),
            "adarms_bias_step_buffers": sum(
                sum(bias is not None for layer_bias_tuple in layer_biases for bias in layer_bias_tuple)
                + int(final_bias is not None)
                for layer_biases, final_bias in self._action_denoise_adarms_bias_steps
            ),
            "adarms_bias_binding": "linear_argument",
            "dtype": str(scale_table.dtype),
            "device": str(scale_table.device),
        }

    def _action_expert_adarms_layernorms(self) -> list[nn.Module]:
        """按 denoise forward 的消费顺序列出 action expert 中所有 AdaRMS 层。"""
        # AdaRMS 调制表的层顺序必须和 denoise 计算顺序严格一致。
        # 这里固定按 action expert 的 layer input/post norm，再接 final norm 展开。
        layernorms: list[nn.Module] = []
        for layer in self.paligemma_with_expert.gemma_expert.model.layers:
            layernorms.append(layer.input_layernorm)
            layernorms.append(layer.post_attention_layernorm)
        layernorms.append(self.paligemma_with_expert.gemma_expert.model.norm)
        return layernorms

    def _adarms_bias_fusion_incompatible_targets(self, stage: str) -> tuple[str, ...]:
        """List DiT projections that cannot accept exact AdaRMS shift folding.

        A checkpoint may quantize the ViT or PaliGemma prefix without affecting
        this optimization. Only the downstream projections on the DiT denoise
        path participate in the folding identity and therefore need FP weights.
        """
        use_qkv = stage in {"qkv", "qkv_mlp", "all"}
        use_mlp = stage in {"qkv_mlp", "mlp_action_out", "all"}
        use_action_out = stage in {"mlp_action_out", "all"}
        targets: list[tuple[str, nn.Module]] = []
        for layer_idx, layer in enumerate(self.paligemma_with_expert.gemma_expert.model.layers):
            if use_qkv:
                qkv = getattr(layer.self_attn, "qkv", None)
                if qkv is not None:
                    targets.append((f"dit.layers.{layer_idx}.self_attn.qkv", qkv))
                else:
                    targets.extend(
                        (
                            (f"dit.layers.{layer_idx}.self_attn.q_proj", layer.self_attn.q_proj),
                            (f"dit.layers.{layer_idx}.self_attn.k_proj", layer.self_attn.k_proj),
                            (f"dit.layers.{layer_idx}.self_attn.v_proj", layer.self_attn.v_proj),
                        )
                    )
            if use_mlp:
                targets.extend(
                    (
                        (f"dit.layers.{layer_idx}.mlp.gate_proj", layer.mlp.gate_proj),
                        (f"dit.layers.{layer_idx}.mlp.up_proj", layer.mlp.up_proj),
                    )
                )
        if use_action_out:
            targets.append(("dit.action_out_proj", self.action_out_proj))
        return tuple(name for name, module in targets if not _is_fp_linear_for_adarms_bias_fusion(module))

    def _adarms_shift_is_folded(self, layernorm_idx: int, num_layernorms: int) -> bool:
        """返回当前 layernorm 的 shift 是否已进入紧随其后的 Linear bias。"""
        if not self._adarms_bias_fusion_enabled:
            return False
        if layernorm_idx == num_layernorms - 1:
            return self._adarms_bias_fusion_stage in {"mlp_action_out", "all"}
        if layernorm_idx % 2 == 0:
            return self._adarms_bias_fusion_stage in {"qkv", "qkv_mlp", "all"}
        return self._adarms_bias_fusion_stage in {"qkv_mlp", "mlp_action_out", "all"}

    def _refresh_adarms_modulation_step_buffers(
        self,
        scale_weight_table: torch.Tensor,
        shift_table: torch.Tensor,
        gate_table: torch.Tensor,
    ) -> None:
        """构造 folding-aware 的 step/layer buffer，避免热路径传入已折叠 shift。"""
        self._remove_denoise_lookup_buffers(
            (
                "_action_denoise_adarms_scale_weight_s",
                "_action_denoise_adarms_shift_s",
                "_action_denoise_adarms_gate_s",
            )
        )
        step_modulations: list[tuple[AdaRMSModulation, ...]] = []
        num_steps, num_layernorms = scale_weight_table.shape[:2]
        for step in range(num_steps):
            layer_modulations: list[AdaRMSModulation] = []
            for layer_idx in range(num_layernorms):
                scale_name = f"_action_denoise_adarms_scale_weight_s{step}_l{layer_idx}"
                shift_name = f"_action_denoise_adarms_shift_s{step}_l{layer_idx}"
                gate_name = f"_action_denoise_adarms_gate_s{step}_l{layer_idx}"
                self._set_denoise_lookup_buffer(
                    scale_name,
                    scale_weight_table[step, layer_idx].contiguous(),
                )
                self._set_denoise_lookup_buffer(
                    gate_name,
                    gate_table[step, layer_idx].contiguous(),
                )
                scale_buffer = getattr(self, scale_name)
                gate_buffer = getattr(self, gate_name)
                if self._adarms_shift_is_folded(layer_idx, num_layernorms):
                    # shift 只保留在独立整表中供一次性 bias 生成，不进入编译热路径。
                    layer_modulations.append((scale_buffer, gate_buffer))
                else:
                    self._set_denoise_lookup_buffer(
                        shift_name,
                        shift_table[step, layer_idx].contiguous(),
                    )
                    layer_modulations.append((scale_buffer, getattr(self, shift_name), gate_buffer))
            step_modulations.append(tuple(layer_modulations))
        self._action_denoise_adarms_modulation_steps = tuple(step_modulations)

    def _rebuild_adarms_modulation_step_refs(self) -> None:
        """根据已注册 buffer 重建 Python tuple 索引，保持设备迁移后引用有效。"""
        scale_table = self._buffers.get("_action_denoise_adarms_scale_weight_table")
        if scale_table is None or scale_table.numel() == 0:
            self._action_denoise_adarms_modulation_steps = ()
            return
        step_modulations: list[tuple[AdaRMSModulation, ...]] = []
        num_steps, num_layernorms = scale_table.shape[:2]
        for step in range(num_steps):
            layer_modulations: list[AdaRMSModulation] = []
            for layer_idx in range(num_layernorms):
                scale_name = f"_action_denoise_adarms_scale_weight_s{step}_l{layer_idx}"
                shift_name = f"_action_denoise_adarms_shift_s{step}_l{layer_idx}"
                gate_name = f"_action_denoise_adarms_gate_s{step}_l{layer_idx}"
                if scale_name not in self._buffers or gate_name not in self._buffers:
                    self._action_denoise_adarms_modulation_steps = ()
                    return
                scale_buffer = getattr(self, scale_name)
                gate_buffer = getattr(self, gate_name)
                if self._adarms_shift_is_folded(layer_idx, num_layernorms):
                    layer_modulations.append((scale_buffer, gate_buffer))
                else:
                    if shift_name not in self._buffers:
                        self._action_denoise_adarms_modulation_steps = ()
                        return
                    layer_modulations.append((scale_buffer, getattr(self, shift_name), gate_buffer))
            step_modulations.append(tuple(layer_modulations))
        self._action_denoise_adarms_modulation_steps = tuple(step_modulations)

    def _clear_adarms_bias_step_buffers(self, *, clear_tables: bool = True) -> None:
        """清理旧的 step bias 绑定；关闭 folding 时同时清空整表。"""
        self._remove_denoise_lookup_buffers(
            (
                "_action_denoise_qkv_bias_s",
                "_action_denoise_mlp_gate_bias_s",
                "_action_denoise_mlp_up_bias_s",
                "_action_denoise_action_out_bias_s",
            )
        )
        self._action_denoise_adarms_bias_steps = ()
        if clear_tables:
            for name in (
                "_action_denoise_qkv_bias_table",
                "_action_denoise_mlp_gate_bias_table",
                "_action_denoise_mlp_up_bias_table",
                "_action_denoise_action_out_bias_table",
            ):
                current = getattr(self, name)
                setattr(self, name, current.new_empty((0,)))

    def _refresh_adarms_bias_step_buffers(
        self,
        qkv_bias_table: torch.Tensor,
        mlp_gate_bias_table: torch.Tensor,
        mlp_up_bias_table: torch.Tensor,
        action_out_bias_table: torch.Tensor,
    ) -> None:
        """将每一步的固定 bias 注册为查表 buffer，并作为 Linear 参数传入。"""
        self._clear_adarms_bias_step_buffers(clear_tables=False)
        bias_steps = []
        use_qkv = self._adarms_bias_fusion_stage in {"qkv", "qkv_mlp", "all"}
        use_mlp = self._adarms_bias_fusion_stage in {"qkv_mlp", "mlp_action_out", "all"}
        use_action_out = self._adarms_bias_fusion_stage in {"mlp_action_out", "all"}
        num_steps, num_layers = qkv_bias_table.shape[:2]
        for step in range(num_steps):
            layer_biases = []
            for layer_idx in range(num_layers):
                names_and_values = (
                    (f"_action_denoise_qkv_bias_s{step}_l{layer_idx}", qkv_bias_table[step, layer_idx]),
                    (
                        f"_action_denoise_mlp_gate_bias_s{step}_l{layer_idx}",
                        mlp_gate_bias_table[step, layer_idx],
                    ),
                    (
                        f"_action_denoise_mlp_up_bias_s{step}_l{layer_idx}",
                        mlp_up_bias_table[step, layer_idx],
                    ),
                )
                for name, value in names_and_values:
                    # 这里只缓存查表结果，不标记静态地址；denoise 展开图
                    # 仍通过普通 bias 参数把当前 step 的 effective bias 传给 Linear。
                    self._set_denoise_lookup_buffer(name, value.contiguous())
                layer_biases.append(
                    (
                        getattr(self, names_and_values[0][0]) if use_qkv else None,
                        getattr(self, names_and_values[1][0]) if use_mlp else None,
                        getattr(self, names_and_values[2][0]) if use_mlp else None,
                    )
                )
            final_name = f"_action_denoise_action_out_bias_s{step}"
            self._set_denoise_lookup_buffer(
                final_name,
                action_out_bias_table[step].contiguous(),
            )
            bias_steps.append((tuple(layer_biases), getattr(self, final_name) if use_action_out else None))
        self._action_denoise_adarms_bias_steps = tuple(bias_steps)

    def _rebuild_adarms_bias_step_refs(self) -> None:
        qkv_bias_table = self._buffers.get("_action_denoise_qkv_bias_table")
        if qkv_bias_table is None or qkv_bias_table.numel() == 0:
            self._action_denoise_adarms_bias_steps = ()
            return

        bias_steps = []
        use_qkv = self._adarms_bias_fusion_stage in {"qkv", "qkv_mlp", "all"}
        use_mlp = self._adarms_bias_fusion_stage in {"qkv_mlp", "mlp_action_out", "all"}
        use_action_out = self._adarms_bias_fusion_stage in {"mlp_action_out", "all"}
        num_steps, num_layers = qkv_bias_table.shape[:2]
        for step in range(num_steps):
            layer_biases = []
            for layer_idx in range(num_layers):
                names = (
                    f"_action_denoise_qkv_bias_s{step}_l{layer_idx}",
                    f"_action_denoise_mlp_gate_bias_s{step}_l{layer_idx}",
                    f"_action_denoise_mlp_up_bias_s{step}_l{layer_idx}",
                )
                if any(name not in self._buffers for name in names):
                    self._action_denoise_adarms_bias_steps = ()
                    return
                layer_biases.append(
                    (
                        getattr(self, names[0]) if use_qkv else None,
                        getattr(self, names[1]) if use_mlp else None,
                        getattr(self, names[2]) if use_mlp else None,
                    )
                )
            final_name = f"_action_denoise_action_out_bias_s{step}"
            if final_name not in self._buffers:
                self._action_denoise_adarms_bias_steps = ()
                return
            bias_steps.append((tuple(layer_biases), getattr(self, final_name) if use_action_out else None))
        self._action_denoise_adarms_bias_steps = tuple(bias_steps)

    @torch.no_grad()
    def _refresh_adarms_bias_tables(self) -> None:
        """将固定 AdaRMS shift 折入对应下游 BF16 Linear 的 bias。"""
        if not self.paligemma_with_expert._qkv_weights_fused:
            raise RuntimeError("AdaRMS bias fusion requires fused QKV weights")
        incompatible_targets = self._adarms_bias_fusion_incompatible_targets(self._adarms_bias_fusion_stage)
        if incompatible_targets:
            raise RuntimeError(
                "AdaRMS bias fusion found incompatible DiT projections after preparation: "
                + ", ".join(incompatible_targets)
            )

        shift_vectors = self._action_denoise_adarms_shift_table[:, :, 0, 0, :]
        qkv_bias_tables = []
        mlp_gate_bias_tables = []
        mlp_up_bias_tables = []
        for layer_idx, layer in enumerate(self.paligemma_with_expert.gemma_expert.model.layers):
            input_shift = shift_vectors[:, layer_idx * 2]
            post_shift = shift_vectors[:, layer_idx * 2 + 1]
            qkv_bias_tables.append(F.linear(input_shift, layer.self_attn.qkv.weight, layer.self_attn.qkv.bias))
            mlp_gate_bias_tables.append(F.linear(post_shift, layer.mlp.gate_proj.weight, layer.mlp.gate_proj.bias))
            mlp_up_bias_tables.append(F.linear(post_shift, layer.mlp.up_proj.weight, layer.mlp.up_proj.bias))

        # Ascend MatMulV2 消费 FP32 bias。先用 BF16 计算以保持原折叠数值，
        # 再统一提升固定表为 FP32，避免 GE 为每次 Linear 插入 BF16->FP32 Cast。
        qkv_bias_table = torch.stack(qkv_bias_tables, dim=1).to(dtype=torch.float32).contiguous()
        mlp_gate_bias_table = torch.stack(mlp_gate_bias_tables, dim=1).to(dtype=torch.float32).contiguous()
        mlp_up_bias_table = torch.stack(mlp_up_bias_tables, dim=1).to(dtype=torch.float32).contiguous()
        action_out_bias_table = (
            F.linear(
                shift_vectors[:, -1],
                self.action_out_proj.weight,
                self.action_out_proj.bias,
            )
            .to(dtype=torch.float32)
            .contiguous()
        )

        self._action_denoise_qkv_bias_table = qkv_bias_table
        self._action_denoise_mlp_gate_bias_table = mlp_gate_bias_table
        self._action_denoise_mlp_up_bias_table = mlp_up_bias_table
        self._action_denoise_action_out_bias_table = action_out_bias_table
        self._refresh_adarms_bias_step_buffers(
            qkv_bias_table,
            mlp_gate_bias_table,
            mlp_up_bias_table,
            action_out_bias_table,
        )

    @torch.no_grad()
    def _refresh_fixed_denoise_lookup_tables(self) -> None:
        """按当前求解器预计算固定 denoise 的 timestep、AdaRMS cond 和调制查表。"""
        device = _module_device(self.time_mlp_in)
        target_dtype = self._action_compute_dtype(device)
        denoise_steps = self._graph_denoise_steps()
        timestep_table_f32 = (
            1.0
            - torch.arange(
                denoise_steps,
                dtype=torch.float32,
                device=device,
            )
            / denoise_steps
        )
        time_embedding_f32 = create_sinusoidal_pos_embedding(
            timestep_table_f32,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=device,
        )
        time_embedding = time_embedding_f32.to(dtype=linear_weight_dtype(self.time_mlp_in))
        adarms_cond = self.time_mlp_in(time_embedding)
        adarms_cond = F.silu(adarms_cond)
        adarms_cond = self.time_mlp_out(adarms_cond)
        adarms_cond = F.silu(adarms_cond).to(dtype=target_dtype).contiguous()

        scale_weight_tables: list[torch.Tensor] = []
        shift_tables: list[torch.Tensor] = []
        gate_tables: list[torch.Tensor] = []
        for layernorm in self._action_expert_adarms_layernorms():
            modulation = layernorm.dense(adarms_cond)
            scale, shift, gate = modulation.reshape(denoise_steps, 3, -1).unbind(dim=1)
            # 提前按后续计算整理形状，减少 denoise 热路径中的重排：
            # scale 存为 RMSNorm 直接消费的一维 1+scale，shift/gate 存为广播形状。
            scale_weight_tables.append((1 + scale).to(dtype=target_dtype).contiguous())
            shift_tables.append(shift[:, None, None, :].to(dtype=target_dtype).contiguous())
            gate_tables.append(gate[:, None, None, :].to(dtype=target_dtype).contiguous())
        # 调制量拆成三张表，并额外注册 per-step/per-layer buffer。
        # 编译图热路径直接消费 buffer tuple，避免 Tensor 链式索引产生大量 GatherV2。
        adarms_scale_weight_table = torch.stack(scale_weight_tables, dim=1).contiguous()
        adarms_shift_table = torch.stack(shift_tables, dim=1).contiguous()
        adarms_gate_table = torch.stack(gate_tables, dim=1).contiguous()

        self._action_denoise_dt = torch.tensor(
            -1.0 / denoise_steps,
            dtype=target_dtype,
            device=device,
        )
        self._action_denoise_timestep_table = timestep_table_f32.to(dtype=target_dtype)
        self._action_denoise_suffix_position_ids = torch.arange(
            self.config.chunk_size,
            dtype=torch.int64,
            device=device,
        )
        self._action_denoise_adarms_cond_table = adarms_cond
        self._action_denoise_adarms_scale_weight_table = adarms_scale_weight_table
        self._action_denoise_adarms_shift_table = adarms_shift_table
        self._action_denoise_adarms_gate_table = adarms_gate_table
        self._refresh_adarms_modulation_step_buffers(
            adarms_scale_weight_table,
            adarms_shift_table,
            adarms_gate_table,
        )

    def _force_eager_attention_for_graph_compile(self) -> None:
        """TorchAir 捕获前强制使用图安全 eager/bmm attention 实现。"""
        try:
            from transformers import modeling_utils
        except Exception:
            attention_functions = None
        else:
            attention_functions = getattr(modeling_utils, "ALL_ATTENTION_FUNCTIONS", None)
        if attention_functions is not None:
            attention_functions.register("eager_bmm", _npu_graph_safe_gemma_attention_forward)
        gemma_attention_functions = getattr(modeling_gemma, "ALL_ATTENTION_FUNCTIONS", None)
        if gemma_attention_functions is not None:
            gemma_attention_functions.register("eager_bmm", _npu_graph_safe_gemma_attention_forward)

        modules = [
            self.paligemma_with_expert.paligemma,
            self.paligemma_with_expert.paligemma.model.vision_tower,
        ]
        for module in modules:
            config = getattr(module, "config", None)
            if config is not None and hasattr(config, "_attn_implementation"):
                config._attn_implementation = "eager"

        gemma_modules = [
            self.paligemma_with_expert.paligemma.model.language_model,
            self.paligemma_with_expert.gemma_expert.model,
        ]
        for module in gemma_modules:
            config = getattr(module, "config", None)
            if config is not None and hasattr(config, "_attn_implementation"):
                config._attn_implementation = "eager_bmm"

    def _patch_linear_for_npu_graph_compile(self) -> None:
        """把所有 Linear forward 替换成 NPU 图安全版本。"""
        fp32_bias_buffers = 0
        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue

            if module.bias is not None:
                bias_fp32 = module.bias.detach().to(dtype=torch.float32).contiguous()
                if "_pi05_graph_bias_fp32" in module._buffers:
                    module._pi05_graph_bias_fp32 = bias_fp32
                else:
                    module.register_buffer(
                        "_pi05_graph_bias_fp32",
                        bias_fp32,
                        persistent=False,
                    )
                fp32_bias_buffers += 1

            if not getattr(module, "_pi05_graph_safe_linear", False):
                module.forward = types.MethodType(_npu_graph_safe_linear_forward, module)
                module._pi05_graph_safe_linear = True
        self._npu_graph_fp32_linear_bias_count = fp32_bias_buffers

    def _patch_gemma_attention_for_npu_graph_compile(self) -> None:
        """把 GemmaAttention forward 替换为 bmm 版本以便 NPU 图编译。"""
        gemma_attention_cls = getattr(modeling_gemma, "GemmaAttention", None)
        if gemma_attention_cls is None:
            return
        for module in self.modules():
            if isinstance(module, gemma_attention_cls) and not getattr(module, "_pi05_graph_safe_attention", False):
                module.forward = types.MethodType(_npu_graph_safe_gemma_attention_module_forward, module)
                module._pi05_graph_safe_attention = True

    def _patch_siglip_attention_for_npu_graph_compile(self) -> None:
        """把 SigLIP vision attention forward 替换为图安全 bmm 版本。"""
        try:
            from transformers.models.siglip.modeling_siglip import SiglipAttention
        except Exception:
            return
        for module in self.modules():
            if isinstance(module, SiglipAttention) and not getattr(module, "_pi05_graph_safe_attention", False):
                module.forward = types.MethodType(_npu_graph_safe_siglip_attention_module_forward, module)
                module._pi05_graph_safe_attention = True

    def _patch_gemma_rotary_for_npu_graph_compile(self) -> None:
        """把 Gemma RotaryEmbedding forward 替换为显式 RoPE 实现。"""
        rotary_cls = getattr(modeling_gemma, "GemmaRotaryEmbedding", None)
        if rotary_cls is None:
            return
        for module in self.modules():
            if isinstance(module, rotary_cls) and not getattr(module, "_pi05_graph_safe_rotary", False):
                module.forward = types.MethodType(_npu_graph_safe_gemma_rotary_forward, module)
                module._pi05_graph_safe_rotary = True

    def enable_sample_actions_graph_compile(
        self,
        *,
        backend=None,
        fullgraph: bool | None = None,
        dynamic: bool | None = None,
        frozen_parameter: bool | None = None,
        tiling_schedule_optimize: bool | None = None,
    ) -> dict[str, Any]:
        """编译 PI0.5 推理的 prefix 图和当前求解器对应的固定 denoise 图。"""
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch build")
        self._force_eager_attention_for_graph_compile()
        if next(self.parameters()).device.type == "npu":
            self._patch_linear_for_npu_graph_compile()
            self._patch_gemma_attention_for_npu_graph_compile()
            self._patch_siglip_attention_for_npu_graph_compile()
            self._patch_gemma_rotary_for_npu_graph_compile()
        if (
            self._action_denoise_adarms_scale_weight_table.numel() == 0
            or not self._action_denoise_adarms_modulation_steps
        ):
            self._refresh_fixed_denoise_lookup_tables()

        fullgraph = self.config.compile_inference_fullgraph if fullgraph is None else fullgraph
        dynamic = self.config.compile_inference_dynamic if dynamic is None else dynamic
        frozen_parameter = self.config.compile_frozen_parameter if frozen_parameter is None else frozen_parameter
        tiling_schedule_optimize = (
            self.config.compile_tiling_schedule_optimize
            if tiling_schedule_optimize is None
            else tiling_schedule_optimize
        )
        compile_kwargs, backend_name = self._build_inference_compile_kwargs(
            backend=backend,
            fullgraph=fullgraph,
            dynamic=dynamic,
            frozen_parameter=frozen_parameter,
            tiling_schedule_optimize=tiling_schedule_optimize,
        )

        # prefix 与 denoise 拆成两张图；AB2/6 是独立的完整优化图，
        # 与 Euler/10 共享 QKV/PFA、FIAS、静态 norm 和 AdaRMS 查表等准备步骤。
        self._compiled_action_prefix_forward = torch.compile(
            self._action_prefix_forward_for_compile,
            **compile_kwargs,
        )
        denoise_solver = self._denoise_solver()
        if denoise_solver == "ab2":
            denoise_target = self._action_denoise_ab2_6_steps_for_compile
            denoise_target_name = "_action_denoise_ab2_6_steps_for_compile"
            self._compiled_action_denoise_ab2_6_steps = torch.compile(
                denoise_target,
                **compile_kwargs,
            )
        else:
            denoise_target = self._action_denoise_10_steps_for_compile
            denoise_target_name = "_action_denoise_10_steps_for_compile"
            self._compiled_action_denoise_10_steps = torch.compile(
                denoise_target,
                **compile_kwargs,
            )
        denoise_steps = self._graph_denoise_steps()
        self._sample_actions_graph_compile_enabled = True
        self._sample_actions_graph_compile_info = {
            "enabled": True,
            "backend": backend_name,
            "dynamic": dynamic,
            "fullgraph": fullgraph,
            "graph_path": "split_prefix_denoise",
            "prefix_compiled": True,
            "denoise_solver": denoise_solver,
            "denoise_steps": denoise_steps,
            "denoise_compilation": "single_graph",
            "lossy_solver": denoise_solver == "ab2",
            "frozen_parameter": frozen_parameter,
            "tiling_schedule_optimize": tiling_schedule_optimize,
            "linear_bias_cast_elimination": True,
            "fp32_linear_bias_buffers": getattr(self, "_npu_graph_fp32_linear_bias_count", 0),
            "targets": ["_action_prefix_forward_for_compile", denoise_target_name],
        }
        return dict(self._sample_actions_graph_compile_info)

    def _build_inference_compile_kwargs(
        self,
        *,
        backend=None,
        fullgraph: bool | None = None,
        dynamic: bool | None = None,
        frozen_parameter: bool | None = None,
        tiling_schedule_optimize: bool | None = None,
    ) -> tuple[dict[str, Any], str]:
        """根据配置生成 torch.compile 参数，并选择 TorchAir/npugraph/inductor 后端。"""
        fullgraph = self.config.compile_inference_fullgraph if fullgraph is None else fullgraph
        dynamic = self.config.compile_inference_dynamic if dynamic is None else dynamic
        frozen_parameter = self.config.compile_frozen_parameter if frozen_parameter is None else frozen_parameter
        tiling_schedule_optimize = (
            self.config.compile_tiling_schedule_optimize
            if tiling_schedule_optimize is None
            else tiling_schedule_optimize
        )

        compile_kwargs: dict[str, Any] = {"dynamic": dynamic, "fullgraph": fullgraph}
        backend_name = self.config.compile_inference_backend or "auto"
        if backend is None:
            device_type = next(self.parameters()).device.type
            if backend_name in {None, "auto"}:
                if device_type == "npu":
                    backend = self._build_torchair_backend(
                        frozen_parameter=frozen_parameter,
                        tiling_schedule_optimize=tiling_schedule_optimize,
                    )
                    backend_name = "torchair"
                else:
                    compile_kwargs["mode"] = self.config.compile_mode
                    backend_name = "inductor"
            elif backend_name == "torchair":
                backend = self._build_torchair_backend(
                    frozen_parameter=frozen_parameter,
                    tiling_schedule_optimize=tiling_schedule_optimize,
                )
            elif backend_name == "npugraph_ex":
                backend = self._build_npugraph_ex_backend()
            elif backend_name == "inductor":
                compile_kwargs["mode"] = self.config.compile_mode
            else:
                backend = backend_name
        else:
            backend_name = type(backend).__name__

        if backend is not None:
            compile_kwargs["backend"] = backend
        return compile_kwargs, backend_name

    def _build_npugraph_ex_backend(self):
        """构造 npugraph_ex 后端对象，用于可选 NPU 图编译实验路径。"""
        try:
            import torch_npu  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError("npugraph_ex backend requires torch_npu") from exc

        try:
            import npugraph_ex  # type: ignore
        except ModuleNotFoundError:
            try:
                from torch_npu.dynamo import npugraph_ex  # type: ignore
            except Exception as exc:
                raise RuntimeError("npugraph_ex backend requires torch_npu.dynamo.npugraph_ex") from exc
            sys.modules.setdefault("npugraph_ex", npugraph_ex)

        config_cls = getattr(npugraph_ex, "CompilerConfig", None)
        if config_cls is None:
            from npugraph_ex.configs.compiler_config import CompilerConfig  # type: ignore

            config_cls = CompilerConfig

        compiler_config = config_cls()
        if hasattr(compiler_config, "mode"):
            compiler_config.mode = "npugraph_ex"
        return npugraph_ex.get_npu_backend(compiler_config=compiler_config)

    def _build_torchair_backend(
        self,
        *,
        frozen_parameter: bool = True,
        tiling_schedule_optimize: bool = True,
    ):
        """构造 TorchAir 后端，并配置 frozen_parameter 与 tiling 调度优化。"""
        try:
            import torch_npu  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError("TorchAir backend requires torch_npu") from exc

        try:
            import torchair  # type: ignore
        except ModuleNotFoundError:
            try:
                from torch_npu.dynamo import torchair  # type: ignore
            except Exception as exc:
                raise RuntimeError("TorchAir backend requires torchair") from exc
            sys.modules.setdefault("torchair", torchair)

        config_cls = getattr(torchair, "CompilerConfig", None)
        if config_cls is None:
            from torchair.configs.compiler_config import CompilerConfig  # type: ignore

            config_cls = CompilerConfig

        compiler_config = config_cls()
        experimental_config = getattr(compiler_config, "experimental_config", None)
        if experimental_config is not None:
            if hasattr(experimental_config, "frozen_parameter"):
                experimental_config.frozen_parameter = frozen_parameter
            if hasattr(experimental_config, "tiling_schedule_optimize"):
                experimental_config.tiling_schedule_optimize = tiling_schedule_optimize
        return torchair.get_npu_backend(compiler_config=compiler_config)

    def _can_use_compiled_action_inference(self, rtc_kwargs: dict[str, Any], *, num_steps: int) -> bool:
        """判断本次 sample_actions 是否可走已编译 prefix/denoise 图。"""
        return (
            int(num_steps) == self._graph_denoise_steps()
            and self._sample_actions_graph_compile_enabled
            and self._compiled_action_prefix_forward is not None
            and self._compiled_action_denoise_for_solver() is not None
            and self.rtc_processor is None
            and not rtc_kwargs
        )

    def _make_action_graph_att_2d_masks(self, pad_masks, att_masks):
        """图编译路径下构造 action/prefix 使用的二维 attention mask。"""
        cumsum = torch.cumsum(att_masks, dim=1)
        att_2d_masks = cumsum.unsqueeze(1) <= cumsum.unsqueeze(2)
        pad_2d_masks = pad_masks.unsqueeze(1) & pad_masks.unsqueeze(2)
        return att_2d_masks & pad_2d_masks

    def gradient_checkpointing_enable(self):
        """训练时开启 gradient checkpointing，降低显存/显存等价内存占用。"""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05Pytorch model")

    def gradient_checkpointing_disable(self):
        """关闭 gradient checkpointing，恢复普通前向。"""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05Pytorch model")

    def _rtc_enabled(self):
        """判断当前配置是否启用 RTC denoise 调度。"""
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """训练时按需对给定子函数应用 gradient checkpointing。"""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """把二维可见性 mask 转为 transformer attention 接收的四维 additive mask。"""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device):
        """推理/训练时按动作 chunk 形状采样初始高斯噪声。"""
        target_dtype = self._action_compute_dtype(device)
        normal_dtype = target_dtype
        if torch.device(device).type == "npu" and target_dtype == torch.bfloat16:
            normal_dtype = torch.float32
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=normal_dtype,
            device=device,
        )
        if noise.dtype != target_dtype:
            return noise.to(dtype=target_dtype)
        return noise

    def sample_time(self, bsize, device):
        """训练时为 batch 采样 flow matching 时间标量。"""
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(self, images, img_masks, tokens, masks) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """构造 prefix embedding：多相机图像 embedding 拼接语言 token embedding。"""
        embs = []
        pad_masks = []

        # 与 pi0 推理保持一致，把多个相机视角合并成一次
        # vision forward，避免每个视角单独调用 SigLIP 造成重复调度开销。
        def image_embed_func(stacked_images):
            """把多视角图像展平后一次送入视觉塔，再恢复 batch 维度。"""
            bsize, num_views = stacked_images.shape[:2]
            flat_images = stacked_images.reshape(bsize * num_views, *stacked_images.shape[2:])
            flat_img_emb = self.paligemma_with_expert.embed_image(flat_images)
            num_img_embs, hidden_dim = flat_img_emb.shape[1:]
            return flat_img_emb.reshape(bsize, num_views * num_img_embs, hidden_dim)

        stacked_images = torch.stack(images, dim=1)
        img_emb = self._apply_checkpoint(image_embed_func, stacked_images)
        bsize = img_emb.shape[0]
        num_img_embs = img_emb.shape[1] // len(images)

        embs.append(img_emb)
        stacked_img_masks = torch.stack(img_masks, dim=1)
        pad_masks.append(stacked_img_masks[:, :, None].expand(bsize, len(images), num_img_embs).reshape(bsize, -1))

        # Process language tokens
        def lang_embed_func(tokens):
            """执行语言 token embedding，用于和图像 token 拼成 prefix。"""
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            return lang_emb

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)

        bsize = pad_masks.shape[0]
        # prefix token 全部属于同一个 attention block，直接在设备侧
        # 构造全 0 mask，避免 TorchAir 捕获到 Python list -> NPU tensor 拷贝。
        att_masks = torch.zeros(bsize, pad_masks.shape[1], dtype=torch.bool, device=pad_masks.device)

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """构造 denoise suffix embedding，并生成当前 timestep 的 AdaRMS 条件向量。"""
        embs = []
        pad_masks = []

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        # sin-cos 表在 fp32 下构造；eager denoise 路径里 time MLP 可能已被
        # _prepare_action_compute_dtype 调成 bf16（图编译路径走 AdaRMS 查表不经过此处），
        # 输入必须对齐权重 dtype，否则 F.linear 报 dtype 不匹配。
        time_emb = time_emb.to(dtype=linear_weight_dtype(self.time_mlp_in))

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            """把 noisy action 投影到 action expert hidden 维度。"""
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb):
            """把 sin-cos timestep embedding 转换为 AdaRMS 条件向量。"""
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        # 第一个 action token 开启新 attention block，后续 action token
        # 共享 causal block；全程在设备侧构造，便于 TorchAir 捕获。
        first_action_att_mask = torch.ones(bsize, 1, dtype=embs.dtype, device=embs.device)
        remaining_action_att_mask = torch.zeros(
            bsize,
            action_time_dim - 1,
            dtype=embs.dtype,
            device=embs.device,
        )
        att_masks = torch.cat([first_action_att_mask, remaining_action_att_mask], dim=1)

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, images, img_masks, tokens, masks, actions, noise, time) -> Tensor:
        """训练前向：构造 noisy action，预测 velocity，并返回逐元素 MSE loss。"""
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            linear_weight_dtype(self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj)
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            """训练路径联合运行 prefix+suffix transformer，取 suffix 输出。"""
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            """把 action expert hidden 输出投影回动作 velocity 维度。"""
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    def _action_prefix_embed_for_compile(self, images, img_masks, tokens, masks):
        """TorchAir prefix 图入口：只负责生成 prefix embedding 和 mask。"""
        # prefix 子图第一部分，包含视觉塔、多模态投影和语言 token embedding。
        return self.embed_prefix(images, img_masks, tokens, masks)

    def _action_prefix_masks_for_compile(self, prefix_pad_masks, prefix_att_masks):
        """TorchAir prefix 图内根据 pad/att mask 生成二维 mask 和 position ids。"""
        # 使用和 pi0 一致的 mask 构造函数，避免图内 Python list/tensor 构造。
        prefix_att_2d_masks = self._make_action_graph_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        return prefix_att_2d_masks, prefix_position_ids

    def _action_prefix_prefill_for_compile(
        self,
        prefix_embs,
        prefix_att_2d_masks_4d,
        prefix_position_ids,
    ):
        """TorchAir prefix 图内执行 prefill，并产出 denoise 复用的 KV cache。"""
        # prefix 子图末尾生成 KV cache，供 denoise 10 步图只读复用。
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return past_key_values

    def _action_prefix_forward_for_compile(self, images, img_masks, tokens, masks):
        """完整 prefix 编译图：embedding、mask、position 和 prefill/KV cache。"""
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._action_prefix_embed_for_compile(
            images, img_masks, tokens, masks
        )
        prefix_att_2d_masks, prefix_position_ids = self._action_prefix_masks_for_compile(
            prefix_pad_masks, prefix_att_masks
        )
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        past_key_values = self._action_prefix_prefill_for_compile(
            prefix_embs,
            prefix_att_2d_masks_4d,
            prefix_position_ids,
        )
        return prefix_pad_masks, past_key_values

    def _embed_suffix_embs_with_adarms_cond(self, noisy_actions, adarms_cond):
        """使用预计算 AdaRMS cond 构造单个 denoise step 的 action suffix embedding。"""
        noisy_actions = noisy_actions.to(dtype=linear_weight_dtype(self.action_in_proj))

        def action_proj_func(noisy_actions):
            """编译 denoise 路径中仅保留 action 投影，不重复计算时间 MLP。"""
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)
        bsize = action_emb.shape[0]
        device = action_emb.device

        if adarms_cond.ndim == 1:
            adarms_cond = adarms_cond[None, :].expand(bsize, -1)
        adarms_cond = adarms_cond.to(dtype=action_emb.dtype, device=device)
        return action_emb, adarms_cond

    def embed_suffix_with_adarms_cond(self, noisy_actions, adarms_cond):
        """构造使用预计算 AdaRMS condition 的完整 action suffix 上下文。"""
        # 固定图热路径只需要 embedding/condition，因此底层实现已拆成
        # `_embed_suffix_embs_with_adarms_cond`。兼容 eager helper 仍需要 pad/attention
        # mask；这里复用同一投影与 dtype 逻辑，再补齐与 embed_suffix 完全一致的 mask。
        suffix_embs, adarms_cond = self._embed_suffix_embs_with_adarms_cond(
            noisy_actions,
            adarms_cond,
        )
        bsize, suffix_len = suffix_embs.shape[:2]
        device = suffix_embs.device
        suffix_pad_masks = torch.ones(bsize, suffix_len, dtype=torch.bool, device=device)
        first_action_att_mask = torch.ones(bsize, 1, dtype=suffix_embs.dtype, device=device)
        remaining_action_att_mask = torch.zeros(
            bsize,
            suffix_len - 1,
            dtype=suffix_embs.dtype,
            device=device,
        )
        suffix_att_masks = torch.cat([first_action_att_mask, remaining_action_att_mask], dim=1)
        return suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond

    def _denoise_step_from_suffix_embs(
        self,
        prefix_pad_masks,
        past_key_values,
        suffix_embs,
        suffix_pad_masks,
        suffix_att_masks,
        adarms_cond,
        adarms_modulation=None,
        adarms_bias_plan=None,
    ):
        """eager denoise step：根据 suffix embedding 构造 mask/position 并执行 expert。"""
        # eager 路径先按当前 suffix 形状现算 mask，再交给统一的
        # denoise forward helper，避免和编译路径重复一套 forward 代码。
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        attn_implementation = (
            "eager_bmm" if self._sample_actions_graph_compile_enabled and suffix_embs.device.type == "npu" else "eager"
        )
        return self._run_action_denoise_forward(
            past_key_values=past_key_values,
            suffix_embs=suffix_embs,
            adarms_cond=adarms_cond,
            adarms_modulation=adarms_modulation,
            adarms_bias_plan=adarms_bias_plan,
            full_att_2d_masks_4d=full_att_2d_masks_4d,
            position_ids=position_ids,
            attn_implementation=attn_implementation,
        )

    def denoise_step_with_adarms_cond(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        adarms_cond,
        adarms_modulation=None,
        adarms_bias_plan=None,
    ):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix_with_adarms_cond(
            x_t,
            adarms_cond,
        )
        return self._denoise_step_from_suffix_embs(
            prefix_pad_masks,
            past_key_values,
            suffix_embs,
            suffix_pad_masks,
            suffix_att_masks,
            adarms_cond,
            adarms_modulation,
            adarms_bias_plan,
        )

    def _run_action_denoise_forward(
        self,
        past_key_values,
        suffix_embs,
        adarms_cond,
        adarms_modulation,
        adarms_bias_plan,
        full_att_2d_masks_4d,
        position_ids,
        attn_implementation,
        prepared_npu_attention_mask=None,
        prepared_npu_rotary_cache=None,
    ):
        """运行 action expert denoise forward，并把 suffix hidden 投影成 velocity。"""
        # 统一的 denoise 前向执行入口，专门承接 eager/graph 两条路径
        # 共同的 cache clone、forward 和 action head 投影，减少重复代码。
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = (  # noqa: SLF001
            attn_implementation
        )
        past_key_values = clone_past_key_values(past_key_values)
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
            adarms_modulations=[None, adarms_modulation],
            adarms_bias_plans=[None, adarms_bias_plan],
            prepared_npu_attention_mask=prepared_npu_attention_mask,
            prepared_npu_rotary_cache=prepared_npu_rotary_cache,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=linear_weight_dtype(self.action_out_proj))
        action_out_bias = adarms_bias_plan[1] if adarms_bias_plan is not None else None
        if action_out_bias is None:
            return self.action_out_proj(suffix_out)
        return _npu_graph_safe_linear_with_bias(self.action_out_proj, suffix_out, action_out_bias)

    def _prepare_denoise_mask_context_for_compile(self, prefix_pad_masks, x_t):
        """为固定 10 步 denoise 编译图预先构造不随 step 变化的 mask/position。"""
        # 固定 10 步 denoise 中 prefix/suffix mask 与 position_ids
        # 不随 x_t 更新而变化；在循环外预先构造，避免每步重复 cat/cumsum/sum。
        batch_size, suffix_len = x_t.shape[:2]
        prefix_len = prefix_pad_masks.shape[1]
        device = x_t.device
        suffix_pad_masks = torch.ones(batch_size, suffix_len, dtype=torch.bool, device=device)
        first_action_att_mask = torch.ones(batch_size, 1, dtype=x_t.dtype, device=device)
        remaining_action_att_mask = torch.zeros(
            batch_size,
            suffix_len - 1,
            dtype=x_t.dtype,
            device=device,
        )
        suffix_att_masks = torch.cat([first_action_att_mask, remaining_action_att_mask], dim=1)

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        if self._action_denoise_suffix_position_ids.numel() != suffix_len:
            # 固定 10 步图默认复用预计算 [0..chunk_size-1] buffer；
            # 只有 suffix 长度与配置不一致时，才退回现场构造。
            suffix_position_ids = torch.arange(suffix_len, dtype=torch.int64, device=device)
        else:
            suffix_position_ids = self._action_denoise_suffix_position_ids
        position_ids = prefix_offsets + suffix_position_ids[None, :]
        return full_att_2d_masks_4d, position_ids

    def _denoise_step_with_prepared_mask_context(
        self,
        past_key_values,
        x_t,
        adarms_cond,
        adarms_modulation,
        adarms_bias_plan,
        full_att_2d_masks_4d,
        position_ids,
        prepared_npu_attention_mask,
        prepared_npu_rotary_cache,
    ):
        """编译 denoise 热路径：复用 mask/position，仅更新 x_t 和 AdaRMS step。"""
        # TorchAir denoise 图专用热路径。mask/position 已在 10 步循环外
        # 准备好，这里只保留随 step 变化的 action embedding 与 AdaRMS 表查找，
        # 避免每步重复构造 suffix mask。
        suffix_embs, adarms_cond = self._embed_suffix_embs_with_adarms_cond(x_t, adarms_cond)
        attn_implementation = "eager_bmm" if suffix_embs.device.type == "npu" else "eager"
        return self._run_action_denoise_forward(
            past_key_values=past_key_values,
            suffix_embs=suffix_embs,
            adarms_cond=adarms_cond,
            adarms_modulation=adarms_modulation,
            adarms_bias_plan=adarms_bias_plan,
            full_att_2d_masks_4d=full_att_2d_masks_4d,
            position_ids=position_ids,
            attn_implementation=attn_implementation,
            prepared_npu_attention_mask=prepared_npu_attention_mask,
            prepared_npu_rotary_cache=prepared_npu_rotary_cache,
        )

    def _action_denoise_10_steps_for_compile(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
    ):
        """固定展开 10 个 denoise step 的 TorchAir 编译图入口。"""
        # 固定 10 步 denoise 整体编译成一张图；每步 AdaRMS 条件
        # 和 scale/shift/gate 调制量来自预计算表，prefix KV cache 在 10 步内只读复用。
        dt_tensor = self._action_denoise_dt.to(dtype=x_t.dtype)
        adarms_cond_table = self._action_denoise_adarms_cond_table
        adarms_modulation_steps = self._action_denoise_adarms_modulation_steps
        adarms_bias_steps = self._action_denoise_adarms_bias_steps
        full_att_2d_masks_4d, position_ids = self._prepare_denoise_mask_context_for_compile(prefix_pad_masks, x_t)
        # 这两组张量仅由固定 mask、position_ids 和 head 配置决定，
        # 在 10 个 Euler step 外构造一次，循环内只读复用。
        prepared_npu_attention_mask = self.paligemma_with_expert._npu_attention_mask(full_att_2d_masks_4d)
        expert_first_attn = self.paligemma_with_expert.gemma_expert.model.layers[0].self_attn
        prepared_npu_rotary_cache = self.paligemma_with_expert._prepare_npu_rotary_cache(
            position_ids,
            expert_first_attn,
            self.paligemma_with_expert.gemma_expert.model.config.num_attention_heads,
            self.paligemma_with_expert.gemma_expert.model.config.num_key_value_heads,
        )
        for step in range(PI05_GRAPH_DENOISE_STEPS):
            v_t = self._denoise_step_with_prepared_mask_context(
                past_key_values=past_key_values,
                x_t=x_t,
                adarms_cond=adarms_cond_table[step],
                adarms_modulation=adarms_modulation_steps[step],
                adarms_bias_plan=adarms_bias_steps[step] if adarms_bias_steps else None,
                full_att_2d_masks_4d=full_att_2d_masks_4d,
                position_ids=position_ids,
                prepared_npu_attention_mask=prepared_npu_attention_mask,
                prepared_npu_rotary_cache=prepared_npu_rotary_cache,
            )
            x_t = x_t + dt_tensor * v_t
        return x_t

    def _action_denoise_ab2_6_steps_for_compile(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
    ):
        """固定展开六个 raw-velocity NFE 的 AB2 TorchAir 编译图入口。"""
        # AB2 首步按 Euler 启动，后续仅组合当前与上一轮的原始
        # velocity；不得把组合后的更新量写回历史，否则会改变积分器语义。
        dt_tensor = self._action_denoise_dt.to(dtype=x_t.dtype)
        adarms_cond_table = self._action_denoise_adarms_cond_table
        adarms_modulation_steps = self._action_denoise_adarms_modulation_steps
        adarms_bias_steps = self._action_denoise_adarms_bias_steps
        full_att_2d_masks_4d, position_ids = self._prepare_denoise_mask_context_for_compile(prefix_pad_masks, x_t)
        prepared_npu_attention_mask = self.paligemma_with_expert._npu_attention_mask(full_att_2d_masks_4d)
        expert_first_attn = self.paligemma_with_expert.gemma_expert.model.layers[0].self_attn
        prepared_npu_rotary_cache = self.paligemma_with_expert._prepare_npu_rotary_cache(
            position_ids,
            expert_first_attn,
            self.paligemma_with_expert.gemma_expert.model.config.num_attention_heads,
            self.paligemma_with_expert.gemma_expert.model.config.num_key_value_heads,
        )
        previous_raw_v_t = None
        for step in range(PI05_AB2_DENOISE_STEPS):
            v_t = self._denoise_step_with_prepared_mask_context(
                past_key_values=past_key_values,
                x_t=x_t,
                adarms_cond=adarms_cond_table[step],
                adarms_modulation=adarms_modulation_steps[step],
                adarms_bias_plan=adarms_bias_steps[step] if adarms_bias_steps else None,
                full_att_2d_masks_4d=full_att_2d_masks_4d,
                position_ids=position_ids,
                prepared_npu_attention_mask=prepared_npu_attention_mask,
                prepared_npu_rotary_cache=prepared_npu_rotary_cache,
            )
            x_t = _apply_ab2_denoise_update(x_t, v_t, previous_raw_v_t, dt_tensor)
            previous_raw_v_t = v_t
        return x_t

    def _sample_actions_graph_inference(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise,
    ):
        """已编译推理路径：prefix 图生成 cache，所选 denoise 图一次完成采样。"""
        x_t = self._cast_action_tensor(noise)
        compiled_denoise = self._compiled_action_denoise_for_solver()
        if self._compiled_action_prefix_forward is None or compiled_denoise is None:
            raise RuntimeError("graph inference requires compiled prefix and selected denoise callables")

        prefix_pad_masks, past_key_values = self._compiled_action_prefix_forward(images, img_masks, tokens, masks)
        x_t = compiled_denoise(prefix_pad_masks, past_key_values, x_t)
        return x_t

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """完整推理采样入口：从噪声开始迭代 denoise，输出一个 action chunk。"""
        num_steps = self._resolve_denoise_steps(num_steps)
        denoise_solver = self._denoise_solver()
        if denoise_solver == "ab2" and self._rtc_enabled():
            raise ValueError("AB2/6 inference does not support RTC denoise guidance")

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)
        elif noise.dtype.is_floating_point:
            noise = self._cast_action_tensor(noise)

        if self._can_use_compiled_action_inference(kwargs, num_steps=num_steps):
            # 已启用图编译且无 RTC 动态参数时，选择 Euler/10 或 AB2/6
            # 对应的 prefix + 固定 denoise 完整优化路径。
            return self._sample_actions_graph_inference(
                images,
                img_masks,
                tokens,
                masks,
                noise,
            ).to(dtype=torch.float32)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps

        x_t = noise
        previous_raw_v_t = None
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                """单步 denoise 闭包，供普通采样和 RTC 调度共同调用。"""
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            if denoise_solver == "ab2":
                x_t = _apply_ab2_denoise_update(x_t, v_t, previous_raw_v_t, dt)
                previous_raw_v_t = v_t
            else:
                x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """执行单个 denoise step：构造 suffix/AdaRMS 条件并预测 velocity。"""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)
        return self._denoise_step_from_suffix_embs(
            prefix_pad_masks,
            past_key_values,
            suffix_embs,
            suffix_pad_masks,
            suffix_att_masks,
            adarms_cond,
        )


class PI05Policy(PreTrainedPolicy):
    """LeRobot policy 封装层，负责 batch 预处理、action chunk 推理和训练 loss。"""

    config_class = PI05Config
    name = "pi05"

    def __init__(
        self,
        config: PI05Config,
        **kwargs,
    ):
        """初始化 policy、RTC processor、核心 PI05 模型和内部 action 队列。"""
        require_package("transformers", extra="pi")
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize the core PI05 model
        self.init_rtc_processor()
        self.model = PI05Pytorch(config, rtc_processor=self.rtc_processor)

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        self.reset()
        # PI0.5 推理默认使用 eval 模式，避免每次 predict_action_chunk
        # 都重复切换模块状态。
        self.eval()

    def prepare_inference_optimizations(
        self,
        *,
        enable_npu_fused_ops: bool = False,
        enable_graph_compile: bool | None = None,
        enable_qkv_fusion: bool | None = None,
        enable_mlp_fusion: bool = False,
        mlp_fusion_scope: Literal["all", "prefix"] = "all",
        enable_shared_prefix_fias: bool | None = None,
        enable_adarms_bias_fusion: bool | None = None,
        adarms_bias_fusion_stage: str | None = None,
    ) -> dict[str, Any]:
        """policy 层推理优化入口，转发到核心模型准备 QKV/PFA/TorchAir 优化。"""
        # policy 层提供统一入口，调用方可显式选择 eager QKV/PFA
        # 或 TorchAir prefix/denoise 双图编译路径。
        effective_graph_compile = (
            bool(getattr(self.config, "compile_inference_graph", False))
            if enable_graph_compile is None
            else bool(enable_graph_compile)
        )
        effective_qkv_fusion = (
            bool(enable_npu_fused_ops or effective_graph_compile)
            if enable_qkv_fusion is None
            else bool(enable_qkv_fusion)
        )
        if self.config.quantization is not None and (effective_qkv_fusion or enable_mlp_fusion):
            # 只允许 Selective-99 no-smooth INT8 进入 QKV 与 Gate/Up 融合。
            from .quantization import assert_qkv_fusion_supported

            assert_qkv_fusion_supported(self.config.quantization)
        self.eval()
        return self.model.prepare_inference_optimizations(
            enable_npu_fused_ops=enable_npu_fused_ops,
            enable_graph_compile=enable_graph_compile,
            enable_qkv_fusion=enable_qkv_fusion,
            enable_mlp_fusion=enable_mlp_fusion,
            mlp_fusion_scope=mlp_fusion_scope,
            enable_shared_prefix_fias=enable_shared_prefix_fias,
            enable_adarms_bias_fusion=enable_adarms_bias_fusion,
            adarms_bias_fusion_stage=adarms_bias_fusion_stage,
        )

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """加载 PI0.5 checkpoint，并处理 OpenPI 到 LeRobot 命名差异。"""
        print(
            "The PI05 model is a direct port of the OpenPI implementation. \n"
            "This implementation follows the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            from .provider import load_pi05_ascend_910b_config

            runtime_dtype = kwargs.pop("model_dtype", "bf16")
            config = load_pi05_ascend_910b_config(
                bundle_root=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                model_dtype=runtime_dtype,
            )

        # Initialize model without loading weights
        # Check if dataset_stats were provided in kwargs
        model = cls(config, **kwargs)

        if config.quantization is not None:
            # 量化 checkpoint——先把匹配的 nn.Linear 换成量化实现，
            # 再加载权重，scale 张量随 state dict 一起进来（见本包 quantization）。
            from .quantization import apply_quantization

            apply_quantization(model, config.quantization)

        # Load state dict (expects keys with "model." prefix)
        try:
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    revision=revision,
                    local_files_only=local_files_only,
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                raise RuntimeError("PI05 Ascend910B checkpoint weights could not be loaded") from e

            # First, fix any key differences (see openpi model.py, _fix_pytorch_state_dict_keys)
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            # Load the remapped state dict into the model
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=strict)

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        except Exception as e:
            raise RuntimeError("PI05 Ascend910B checkpoint loading failed") from e

        if config.quantization is not None:
            # 量化模型还要显式校验 scale/qweight，防止损坏但键名完整的
            # checkpoint 使用 NaN scale 或全零权重进入推理。
            from .quantization import validate_quantized

            validate_quantized(model)
        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """修正 checkpoint key，使其匹配当前 PI0.5 PyTorch 模型结构。"""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False)
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False)
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi05
            # pi05 model expects time_mlp_*, but checkpoint might have action_time_mlp_*
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            # Also handle state_proj which shouldn't exist in pi05
            if key.startswith("state_proj."):
                logging.warning(f"Skipping state_proj key in pi05 mode: {key}")
                continue

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            if (
                key == "model.paligemma_with_expert.paligemma.lm_head.weight"
                or key == "paligemma_with_expert.paligemma.lm_head.weight"
            ):
                fixed_state_dict["model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"] = (
                    value.clone()
                )

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        """返回优化器需要更新的参数集合。"""
        return self.parameters()

    def reset(self):
        """环境 reset 时清空 action 队列和内部缓存状态。"""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """根据配置初始化 RTC processor，并挂到核心模型上。"""
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        """判断 policy 当前是否启用 RTC 推理。"""
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    @torch.no_grad()
    def preprocess_raw_image_for_inference(
        self,
        image: Tensor,
    ) -> Tensor:
        """将环境原始 BHWC uint8 图像尽早搬到模型设备并转换为 BCHW/[0,1]。

        在线 rollout 在 NumPy 图像转为 Tensor 后立即调用本入口。原始 uint8
        H2D 后才在 NPU 上做 layout、float32 和 /255，避免 CPU 大图处理以及
        float32 图像四倍传输量。相机仍逐张处理，不改变多相机 batch 语义。

        本入口不执行 resize；环境特定的方向变换仍先于 _preprocess_images
        中的 resize，避免改变插值结果。
        """
        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError(f"expected batched BHWC RGB image, got shape={tuple(image.shape)}")
        if image.dtype != torch.uint8:
            raise TypeError(f"expected uint8 raw image, got dtype={image.dtype}")

        device = next(self.parameters()).device
        if image.device != device:
            # 必须在 cast 前搬运，uint8 传输量仅为 float32 的四分之一。
            image = image.to(device)
        image = image.permute(0, 3, 1, 2).contiguous()
        image = image.to(dtype=torch.float32).div_(255.0)
        return image

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """把 LeRobot 图像 batch 转换为 PaliGemma prefix 可直接消费的图像列表和 mask。

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images: list[Tensor] = []
        img_masks: list[Tensor] = []

        device = next(self.parameters()).device
        target_resolution = tuple(self.config.image_resolution)

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        for key in present_img_keys:
            image = batch[key]

            if image.ndim != 4:
                raise ValueError(f"expected a 4D image tensor for {key}, got shape={tuple(image.shape)}")

            if image.device != device:
                image = image.to(device)

            # 入口统一为 BCHW，后续 resize、归一化和视觉塔均保持同一 layout。
            # 标准 LeRobot 输入无需 permute；BHWC 输入最多转换一次。
            if image.shape[1] != 3:
                if image.shape[-1] != 3:
                    raise ValueError(
                        f"expected RGB image in BCHW or BHWC format for {key}, got shape={tuple(image.shape)}"
                    )
                image = image.permute(0, 3, 1, 2).contiguous()

            if image.dtype != torch.float32:
                image = image.to(dtype=torch.float32)

            if tuple(image.shape[-2:]) != target_resolution:
                image = resize_with_pad_torch(image, *target_resolution)

            # 先创建归一化输出，再原地减一；不修改 batch 中的输入 tensor。
            image = image.mul(2.0).sub_(1.0)

            images.append(image)
            img_masks.append(torch.ones(image.shape[0], dtype=torch.bool, device=device))

        for _ in missing_img_keys:
            # SigLIP 的空相机像素为 -1，mask 为 False。
            images.append(torch.full_like(images[-1], -1.0))
            img_masks.append(torch.zeros_like(img_masks[-1]))

        return images, img_masks

    def prepare_action(self, batch):
        """把真实 action padding 到模型内部 max_action_dim。"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """在线控制入口：从 action chunk 队列中取出单步动作。"""
        assert not self._rtc_enabled(), "RTC is not supported for select_action, use it with predict_action_chunk"

        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """根据当前观测预测一整个 action chunk，并裁剪回真实动作维度。"""
        # self.eval()

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        # Sample actions using the model (pass through RTC kwargs, no separate state needed for PI05)
        actions = self.model.sample_actions(images, img_masks, tokens, masks, **kwargs)

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """训练入口：预处理 batch、采样 noise/time，并计算 flow matching loss。

        Args:
            batch: Training batch containing observations and actions.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.prepare_action(batch)

        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)

        # Compute loss (no separate state needed for PI05)
        losses = self.model.forward(images, img_masks, tokens, masks, actions, noise, time)

        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def _get_default_peft_targets(self) -> dict[str, any]:
        """返回 PI0.5 微调默认 LoRA/PEFT 目标模块匹配规则。"""
        common_projections = "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        target_modules = rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }


class PI05Ascend910BPolicy(PI05Policy):
    """PI0.5 policy with the repository-owned Ascend910B runtime contract."""
