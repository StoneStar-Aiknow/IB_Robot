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

"""
π0.5 VLM (Vision-Language Model) part — prefix encoding only.

This module is the VLM half of the PI05 model split for deployment.
It processes images + language tokens → KV cache, which is then consumed
by the Action Expert module (modeling_pi05_action_expert.py).

Original full model: modeling_pi05.py
"""

import builtins
import logging
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn.functional as F  # noqa: N812
from lerobot.utils.import_utils import _transformers_available
from torch import Tensor, nn

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma
    from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
    from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
else:
    CONFIG_MAPPING = None
    modeling_gemma = None
    GemmaForCausalLM = None
    PaliGemmaForConditionalGeneration = None

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OPENPI_ATTENTION_MASK_VALUE,
)


# LoRA Implementation
class LoRALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # LoRA matrices
        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)

        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(x)
        lora_output = self.lora_B(self.lora_A(x))
        return self.alpha * lora_output


class LoRALinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, rank: int, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(linear_layer.in_features, linear_layer.out_features, rank, alpha, dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x) + self.lora(x)


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
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
    """Computes sine-cosine positional embedding vectors for scalar positions."""
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
    # use sin to represent cos to avoid precision issues
    cos_value = torch.sin(sin_input + math.pi / 2)
    pos_emb = torch.cat([torch.sin(sin_input), cos_value], dim=1)
    return pos_emb


def flatten_kv(past_key_values):
    """
    将 past_key_values（可能是 dict、list 或 DynamicCache）统一展平为一个 Tensor:
    形状为 (L, 2, B, H, S, D)，方便导出到 ONNX。

    注意：transformers 的 KV cache shape 是 (B, H, S, D)，即 batch, heads, seq, head_dim
    """
    # 处理 transformers DynamicCache 对象
    try:
        from transformers.cache_utils import DynamicCache

        if isinstance(past_key_values, DynamicCache):
            # DynamicCache 有 key_cache 和 value_cache 属性
            # 它们是 list[Tensor]，每个 tensor shape = (B, H, S, D)
            # 参考 cache_utils.py 中的 _flatten_dynamic_cache 函数
            key_cache = past_key_values.key_cache  # list of (B, H, S, D)
            value_cache = past_key_values.value_cache  # list of (B, H, S, D)
            keys = torch.stack(key_cache, dim=0)  # (L, B, H, S, D)
            values = torch.stack(value_cache, dim=0)  # (L, B, H, S, D)
            flat = torch.stack([keys, values], dim=1)  # (L, 2, B, H, S, D)
            return flat
    except ImportError:
        pass

    # 处理旧格式：dict 或 list
    if isinstance(past_key_values, dict):
        past_key_values = [past_key_values[i] for i in sorted(past_key_values.keys())]
    elif not isinstance(past_key_values, (list, tuple)):
        raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)}")

    keys = [kv["key_states"] for kv in past_key_values]
    values = [kv["value_states"] for kv in past_key_values]

    keys = torch.stack(keys, dim=0)  # (L, B, H, S, D)
    values = torch.stack(values, dim=0)  # (L, B, H, S, D)

    flat = torch.stack([keys, values], dim=1)  # (L, 2, B, H, S, D)
    return flat


def unflatten_kv(flat_tensor):
    """
    把 (L, 2, B, H, S, D) 的 Tensor 还原回 list[dict]
    供模型继续使用。

    注意：shape 是 (L, 2, B, H, S, D)，其中 H=heads, S=seq_len, D=head_dim
    """
    if not isinstance(flat_tensor, torch.Tensor):
        raise TypeError(f"Expected Tensor, got {type(flat_tensor)}")

    # flat_tensor shape: (L, 2, B, H, S, D)
    num_layers = flat_tensor.shape[0]
    keys = flat_tensor[:, 0]  # (L, B, H, S, D)
    values = flat_tensor[:, 1]

    # Prefer to return a transformers.DynamicCache when available so
    # downstream code that expects DynamicCache (used in flatten_kv)
    # will interoperate seamlessly.
    try:
        from transformers.cache_utils import DynamicCache

        dyn = DynamicCache()
        # DynamicCache.key_cache and .value_cache are lists of tensors
        dyn.key_cache = [keys[i] for i in range(num_layers)]
        dyn.value_cache = [values[i] for i in range(num_layers)]
        return dyn
    except Exception:
        # Fallback: return the old list-of-dicts format
        out = []
        for i in range(num_layers):
            out.append(
                {
                    "key_states": keys[i],
                    "value_states": values[i],
                }
            )
        return out


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

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

    att_masks = att_masks.to(dtype=torch.int)
    cumsum = torch.cumsum(att_masks, dim=1)
    att_masks = att_masks.to(dtype=torch.bool)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


# Stage A — Plan A.
# Host-side helper that pre-computes the additive 4D attention mask
# (the one normally produced inside ``sample_actions`` via
# ``make_att_2d_masks`` + ``_prepare_attention_masks_4d``).
#
# Why this exists:
#   The 4D mask contains the constant ``OPENPI_ATTENTION_MASK_VALUE``
#   (``-2.3819763e38``, near the fp32 lower bound).  When the surrounding
#   graph is exported to ONNX and then compiled to OM, ATC's fp16 mixed
#   precision pass is prone to fuse / re-order the producing ``Where``
#   node and either truncate the constant to ``-65504`` or hoist a Cast
#   that clobbers it to ``-inf``.  Either case weakens the padding mask
#   and silently corrupts attention outputs (observed: VLM cosine 0.03).
#
#   By moving this computation to the host and feeding the resulting
#   tensor as a regular model input, the dangerous constant never appears
#   in the ONNX/OM graph at all.
def build_prefix_att_2d_masks_4d(
    pad_masks: Tensor,
    att_masks: Tensor,
    *,
    mask_value: float = OPENPI_ATTENTION_MASK_VALUE,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Compute the 4D additive attention mask outside the ONNX/OM graph.

    Args:
        pad_masks: ``(B, S)`` bool tensor — ``True`` for valid prefix
            tokens, ``False`` for padding.
        att_masks: ``(B, S)`` int/bool tensor — block boundary markers
            in the same convention as :func:`make_att_2d_masks`.
        mask_value: Value used for masked-out positions in the additive
            mask.  Defaults to PI05's ``OPENPI_ATTENTION_MASK_VALUE``;
            override (e.g. to ``-1e4``) only when validating the fp16
            truncation hypothesis.
        dtype: Output dtype.  Keep ``float32`` so PaliGemma's attention
            ``add(scores_fp16, mask_fp32)`` upcasts safely.

    Returns:
        ``(B, 1, S, S)`` tensor with ``0.0`` on attendable positions and
        ``mask_value`` elsewhere — the exact equivalent of running
        ``self._prepare_attention_masks_4d(make_att_2d_masks(...))``.
    """
    att_2d = make_att_2d_masks(pad_masks, att_masks)
    att_2d_4d = att_2d[:, None, :, :].to(torch.bool)
    return torch.where(att_2d_4d, 0.0, mask_value).to(dtype)


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
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
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else -1.0
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


class GemmaConfig:  # see openpi `gemma.py: Config`
    """Configuration for Gemma model variants."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
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


class PaliGemmaWithExpertModel(
    nn.Module
):  # see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
    """PaliGemma model with action expert for PI05."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        lora_config: dict[str, Any] | None = None,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

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
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        # Store LoRA configuration
        self.lora_config = lora_config

        # Apply LoRA if configured
        if lora_config and lora_config.get("use_lora", False):
            self._apply_lora()

        self.to_bfloat16_for_selected_params(precision)

    def _apply_lora(self):
        """Apply LoRA to the model based on configuration."""
        lora_r = self.lora_config.get("lora_r", 8)
        lora_alpha = self.lora_config.get("lora_alpha", 16)
        lora_dropout = self.lora_config.get("lora_dropout", 0.1)
        target_modules = self.lora_config.get("target_modules", ["q_proj", "v_proj"])

        logging.info(
            f"Applying LoRA with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}, target_modules={target_modules}"
        )

        total_lora_layers = 0

        # Apply LoRA to PaliGemma language model
        paligemma_layers = self._apply_lora_to_model(
            self.paligemma.language_model, lora_r, lora_alpha, lora_dropout, target_modules
        )
        total_lora_layers += paligemma_layers

        # Apply LoRA to Gemma expert model
        gemma_expert_layers = self._apply_lora_to_model(
            self.gemma_expert.model, lora_r, lora_alpha, lora_dropout, target_modules
        )
        total_lora_layers += gemma_expert_layers

        logging.info(f"Applied LoRA to {total_lora_layers} layers in total")

    def _apply_lora_to_model(self, model, lora_r, lora_alpha, lora_dropout, target_modules):
        """Apply LoRA to a specific model component."""
        lora_applied_count = 0
        for name, module in model.named_children():
            if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
                # Replace the linear layer with LoRALinear
                lora_linear = LoRALinear(module, lora_r, lora_alpha, lora_dropout)
                setattr(model, name, lora_linear)
                lora_applied_count += 1
            else:
                # Recursively apply to child modules
                lora_applied_count += self._apply_lora_to_model(
                    module, lora_r, lora_alpha, lora_dropout, target_modules
                )

        return lora_applied_count

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        """VLM forward: only prefix path (inputs_embeds[1] is None)."""
        if adarms_cond is None:
            adarms_cond = [None, None]

        # In VLM mode, we only run the prefix (paligemma language model)
        prefix_output = self.paligemma.language_model.forward(
            inputs_embeds=inputs_embeds[0],
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
        )
        prefix_past_key_values = prefix_output.past_key_values
        prefix_output_hidden = prefix_output.last_hidden_state

        return [prefix_output_hidden, None], prefix_past_key_values


