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
else:
    CONFIG_MAPPING = None
    modeling_gemma = None

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OPENPI_ATTENTION_MASK_VALUE,
)

from inference_service.pi05_image_preprocess import resize_with_pad_torch
from model_utils.pi05_export.pi_gemma import PaliGemmaForConditionalGenerationWithPiGemma, PiGemmaForCausalLM


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
            # TF5.3: DynamicCache is iterable, yields (key, value, None) tuples
            kv_list = list(past_key_values)
            key_cache = [kv[0] for kv in kv_list]  # list of (B, H, S, D)
            value_cache = [kv[1] for kv in kv_list]  # list of (B, H, S, D)
            keys = torch.stack(key_cache, dim=0)  # (L, B, H, S, D)
            values = torch.stack(value_cache, dim=0)  # (L, B, H, S, D)
            flat = torch.stack([keys, values], dim=1)  # (L, 2, B, H, S, D)
            return flat
    except ImportError:
        pass

    # 处理旧格式：dict 或 list
    if isinstance(past_key_values, dict):
        past_key_values = [past_key_values[i] for i in sorted(past_key_values.keys())]
    elif not isinstance(past_key_values, (list, tuple)):  # noqa: UP038
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
        # TF5.3: use update() to add key-value pairs per layer
        for i in range(num_layers):
            dyn.update(keys[i], values[i], layer_idx=i)
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

        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
        self.gemma_expert = PiGemmaForCausalLM(config=action_expert_config_hf)
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
            self.paligemma.model.language_model, lora_r, lora_alpha, lora_dropout, target_modules
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
        image_outputs = self.paligemma.model.vision_tower(image, return_dict=True)
        return self.paligemma.model.multi_modal_projector(image_outputs.last_hidden_state)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.model.language_model.embed_tokens(tokens)

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
        prefix_output = self.paligemma.model.language_model.forward(
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


class PI05VLMPytorch(nn.Module):
    """PI05 VLM part — encodes images + language into KV cache."""

    def __init__(self, config: PI05Config):
        super().__init__()
        self.config = config

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        # Extract LoRA configuration from main config
        lora_config = {
            "use_lora": getattr(config, "use_lora", False),
            "lora_r": getattr(config, "lora_r", 8),
            "lora_alpha": getattr(config, "lora_alpha", 16),
            "lora_dropout": getattr(config, "lora_dropout", 0.1),
            "target_modules": getattr(config, "lora_target_modules", ["q_proj", "v_proj"]),
        }

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            lora_config=lora_config,
        )

        # Action projection layers (needed for config but not used in VLM forward)
        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05VLMPytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05VLMPytorch model")

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device):
        # Zero noise for deterministic inference
        return torch.zeros(shape, dtype=torch.float16, device=device)

    def embed_prefix(self, images, img_masks, tokens, masks) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            img_emb = img_emb.to(dtype=torch.float16)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        prefix_att_2d_masks_4d: torch.Tensor | None = None,
    ) -> Tensor:
        """Encode prefix (images + language) and return KV cache + pad masks.

        Args:
            prefix_att_2d_masks_4d: Optional pre-computed additive 4D
                attention mask of shape ``(B, 1, S, S)``.  When supplied,
                the in-graph ``make_att_2d_masks`` +
                ``_prepare_attention_masks_4d`` chain is **bypassed** —
                this is the Plan A path that keeps
                ``OPENPI_ATTENTION_MASK_VALUE`` (-2.38e38) out of the
                exported ONNX/OM graph.
                Pass ``None`` to keep the original behaviour (PyTorch
                training & legacy inference paths).
        """
        bsize = tokens.shape[0]  # noqa: F841
        device = tokens.device  # noqa: F841

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)

        prefix_pad_masks = prefix_pad_masks.to(dtype=torch.int)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_pad_masks = prefix_pad_masks.to(dtype=torch.bool)

        # Stage A — Plan A: prefer the host-supplied 4D mask so the
        # exported graph never contains the dangerous fp32 constant.
        if prefix_att_2d_masks_4d is None:
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        past_kv_tensor = flatten_kv(past_key_values)
        return past_kv_tensor, prefix_pad_masks


class PI05VLMPolicy(PreTrainedPolicy):
    """PI05 VLM Policy — encodes images + language, outputs KV cache."""

    config_class = PI05Config
    name = "pi05"

    def __init__(
        self,
        config: PI05Config,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.kd = getattr(config, "kd", False)
        logging.info(f"Knowledge distillation (kd) is set to: {self.kd}")

        torch.cuda.empty_cache()

        with torch.device("cpu"):
            self.model = PI05VLMPytorch(config)

        if config.device and config.device != "cpu":
            self.model = self.model.to(config.device)

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Freeze base model parameters if using LoRA
        if getattr(config, "use_lora", False) and getattr(config, "lora_freeze_base", True):
            self._freeze_base_model_for_lora()

        self.model.to(config.device)

        self.reset()

    def _freeze_base_model_for_lora(self):
        """Freeze base model parameters and only train LoRA parameters."""
        logging.info("Freezing base model parameters for LoRA training")

        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            if any(keyword in name for keyword in ["lora", "action_in_proj", "action_out_proj", "time_mlp"]):
                param.requires_grad = True
                logging.debug(f"Training parameter: {name}")

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logging.info(
            f"LoRA training: {trainable_params:,} trainable parameters out of {total_params:,} ({100 * trainable_params / total_params:.2f}%)"
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
        """Override the from_pretrained method to handle key remapping."""
        print(
            "The PI05 VLM model is a direct port of the OpenPI implementation. \n"
            "This implementation follows the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        # Initialize model without loading weights
        # Check if dataset_stats were provided in kwargs
        model = cls(config, **kwargs)

        # Now manually load and remap the state dict
        try:
            # Try to load the pytorch_model.bin or model.safetensors file
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                # Try safetensors first
                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    use_auth_token=kwargs.get("use_auth_token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                print(f"Could not load state dict from remote files: {e}")
                print("Returning model without loading pretrained weights")
                return model

            # First, fix any key differences # see openpi `model.py, _fix_pytorch_state_dict_keys`
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                    if remap_count <= 10:  # Only print first 10 to avoid spam
                        print(f"Remapped: {key} -> {new_key}")
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
            print(f"Warning: Could not remap state dict keys: {e}")

        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
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

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        """Get parameters for optimization, considering LoRA configuration."""
        if getattr(self.config, "use_lora", False):
            # Only return parameters that require gradients (LoRA parameters and projection layers)
            trainable_params = []
            for _, param in self.named_parameters():
                if param.requires_grad:
                    trainable_params.append(param)
            logging.info(f"Total trainable parameters with LoRA: {len(trainable_params)}")
            return trainable_params
        else:
            # Return all parameters for full fine-tuning
            return self.parameters()

    def reset(self):
        """Reset internal state."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []

        # Get device from model parameters
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            # Ensure tensor is on the same device as the model
            if img.device != device:
                img = img.to(device)

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)

            # from openpi preprocess_observation_pytorch: Resize with padding if needed
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """Use language tokens and masks directly from the batch (pre-tokenized)."""
        device = next(self.parameters()).device

        if OBS_LANGUAGE_TOKENS in batch:
            lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        elif "lang_tokens" in batch:
            lang_tokens = batch["lang_tokens"]
        elif "lang_token" in batch:
            lang_tokens = batch["lang_token"]
        else:
            raise KeyError(
                "Missing language tokens in batch. Expected 'observation.language_tokens', 'lang_tokens' or 'lang_token'."
            )

        if OBS_LANGUAGE_ATTENTION_MASK in batch:
            lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        elif "lang_masks" in batch:
            lang_masks = batch["lang_masks"]
        elif "lang_mask" in batch:
            lang_masks = batch["lang_mask"]
        else:
            pad_token_id = getattr(self.config, "pad_token_id", 0)
            lang_masks = lang_tokens != pad_token_id

        lang_tokens = lang_tokens.to(device=device, dtype=torch.long)
        if lang_masks.dtype is torch.bool:
            lang_masks = lang_masks.to(device=device)
        else:
            lang_masks = lang_masks.to(device=device) != 0

        return lang_tokens, lang_masks

    @torch.no_grad()
    def compute_prefix_att_2d_masks_4d(
        self,
        batch: dict[str, Tensor],
        *,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Stage B — Plan A helper: build the 4D additive prefix mask
        on the host so callers can feed it into ONNX/OM as a model input
        instead of letting the constant ``OPENPI_ATTENTION_MASK_VALUE``
        live inside the exported graph.

        Mirrors the prefix construction in ``sample_actions``:
        ``embed_prefix`` produces ``(prefix_pad_masks, prefix_att_masks)``
        whose layout is ``[image tokens × N_cams | language tokens]``;
        we then call :func:`build_prefix_att_2d_masks_4d` to combine
        them and apply the additive mask value.

        Args:
            batch: The same dict consumed by :meth:`select_action`
                (raw images + tokenised language).
            dtype: Target dtype for the additive mask (default fp32).

        Returns:
            Tensor of shape ``(B, 1, S, S)`` with ``0.0`` on visible
            positions and ``OPENPI_ATTENTION_MASK_VALUE`` on masked ones.
        """
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = self.prepare_language(batch)
        _, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(images, img_masks, tokens, masks)
        # ``embed_prefix`` returns pad_masks as bool already.
        return build_prefix_att_2d_masks_4d(
            prefix_pad_masks.to(torch.bool),
            prefix_att_masks,
            dtype=dtype,
        )

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode prefix and return KV cache + pad masks."""

        images, img_masks = self._preprocess_images(batch)
        tokens, masks = self.prepare_language(batch)

        # Stage A — Plan A: forward the optional pre-computed 4D
        # attention mask if the caller put one in the batch.  Absent
        # → ``sample_actions`` falls back to the in-graph computation.
        prefix_att_2d_masks_4d = batch.get("prefix_att_2d_masks_4d")

        past_kv_tensor, prefix_pad_masks = self.model.sample_actions(
            images,
            img_masks,
            tokens,
            masks,
            prefix_att_2d_masks_4d=prefix_att_2d_masks_4d,
        )

        return past_kv_tensor, prefix_pad_masks

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Not applicable for VLM-only part."""
        raise NotImplementedError("VLM part only produces KV cache, not actions.")

    def forward(self, batch: dict[str, Tensor], teacher_policy=None) -> tuple[Tensor, dict]:
        """Not applicable for VLM-only part during deployment."""
        raise NotImplementedError("VLM part is for inference only. Use the full PI05Policy for training.")
