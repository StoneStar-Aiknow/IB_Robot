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
π0.5 Action Expert part — denoising / action generation only.

This module is the Action Expert half of the PI05 model split for deployment.
It receives a KV cache (from the VLM part) + state + time + noise,
and performs a single Euler denoising step to produce actions.

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
    ACTION,
)

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

    # Use float32 instead of float64 for the sinusoidal computation. The result is
    # ultimately cast down to the (fp16/fp32) model dtype, so double precision is
    # wasted here, and Ascend 310P AICore has no float64 vector kernels — exporting
    # this sub-graph as float64 makes ATC fall back to a generic/AICPU path and emit
    # "W11001 ... does not hit the high-priority operator information library" for
    # every Sin/Cast/Mul/Add/Concat op. fp32 matches the openpi reference and is
    # fully covered by the high-priority op library.
    dtype = torch.float32
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
    将 past_key_values（可能是 dict 或 list）统一展平为一个 Tensor:
    形状为 (L, 2, B, H, S, D)，方便导出到 ONNX。

    注意：transformers 的 KV cache shape 是 (B, H, S, D)，即 batch, heads, seq, head_dim
    """
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
        # TF5.3: get_image_features returns BaseModelOutputWithPooling, use pooler_output
        # and multiply back by hidden_size**0.5 to restore pre-v5 scale
        image_output = self.paligemma.model.get_image_features(image)
        hidden_size = self.paligemma.config.text_config.hidden_size
        return image_output.pooler_output * (hidden_size**0.5)

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
        """Action Expert forward: only suffix path (inputs_embeds[0] is None)."""
        if adarms_cond is None:
            adarms_cond = [None, None]

        # In Action Expert mode, we only run the suffix (gemma expert)
        suffix_output = self.gemma_expert.model.forward(
            inputs_embeds=inputs_embeds[1],
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
        )
        suffix_output_hidden = suffix_output.last_hidden_state

        return [None, suffix_output_hidden], None


class PI05ActionExpertPytorch(nn.Module):
    """PI05 Action Expert — takes KV cache + state + time + noise, outputs actions."""

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
        logging.info("Enabled gradient checkpointing for PI05ActionExpertPytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05ActionExpertPytorch model")

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer.

        Uses a fp16-safe masked sentinel (``finfo(fp16).min`` = -65504) instead
        of ``OPENPI_ATTENTION_MASK_VALUE`` (-2.38e38).  When the attention runs
        in fp16 (action-expert export), the eager-attention patch casts this
        mask to fp16; -2.38e38 would overflow to ``-inf`` and NaN any
        fully-masked row, whereas -65504 stays representable.  After softmax
        both sentinels drive masked logits to ~0, so the result is unchanged
        for fp32 too.
        """
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        masked_value = torch.finfo(torch.float16).min
        return torch.where(att_2d_masks_4d, 0.0, masked_value)

    def sample_noise(self, shape, device):
        # Zero noise for deterministic inference
        return torch.zeros(shape, dtype=torch.float16, device=device)

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float16, device=device)

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb):
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

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    @torch.no_grad()
    def sample_actions(self, past_kv_tensor, prefix_pad_masks, time, noise) -> Tensor:
        """Perform a single Euler denoising step using KV cache from VLM."""
        bsize = noise.shape[0]
        device = noise.device

        past_key_values = unflatten_kv(past_kv_tensor)

        dt = -1.0 / self.config.num_inference_steps
        dt = torch.tensor(dt, dtype=torch.float16, device=device)

        x_t = noise

        expanded_time = time.expand(bsize)
        v_t = self.denoise_step(
            prefix_pad_masks,
            past_key_values,
            x_t,
            expanded_time,
        )

        # Single Euler step
        x_t = x_t + dt * v_t
        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]

        suffix_pad_masks = suffix_pad_masks.to(dtype=torch.int)
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        suffix_pad_masks = suffix_pad_masks.to(dtype=torch.bool)

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Cast to the same dtype as action_out_proj weights so that
        # the linear layer works in both fp16-export and fp32-export modes.
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        return v_t


class PI05ActionExpertPolicy(PreTrainedPolicy):
    """PI05 Action Expert Policy — takes KV cache, outputs actions."""

    config_class = PI05Config
    name = "pi05"

    def __init__(
        self,
        config: PI05Config,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.kd = getattr(config, "kd", False)
        logging.info(f"Knowledge distillation (kd) is set to: {self.kd}")

        torch.cuda.empty_cache()

        with torch.device("cpu"):
            self.model = PI05ActionExpertPytorch(config)

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

        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze LoRA parameters and projection layers
        for name, param in self.model.named_parameters():
            if any(keyword in name for keyword in ["lora", "action_in_proj", "action_out_proj", "time_mlp"]):
                param.requires_grad = True
                logging.debug(f"Training parameter: {name}")

        # Log trainable parameters
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
            "The PI05 Action Expert model is a direct port of the OpenPI implementation. \n"
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

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Perform a single denoising step given KV cache from VLM.

        Expected batch keys:
        - past_kv_tensor: KV cache from VLM part
        - prefix_pad_masks: padding masks from VLM part
        - time: current denoising timestep
        - noise: current noisy action (x_t)
        """

        past_kv_tensor = batch["past_kv_tensor"]
        prefix_pad_masks = batch["prefix_pad_masks"]
        time = batch["time"]
        noise = batch["noise"]

        actions = self.model.sample_actions(past_kv_tensor, prefix_pad_masks, time, noise=noise)
        return actions

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Not applicable for action-expert-only part without full denoising loop."""
        raise NotImplementedError(
            "Action Expert performs single denoising steps. Use select_action in a loop for full denoising."
        )

    def forward(self, batch: dict[str, Tensor], teacher_policy=None) -> tuple[Tensor, dict]:
        """Not applicable for Action Expert during deployment."""
        raise NotImplementedError("Action Expert part is for inference only. Use the full PI05Policy for training.")
