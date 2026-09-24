#!/usr/bin/env python

"""Local SigLIP vision tower used by PI0.5 NPU inference optimizations."""

from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling

PI05_VISION_PFA_FORCE_FP16_ENV = "LEROBOT_PI05_VISION_PFA_FORCE_FP16"


def _import_torch_npu():
    try:
        import torch_npu  # type: ignore
    except ModuleNotFoundError:
        return None
    return torch_npu


def _blocked_attention_mask(attention_mask: torch.Tensor | None) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.dtype == torch.bool:
        blocked_mask = torch.logical_not(attention_mask)
    else:
        blocked_mask = attention_mask < 0
    return blocked_mask.to(dtype=torch.int8, memory_format=torch.contiguous_format)


class PI05SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        num_patches = embeddings.shape[1]
        num_positions = self.position_embedding.weight.shape[0]
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)

        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)
        dim = embeddings.shape[-1]
        new_height = height // self.patch_size
        new_width = width // self.patch_size
        sqrt_num_positions = int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class PI05SiglipAttention(nn.Module):
    """SigLIP self-attention with selectable default/PFA/FIAS inference backend."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads, got {self.embed_dim} and {self.num_heads}")
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        # INT8 W8A8 融合后为 Int8W8A8Linear，FP fallback 才是 nn.Linear。
        self.qkv: nn.Module | None = None
        self._npu_attention_backend = "default"
        self._npu_pfa_force_fp16 = False

    @torch.no_grad()
    def fuse_qkv_weights(self) -> None:
        if self.qkv is not None:
            return
        # Selective-99 INT8 拼接 qweight/weight_scale/bias；BF16 拼接浮点
        # 权重。两种产物均满足 self.qkv(x).split() 消费接口。
        from .quantization import Int8W8A8Linear, fuse_qkv_int8

        if isinstance(self.q_proj, Int8W8A8Linear):
            self.qkv = fuse_qkv_int8(self.q_proj, self.k_proj, self.v_proj)
            return
        qkv_weight = torch.cat(
            [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight],
            dim=0,
        ).contiguous()
        qkv_bias = torch.cat(
            [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias],
            dim=0,
        ).contiguous()
        self.qkv = nn.Linear(
            qkv_weight.shape[1],
            qkv_weight.shape[0],
            bias=True,
            device=qkv_weight.device,
            dtype=qkv_weight.dtype,
        )
        self.qkv.weight.copy_(qkv_weight)
        self.qkv.bias.copy_(qkv_bias)
        self.qkv.weight.requires_grad_(False)
        self.qkv.bias.requires_grad_(False)

    def enable_npu_pfa(self, enabled: bool = True) -> None:
        self._npu_attention_backend = "pfa" if enabled else "default"

    def set_npu_attention_backend(self, backend: str) -> None:
        backend = str(backend).lower()
        if backend not in {"default", "pfa", "fias"}:
            raise ValueError(f"unsupported SigLIP NPU attention backend: {backend!r}")
        self._npu_attention_backend = backend
        self._npu_pfa_force_fp16 = os.environ.get(PI05_VISION_PFA_FORCE_FP16_ENV, "0").strip() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _project_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_shape = (*hidden_states.shape[:-1], self.num_heads, self.head_dim)
        if self.qkv is not None:
            qkv = self.qkv(hidden_states)
            query, key, value = qkv.split([self.embed_dim, self.embed_dim, self.embed_dim], dim=-1)
        else:
            query = self.q_proj(hidden_states)
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)
        return query.view(hidden_shape), key.view(hidden_shape), value.view(hidden_shape)

    def _can_use_npu_fused_attention(self, query: torch.Tensor) -> bool:
        return (
            self._npu_attention_backend in {"pfa", "fias"}
            and not self.training
            and query.device.type == "npu"
            and query.dtype in {torch.float16, torch.bfloat16}
            and _import_torch_npu() is not None
        )

    def _eager_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_bnsd = query.transpose(1, 2)
        key_bnsd = key.transpose(1, 2)
        value_bnsd = value.transpose(1, 2)
        batch_size, num_heads, query_len, head_dim = query_bnsd.shape
        key_len = key_bnsd.shape[-2]
        query_3d = query_bnsd.reshape(batch_size * num_heads, query_len, head_dim)
        key_3d = key_bnsd.reshape(batch_size * num_heads, key_len, head_dim)
        attn_weights = torch.bmm(query_3d, key_3d.transpose(1, 2)) * self.scale
        attn_weights = attn_weights.reshape(batch_size, num_heads, query_len, key_len)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(
            attn_weights.reshape(batch_size * num_heads, query_len, key_len),
            value_bnsd.reshape(batch_size * num_heads, key_len, head_dim),
        ).reshape(batch_size, num_heads, query_len, head_dim)
        return attn_output.transpose(1, 2).contiguous(), attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        query, key, value = self._project_qkv(hidden_states)

        if self._can_use_npu_fused_attention(query):
            torch_npu = _import_torch_npu()
            output_dtype = hidden_states.dtype
            if self._npu_attention_backend == "fias":
                # Keep Vision Q/K/V in BF16 to isolate FIAS from the PFA Cast
                # overhead. CANN 9.2 on 910B3 requires BSND head_dim to be
                # 16-aligned when Qs>1, so SigLIP 72 is padded to 80 and sliced
                # back after FIAS. The FIAS arguments otherwise match official
                # LeRobot.
                aligned_head_dim = ((self.head_dim + 15) // 16) * 16
                head_pad = aligned_head_dim - self.head_dim
                fias_query, fias_key, fias_value = query, key, value
                if head_pad:
                    fias_query = F.pad(fias_query, (0, head_pad))
                    fias_key = F.pad(fias_key, (0, head_pad))
                    fias_value = F.pad(fias_value, (0, head_pad))
                attn_output = torch_npu.npu_fused_infer_attention_score(
                    fias_query.contiguous(),
                    fias_key.contiguous(),
                    fias_value.contiguous(),
                    num_heads=self.num_heads,
                    input_layout="BSND",
                    scale=1.0 / math.sqrt(self.head_dim),
                    pre_tokens=65535,
                    next_tokens=65535,
                    atten_mask=_blocked_attention_mask(attention_mask),
                    num_key_value_heads=self.num_heads,
                    sparse_mode=0,
                    inner_precise=0,
                )[0]
                if head_pad:
                    attn_output = attn_output[..., : self.head_dim]
            else:
                # 910B3/CANN 9.2 的 PFA 可直接消费 SigLIP head_dim=72
                # BF16 Q/K/V。保持原 dtype，避免每层 3 个输入 Cast 和
                # 1 个输出 Cast。
                pfa_query, pfa_key, pfa_value = query, key, value
                if self._npu_pfa_force_fp16:
                    # Accuracy A/B compatibility mode for the former 910B path.
                    pfa_query = pfa_query.to(torch.float16)
                    pfa_key = pfa_key.to(torch.float16)
                    pfa_value = pfa_value.to(torch.float16)
                attn_output = torch_npu.npu_prompt_flash_attention(
                    pfa_query.contiguous(),
                    pfa_key.contiguous(),
                    pfa_value.contiguous(),
                    num_heads=self.num_heads,
                    input_layout="BSND",
                    scale_value=1.0 / math.sqrt(self.head_dim),
                    pre_tokens=65535,
                    next_tokens=65535,
                    atten_mask=_blocked_attention_mask(attention_mask),
                    num_key_value_heads=self.num_heads,
                )
            if attn_output.dtype != output_dtype:
                attn_output = attn_output.to(output_dtype)
            attn_weights = None
        else:
            attn_output, attn_weights = self._eager_attention(query, key, value, attention_mask)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


class PI05SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class PI05SiglipEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = PI05SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = PI05SiglipMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class PI05SiglipEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([PI05SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> BaseModelOutput:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
                **kwargs,
            )
        return BaseModelOutput(last_hidden_state=hidden_states)


class PI05SiglipMultiheadAttentionPoolingHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            batch_first=True,
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = PI05SiglipMLP(config)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)
        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)
        return hidden_state[:, 0]


class PI05SiglipVisionTransformer(nn.Module):
    _input_embed_layer = "patch_embedding"

    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = PI05SiglipVisionEmbeddings(config)
        self.encoder = PI05SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = PI05SiglipMultiheadAttentionPoolingHead(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool | None = False,
        **kwargs: Any,
    ) -> BaseModelOutputWithPooling:
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=bool(interpolate_pos_encoding))
        encoder_outputs = self.encoder(inputs_embeds=hidden_states, **kwargs)
        last_hidden_state = self.post_layernorm(encoder_outputs.last_hidden_state)
        pooler_output = self.head(last_hidden_state) if self.use_head else None
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=pooler_output)


class PI05SiglipVisionModel(nn.Module):
    main_input_name = "pixel_values"
    input_modalities = ("image",)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.vision_model = PI05SiglipVisionTransformer(config)

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def fuse_qkv_weights(self) -> None:
        for layer in self.vision_model.encoder.layers:
            layer.self_attn.fuse_qkv_weights()

    def enable_npu_pfa(self, enabled: bool = True) -> None:
        for layer in self.vision_model.encoder.layers:
            layer.self_attn.enable_npu_pfa(enabled)

    def set_npu_attention_backend(self, backend: str) -> None:
        for layer in self.vision_model.encoder.layers:
            layer.self_attn.set_npu_attention_backend(backend)

    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = False,
        **kwargs: Any,
    ) -> BaseModelOutputWithPooling:
        return self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )
