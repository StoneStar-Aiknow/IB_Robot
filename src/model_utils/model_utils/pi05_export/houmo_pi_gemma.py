"""Houmo-specialized PI0.5 Gemma adapters for HMONNX export."""

from __future__ import annotations

import torch
from torch import nn


class HoumoPiGemmaRMSNorm(nn.Module):
    """Preserve PI Gemma norm semantics with Houmo RMSNorm operators."""

    def __init__(self, source: nn.Module):
        super().__init__()
        from xhquant import nn as xhnn

        self.dense = source.dense
        self.dim = int(source.dim)
        if self.dense is None:
            self.norm = xhnn.RMSNorm(self.dim, eps=float(source.eps))
            with torch.no_grad():
                self.norm.weight.copy_(1.0 + source.weight.detach().float())
        else:
            self.norm = xhnn.AdaRMSNorm(self.dim, eps=float(source.eps))
            self.scale_slice = xhnn.Slice([0], [self.dim], [-1], [1])
            self.shift_slice = xhnn.Slice([self.dim], [self.dim * 2], [-1], [1])
            self.gate_slice = xhnn.Slice([self.dim * 2], [self.dim * 3], [-1], [1])

    def forward(
        self, hidden_states: torch.Tensor, condition: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.dense is None:
            return self.norm(hidden_states), None
        if condition is None:
            raise ValueError("PI0.5 expert normalization requires a condition tensor")

        modulation = self.dense(condition)
        if hidden_states.ndim == 3:
            modulation = modulation.unsqueeze(1)
        scale = self.scale_slice(modulation)
        shift = self.shift_slice(modulation)
        gate = self.gate_slice(modulation)
        normalized = self.norm(hidden_states, 1.0 + scale)
        return normalized + shift, gate


class HoumoPiGemmaAttention(nn.Module):
    """Gemma GQA using Houmo RoPE and grouped matrix multiplication."""

    def __init__(self, source: nn.Module, *, use_past: bool):
        super().__init__()
        from xhquant import nn as xhnn

        self.q_proj = source.q_proj
        self.k_proj = source.k_proj
        self.v_proj = source.v_proj
        self.o_proj = source.o_proj
        self.head_dim = int(source.head_dim)
        self.num_heads = int(source.config.num_attention_heads)
        self.num_key_value_heads = int(source.config.num_key_value_heads)
        self.scaling = float(source.scaling)
        self.use_past = use_past
        groups = self.num_heads // self.num_key_value_heads
        self.rope = xhnn.Rope()
        self.qk_matmul = xhnn.GroupMatMul(groups=groups)
        self.pv_matmul = xhnn.GroupMatMul(groups=groups)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key: torch.Tensor | None = None,
        past_value: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        query = self.rope(query, cos, sin)
        key = self.rope(key, cos, sin)

        if self.use_past:
            if past_key is None or past_value is None:
                raise ValueError("PI0.5 decode requires key/value cache tensors")
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)

        scores = self.qk_matmul(query * self.scaling, key.transpose(2, 3))
        scores = scores + attention_mask
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        output = self.pv_matmul(probabilities, value)
        output = output.transpose(1, 2).contiguous().reshape(batch_size, sequence_length, -1)
        return self.o_proj(output), key, value


class HoumoPiGemmaDecoderLayer(nn.Module):
    def __init__(self, source: nn.Module, *, use_past: bool):
        super().__init__()
        self.input_layernorm = HoumoPiGemmaRMSNorm(source.input_layernorm)
        self.self_attn = HoumoPiGemmaAttention(source.self_attn, use_past=use_past)
        self.post_attention_layernorm = HoumoPiGemmaRMSNorm(source.post_attention_layernorm)
        self.mlp = source.mlp

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        condition: torch.Tensor | None = None,
        past_key: torch.Tensor | None = None,
        past_value: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states, gate = self.input_layernorm(hidden_states, condition)
        hidden_states, key, value = self.self_attn(hidden_states, attention_mask, cos, sin, past_key, past_value)
        hidden_states = residual + hidden_states if gate is None else residual + hidden_states * gate

        residual = hidden_states
        hidden_states, gate = self.post_attention_layernorm(hidden_states, condition)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states if gate is None else residual + hidden_states * gate
        return hidden_states, key, value


class HoumoPiGemmaModel(nn.Module):
    """Static PI Gemma graph that exposes or consumes interleaved cache tensors."""

    def __init__(self, source: nn.Module, *, use_past: bool):
        super().__init__()
        from xhquant import nn as xhnn

        self.layers = nn.ModuleList(HoumoPiGemmaDecoderLayer(layer, use_past=use_past) for layer in source.layers)
        self.norm = HoumoPiGemmaRMSNorm(source.norm)
        self.use_past = use_past
        rotary = source.rotary_emb
        inv_freq = rotary.inv_freq.detach().float()
        positions = torch.arange(int(rotary.max_seq_len_cached), dtype=torch.float32, device=inv_freq.device)
        frequencies = torch.outer(positions, inv_freq)
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cos_cached", embeddings.cos() * rotary.attention_scaling, persistent=False)
        self.register_buffer("sin_cached", embeddings.sin() * rotary.attention_scaling, persistent=False)
        self.position_gather = xhnn.Gather(0)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        condition: torch.Tensor | None,
        *flat_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if self.use_past and len(flat_cache) != len(self.layers) * 2:
            raise ValueError(f"Expected {len(self.layers) * 2} cache tensors, got {len(flat_cache)}")
        if not self.use_past and flat_cache:
            raise ValueError("PI0.5 prefill does not accept cache tensors")

        hidden_states = inputs_embeds
        cos = self.position_gather(self.cos_cached, position_ids)
        sin = self.position_gather(self.sin_cached, position_ids)
        output_cache: list[torch.Tensor] = []
        for index, layer in enumerate(self.layers):
            past_key = flat_cache[index * 2] if self.use_past else None
            past_value = flat_cache[index * 2 + 1] if self.use_past else None
            hidden_states, key, value = layer(
                hidden_states,
                attention_mask,
                cos,
                sin,
                condition,
                past_key,
                past_value,
            )
            output_cache.extend((key, value))
        hidden_states, _ = self.norm(hidden_states, condition)
        if self.use_past:
            return (hidden_states,)
        return tuple(output_cache)


class HoumoPI05PrefillWrapper(nn.Module):
    def __init__(self, language_model: nn.Module):
        super().__init__()
        self.model = HoumoPiGemmaModel(language_model, use_past=False)

    def forward(
        self,
        prefix_embs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return self.model(prefix_embs, attention_mask, position_ids, None)


class HoumoPI05DecodeWrapper(nn.Module):
    def __init__(self, expert_model: nn.Module):
        super().__init__()
        self.model = HoumoPiGemmaModel(expert_model, use_past=True)

    def forward(
        self,
        action_embs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        condition: torch.Tensor,
        *flat_cache: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(action_embs, attention_mask, position_ids, condition, *flat_cache)[0]
