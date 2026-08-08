# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export-only GraspGen modules without CUDA PointNet2 dependencies.

The upstream inference graph mixes CUDA geometry operators, a Python DDPM
loop, and ordinary neural-network layers. This module reconstructs only the
neural-network portions directly from checkpoint tensors so ONNX export does
not import or execute ``pointnet2_ops``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

GENERATOR_SA_PREFIXES = (
    "object_encoder.obj_SA_modules.0.mlps.0.",
    "object_encoder.obj_SA_modules.1.mlps.0.",
    "object_encoder.obj_SA_modules.2.mlps.0.",
)
ENCODER_HEAD_PREFIX = "object_encoder.prediction_head."
DIFFUSION_PREFIX = "diffusion_head."


def load_checkpoint_state(path: str) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint must be a mapping, got {type(checkpoint).__name__}")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint['model'] must be a mapping")
    return {str(key): value.detach().cpu() for key, value in state.items() if torch.is_tensor(value)}


def _substate(state: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    values = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    if not values:
        raise KeyError(f"checkpoint contains no tensors under prefix {prefix!r}")
    return values


class PointNetSAMLP(nn.Module):
    """PointNet++ shared MLP after CPU FPS/ball-query/grouping."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    @classmethod
    def from_state_dict(cls, state: Mapping[str, torch.Tensor], prefix: str) -> PointNetSAMLP:
        values = _substate(state, prefix)
        first_weight = values["0.weight"]
        second_weight = values["3.weight"]
        module = cls(
            in_channels=int(first_weight.shape[1]),
            hidden_channels=int(first_weight.shape[0]),
            out_channels=int(second_weight.shape[0]),
        )
        module.mlp.load_state_dict(values, strict=True)
        return module.eval()

    def forward(self, grouped_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(grouped_features).amax(dim=-1)


class PointNetEncoderHead(nn.Module):
    """Last PointNet++ SA MLP followed by the object embedding MLP."""

    def __init__(self, sa_mlp: PointNetSAMLP, prediction_head: nn.Sequential):
        super().__init__()
        self.sa_mlp = sa_mlp
        self.prediction_head = prediction_head

    @classmethod
    def from_state_dict(cls, state: Mapping[str, torch.Tensor]) -> PointNetEncoderHead:
        sa_mlp = PointNetSAMLP.from_state_dict(state, GENERATOR_SA_PREFIXES[2])
        head_state = _substate(state, ENCODER_HEAD_PREFIX)
        first_weight = head_state["0.weight"]
        second_weight = head_state["2.weight"]
        third_weight = head_state["4.weight"]
        prediction_head = nn.Sequential(
            nn.Linear(int(first_weight.shape[1]), int(first_weight.shape[0])),
            nn.ReLU(inplace=False),
            nn.Linear(int(second_weight.shape[1]), int(second_weight.shape[0])),
            nn.ReLU(inplace=False),
            nn.Linear(int(third_weight.shape[1]), int(third_weight.shape[0])),
        )
        prediction_head.load_state_dict(head_state, strict=True)
        return cls(sa_mlp, prediction_head).eval()

    def forward(self, grouped_features: torch.Tensor) -> torch.Tensor:
        features = self.sa_mlp(grouped_features).squeeze(-1)
        return self.prediction_head(features)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        half_dim = dim // 2
        scale = math.log(10000.0) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -scale)
        self.frequencies: torch.Tensor
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = timestep.reshape(-1)
        embedding = timestep[:, None] * self.frequencies[None, :]
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1)


class SimplifiedTransformerBlock(nn.Module):
    """Exact GraspGen self-attention block for sequence length one.

    GraspGen presents one token per grasp to ``MultiheadAttention``. Softmax is
    therefore exactly one, making Q/K projections dead computation. The value
    path and both residual LayerNorms are retained without approximation.
    """

    def __init__(self, embed_dim: int, feedforward_dim: int):
        super().__init__()
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.ff1 = nn.Linear(embed_dim, feedforward_dim)
        self.ff2 = nn.Linear(feedforward_dim, embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, torch.Tensor],
        attention_prefix: str,
        ffn_prefix: str,
    ) -> SimplifiedTransformerBlock:
        in_proj_weight = state[f"{attention_prefix}attn.in_proj_weight"]
        in_proj_bias = state[f"{attention_prefix}attn.in_proj_bias"]
        embed_dim = int(in_proj_weight.shape[1])
        ff1_weight = state[f"{ffn_prefix}ff.0.weight"]
        module = cls(embed_dim=embed_dim, feedforward_dim=int(ff1_weight.shape[0]))

        module.value_projection.weight.data.copy_(in_proj_weight[2 * embed_dim :])
        module.value_projection.bias.data.copy_(in_proj_bias[2 * embed_dim :])
        module.output_projection.weight.data.copy_(state[f"{attention_prefix}attn.out_proj.weight"])
        module.output_projection.bias.data.copy_(state[f"{attention_prefix}attn.out_proj.bias"])
        module.attention_norm.weight.data.copy_(state[f"{attention_prefix}norm.weight"])
        module.attention_norm.bias.data.copy_(state[f"{attention_prefix}norm.bias"])
        module.ff1.weight.data.copy_(ff1_weight)
        module.ff1.bias.data.copy_(state[f"{ffn_prefix}ff.0.bias"])
        module.ff2.weight.data.copy_(state[f"{ffn_prefix}ff.2.weight"])
        module.ff2.bias.data.copy_(state[f"{ffn_prefix}ff.2.bias"])
        module.ffn_norm.weight.data.copy_(state[f"{ffn_prefix}norm.weight"])
        module.ffn_norm.bias.data.copy_(state[f"{ffn_prefix}norm.bias"])
        return module.eval()

    def forward(self, embedding: torch.Tensor, query_position: torch.Tensor) -> torch.Tensor:
        value = embedding + query_position
        attention_output = self.output_projection(self.value_projection(value))
        embedding = self.attention_norm(embedding + attention_output)
        feedforward = self.ff2(F.gelu(self.ff1(embedding)))
        return self.ffn_norm(embedding + feedforward)


class GraspGenDenoiser(nn.Module):
    """Noise prediction network used once per DDPM step."""

    def __init__(
        self,
        diffusion_step_encoder: nn.Sequential,
        sample_encoder: nn.Sequential,
        query_position: torch.Tensor,
        blocks: list[SimplifiedTransformerBlock],
        prediction_head: nn.Sequential,
    ):
        super().__init__()
        self.diffusion_step_encoder = diffusion_step_encoder
        self.sample_encoder = sample_encoder
        self.query_position = nn.Parameter(query_position.clone(), requires_grad=False)
        self.blocks = nn.ModuleList(blocks)
        self.prediction_head = prediction_head

    @classmethod
    def from_state_dict(cls, state: Mapping[str, torch.Tensor]) -> GraspGenDenoiser:
        prefix = DIFFUSION_PREFIX
        time_state = _substate(state, f"{prefix}diffusion_step_encoder.")
        time_hidden = int(time_state["1.weight"].shape[0])
        time_dim = int(time_state["1.weight"].shape[1])
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_hidden),
            nn.Mish(),
            nn.Linear(time_hidden, int(time_state["3.weight"].shape[0])),
        )
        diffusion_step_encoder.load_state_dict(time_state, strict=True)

        sample_state = _substate(state, f"{prefix}sample_encoder.")
        sample_encoder = nn.Sequential(
            nn.Linear(int(sample_state["0.weight"].shape[1]), int(sample_state["0.weight"].shape[0])),
            nn.ReLU(inplace=False),
            nn.Linear(int(sample_state["2.weight"].shape[1]), int(sample_state["2.weight"].shape[0])),
        )
        sample_encoder.load_state_dict(sample_state, strict=True)

        layer_indices = sorted(
            {int(key.split(".")[2]) for key in state if key.startswith(f"{prefix}self_attention_layers.")}
        )
        blocks = [
            SimplifiedTransformerBlock.from_state_dict(
                state,
                attention_prefix=f"{prefix}self_attention_layers.{index}.",
                ffn_prefix=f"{prefix}ffn_layers.{index}.",
            )
            for index in layer_indices
        ]

        prediction_state = _substate(state, f"{prefix}prediction_head.")
        prediction_head = nn.Sequential(
            nn.Linear(
                int(prediction_state["0.weight"].shape[1]),
                int(prediction_state["0.weight"].shape[0]),
            ),
            nn.ReLU(inplace=False),
            nn.Linear(
                int(prediction_state["2.weight"].shape[1]),
                int(prediction_state["2.weight"].shape[0]),
            ),
            nn.ReLU(inplace=False),
            nn.Linear(
                int(prediction_state["4.weight"].shape[1]),
                int(prediction_state["4.weight"].shape[0]),
            ),
        )
        prediction_head.load_state_dict(prediction_state, strict=True)
        query_position = state[f"{prefix}query_pos_enc.weight"]

        return cls(
            diffusion_step_encoder=diffusion_step_encoder,
            sample_encoder=sample_encoder,
            query_position=query_position,
            blocks=blocks,
            prediction_head=prediction_head,
        ).eval()

    def forward(
        self,
        object_embedding: torch.Tensor,
        sample: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        sample_embedding = self.sample_encoder(sample)
        batch_size = sample_embedding.shape[0]
        timestep_embedding = self.diffusion_step_encoder(timestep)
        timestep_embedding = timestep_embedding.expand(batch_size, timestep_embedding.shape[1])
        object_embedding = object_embedding.expand(batch_size, object_embedding.shape[1])
        embedding = torch.cat((sample_embedding, timestep_embedding, object_embedding), dim=-1)
        for block in self.blocks:
            embedding = block(embedding, self.query_position)
        return self.prediction_head(embedding)


class GraspGenDiscriminatorHead(nn.Module):
    """Grasp scoring MLP after host-side SO(3) conversion."""

    def __init__(self, sample_encoder: nn.Sequential, prediction_head: nn.Sequential):
        super().__init__()
        self.sample_encoder = sample_encoder
        self.prediction_head = prediction_head

    @classmethod
    def from_state_dict(cls, state: Mapping[str, torch.Tensor]) -> GraspGenDiscriminatorHead:
        sample_state = _substate(state, "sample_encoder.")
        sample_encoder = nn.Sequential(
            nn.Linear(int(sample_state["0.weight"].shape[1]), int(sample_state["0.weight"].shape[0])),
            nn.ReLU(inplace=False),
            nn.Linear(int(sample_state["2.weight"].shape[1]), int(sample_state["2.weight"].shape[0])),
        )
        sample_encoder.load_state_dict(sample_state, strict=True)

        prediction_state = _substate(state, "prediction_head.")
        prediction_head = nn.Sequential(
            nn.Linear(
                int(prediction_state["0.weight"].shape[1]),
                int(prediction_state["0.weight"].shape[0]),
            ),
            nn.ReLU(inplace=False),
            nn.Linear(
                int(prediction_state["2.weight"].shape[1]),
                int(prediction_state["2.weight"].shape[0]),
            ),
            nn.ReLU(inplace=False),
            nn.Linear(
                int(prediction_state["4.weight"].shape[1]),
                int(prediction_state["4.weight"].shape[0]),
            ),
        )
        prediction_head.load_state_dict(prediction_state, strict=True)
        return cls(sample_encoder, prediction_head).eval()

    def forward(
        self,
        object_embedding: torch.Tensor,
        grasp_rt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sample_embedding = self.sample_encoder(grasp_rt)
        object_embedding = object_embedding.expand(sample_embedding.shape[0], object_embedding.shape[1])
        logits = self.prediction_head(torch.cat((sample_embedding, object_embedding), dim=-1))
        return logits, logits.sigmoid()


def build_pointnet_segments(
    state: Mapping[str, torch.Tensor],
) -> tuple[PointNetSAMLP, PointNetSAMLP, PointNetEncoderHead]:
    return (
        PointNetSAMLP.from_state_dict(state, GENERATOR_SA_PREFIXES[0]),
        PointNetSAMLP.from_state_dict(state, GENERATOR_SA_PREFIXES[1]),
        PointNetEncoderHead.from_state_dict(state),
    )
