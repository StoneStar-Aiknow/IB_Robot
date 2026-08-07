# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Parity between native GraspGen subgraphs and the export-only reconstruction.

``model_utils.graspgen_export.modeling`` rebuilds the neural parts of GraspGen from
raw checkpoint tensors so ONNX export never imports ``pointnet2_ops``. That
reconstruction is a hand-written re-derivation, and the ONNX verification inside
``export_onnx`` only proves the exported graph matches *that* re-derivation - if the
re-derivation itself diverges from GraspGen, every downstream check still passes and
the OM silently computes the wrong grasps.

This module closes that gap independently. The ``_Native*`` classes below are
transcribed from upstream GraspGen:

* ``AttentionLayer`` / ``FFNLayer`` / ``SinusoidalPosEmb`` / ``repeat_new_axis``
  from ``grasp_gen/models/model_utils.py``
* ``DiffusionNoisePredictionNet`` (``pose_repr="mlp"``, self-attention branch)
  from ``grasp_gen/models/generator.py``
* ``GraspGenDiscriminator``'s sample encoder and prediction head
  from ``grasp_gen/models/discriminator.py``
* the PointNet++ set-abstraction tail (shared MLP then max over the sampled
  neighbourhood) from ``PointNetPlusPlus`` and ``PointnetSAModule``

They use the real ``nn.MultiheadAttention`` and ``F.max_pool2d`` that the export-only
modules claim to be equivalent to. Each test builds a native module, serialises it into
checkpoint-shaped tensors, loads those same tensors through the export path, and feeds
both identical inputs. Nothing is shared between the two implementations except the
weights, so a wrong weight slice, a dropped positional encoding, or a wrong reduction
axis shows up as a numerical difference.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as functional
from torch import nn

from model_utils.graspgen_export.modeling import (
    GENERATOR_SA_PREFIXES,
    GraspGenDenoiser,
    GraspGenDiscriminatorHead,
    PointNetEncoderHead,
    PointNetSAMLP,
)

# float32 CPU tolerance: the two implementations run the same layers in a different
# order (fused value projection versus MultiheadAttention, amax versus max_pool2d), so
# only reassociation noise may differ. Any structural mismatch is orders of magnitude
# larger than this.
_RTOL = 1e-5
_ATOL = 1e-6


def _repeat_new_axis(tensor: torch.Tensor, rep: int, dim: int) -> torch.Tensor:
    """Upstream ``grasp_gen.models.model_utils.repeat_new_axis``."""
    reps = [1] * len(tensor.shape)
    reps.insert(dim, rep)
    return tensor.unsqueeze(dim).repeat(*reps)


class _NativeSinusoidalPosEmb(nn.Module):
    """Upstream ``grasp_gen.models.model_utils.SinusoidalPosEmb``."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :] if len(x.shape) == 1 else x[:, :, None] * emb[None, None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.reshape([batch_size, -1])


class _NativeAttentionLayer(nn.Module):
    """Upstream ``grasp_gen.models.model_utils.AttentionLayer``."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value, query_pos_enc, key_pos_enc, attn_mask=None):
        output, _ = self.attn(query + query_pos_enc, key + key_pos_enc, value, attn_mask=attn_mask)
        return self.norm(query + output)


class _NativeFFNLayer(nn.Module):
    """Upstream ``grasp_gen.models.model_utils.FFNLayer`` with the GELU activation."""

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.ff(x))


class _NativeDenoiser(nn.Module):
    """Upstream ``DiffusionNoisePredictionNet`` for ``pose_repr="mlp"`` with self attention."""

    def __init__(
        self,
        sample_dim: int,
        sample_embed_dim: int,
        time_dim: int,
        observation_embed_dim: int,
        num_layers: int,
        num_heads: int,
        feedforward_dim: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.diffusion_step_encoder = nn.Sequential(
            _NativeSinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.sample_encoder = nn.Sequential(
            nn.Linear(sample_dim, sample_embed_dim),
            nn.ReLU(),
            nn.Linear(sample_embed_dim, sample_embed_dim),
        )
        embed_dim = sample_embed_dim + time_dim + observation_embed_dim
        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, sample_dim),
        )
        self.query_pos_enc = nn.Embedding(1, embed_dim)
        self.self_attention_layers = nn.ModuleList(
            _NativeAttentionLayer(embed_dim, num_heads) for _ in range(num_layers)
        )
        self.ffn_layers = nn.ModuleList(_NativeFFNLayer(embed_dim, feedforward_dim) for _ in range(num_layers))

    def forward(
        self,
        observation_embedding: torch.Tensor,
        timesteps: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        timestep_embedding = self.diffusion_step_encoder(timesteps)
        sample_embedding = self.sample_encoder(sample)
        embed = torch.cat([sample_embedding, timestep_embedding, observation_embedding], axis=-1)
        embed = embed.unsqueeze(0)
        query_pos_enc = _repeat_new_axis(self.query_pos_enc.weight, embed.shape[1], dim=1)
        for index in range(self.num_layers):
            embed = self.self_attention_layers[index](
                embed,
                embed,
                embed + query_pos_enc,
                query_pos_enc,
                query_pos_enc,
            )
            embed = self.ffn_layers[index](embed)
        return self.prediction_head(embed.squeeze(0))


class _NativeDiscriminator(nn.Module):
    """Upstream ``GraspGenDiscriminator`` for ``pose_repr="mlp"``, scoring path only."""

    def __init__(self, sample_dim: int, sample_embed_dim: int, observation_embed_dim: int):
        super().__init__()
        self.sample_encoder = nn.Sequential(
            nn.Linear(sample_dim, sample_embed_dim),
            nn.ReLU(),
            nn.Linear(sample_embed_dim, sample_embed_dim),
        )
        total_input_dim = sample_embed_dim + observation_embed_dim
        self.prediction_head = nn.Sequential(
            nn.Linear(total_input_dim, total_input_dim // 2),
            nn.ReLU(),
            nn.Linear(total_input_dim // 2, total_input_dim // 4),
            nn.ReLU(),
            nn.Linear(total_input_dim // 4, 1),
        )

    def forward(self, grasps_input: torch.Tensor, object_embedding: torch.Tensor):
        sample_embedding = self.sample_encoder(grasps_input)
        logits = self.prediction_head(torch.cat([sample_embedding, object_embedding], axis=-1))
        return logits, logits.sigmoid()


class _NativeSharedMLP(nn.Module):
    """Upstream PointNet++ set-abstraction tail: shared MLP then max over the neighbourhood."""

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

    def forward(self, grouped_features: torch.Tensor) -> torch.Tensor:
        features = self.mlp(grouped_features)
        return functional.max_pool2d(features, kernel_size=[1, features.size(3)]).squeeze(-1)


class _NativeEncoderHead(nn.Module):
    """Last set-abstraction stage plus ``PointNetPlusPlus.prediction_head``."""

    def __init__(self, shared_mlp: _NativeSharedMLP, embedding_dim: int, in_channels: int):
        super().__init__()
        self.shared_mlp = shared_mlp
        self.prediction_head = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.ReLU(),
            nn.Linear(in_channels * 2, in_channels * 2),
            nn.ReLU(),
            nn.Linear(in_channels * 2, embedding_dim),
        )

    def forward(self, grouped_features: torch.Tensor) -> torch.Tensor:
        features = self.shared_mlp(grouped_features)
        return self.prediction_head(features.squeeze(axis=-1))


def _randomise_batchnorm(module: nn.Module) -> None:
    """Give every BatchNorm non-default running statistics.

    Freshly constructed BatchNorm is the identity in eval mode (mean 0, var 1, weight 1,
    bias 0), which would let a reconstruction that ignored the normalisation entirely
    still pass parity.
    """
    for layer in module.modules():
        if isinstance(layer, nn.BatchNorm2d):
            with torch.no_grad():
                layer.running_mean.normal_(0.0, 0.5)
                layer.running_var.uniform_(0.5, 2.0)
                layer.weight.normal_(1.0, 0.2)
                layer.bias.normal_(0.0, 0.2)


def _checkpoint_state(module: nn.Module, prefix: str) -> dict[str, torch.Tensor]:
    """Serialise a native module the way ``load_checkpoint_state`` returns it."""
    return {f"{prefix}{key}": value.detach().clone() for key, value in module.state_dict().items()}


_SAMPLE_DIM = 6
_SAMPLE_EMBED_DIM = 16
_TIME_DIM = 16
_OBSERVATION_EMBED_DIM = 16
_NUM_LAYERS = 3
_NUM_HEADS = 8
_FEEDFORWARD_DIM = 32
_GRASP_BATCH = 5


@pytest.fixture(autouse=True)
def _deterministic_weights():
    torch.manual_seed(20260807)


def test_denoiser_reconstruction_matches_the_native_diffusion_network():
    """The export-only denoiser must reproduce GraspGen's attention stack exactly.

    ``SimplifiedTransformerBlock`` drops the Q/K projections on the argument that the
    sequence length is one, so softmax is identically one. This compares it against the
    real ``nn.MultiheadAttention`` loaded from the same tensors.
    """
    native = _NativeDenoiser(
        sample_dim=_SAMPLE_DIM,
        sample_embed_dim=_SAMPLE_EMBED_DIM,
        time_dim=_TIME_DIM,
        observation_embed_dim=_OBSERVATION_EMBED_DIM,
        num_layers=_NUM_LAYERS,
        num_heads=_NUM_HEADS,
        feedforward_dim=_FEEDFORWARD_DIM,
    ).eval()
    exported = GraspGenDenoiser.from_state_dict(_checkpoint_state(native, "diffusion_head."))

    object_embedding = torch.randn(1, _OBSERVATION_EMBED_DIM)
    sample = torch.randn(_GRASP_BATCH, _SAMPLE_DIM)
    timestep = torch.full((1,), 7.0)

    with torch.no_grad():
        # The native network is fed one row per grasp; the export graph broadcasts a
        # single object embedding and a single timestep, which must be equivalent.
        expected = native(
            object_embedding.expand(_GRASP_BATCH, _OBSERVATION_EMBED_DIM),
            timestep.expand(_GRASP_BATCH),
            sample,
        )
        actual = exported(object_embedding, sample, timestep)

    assert actual.shape == (_GRASP_BATCH, _SAMPLE_DIM)
    torch.testing.assert_close(actual, expected, rtol=_RTOL, atol=_ATOL)


def test_denoiser_reconstruction_is_sensitive_to_the_query_position_encoding():
    """Parity must not be an accident of an unused tensor.

    Upstream passes ``embed + query_pos_enc`` as the attention value, so the query
    position encoding reaches the output. Perturbing it in the checkpoint has to change
    the reconstruction's prediction, otherwise the parity assertion above would still
    hold for a reconstruction that silently ignored it.
    """
    native = _NativeDenoiser(
        sample_dim=_SAMPLE_DIM,
        sample_embed_dim=_SAMPLE_EMBED_DIM,
        time_dim=_TIME_DIM,
        observation_embed_dim=_OBSERVATION_EMBED_DIM,
        num_layers=_NUM_LAYERS,
        num_heads=_NUM_HEADS,
        feedforward_dim=_FEEDFORWARD_DIM,
    ).eval()
    state = _checkpoint_state(native, "diffusion_head.")
    perturbed = dict(state)
    perturbed["diffusion_head.query_pos_enc.weight"] = state["diffusion_head.query_pos_enc.weight"] + 1.0

    object_embedding = torch.randn(1, _OBSERVATION_EMBED_DIM)
    sample = torch.randn(_GRASP_BATCH, _SAMPLE_DIM)
    timestep = torch.full((1,), 7.0)
    with torch.no_grad():
        baseline = GraspGenDenoiser.from_state_dict(state)(object_embedding, sample, timestep)
        shifted = GraspGenDenoiser.from_state_dict(perturbed)(object_embedding, sample, timestep)

    assert not torch.allclose(baseline, shifted, rtol=_RTOL, atol=_ATOL)


def test_discriminator_head_reconstruction_matches_the_native_scorer():
    native = _NativeDiscriminator(_SAMPLE_DIM, _SAMPLE_EMBED_DIM, _OBSERVATION_EMBED_DIM).eval()
    exported = GraspGenDiscriminatorHead.from_state_dict(_checkpoint_state(native, ""))

    object_embedding = torch.randn(1, _OBSERVATION_EMBED_DIM)
    grasp_rt = torch.randn(_GRASP_BATCH, _SAMPLE_DIM)

    with torch.no_grad():
        expected_logits, expected_confidence = native(
            grasp_rt, object_embedding.expand(_GRASP_BATCH, _OBSERVATION_EMBED_DIM)
        )
        logits, confidence = exported(object_embedding, grasp_rt)

    torch.testing.assert_close(logits, expected_logits, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(confidence, expected_confidence, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize(
    ("prefix_index", "in_channels", "hidden_channels", "out_channels", "npoint", "nsample"),
    [
        (0, 3, 8, 12, 16, 6),
        (1, 12, 16, 24, 8, 10),
    ],
)
def test_set_abstraction_mlp_matches_the_native_max_pool(
    prefix_index, in_channels, hidden_channels, out_channels, npoint, nsample
):
    """``amax(dim=-1)`` must equal upstream's ``max_pool2d`` over the neighbourhood axis."""
    native = _NativeSharedMLP(in_channels, hidden_channels, out_channels)
    _randomise_batchnorm(native)
    native = native.eval()
    # The checkpoint stores the shared MLP's layers directly under the SA prefix, so the
    # native module's own ``mlp.`` attribute name is stripped here, not carried across.
    exported = PointNetSAMLP.from_state_dict(
        _checkpoint_state(native.mlp, GENERATOR_SA_PREFIXES[prefix_index]),
        GENERATOR_SA_PREFIXES[prefix_index],
    )

    grouped = torch.randn(1, in_channels, npoint, nsample)
    with torch.no_grad():
        expected = native(grouped)
        actual = exported(grouped)

    assert actual.shape == (1, out_channels, npoint)
    torch.testing.assert_close(actual, expected, rtol=_RTOL, atol=_ATOL)


def test_encoder_head_reconstruction_matches_the_native_object_embedding():
    in_channels, hidden_channels, out_channels, embedding_dim = 24, 32, 48, 20
    shared_mlp = _NativeSharedMLP(in_channels, hidden_channels, out_channels)
    native = _NativeEncoderHead(shared_mlp, embedding_dim, out_channels)
    _randomise_batchnorm(native)
    native = native.eval()

    state = _checkpoint_state(native.shared_mlp.mlp, GENERATOR_SA_PREFIXES[2])
    state.update(_checkpoint_state(native.prediction_head, "object_encoder.prediction_head."))
    exported = PointNetEncoderHead.from_state_dict(state)

    grouped = torch.randn(1, in_channels, 1, 12)
    with torch.no_grad():
        expected = native(grouped)
        actual = exported(grouped)

    assert actual.shape == (1, embedding_dim)
    torch.testing.assert_close(actual, expected, rtol=_RTOL, atol=_ATOL)
