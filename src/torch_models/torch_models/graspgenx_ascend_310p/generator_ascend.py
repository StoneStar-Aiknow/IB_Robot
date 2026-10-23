#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ascend inference variant of the GraspGen generator.

The network definition, checkpoint layout, training path, and public output
contract remain in ``generator.py``. This subclass owns inference-only changes
for Ascend, where a 400-grasp request issues about 5500 kernels for 100 ms of
device time: operator dispatch, not arithmetic, is what the diffusion loop
costs. So the work here removes operators rather than FLOPs.

* The ``cat_attn`` denoising head is rewritten for the deployed configuration:
  attention at sequence length 1 collapses into one affine map, the constant
  position encoding folds into its bias, residual adds fuse into their
  LayerNorms, and the timestep embedding is computed once per timestep instead
  of once per grasp. 60 operators per step become about 30.
* The likelihood uses the Normal log-density in closed form with per-timestep
  constants prepared once per request, instead of constructing a distribution
  twice per step.
* The pose conversion skips the dead intermediates of ``so3_exp_map``.
* ``record_history`` makes the per-step grasp history optional for deployments
  that do not read it.

Every rewrite is an identity in exact arithmetic, but folding matrices and
reassociating reductions move results within float32 rounding: outputs are no
longer bitwise identical to ``generator.py``, which the baseline comparison
reports and ``ascend/tests/test_graspgenx_baseline.py`` pins with tolerances.
History remains a CPU tensor to preserve the original contract.
PTV3 placement is independent and is selected by the caller's model assembly.
"""

import math
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import KDTree

try:  # The module also imports on hosts without the Ascend stack.
    import torch_npu
except ImportError:  # pragma: no cover - exercised only off-device
    torch_npu = None

from graspgenx.metrics import (
    compute_metrics_given_two_sets_of_xgripper_poses,
    compute_recall,
)
from graspgenx.models.generator import GraspGenGenerator
from graspgenx.models.model_utils import (
    convert_to_ptv3_pc_format,
    offset2batch,
)
from graspgenx.utils.transformations import (
    axis_angle_to_matrix,
)
from graspgenx.utils.transformations import (
    rt_to_matrix as _reference_rt_to_matrix,
)

from .diffusion_scheduler import AscendDDPMStep as _AscendDDPMStep
from .diffusion_scheduler import host_noise as _host_noise

_LOG_SQRT_TWO_PI = math.log(math.sqrt(2 * math.pi))
_HOMOGENEOUS_ROWS: dict = {}


def _homogeneous_row(count, dtype, device):
    """Cache the constant bottom row of a batch of 4x4 transforms.

    Inference callers reuse one device tensor across requests; autograd callers
    build their own, so a tensor created under inference mode never reaches a
    graph that needs to differentiate it.
    """
    key = (count, dtype, str(device))
    row = _HOMOGENEOUS_ROWS.get(key)
    if row is not None and not torch.is_grad_enabled():
        return row
    row = torch.zeros((count, 1, 4), dtype=dtype, device=device)
    row[:, :, 3] = 1.0
    if not torch.is_grad_enabled():
        _HOMOGENEOUS_ROWS[key] = row
    return row


def rt_to_matrix(input, grasp_repr: str, kappa=None):
    """Assemble the pose matrix without the discarded ``so3_exp_map`` work.

    ``so3.so3_exp_map`` keeps only the first element of ``_so3_exp_map``, so the
    norm, clamp, hat operator and its batched square are computed and thrown
    away -- roughly ten extra kernels per call. The rotation itself still comes
    from ``axis_angle_to_matrix`` exactly as in the reference, and the matrix is
    assembled by concatenation instead of eye/repeat plus indexed writes.
    Other grasp representations keep the reference implementation.
    """
    if grasp_repr != "r3_so3":
        return _reference_rt_to_matrix(input, grasp_repr, kappa)
    rotation = axis_angle_to_matrix(input[:, 3:] * torch.pi)
    translation = input[:, :3]
    if kappa is not None:
        translation = translation * (1.0 / kappa)
    return torch.cat(
        [
            torch.cat([rotation, translation.unsqueeze(-1)], dim=-1),
            _homogeneous_row(input.shape[0], rotation.dtype, rotation.device),
        ],
        dim=1,
    )


@contextmanager
def _single_op_jit_compile():
    """Re-enable single-operator JIT for the denoising head only.

    ``PointTransformerV3Ascend`` sets ``jit_compile=False`` process-wide, which
    its own custom operators require. Under that flag the aten ``layer_norm``
    and ``gelu`` kernels cost a flat ~1 s per call each on Ascend310P1 /
    CANN 8.1.RC1, independent of shape and of how much they compute, while
    ``relu`` and elementwise ops stay at ~0.2 ms. The head runs six LayerNorms
    and three GELUs per diffusion step, so 20 steps spent ~180 s in those nine
    calls alone.

    Scoping the flag back to its JIT default keeps aten ``layer_norm`` and
    ``gelu`` usable in the head -- the fused residual norm replaces six of the
    LayerNorms, but the FFN's GELU has no float32 fused equivalent -- and
    restores PTV3's mode on exit. The flag is process-global, so this must not
    wrap any PTV3 call.
    """
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        yield
        return
    disabled = torch.npu.is_jit_compile_false()
    if disabled:
        torch.npu.set_compile_mode(jit_compile=True)
    try:
        yield
    finally:
        if disabled:
            torch.npu.set_compile_mode(jit_compile=False)


_FRACTAL_NZ = 29


def _pack_matmul_format(module):
    """Store this subtree's matmul weights in the layout the cube unit wants.

    Ascend310P1 has no float32 cube, so a float32 ``addmm`` casts both operands
    to float16 and converts them to FRACTAL_NZ on **every call**: for the head's
    2048x2048 projection that is 60 wasted conversions of the same weight per
    request, 11 ms of device time and 120 dispatches. Converting the layout once
    removes the per-call ``TransData`` for the weight and leaves the result
    bitwise identical -- the cast to float16 operands and the float32
    accumulation are unchanged, only the memory layout is prepared in advance.

    The weights keep their float32 dtype, shape and values; only their private
    device format changes, and reading them back to the host restores ND.
    """
    if torch_npu is None:
        return module
    with torch.inference_mode(False), torch.no_grad():
        for layer in module.modules():
            weight = getattr(layer, "weight", None)
            if not isinstance(layer, torch.nn.Linear) or weight is None:
                continue
            if weight.device.type != "npu" or weight.dim() != 2:
                continue
            if torch_npu.get_npu_format(weight) == _FRACTAL_NZ:
                continue
            # Rebind rather than mutate: an in-place update of a parameter that
            # still requires grad is not allowed under inference mode.
            weight.data = torch_npu.npu_format_cast(
                weight.data.contiguous(),
                _FRACTAL_NZ,
            )
    return module


def _fused_add_norm(residual, update, norm):
    """``norm(residual + update)`` as one operator where CANN provides it.

    ``npu_add_layer_norm`` accepts float32 and stays within 1e-6 of the eager
    pair at the head's shapes. The head issues six of these per diffusion step,
    and a request is dominated by operator dispatch, not by arithmetic.
    """
    if torch_npu is not None and residual.device.type == "npu":
        return torch_npu.npu_add_layer_norm(
            residual,
            update,
            norm.weight,
            norm.bias,
            epsilon=norm.eps,
        )[0]
    return norm(residual + update)


class _CollapsedSelfAttention(torch.nn.Module):
    """``AttentionLayer`` reduced to the linear map it computes at length 1.

    The ``cat_attn`` head stacks every grasp into the batch dimension and keeps
    the sequence dimension at 1, so ``softmax(q k^T / sqrt(d))`` is the scalar 1
    for every head and the query and key projections cannot influence the
    result. Attention therefore reduces exactly to ``out_proj(W_v v + b_v)``,
    which folds into the single projection ``W_o W_v v + (W_o b_v + b_o)``.
    The caller's constant position encoding folds into the bias as well, since
    ``W (x + p) + b`` is ``W x + (W p + b)``.

    Per diffusion step this removes two thirds of each 1536x4608 input
    projection, one of the two remaining 1536x1536 products, and the position
    add. Folding reassociates the products, so results move within float32
    rounding of the reference.
    """

    def __init__(self, layer, position):
        super().__init__()
        attention = layer.attn
        if not (
            attention._qkv_same_embed_dim
            and attention.in_proj_bias is not None
            and attention.bias_k is None
            and attention.bias_v is None
            and not attention.add_zero_attn
        ):
            raise ValueError("unsupported MultiheadAttention layout for collapse")
        if attention.batch_first:
            raise ValueError("collapse expects sequence-first attention inputs")
        dim = attention.embed_dim
        output_projection = attention.out_proj
        with torch.inference_mode(False), torch.no_grad():
            weight = output_projection.weight.detach() @ attention.in_proj_weight.detach()[2 * dim :]
            bias = (
                output_projection.weight.detach() @ attention.in_proj_bias.detach()[2 * dim :]
                + output_projection.bias.detach()
                + F.linear(position.detach().reshape(dim), weight)
            )
        if torch_npu is not None and weight.device.type == "npu":
            weight = torch_npu.npu_format_cast(weight.contiguous(), _FRACTAL_NZ)
        self.register_buffer("weight", weight, persistent=False)
        self.register_buffer("bias", bias, persistent=False)
        self.norm = layer.norm

    def forward(self, features):
        return _fused_add_norm(features, F.linear(features, self.weight, self.bias), self.norm)


class _FusedResidualFFN(torch.nn.Module):
    """``FFNLayer`` with its residual add and LayerNorm fused."""

    def __init__(self, layer):
        super().__init__()
        self.ff = layer.ff
        self.norm = layer.norm

    def forward(self, features):
        return _fused_add_norm(features, self.ff(features), self.norm)


class _AscendDiffusionHead(torch.nn.Module):
    """The deployed denoising head with only the per-step work it needs.

    ``DiffusionNoisePredictionNet.forward`` covers every configuration the
    checkpoints may carry; this reimplements the one the release config selects
    (``pose_repr=mlp``, ``attention=cat_attn`` without cross attention) and is
    installed only for it. Two things leave the per-step path:

    * The timestep embedding depends on nothing else, and the twenty eval
      timesteps repeat every request, so it is computed once per timestep at
      batch 1 instead of once per step across every grasp. The reference
      expands a scalar timestep across the batch, so all rows were identical.
    * The query position encoding is constant and folds into the attention
      bias, and ``repeat_new_axis`` materialized it per step.

    Together with the collapsed attention and the fused residual norms this
    takes the head from 60 operators per step to about 30. At 400 grasps the
    request issues roughly 5500 kernels for 100 ms of device time, so operator
    dispatch is what the head costs.
    """

    def __init__(self, head):
        super().__init__()
        self.head = head
        position = head.query_pos_enc.weight
        self.attention = torch.nn.ModuleList(
            _CollapsedSelfAttention(layer, position) for layer in head.self_attention_layers
        )
        self.ffn = torch.nn.ModuleList(_FusedResidualFFN(layer) for layer in head.ffn_layers)
        self.step_embeddings: dict = {}
        _pack_matmul_format(self)

    @staticmethod
    def supports(head) -> bool:
        return (
            getattr(head, "pose_repr", None) == "mlp"
            and isinstance(getattr(head, "attention", None), str)
            and head.attention.find("attn") >= 0
            and head.attention.find("cross") < 0
            and getattr(head, "self_attention_layers", None) is not None
            and getattr(head, "ffn_layers", None) is not None
            and getattr(head, "cross_attention_layers", None) is None
        )

    def _step_embedding(self, timesteps, device):
        def index():
            # A zero-dimensional timestep is the inference loop's own tensor;
            # rebuilding it on a cache hit costs two dispatches per step.
            if torch.is_tensor(timesteps) and timesteps.dim() == 0:
                return timesteps[None].to(device)
            return torch.as_tensor([timesteps], device=device)

        if torch.is_grad_enabled():
            # Keep training and autograd callers off the cached inference value.
            return self.head.diffusion_step_encoder(index())
        key = (int(timesteps), str(device))
        embedding = self.step_embeddings.get(key)
        if embedding is None:
            embedding = self.head.diffusion_step_encoder(index()).detach()
            self.step_embeddings[key] = embedding
        return embedding

    def forward(self, observation_embedding, timesteps, sample=None):
        if sample is None:
            raise RuntimeError("the mlp pose representation requires a sample")
        step = self._step_embedding(timesteps, observation_embedding.device)
        features = torch.cat(
            [
                self.head.sample_encoder(sample),
                step.expand(observation_embedding.shape[0], -1),
                observation_embedding,
            ],
            dim=-1,
        )
        # strict: a length mismatch here would silently run fewer transformer
        # blocks than the checkpoint carries, and still return a plausible
        # prediction.
        for attention, ffn in zip(self.attention, self.ffn, strict=True):
            features = ffn(attention(features))
        return self.head.prediction_head(features)


class GraspGenGeneratorAscend(GraspGenGenerator):
    """Reference-compatible generator with the default Ascend inference path."""

    # Per-step history costs one pose conversion plus one device-to-host write
    # per diffusion step -- about 85 ms of the 400-grasp request on Ascend310P1
    # -- and only ``scripts/inference_xgrasp.py`` visualization reads it. The
    # default records it, so the frozen-baseline contract is unchanged; a
    # deployment that does not read it sets this to False and then receives
    # ``grasps_per_iteration`` with zero recorded iterations. Nothing in the
    # loop reads the history back, so the grasps, likelihood and confidence are
    # identical either way.
    record_history = True

    def _install_inference_head(self):
        """Rewrite the denoising head once, after the checkpoint is loaded.

        Construction predates ``load_state_dict``, so the folded projections are
        built on first inference instead of in ``__init__``. Configurations the
        rewrite does not cover keep the reference head untouched.
        """
        if getattr(self, "_inference_head_ready", False):
            return
        head = self.diffusion_head
        # Benchmarks may already have wrapped the head for stage timing.
        while not _AscendDiffusionHead.supports(head) and hasattr(head, "module"):
            head = head.module
        if _AscendDiffusionHead.supports(head):
            self.diffusion_head = _AscendDiffusionHead(head)
        self._inference_head_ready = True

    def _install_inference_schedulers(self):
        """Give each scheduler the fast step, unless a caller supplied one.

        A benchmark that replays frozen noise assigns ``step`` on the instance;
        that assignment wins, so determinism harnesses keep their own step.
        """
        if getattr(self, "_inference_schedulers_ready", False):
            return
        for name in ("noise_scheduler", "noise_scheduler_pos", "noise_scheduler_rot"):
            scheduler = getattr(self, name, None)
            if scheduler is None or "step" in vars(scheduler):
                continue
            fast = _AscendDDPMStep(scheduler)
            if fast.supported():
                scheduler.step = fast
        self._inference_schedulers_ready = True

    # An offline-model diffusion loop (graspgenx.models.diffusion_loop_om), set
    # by the deployment that built one. The eager loop runs when it is absent or
    # does not match the request.
    om_loop = None
    # Overridden by determinism harnesses to replay frozen noise; the default
    # draws on the host because the device generator is unusable (see
    # _host_noise). The signature is (steps, rows, components, dtype, device).
    om_noise = None
    # The per-step traces of the last offline-model run, for baseline comparison.
    om_traces: dict = {}

    def _om_loop_applies(self, rows: int, steps: int, device) -> bool:
        loop = self.om_loop
        return (
            loop is not None
            and self.compositional_schedular
            and self.pose_repr == "mlp"
            and self.grasp_repr == "r3_so3"
            and self.output_dim % 2 == 0
            and loop.matches(rows, steps)
            and torch.device(getattr(loop, "device", device)).type == device.type
        )

    def _run_om_loop(self, obs_embedding, noisy_init, timesteps, device):
        """Run the whole loop as one offline model, or return None.

        The graph carries the head, both schedulers and the likelihood, so the
        noise for every step is needed before it starts: the harness supplies
        the frozen streams, a deployment draws them on the host.
        """
        rows, half = noisy_init.shape[0], self.output_dim // 2
        if not self._om_loop_applies(rows, len(timesteps), device):
            return None
        draw = self.om_noise or (
            lambda steps, count, components, dtype, target: tuple(
                _host_noise([steps, count, components], dtype, target) for _ in range(2)
            )
        )
        position_noise, rotation_noise = draw(
            len(timesteps) - 1,
            rows,
            half,
            noisy_init.dtype,
            device,
        )
        grasps, likelihood, predictions, latents = self.om_loop.run(
            obs_embedding,
            noisy_init,
            position_noise,
            rotation_noise,
        )
        self.om_traces = {"noise_predictions": predictions, "latents": latents}
        return grasps, likelihood, latents

    def _prepare_likelihood_scales(self, betas, device):
        """Per-timestep likelihood constants: one device scalar, one host float.

        Validating the fixed scheduler constants once keeps the device-to-host
        boolean checks out of every step, and the closed form then needs four
        elementwise operators instead of constructing a distribution. Building
        the twenty pairs costs 160 dispatches, so the result is cached per
        scheduler tensor: betas do not change between requests, and the cache
        keys on the tensor's identity and the device it was prepared for.
        """
        cache = self.__dict__.setdefault("_likelihood_scales", {})
        key = (id(betas), str(device))
        cached = cache.get(key)
        if cached is not None and cached[0] is betas:
            return cached[1]
        host_betas = betas.detach().cpu()
        if not bool(torch.isfinite(host_betas).all() and (host_betas > 0).all()):
            raise ValueError("Likelihood scheduler betas must be finite and positive")
        variances = betas.to(device=device)
        scales = [
            (
                -2.0 * variances[index],
                -0.5 * math.log(float(host_betas[index])) - _LOG_SQRT_TWO_PI,
            )
            for index in range(len(host_betas))
        ]
        # Hold the tensor so the identity cannot be recycled by a new one.
        cache[key] = (betas, scales)
        return scales

    def _inference_likelihood(self, sample, mean, timestep, scales):
        negative_double_variance, offset = scales[timestep]
        difference = sample - mean
        squared = (difference * difference).sum(-1, keepdim=True)
        return squared / negative_double_variance + sample.shape[-1] * offset

    def _accumulate_inference_likelihood(self, likelihood, position, rotation=None):
        likelihood += position if rotation is None else position + rotation

    def _allocate_inference_history(self, shape):
        return torch.zeros(shape, device="cpu")

    def _write_inference_history(self, history, timestep, grasps):
        history[:, timestep, :, ::] = grasps

    def forward_inference(self, data, return_metrics=False):
        """Reference inference flow with precomputed likelihood scales."""
        self._install_inference_head()
        self._install_inference_schedulers()
        device = data["points"].device

        num_objects_in_batch = len(data["points"])
        if "grasps" in data:
            if isinstance(data["grasps"][0], list):
                data["grasps"][0] = np.array(data["grasps"][0])
            num_grasps_per_batch = data["grasps"][0].shape[0]
        else:
            num_grasps_per_batch = self.num_grasps_per_object
            return_metrics = False

        batch_size = data["points"].shape[0] * num_grasps_per_batch
        depth = data["points"]
        num_points = depth.shape[-2]
        depth = depth.reshape([-1, num_points, 3]).to(device)
        grasps_init_size = [num_objects_in_batch, num_grasps_per_batch, 4, 4]

        if self.kappa is not None:
            depth = self.kappa * depth
        if self.object_backbone in ("ptv3", "ptv3vanilla"):
            depth = convert_to_ptv3_pc_format(depth, grid_size=self.grid_size)

        grasps_per_iteration = self._allocate_inference_history(
            [
                num_objects_in_batch,
                self.num_diffusion_iters_eval if self.record_history else 0,
                num_grasps_per_batch,
                4,
                4,
            ],
        )

        with torch.no_grad():
            if "initial_noise" in data:
                noisy_init = data["initial_noise"].to(device=device, dtype=data["points"].dtype)
                noisy_init = noisy_init.reshape(batch_size, self.output_dim).clone()
            else:
                # The device generator terminates the process here; see
                # _host_noise.
                noisy_init = _host_noise(
                    [batch_size, self.output_dim],
                    data["points"].dtype,
                    device,
                )
            noisy_grasps = noisy_init
            likelihood = torch.zeros((batch_size, 1), device=device)

            offset = torch.tensor([num_grasps_per_batch]).repeat(num_objects_in_batch).cumsum(dim=0).to(device)
            mask_batch = offset2batch(offset)

            if self.pose_repr == "mlp":
                object_embedding = self.object_encoder(depth)
                object_embedding = object_embedding[mask_batch]

            if self.gripper_backbone == "onehot":
                gripper_embedding = self.gripper_encoder(data["onehot"])
            elif self.gripper_backbone == "gripper_type":
                gripper_type = torch.eye(3, dtype=torch.float32, device=object_embedding.device)[
                    torch.tensor(
                        data["gripper_type"],
                        dtype=torch.long,
                        device=object_embedding.device,
                    )
                ]
                gripper_embedding = self.gripper_encoder(gripper_type)
            elif self.gripper_backbone == "z_offset":
                gripper_embedding = self.gripper_encoder(data["z_offset"])
            elif self.gripper_backbone == "none":
                gripper_embedding = torch.zeros(
                    (num_objects_in_batch, self.num_gripper_dim),
                    dtype=object_embedding.dtype,
                    device=object_embedding.device,
                )
            elif self.gripper_backbone == "sweep_volume":
                gripper_embedding = self.gripper_encoder(data["sweep_volume"])
            elif self.gripper_backbone == "sweep_volume_v2":
                gripper_embedding = self.gripper_encoder(data["sweep_volume_open_and_mid"])
            elif self.gripper_backbone == "gripper_type+sweep_volume_v2":
                gripper_type = torch.eye(3, dtype=torch.float32, device=object_embedding.device)[
                    torch.tensor(
                        data["gripper_type"],
                        dtype=torch.long,
                        device=object_embedding.device,
                    )
                ]
                gripper_embedding = self.gripper_encoder(
                    torch.concat([data["sweep_volume_open_and_mid"], gripper_type], dim=-1)
                )
            elif self.gripper_backbone == "pointcloud":
                gripper_embedding = torch.concat(
                    [
                        self.gripper_encoder(data["gripper_open_ptc"]),
                        self.gripper_encoder(data["gripper_close_ptc"]),
                    ],
                    dim=-1,
                )
            elif self.gripper_backbone == "selected_pointcloud":
                gripper_embedding = self.gripper_encoder(data["gripper_selected_ptc"])
            elif self.gripper_backbone == "selected_pointcloud_v2":
                gripper_embedding = self.gripper_mlp_encoder(
                    torch.concat(
                        [
                            self.gripper_ptc_encoder(data["gripper_selected_open_ptc"]),
                            self.gripper_ptc_encoder(data["gripper_selected_mid_ptc"]),
                            self.gripper_ptc_encoder(data["gripper_selected_close_ptc"]),
                        ],
                        dim=-1,
                    )
                )
            elif self.gripper_backbone == "selected_pointcloud_v3":
                gripper_embedding = torch.concat(
                    [
                        self.gripper_encoder(data["gripper_selected_open_ptc"]),
                        self.gripper_encoder(data["gripper_selected_close_ptc"]),
                    ],
                    dim=-1,
                )
            elif self.gripper_backbone == "volume_tsdf":
                gripper_3d_embedding = self.gripper_backbone_3dconv(data["gripper_vol_tsdf"]).squeeze(-2)
                gripper_2d_embedding = self.gripper_backbone_2dconv(gripper_3d_embedding).reshape(
                    num_objects_in_batch, -1
                )
                gripper_embedding = self.gripper_backbone_mlp(gripper_2d_embedding)
            elif self.gripper_backbone == "pointnet_repr":
                gripper_embedding = self.gripper_encoder(data["gripper_pointnet_repr"])
            else:
                raise NotImplementedError(f"Gripper Backbone {self.gripper_backbone} not implemented")

            gripper_embedding = gripper_embedding[mask_batch]
            obs_embedding = torch.concat([object_embedding, gripper_embedding], dim=-1)

            if self.compositional_schedular:
                self.noise_scheduler_pos.set_timesteps(self.num_diffusion_iters_eval)
                timesteps = self.noise_scheduler_pos.timesteps
                self.noise_scheduler_rot.set_timesteps(self.num_diffusion_iters_eval)
                scales_pos = self._prepare_likelihood_scales(self.noise_scheduler_pos.betas, device)
                scales_rot = self._prepare_likelihood_scales(self.noise_scheduler_rot.betas, device)
            else:
                self.noise_scheduler.set_timesteps(self.num_diffusion_iters_eval)
                timesteps = self.noise_scheduler.timesteps
                scales = self._prepare_likelihood_scales(self.noise_scheduler.betas, device)

            # Truthy om_traces means the last call took the offline path; the
            # harness reads it to recover the per-step traces.
            self.om_traces = {}
            offline = self._run_om_loop(obs_embedding, noisy_init, timesteps, device)
            if offline is not None:
                noisy_grasps, likelihood, latents = offline
                pred_grasps = rt_to_matrix(noisy_grasps, self.grasp_repr, self.kappa)
                if self.record_history:
                    # The eager loop indexes history by timestep value, so the
                    # first iteration lands last; keep that layout.
                    poses = rt_to_matrix(
                        latents.reshape(-1, self.output_dim),
                        self.grasp_repr,
                        self.kappa,
                    ).reshape(len(timesteps), *grasps_init_size)
                    for index, k in enumerate(timesteps):
                        self._write_inference_history(
                            grasps_per_iteration,
                            int(k),
                            poses[index],
                        )
            timesteps = [] if offline is not None else timesteps

            for k in timesteps:
                # The scheduler hands out zero-dimensional tensors, so comparing
                # the timestep in Python saves two dispatches per step.
                step = int(k)
                samples = noisy_grasps if self.pose_repr == "mlp" else None
                if self.pose_repr in ["grasp_cloud", "grasp_cloud_pe", "pc_feature"]:
                    ctrl_pts = self.ctr_pts.to(device=device)
                    noisy_grasps_mat = rt_to_matrix(noisy_grasps, self.grasp_repr, self.kappa)
                    grasp_pc = (noisy_grasps_mat @ ctrl_pts).transpose(-2, -1)[..., :3]

                if self.pose_repr == "pc_feature":
                    depth_full = depth[mask_batch]
                    depth_full = torch.cat([depth_full, grasp_pc], dim=1)
                    pc_feature = torch.cat(
                        [
                            torch.zeros([batch_size, num_points, 1]),
                            torch.ones([batch_size, grasp_pc.shape[1], 1]),
                        ],
                        dim=1,
                    ).to(device=device)
                    object_embedding = torch.cat([depth_full, pc_feature], dim=-1)
                    object_embedding = self.object_encoder(object_embedding)

                with _single_op_jit_compile():
                    noise_pred = self.diffusion_head(obs_embedding, k, samples)
                if self.compositional_schedular:
                    res_pos = self.noise_scheduler_pos.step(
                        model_output=noise_pred[..., :3],
                        timestep=k,
                        sample=noisy_grasps[..., :3],
                    )
                    res_rot = self.noise_scheduler_rot.step(
                        model_output=noise_pred[..., 3 : self.output_dim],
                        timestep=k,
                        sample=noisy_grasps[..., 3 : self.output_dim],
                    )
                    if step > 0:
                        self._accumulate_inference_likelihood(
                            likelihood,
                            self._inference_likelihood(
                                noisy_grasps[..., :3],
                                res_pos.pred_original_sample,
                                step,
                                scales_pos,
                            ),
                            self._inference_likelihood(
                                noisy_grasps[..., 3 : self.output_dim],
                                res_rot.pred_original_sample,
                                step,
                                scales_rot,
                            ),
                        )
                    noisy_grasps = torch.hstack([res_pos.prev_sample, res_rot.prev_sample])
                else:
                    res = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=noisy_grasps)
                    if step > 0:
                        self._accumulate_inference_likelihood(
                            likelihood,
                            self._inference_likelihood(noisy_grasps, res.pred_original_sample, step, scales),
                        )
                    noisy_grasps = res.prev_sample

                if self.record_history:
                    pred_grasps = rt_to_matrix(noisy_grasps, self.grasp_repr, self.kappa)
                    self._write_inference_history(grasps_per_iteration, step, pred_grasps.reshape(grasps_init_size))

            if not self.record_history and offline is None:
                pred_grasps = rt_to_matrix(noisy_grasps, self.grasp_repr, self.kappa)

        grasps_pred = pred_grasps.reshape(grasps_init_size)
        stats_batch = []
        if return_metrics:
            all_stats = []
            for i in range(num_objects_in_batch):
                grasps_pred_i = grasps_pred[i].cpu().numpy()
                grasps_gt_i = data["grasps_highres"][i].cpu().numpy()
                tree = KDTree(grasps_gt_i[:, :3, 3])
                dist, idx = tree.query(grasps_pred_i[:, :3, 3])
                matched = dist < 4.0
                grasps_pred_matched = grasps_pred_i[matched]
                grasps_gt_for_pred = grasps_gt_i[idx[matched]]
                gripper_depth = (
                    torch.tensor([data["gripper_depth"][i]], device=grasps_pred.device)
                    .unsqueeze(0)
                    .repeat_interleave(len(grasps_gt_for_pred), dim=0)
                )
                gripper_sym = (
                    torch.tensor([data["gripper_symmetry"][i]], device=grasps_pred.device)
                    .unsqueeze(0)
                    .repeat_interleave(len(grasps_gt_for_pred), dim=0)
                )
                stats = compute_metrics_given_two_sets_of_xgripper_poses(
                    torch.from_numpy(grasps_gt_for_pred).to(grasps_pred.device),
                    torch.from_numpy(grasps_pred_matched).to(grasps_pred.device),
                    gripper_depth,
                    gripper_sym,
                    consider_symmetry=True,
                )
                stats["recall"] = torch.tensor(compute_recall(grasps_gt_i, grasps_pred_i)).to(device)
                stats["precision"] = torch.tensor(compute_recall(grasps_pred_i, grasps_gt_i)).to(device)
                all_stats.append(stats)
            stats_batch = {
                key: torch.mean(torch.tensor([stats[key] for stats in all_stats]).to(device)) for key in all_stats[0]
            }

        outputs = {
            "grasps_pred": grasps_pred,
            "grasps_per_iteration": grasps_per_iteration,
            "grasp_confidence": torch.zeros(grasps_pred.shape[:2]),
            "grasping_masks": torch.zeros(grasps_pred.shape[:2]),
            "grasp_contacts": torch.zeros(grasps_pred.shape[:2]),
            "instance_masks": torch.zeros(grasps_pred.shape[:2]),
            "likelihood": likelihood.reshape(num_objects_in_batch, num_grasps_per_batch, 1),
        }
        return outputs, {}, stats_batch
