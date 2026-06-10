# Copyright (c) 2025 Syslong Technology Co., Ltd. All Rights Reserved.
# Copyright (c) 2025 Shanghai Jiao Tong University
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""Runtime monkey-patches applied during ONNX export for Ascend ATC compatibility.

All patches are **temporary**: they are applied via a context manager and
restored when the block exits, so the original library behaviour is never
permanently modified.

Usage::

    from model_utils.pi05_export.ascend_export_patches import ascend_onnx_export_patches

    with ascend_onnx_export_patches():
        torch.onnx.export(...)

To add a new patch:

1. Write a private function ``_patch_xxx()`` that returns a list of
   ``(_module, attr_name, original_value)`` tuples (the undo log).
2. Register it in :data:`_PATCH_REGISTRY`.
"""

from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from typing import Any

import torch

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch: Gemma rotate_half  (Slice → Split for ATC StridedSliceD issue)
# ---------------------------------------------------------------------------


def _patch_rotate_half() -> list[tuple[Any, str, Any]]:
    """Replace ``rotate_half`` in Gemma modules with a chunk-based version.

    The original implementation::

        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]

    exports as ONNX ``Slice`` ops → ATC ``StridedSliceD``, which fails
    when slice boundaries are shape-derived constants.

    The replacement uses ``x.chunk(2, dim=-1)`` which exports as an ONNX
    ``Split`` op that ATC compiles correctly.  The two are **numerically
    identical**.
    """

    def _chunk_rotate_half(x):
        # We deliberately route through aten.split with an explicit
        # *Python int* sizes list so the dynamo exporter constant-folds
        # the boundary into the graph (a single Split node with sizes
        # initializer [H, H]), instead of a Slice with shape-derived
        # Mul outputs as starts/ends.
        #
        # The torch.chunk(x, 2, dim=-1) form ends up calling
        # split_with_sizes where sizes are computed from x.shape[-1]
        # via a ceil-div formula -> Shape/Gather/Add/Div/Mul ops live
        # in the graph and produce dynamic Slice nodes that Ascend ATC
        # mis-compiles when later fused with Transpose+Transdata.
        #
        # int(x.size(-1)) forces evaluation at trace time. Safe because
        # PI05 head_dim is statically 256 and never declared as a
        # dynamic axis during onnx export.
        half = int(x.size(-1)) // 2
        x1, x2 = torch.split(x, [half, half], dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    undo_log: list[tuple[Any, str, Any]] = []

    # Gemma and Gemma2 each have their own module-level rotate_half;
    # apply_rotary_pos_emb resolves via global name lookup, so both
    # must be patched.
    _module_paths = [
        "transformers.models.gemma.modeling_gemma",
        "transformers.models.gemma2.modeling_gemma2",
    ]

    for mod_path in _module_paths:
        try:
            import importlib

            mod = importlib.import_module(mod_path)
        except ImportError:
            continue

        orig = getattr(mod, "rotate_half", None)
        if orig is None:
            continue

        mod.rotate_half = _chunk_rotate_half
        undo_log.append((mod, "rotate_half", orig))
        LOGGER.info("Patched %s.rotate_half -> chunk-based", mod_path)

    if not undo_log:
        LOGGER.warning("rotate_half patch: no target modules found — skipping")

    return undo_log


# ---------------------------------------------------------------------------
# Patch: Remove fp32 promotions in Gemma (matches Reference gemma.patch)
# ---------------------------------------------------------------------------
# ATC's fp16 mixed-precision pass blindly inserts Cast(fp16→fp32) / Cast(fp32→fp16)
# around every op that was originally in fp32 inside an otherwise fp16 graph.
# When Gemma deliberately upcasts to fp32 (for numerical stability in RMSNorm,
# RoPE, attention), the resulting ONNX graph has tiny fp32 "islands" that ATC
# wraps with extra casts, sometimes in the wrong order, corrupting activations.
#
# The fix (same direction as Reference's gemma.patch) is to **eliminate** those
# fp32 islands at export time so the ONNX graph is uniformly fp16/bf16.  The
# numerical accuracy loss within a single layer is negligible (RMSNorm variance
# in fp16 is fine for 2048-dim vectors), but it prevents ATC from generating
# bad Cast chains.
#
# Three patches, applied only during ONNX export:
#   1. AdaRMSNorm.forward: scale/shift use model_dtype instead of float32
#   2. apply_rotary_pos_emb: remove q/k float32 upcast (if present)
#   3. eager_attention_forward: cast value_states to query dtype before matmul


def _patch_gemma_ada_rmsnorm() -> list[tuple[Any, str, Any]]:
    """Patch GemmaRMSNorm.forward so AdaRMSNorm uses model_dtype, not fp32.

    Original::

        normed_inputs = normed_inputs * (1 + scale.to(torch.float32)) + shift.to(torch.float32)

    Patched::

        model_dtype = self.dense.weight.dtype
        normed_inputs = normed_inputs * (1 + scale.to(model_dtype)) + shift.to(model_dtype)

    This prevents ONNX Cast(fp32) nodes inside the adaptive norm block.
    """

    undo_log: list[tuple[Any, str, Any]] = []

    _module_paths = [
        "transformers.models.gemma.modeling_gemma",
        "transformers.models.gemma2.modeling_gemma2",
    ]

    for mod_path in _module_paths:
        try:
            import importlib

            mod = importlib.import_module(mod_path)
        except ImportError:
            continue

        cls = getattr(mod, "GemmaRMSNorm", None)
        if cls is None:
            continue

        orig_forward = cls.forward

        def _patched_forward(self, x, cond=None, _orig=orig_forward):
            import torch as _torch

            dtype = x.dtype
            normed_inputs = self._norm(x)

            if cond is None or self.dense is None:
                # regular RMSNorm — keep original fp32 weight multiply
                normed_inputs = normed_inputs * (1.0 + self.weight.float())
                return normed_inputs.to(dtype), None

            # adaptive RMSNorm
            if cond.shape[-1] != self.cond_dim:
                raise ValueError(f"Expected cond dimension {self.cond_dim}, got {cond.shape[-1]}")
            modulation = self.dense(cond)
            if len(x.shape) == 3:
                modulation = modulation.unsqueeze(1)

            scale, shift, gate = _torch.chunk(modulation, 3, dim=-1)

            # KEY CHANGE: use model_dtype instead of float32
            model_dtype = self.dense.weight.dtype
            normed_inputs = normed_inputs * (1 + scale.to(model_dtype)) + shift.to(model_dtype)

            return normed_inputs.to(dtype), gate.to(dtype)

        cls.forward = _patched_forward
        undo_log.append((cls, "forward", orig_forward))
        LOGGER.info("Patched %s.GemmaRMSNorm.forward (AdaRMSNorm fp32→model_dtype)", mod_path)

    return undo_log


def _patch_gemma_rotary_pos_emb() -> list[tuple[Any, str, Any]]:
    """Replace ``apply_rotary_pos_emb`` with a reshape-based equivalent.

    Background
    ----------
    The HF / GPT-NeoX RoPE convention computes::

        q_embed[..., :h] = a*c - b*s
        q_embed[..., h:] = b*c + a*s

    where ``a, b = q[..., :h], q[..., h:]``.  The textbook implementation
    routes through ``rotate_half`` which exports as a pair of ONNX Slice
    (or Split) nodes.  On Ascend ATC those nodes get fused with the
    surrounding Transpose/TransData ops by
    ``TransdataTransposeTransdataFusionPass``, which mis-compiles the
    StridedSliceD / SplitVD kernel.  Disabling that fusion pass works
    but costs 5-15% on attention.

    Mathematical reformulation
    --------------------------
    Reshape ``q`` from ``(B, H, S, D)`` to ``(B, H, S, 2, h)`` (row-major,
    so dim -2 of size 2 holds [a, b]).  ``cos``/``sin`` were built from
    ``cat((freqs, freqs), -1)`` so the two halves on dim -1 are equal,
    meaning the reshaped ``cos``/``sin`` are constant along the new size-2
    axis.  Define::

        sign = [-1, +1]              # shape (1,1,1,2,1)
        q_sw = swap on dim=-2        # [a,b] -> [b,a], via index_select

    Then::

        q_out_reshaped = q_r * cos_r + q_sw_r * sin_r * sign

    Plug in k=0 (gives a*c - b*s) and k=1 (gives b*c + a*s) — identical to
    the original formula.  Reshape back to ``(B, H, S, D)``.

    Why this avoids the bug
    -----------------------
    The exported graph contains only Reshape, Mul, Add, Gather (size-2 on
    head_dim).  No Slice, no Split, no Transpose between TransData ops.
    The buggy ``Transdata*Transdata`` fusion family has nothing to match,
    so we keep the full GraphFusion + UBFusion stack on.

    Caveat: PI05 head_dim is statically 256, so ``int(q.size(-1)) // 2``
    constant-folds at trace time.  Do NOT enable a dynamic head_dim axis.
    """
    undo_log: list[tuple[Any, str, Any]] = []

    _module_paths = [
        "transformers.models.gemma.modeling_gemma",
        "transformers.models.gemma2.modeling_gemma2",
    ]

    for mod_path in _module_paths:
        try:
            import importlib

            mod = importlib.import_module(mod_path)
        except ImportError:
            continue

        orig = getattr(mod, "apply_rotary_pos_emb", None)
        if orig is None:
            continue

        def _apply(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            # cos, sin: (B, S, D) before unsqueeze
            cos = cos.unsqueeze(unsqueeze_dim)  # (B, 1, S, D)
            sin = sin.unsqueeze(unsqueeze_dim)

            D = int(q.size(-1))  # static for PI05 (256); folded at trace
            half = D // 2

            # Constants: created on q's device/dtype so they become small
            # initializers in the ONNX graph.
            swap_idx = torch.tensor([1, 0], dtype=torch.long, device=q.device)
            sign = torch.tensor([-1.0, 1.0], dtype=q.dtype, device=q.device).view(1, 1, 1, 2, 1)

            def _rope(t, c, s):
                # t: (B, H, S, D)  -> (B, H, S, 2, half)
                B, H, S, _ = t.shape
                tr = t.reshape(B, H, S, 2, half)
                # cos/sin have shape (B, 1, S, D) -> (B, 1, S, 2, half).
                # Both halves on the original last dim are equal, so this
                # reshape just exposes that redundancy as a size-2 axis.
                cr = c.reshape(c.size(0), 1, S, 2, half)
                sr = s.reshape(s.size(0), 1, S, 2, half)
                tr_sw = tr.index_select(-2, swap_idx)  # swap a<->b on dim -2
                out = tr * cr + tr_sw * sr * sign
                return out.reshape(B, H, S, D)

            q_embed = _rope(q, cos, sin)
            k_embed = _rope(k, cos, sin)
            return q_embed, k_embed

        mod.apply_rotary_pos_emb = _apply
        undo_log.append((mod, "apply_rotary_pos_emb", orig))
        LOGGER.info(
            "Patched %s.apply_rotary_pos_emb (reshape-based, no Slice/Split)",
            mod_path,
        )

    return undo_log


# ---------------------------------------------------------------------------
# Patch: apply_rotary_pos_emb -> torch_npu.npu_rotary_mul (native ONNX op)
# ---------------------------------------------------------------------------
# torch_npu ships its own ONNX export support: ``import torch_npu.onnx`` runs
# ``_add_onnx_ops()``, which wraps ``torch_npu.npu_rotary_mul`` so that during
# ``torch.onnx.export`` it emits a single ``NPURotaryMul`` node carrying a
# ``rotary_mode="half"`` attribute.  That node maps 1:1 to the on-device
# ``torch_npu.npu_rotary_mul`` fused kernel, and its "half" mode is exactly the
# GPT-NeoX rotate_half convention HF Gemma uses::
#
#     out = x * cos + rotate_half(x) * sin
#
# The export host therefore needs a working ``torch_npu`` install (it does NOT
# need a physical NPU at trace time — ``forward`` runs the real op, ``symbolic``
# emits the graph node).  torch_npu emits the node under the ``npu`` domain; the
# convert scripts strip that domain (ATC accepts only a single, default domain),
# leaving a bare ``NPURotaryMul`` op_type the ATC custom-op plugin resolves.


def _patch_gemma_rotary_pos_emb_npu() -> list[tuple[Any, str, Any]]:
    """Replace ``apply_rotary_pos_emb`` to use ``torch_npu.npu_rotary_mul``.

    Unlike :func:`_patch_gemma_rotary_pos_emb` (which reshapes RoPE into pure
    Mul/Add/Gather for an ORT-runnable graph), this variant routes RoPE
    through the real NPU fused operator.  ``import torch_npu.onnx`` registers
    the symbolic that makes ``torch.onnx.export`` emit a single
    ``NPURotaryMul`` node per call — so the exported ONNX explicitly states
    the NPU-affine fused kernel is used on device.

    Query and key are rotated by **separate** ``NPURotaryMul`` nodes (no
    Concat/Split merge), keeping the graph free of the Slice/Split ops that
    ATC's StridedSliceD/SplitVD pass mis-compiles.

    Requires a working ``torch_npu`` install on the export host (guaranteed
    when exporting on an NPU device).  If it is unavailable the patch is
    skipped and RoPE keeps transformers' default implementation.  ORT CPU has
    no kernel for ``NPURotaryMul``, so ONNX-vs-PyTorch verification must be
    skipped when this patch is active.
    """
    undo_log: list[tuple[Any, str, Any]] = []

    try:
        import torch_npu  # noqa: F401

        # Registers the ONNX symbolic wrapper for npu_rotary_mul et al.
        import torch_npu.onnx  # noqa: F401
    except ImportError as exc:
        LOGGER.warning(
            "npu rope patch requested but torch_npu is unavailable (%s); "
            "skipping — RoPE will keep transformers' default implementation",
            exc,
        )
        return undo_log

    _module_paths = [
        "transformers.models.gemma.modeling_gemma",
        "transformers.models.gemma2.modeling_gemma2",
    ]

    for mod_path in _module_paths:
        try:
            import importlib

            mod = importlib.import_module(mod_path)
        except ImportError:
            continue

        orig = getattr(mod, "apply_rotary_pos_emb", None)
        if orig is None:
            continue

        def _apply(q, k, cos, sin, position_ids=None, unsqueeze_dim=1, _tn=torch_npu):
            # cos, sin: (B, S, D) -> (B, 1, S, D) so they broadcast over the
            # head axis of q/k (B, H, S, D); npu_rotary_mul broadcasts the
            # head dim natively.  Default rotary_mode="half" matches Gemma's
            # rotate_half convention.
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)
            q_embed = _tn.npu_rotary_mul(q, cos, sin)
            k_embed = _tn.npu_rotary_mul(k, cos, sin)
            return q_embed, k_embed

        mod.apply_rotary_pos_emb = _apply
        undo_log.append((mod, "apply_rotary_pos_emb", orig))
        LOGGER.info(
            "Patched %s.apply_rotary_pos_emb (torch_npu.npu_rotary_mul -> NPURotaryMul)",
            mod_path,
        )

    return undo_log


def _patch_gemma_ada_rmsnorm_npu() -> list[tuple[Any, str, Any]]:
    """Route GemmaRMSNorm through ``torch_npu.npu_rms_norm`` (NPURmsNorm node).

    This is the NPU-affine variant of :func:`_patch_gemma_ada_rmsnorm`.  The
    RMSNorm *core* (``x / rms(x) * gamma``) is replaced by the real fused
    operator ``torch.ops.npu.npu_rms_norm``.  A small **single-output** custom
    autograd Function emits a 1-output ``npu::NPURmsNorm`` node per call — the
    exported ONNX explicitly states the NPU-affine fused kernel is used on
    device.

    Why a custom symbolic instead of torch_npu's ``npu_rms_norm`` wrapper:
    that wrapper emits a 2-output node (y, rstd), and the unused ``rstd``
    output makes the legacy TorchScript exporter crash in
    ``_jit_pass_lower_all_tuples`` (DCE cannot drop one output of a custom
    ``PythonOp``).  Returning a single value sidesteps the dead-output
    problem entirely.

    Two paths, selected at runtime exactly like the original:

    * **regular RMSNorm** (VLM / ``model.norm``; ``self.dense is None``):
      the whole formula is ``normed * (1 + weight)``, so it fully folds into
      the fused op by passing ``gamma = 1 + weight``.  Gemma's unit-offset
      convention (``1 + weight``) differs from npu_rms_norm's plain ``gamma``
      multiply, hence the explicit ``+ 1``.

    * **adaptive RMSNorm** (Action Expert; ``self.dense`` + ``cond``):
      ``scale``/``shift``/``gate`` are computed per-step from ``cond`` and
      cannot fold into a constant ``gamma``.  Only the core ``x / rms(x)`` is
      fused (``gamma = ones``); the ``* (1 + scale) + shift`` modulation and
      ``gate`` stay as ordinary ONNX nodes.

    Requires a working ``torch_npu`` on the export host (guaranteed on an NPU
    device).  If unavailable the patch is skipped and RMSNorm keeps the
    default implementation.  ORT CPU has no ``NPURmsNorm`` kernel, so
    ONNX-vs-PyTorch verification must be skipped when this patch is active.
    """
    undo_log: list[tuple[Any, str, Any]] = []

    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        LOGGER.warning(
            "npu rmsnorm patch requested but torch_npu is unavailable (%s); "
            "skipping — RMSNorm will keep its default implementation",
            exc,
        )
        return undo_log

    import importlib

    import torch as _torch

    # Single-output custom symbolic.
    #
    # torch_npu's ``npu_rms_norm`` wrapper emits a 2-output ``NPURmsNorm`` node
    # (y, rstd).  We only need ``y``; the dead ``rstd`` output makes the legacy
    # TorchScript exporter crash in ``_jit_pass_lower_all_tuples`` (DCE cannot
    # drop one output of a custom ``PythonOp``).  So we declare our own
    # autograd Function that returns a SINGLE value and emits a 1-output
    # ``npu::NPURmsNorm`` node — no dead output, clean jit graph.  ATC matches
    # the op by op_type after domain stripping.
    class _NpuRmsNormSingle(_torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, gamma, epsilon):
            return _torch.ops.npu.npu_rms_norm(x, gamma, epsilon)[0]

        @staticmethod
        def symbolic(g, x, gamma, epsilon):
            return g.op("npu::NPURmsNorm", x, gamma, epsilon_f=epsilon)

    _module_paths = [
        "transformers.models.gemma.modeling_gemma",
        "transformers.models.gemma2.modeling_gemma2",
    ]

    for mod_path in _module_paths:
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue

        cls = getattr(mod, "GemmaRMSNorm", None)
        if cls is None:
            continue

        orig_forward = cls.forward

        def _patched_forward(self, x, cond=None, _rms=_NpuRmsNormSingle):
            import torch as _torch

            dtype = x.dtype

            if cond is None or self.dense is None:
                # regular RMSNorm — fold (1 + weight) into the fused gamma.
                gamma = (1.0 + self.weight.float()).to(dtype)
                normed = _rms.apply(x, gamma, self.eps)
                return normed.to(dtype), None

            # adaptive RMSNorm — fuse the core, keep scale/shift/gate separate.
            if cond.shape[-1] != self.cond_dim:
                raise ValueError(f"Expected cond dimension {self.cond_dim}, got {cond.shape[-1]}")

            ones = _torch.ones(self.dim, dtype=dtype, device=x.device)
            normed_inputs = _rms.apply(x, ones, self.eps)

            modulation = self.dense(cond)
            if len(x.shape) == 3:
                modulation = modulation.unsqueeze(1)

            scale, shift, gate = _torch.chunk(modulation, 3, dim=-1)

            model_dtype = self.dense.weight.dtype
            normed_inputs = normed_inputs * (1 + scale.to(model_dtype)) + shift.to(model_dtype)

            return normed_inputs.to(dtype), gate.to(dtype)

        cls.forward = _patched_forward
        undo_log.append((cls, "forward", orig_forward))
        LOGGER.info(
            "Patched %s.GemmaRMSNorm.forward (torch_npu.npu_rms_norm -> NPURmsNorm)",
            mod_path,
        )

    return undo_log


# ---------------------------------------------------------------------------
# Patch: gelu_pytorch_tanh -> torch_npu.fast_gelu (NPUFastGelu fused node)
# ---------------------------------------------------------------------------
# Gemma's text MLP and SigLIP's vision MLP both use the ``gelu_pytorch_tanh``
# activation, i.e. ``transformers.activations.PytorchGELUTanh`` whose forward is
# ``nn.functional.gelu(x, approximate="tanh")`` — a single aten op.  At opset
# <= 18 the TorchScript ONNX exporter does NOT emit a native ``Gelu`` node
# (that op only exists from opset 20); instead it *decomposes* the tanh
# approximation into the explicit formula, producing a long
# ``Pow → Mul → Add → Mul → Tanh → Add → Mul`` elementwise chain per MLP.  In
# the PI05 VLM profile these decomposed ``Mul`` (~123 us) + ``Add`` (~49 us)
# nodes are the single largest "glue" cost (~172 us, dominating the 10% Mul
# bucket), and ATC's ``GeluFusionPass`` does NOT match this particular
# decomposition (it stays unfused on the hot LLM/SigLIP MLP path).
#
# DOMAIN CONSTRAINT (the deciding factor — same wall that killed Flash Attn).
# ``strings libops_all_onnx_plugin.so | grep -i gelu`` on the 310P CANN 8.1
# install shows the two NPU gelu onnx ops register under DIFFERENT domains:
#     NPUGeluV2   : npu::1 ONLY          (the exact tanh approximation)
#     NPUFastGelu : ai.onnx::11-18 + npu::1  (a sigmoid-style approximation)
# Our convert scripts strip every node to the DEFAULT domain and keep a SINGLE
# opset_import (ATC allows exactly one).  So an npu-domain-only op is unusable:
#   * NPUGeluV2 stripped to default -> "No parser for ai.onnx::NPUGeluV2".
#   * NPUGeluV2 kept in npu domain  -> mixed node domains -> ATC E16005.
# This is the identical constraint that makes Flash Attention infeasible (see
# "NPU 算子替换总结.md" 坑6).  Only NPUFastGelu — double-registered in the
# default domain at opset 11-18, exactly like NPURotaryMul / NPURmsNorm — can
# survive the strip-to-default + single-opset_import flow at our export opset 16.
#
# ACCURACY TRADE-OFF (NOT numerically identical).  Ascend ``FastGelu`` uses the
# sigmoid-style approximation ``x*exp(0.851*(x-|x|))/(1+exp(-1.702*|x|))``,
# which is a DIFFERENT approximation than ``gelu_pytorch_tanh``'s tanh form.
# Measured per-element error of substituting FastGelu for gelu_pytorch_tanh:
#   max ≈ 2.1e-2 (worst near x≈2.3), mean ≈ 7e-3, cosine ≈ 0.99999.
# (For reference gelu_pytorch_tanh itself is within 4.7e-4 of exact erf-gelu, so
# FastGelu is ~40x coarser.)  The error is small and bounded, but this is a VLA
# action policy — it MUST be validated end-to-end (verify_vlm_cpu_vs_om +
# action-error regression) before enabling.  Revert to the default decomposition
# if action accuracy degrades.


def _patch_pytorch_gelu_tanh_npu() -> list[tuple[Any, str, Any]]:
    """Route ``gelu_pytorch_tanh`` activations through ``torch_npu.fast_gelu``.

    Patches ``transformers.activations.PytorchGELUTanh.forward`` (the class
    backing every ``gelu_pytorch_tanh`` activation — Gemma text MLP **and**
    SigLIP vision MLP) so that ``torch.onnx.export`` emits a single
    ``npu::NPUFastGelu`` node per activation instead of the decomposed
    ``Mul/Add/Tanh`` elementwise chain the exporter produces at opset <= 18.

    Like the RoPE / RMSNorm NPU patches, a **single-output** custom autograd
    Function is used: ``forward`` calls the real fused kernel
    ``torch.ops.npu.fast_gelu`` (correct on-device tracing) and ``symbolic``
    emits a clean 1-output ``npu::NPUFastGelu`` node.  ATC matches the op by
    op_type after the convert script strips node domains to the default —
    ``NPUFastGelu`` is registered in the default ``ai.onnx`` domain at opset
    11-18 (unlike ``NPUGeluV2``, which is npu-domain-only and therefore
    unusable under the strip-to-default + single-opset_import flow).

    NOT numerically identical: Ascend ``FastGelu`` is a sigmoid-style
    approximation, distinct from ``gelu_pytorch_tanh``'s tanh form.  The
    substitution introduces a bounded per-element error (max ≈ 2.1e-2, mean
    ≈ 7e-3, cosine ≈ 0.99999) that MUST be validated end-to-end on the VLA
    action output before this patch is relied upon.

    Requires a working ``torch_npu`` on the export host (guaranteed on an NPU
    device).  If unavailable the patch is skipped and the activation keeps its
    default decomposition.  ORT CPU has no ``NPUFastGelu`` kernel, so
    ONNX-vs-PyTorch verification must be skipped when this patch is active.
    """
    undo_log: list[tuple[Any, str, Any]] = []

    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        LOGGER.warning(
            "npu gelu patch requested but torch_npu is unavailable (%s); "
            "skipping — gelu_pytorch_tanh keeps its default decomposition",
            exc,
        )
        return undo_log

    import torch as _torch

    # Single-output custom symbolic.  fast_gelu is already a 1-output op, so
    # there is no dead-output problem (unlike npu_rms_norm's (y, rstd)); we
    # still declare our own Function to keep the symbolic self-contained and
    # independent of ``import torch_npu.onnx`` wrapper-replacement state.
    # Emit npu::NPUFastGelu (default-domain registered at opset 11-18); the
    # convert script's domain strip leaves a bare NPUFastGelu op_type the ATC
    # plugin resolves.
    class _NpuFastGelu(_torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return _torch.ops.npu.fast_gelu(x)

        @staticmethod
        def symbolic(g, x):
            return g.op("npu::NPUFastGelu", x)

    try:
        from transformers.activations import PytorchGELUTanh
    except ImportError as exc:
        LOGGER.warning(
            "npu gelu patch: transformers.activations.PytorchGELUTanh not found (%s); skipping",
            exc,
        )
        return undo_log

    orig_forward = PytorchGELUTanh.forward

    def _patched_forward(self, input, _gelu=_NpuFastGelu):  # noqa: A002
        return _gelu.apply(input)

    PytorchGELUTanh.forward = _patched_forward
    undo_log.append((PytorchGELUTanh, "forward", orig_forward))
    LOGGER.info(
        "Patched transformers.activations.PytorchGELUTanh.forward "
        "(torch_npu.fast_gelu -> NPUFastGelu; sigmoid approx, validate accuracy)"
    )

    return undo_log


def _patch_gemma_eager_attention(
    *, fp16_softmax: bool = False, mqa_broadcast: bool = False
) -> list[tuple[Any, str, Any]]:
    """Cast value_states to query dtype in eager_attention_forward.

    Original::

        attn_output = torch.matmul(attn_weights, value_states)

    Patched::

        attn_output = torch.matmul(attn_weights, value_states.to(query.dtype))

    When KV cache dtype differs from query dtype (e.g. value in fp32,
    query in fp16), the ONNX graph produces a mixed-precision MatMul
    that ATC may compile incorrectly.

    Two optional action-expert optimisations (both off by default, so the
    VLM export is unaffected):

    * ``mqa_broadcast`` — when the layer is multi-query (``num_kv_heads == 1``),
      skip the ``repeat_kv`` expansion that materialises ``num_heads`` physical
      copies of K/V (an ONNX ``Expand`` + extra ``TransData`` per layer) and
      rely on ``matmul`` broadcasting the singleton head dim instead.  This is
      mathematically identical and only changes when the singleton K/V head can
      broadcast (``num_kv_heads == 1``); GQA layers fall back to ``repeat_kv``.

    * ``fp16_softmax`` — keep the score matrix / softmax in the query dtype
      (fp16) instead of upcasting to fp32, removing the per-layer fp32 ``Cast``
      around the (B, H, Sq, Sk) score matrix and halving the softmax cost.
      Requires the additive mask to be fp16-safe (``finfo(fp16).min``); the mask
      is cast to the score dtype here so a fully-masked row stays finite.
    """
    import importlib

    import torch.nn as nn

    undo_log: list[tuple[Any, str, Any]] = []

    _module_paths = [
        "transformers.models.gemma.modeling_gemma",
        "transformers.models.gemma2.modeling_gemma2",
    ]

    for mod_path in _module_paths:
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue

        orig = getattr(mod, "eager_attention_forward", None)
        if orig is None:
            continue

        _repeat_kv = getattr(mod, "repeat_kv", None)
        if _repeat_kv is None:
            continue

        def _make_patched(repeat_kv_fn):
            def _patched_eager_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling,
                dropout=0.0,
                **kwargs,
            ):
                # MQA broadcast: skip materialising num_heads copies of K/V when
                # there is a single KV head — matmul broadcasts the singleton
                # head dim against query's num_heads (mathematically identical
                # to repeat_kv, but drops the per-layer Expand + TransData).
                if mqa_broadcast and key.shape[1] == 1:
                    key_states = key
                    value_states = value
                else:
                    key_states = repeat_kv_fn(key, module.num_key_value_groups)
                    value_states = repeat_kv_fn(value, module.num_key_value_groups)

                attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

                if attention_mask is not None:
                    # NOTE: original code slices ``attention_mask[:, :, :, :key_len]``
                    # which produces a dynamic Slice op.  Non-dynamo export
                    # maps it to StridedSliceD which ATC cannot compile.
                    # For PI05 the mask is already (B,1,S,S) matching key_len,
                    # so the slice is a no-op — use the mask directly.
                    if fp16_softmax:
                        # Keep the Add in the score (fp16) dtype so no fp32 Cast
                        # is inserted around the (B,H,Sq,Sk) score matrix. The
                        # mask must already use a fp16-safe sentinel
                        # (finfo(fp16).min); -2.38e38 would overflow to -inf.
                        attn_weights = attn_weights + attention_mask.to(attn_weights.dtype)
                    else:
                        attn_weights = attn_weights + attention_mask

                if fp16_softmax:
                    attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query.dtype)
                else:
                    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
                attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
                # KEY CHANGE: cast value_states to query dtype
                attn_output = torch.matmul(attn_weights, value_states.to(query.dtype))
                attn_output = attn_output.transpose(1, 2).contiguous()

                return attn_output, attn_weights

            return _patched_eager_attention_forward

        patched = _make_patched(_repeat_kv)
        mod.eager_attention_forward = patched
        undo_log.append((mod, "eager_attention_forward", orig))
        LOGGER.info(
            "Patched %s.eager_attention_forward (value→query dtype, mqa_broadcast=%s, fp16_softmax=%s)",
            mod_path,
            mqa_broadcast,
            fp16_softmax,
        )

    return undo_log


def _patch_gemma_fused_qkv() -> list[tuple[Any, str, Any]]:
    """Fuse q_proj / k_proj / v_proj into a single MatMul in GemmaAttention.

    The three separate projections::

        q = self.q_proj(hidden_states)   # (.., num_heads    * head_dim)
        k = self.k_proj(hidden_states)   # (.., num_kv_heads * head_dim)
        v = self.v_proj(hidden_states)   # (.., num_kv_heads * head_dim)

    are replaced by one fused projection::

        w   = cat([q_w, k_w, v_w], dim=0)          # (q+k+v_dim, hidden)
        qkv = linear(hidden_states, w, b)          # one MatMul
        q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)

    This is **not** an NPU-affine op — it emits ordinary ONNX
    ``MatMul`` / ``Add`` / ``Split`` nodes, needs no ATC custom plugin,
    and stays ORT-verifiable.  The benefit is three GEMM kernel launches
    collapsing into one (plus a single activation read instead of three),
    which mainly helps the launch-bound Action-Expert denoising loop.

    Gemma uses GQA (``num_kv_heads < num_heads``), so the split is
    ``[q_out, k_out, v_out]`` — never an even three-way split.

    The fused weight ``cat`` is performed inside the traced forward; since
    q/k/v weights are constants, ``do_constant_folding=True`` collapses the
    Concat into a single initializer (both export scripts default to
    constant folding).

    LoRA-wrapped projections (``LoRALinear``, which lack a plain ``.weight``)
    are detected per-module and fall back to the original three-projection
    path, so enabling LoRA never breaks the export.
    """
    import importlib

    import torch.nn as nn

    undo_log: list[tuple[Any, str, Any]] = []

    _module_paths = [
        "transformers.models.gemma.modeling_gemma",
        "transformers.models.gemma2.modeling_gemma2",
    ]

    for mod_path in _module_paths:
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue

        cls = getattr(mod, "GemmaAttention", None)
        if cls is None:
            continue

        orig_forward = cls.forward

        def _make_forward(_mod, _orig):
            def _patched_forward(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_value=None,
                cache_position=None,
                use_cache=False,
                **kwargs,
            ):
                # Fall back to the stock forward whenever the projections are
                # not plain Linears (e.g. LoRA-wrapped) — fusing would be
                # incorrect or impossible there.
                if not (
                    isinstance(self.q_proj, nn.Linear)
                    and isinstance(self.k_proj, nn.Linear)
                    and isinstance(self.v_proj, nn.Linear)
                ):
                    return _orig(
                        self,
                        hidden_states,
                        position_embeddings,
                        attention_mask,
                        past_key_value=past_key_value,
                        cache_position=cache_position,
                        use_cache=use_cache,
                        **kwargs,
                    )

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)

                # --- fused qkv projection ---
                split_sizes = [
                    self.q_proj.out_features,
                    self.k_proj.out_features,
                    self.v_proj.out_features,
                ]
                fused_w = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)
                if self.q_proj.bias is not None:
                    fused_b = torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], dim=0)
                else:
                    fused_b = None

                qkv = nn.functional.linear(hidden_states, fused_w, fused_b)
                q, k, v = qkv.split(split_sizes, dim=-1)

                query_states = q.view(hidden_shape).transpose(1, 2)
                key_states = k.view(hidden_shape).transpose(1, 2)
                value_states = v.view(hidden_shape).transpose(1, 2)

                # Resolve RoPE helper at call time so it honours any active
                # apply_rotary_pos_emb patch (reshape-based or npu variant).
                cos, sin = position_embeddings
                _apply_rotary = _mod.apply_rotary_pos_emb
                query_states, key_states = _apply_rotary(query_states, key_states, cos, sin)

                if past_key_value is not None:
                    if use_cache:
                        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                        key_states, value_states = past_key_value.update(
                            key_states, value_states, self.layer_idx, cache_kwargs
                        )
                    else:
                        key_states = torch.cat([past_key_value[self.layer_idx][0], key_states], dim=2)
                        value_states = torch.cat([past_key_value[self.layer_idx][1], value_states], dim=2)

                attention_interface = _mod.eager_attention_forward
                if self.config._attn_implementation != "eager":  # noqa: SLF001
                    attention_interface = _mod.ALL_ATTENTION_FUNCTIONS[
                        self.config._attn_implementation  # noqa: SLF001
                    ]

                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                    **kwargs,
                )

                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output, attn_weights

            return _patched_forward

        cls.forward = _make_forward(mod, orig_forward)
        undo_log.append((cls, "forward", orig_forward))
        LOGGER.info("Patched %s.GemmaAttention.forward (fused qkv projection)", mod_path)

    return undo_log


