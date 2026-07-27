# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
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
    """Patch PiGemmaRMSNorm.forward so AdaRMSNorm uses model_dtype, not fp32.

    Original::

        normed_inputs = normed_inputs * (1 + scale.to(torch.float32)) + shift.to(torch.float32)

    Patched::

        model_dtype = self.dense.weight.dtype
        normed_inputs = normed_inputs * (1 + scale.to(model_dtype)) + shift.to(model_dtype)

    This prevents ONNX Cast(fp32) nodes inside the adaptive norm block.
    """

    undo_log: list[tuple[Any, str, Any]] = []

    # Patch PiGemmaRMSNorm in our adapter module
    try:
        from model_utils.pi05_export.pi_gemma import PiGemmaRMSNorm

        cls = PiGemmaRMSNorm
    except ImportError:
        LOGGER.warning("PiGemmaRMSNorm not found; skipping AdaRMSNorm patch")
        return undo_log

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
    LOGGER.info("Patched PiGemmaRMSNorm.forward (AdaRMSNorm fp32→model_dtype)")

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
    """Route PiGemmaRMSNorm through ``torch_npu.npu_rms_norm`` (NPURmsNorm node).

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

    # Patch PiGemmaRMSNorm in our adapter module
    try:
        from model_utils.pi05_export.pi_gemma import PiGemmaRMSNorm

        cls = PiGemmaRMSNorm
    except ImportError:
        LOGGER.warning("PiGemmaRMSNorm not found; skipping NPU RMSNorm patch")
        return undo_log

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
    LOGGER.info("Patched PiGemmaRMSNorm.forward (torch_npu.npu_rms_norm -> NPURmsNorm)")

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


def _patch_gemma_geglu_npu() -> list[tuple[Any, str, Any]]:
    """Fuse the Gemma text MLP ``gelu(gate_proj(x)) * up_proj(x)`` into one NPUGeglu.

    ``GemmaMLP`` computes ``down_proj(act_fn(gate_proj(x)) * up_proj(x))`` with
    ``act_fn = gelu_pytorch_tanh``. Ascend's ONNX/ATC ``GeGluV2`` fuses the
    activation and gate multiply: given a single ``[..., 2I]`` input it splits
    into halves ``[a, b]`` and returns ``a * gelu(b)`` on 310P. The fused input
    is therefore ordered as ``[up, gate]`` so the OM computes the same result as
    the original Gemma MLP.

    The replacement emits one fused MatMul feeding a single-output
    ``npu::NPUGeglu`` node, using tanh GELU for accuracy. It only patches Gemma
    MLPs; the non-gated SigLIP vision MLP remains unaffected.
    """
    import importlib

    undo_log: list[tuple[Any, str, Any]] = []

    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        LOGGER.warning(
            "npu geglu patch requested but torch_npu is unavailable (%s); "
            "skipping — GemmaMLP keeps its default gate/up/gelu decomposition",
            exc,
        )
        return undo_log

    import torch as _torch
    import torch.nn.functional as _F

    class _NpuGeglu(_torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            out = _torch.ops.npu.npu_geglu(x, -1, 1, False)
            return out[0] if isinstance(out, tuple | list) else out

        @staticmethod
        def symbolic(g, x):
            return g.op("npu::NPUGeglu", x, dim_i=-1, approximate_i=1, activate_left_i=0)

    def _make_forward(geglu):
        def _geglu_forward(self, x):
            # ATC computes left * gelu(right), so concatenate [up, gate].
            fused = getattr(self, "_npu_up_gate_weight", None)
            if fused is None:
                fused = _torch.cat([self.up_proj.weight, self.gate_proj.weight], dim=0)
                self._npu_up_gate_weight = fused
            up_gate = _F.linear(x, fused)
            return self.down_proj(geglu.apply(up_gate))

        return _geglu_forward

    targets = [
        ("transformers.models.gemma.modeling_gemma", "GemmaMLP"),
        ("transformers.models.gemma2.modeling_gemma2", "Gemma2MLP"),
        ("transformers.models.gemma3.modeling_gemma3", "Gemma3MLP"),
    ]
    for mod_path, cls_name in targets:
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue
        cls = getattr(mod, cls_name, None)
        if cls is None:
            continue
        orig_forward = cls.forward
        cls.forward = _make_forward(_NpuGeglu)
        undo_log.append((cls, "forward", orig_forward))
        LOGGER.info("Patched %s.%s.forward (fused gate_up + NPUGeglu, tanh GeGLU)", mod_path, cls_name)

    return undo_log


def _patch_gemma_eager_attention(
    *, softmax_in_model_dtype: bool = False, mqa_broadcast: bool = False
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

    * ``softmax_in_model_dtype`` — keep the score matrix / softmax in the query
      dtype instead of upcasting to fp32, removing the per-layer fp32 ``Cast``
      around the (B, H, Sq, Sk) score matrix and halving the softmax cost.
      Requires the additive mask to use a sentinel representable by the score
      dtype; the mask is cast to the score dtype here so a fully-masked row
      stays finite.
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
                    # PI05 builds the additive mask to exactly (B, 1, Sq, key_len)
                    # in modeling_pi05_*._prepare_attention_masks_4d, so it already
                    # matches attn_weights' (B, H, Sq, key_len): the singleton head
                    # dim broadcasts on the Add, and attention_mask.shape[-1] already
                    # equals key_states.shape[2].  Do NOT slice to key_len or expand
                    # the head dim here — both are numerical no-ops, but each emits a
                    # per-layer Slice + Expand (plus the -1-aware
                    # Equal/ConstantOfShape/Where shape subgraph) that ATC keeps as
                    # real ops, materialising a full (B, H, Sq, key_len) mask in every
                    # one of the 18 expert layers (~0.4ms/inference regression).
                    if softmax_in_model_dtype:
                        # Keep the Add in the score dtype so no fp32 Cast is
                        # inserted around the (B,H,Sq,Sk) score matrix. The mask
                        # must already use a score-dtype-safe sentinel.
                        attn_weights = attn_weights + attention_mask.to(attn_weights.dtype)
                    else:
                        attn_weights = attn_weights + attention_mask

                if softmax_in_model_dtype:
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
            "Patched %s.eager_attention_forward (value→query dtype, mqa_broadcast=%s, softmax_in_model_dtype=%s)",
            mod_path,
            mqa_broadcast,
            softmax_in_model_dtype,
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


def _patch_gemma_flash_attention_npu() -> list[tuple[Any, str, Any]]:
    """Route attention through ``torch_npu.npu_prompt_flash_attention``.

    This swaps the module-level ``eager_attention_forward`` (the
    ``attention_interface`` resolved at call time by both the stock and the
    fused-qkv :func:`_patch_gemma_fused_qkv` forward) for a single
    NPU-affine fused-attention operator.  The exported ONNX then contains
    one ``npu::NPUPromptFlashAttention`` node per layer instead of the
    ``MatMul → Add(mask) → Softmax → MatMul`` subgraph, fusing the whole
    score/softmax/context computation into one on-device kernel.

    Composition with fused qkv
    ---------------------------
    The fused-qkv forward produces ``q/k/v`` already split (GQA, *not*
    repeated) in ``(B, N, S, D)`` / ``(B, Nkv, S, D)`` layout and hands them
    straight to this interface — so the fused projection feeds flash
    attention directly: no ``repeat_kv`` expand, no score-matrix
    materialisation, no intermediate q/k/v re-layout.  This is the pairing
    that makes the standalone qkv ``Split`` worthwhile (it no longer needs to
    land three contiguous buffers for an eager MatMul; flash attention
    consumes the split slices directly).

    Math equivalence vs eager
    -------------------------
    * **scale** — passed through as ``scale_value`` (Gemma's ``self.scaling``,
      already ``head_dim**-0.5`` or ``query_pre_attn_scalar**-0.5``).
    * **mask** — HF hands ``eager_attention_forward`` an *additive* float
      mask (``0`` to attend, ``finfo.min`` to mask).  npu prompt-flash takes
      a *boolean* ``atten_mask`` where **True == masked** (set to ``-inf``
      internally).  So ``atten_mask = attention_mask < 0`` reproduces the
      exact same masking (every masked slot is a large negative additive,
      every attended slot is ``0``).
    * **GQA** — key/value keep ``num_key_value_heads`` heads; the operator
      broadcasts them across query heads internally, matching ``repeat_kv``.

    Output layout
    -------------
    ``input_layout="BNSD"`` ⇒ the operator returns ``(B, N, S, D)``; eager's
    contract is ``(B, S, N, D)`` (it transposes before returning), so we
    ``transpose(1, 2)`` to match.  The caller then ``reshape(*input_shape,
    -1)`` and applies ``o_proj`` unchanged.

    Custom single-output symbolic
    -----------------------------
    torch_npu's own ``npu_prompt_flash_attention`` ONNX wrapper is unusable
    here: its ``symbolic`` references an undefined ``self`` and passes the
    scalar params as positional graph *inputs*.  As with
    :func:`_patch_gemma_ada_rmsnorm_npu`, we declare our own autograd
    Function whose ``forward`` calls the real runtime op (correct tracing on
    device) and whose ``symbolic`` emits a clean single-output
    ``npu::NPUPromptFlashAttention`` node carrying the scalar params as
    attributes.  ATC matches the op by op_type after domain stripping.

    Requires a working ``torch_npu`` on the export host (guaranteed on an NPU
    device).  If unavailable the patch is skipped and attention keeps the
    eager implementation.  ORT CPU has no ``NPUPromptFlashAttention`` kernel,
    so ONNX-vs-PyTorch verification must be skipped when this patch is active.
    """
    import importlib

    undo_log: list[tuple[Any, str, Any]] = []

    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        LOGGER.warning(
            "npu flash-attention patch requested but torch_npu is unavailable "
            "(%s); skipping — attention keeps the eager implementation",
            exc,
        )
        return undo_log

    import torch as _torch

    # Single-output custom symbolic (see docstring — torch_npu's own wrapper
    # is broken).  forward runs the real fused kernel; symbolic emits one
    # npu::NPUPromptFlashAttention node with the scalar params as attributes.
    class _NpuPromptFlashAttn(_torch.autograd.Function):
        @staticmethod
        def forward(ctx, query, key, value, atten_mask, num_heads, num_kv_heads, scale_value):
            # Call the raw aten op directly. ``import torch_npu.onnx`` (done by
            # the npu rope patch) REPLACES the python-level
            # ``torch_npu.npu_prompt_flash_attention`` with a broken ONNX
            # wrapper (extra ``self`` arg, all-positional), so we must NOT go
            # through it. The underlying aten op is untouched and takes the
            # native keyword schema. The op name varies by torch_npu build
            # (``npu_prompt_flash_attention`` vs ``prompt_flash_attention``),
            # so resolve whichever exists.
            _op = getattr(
                _torch.ops.npu,
                "npu_prompt_flash_attention",
                getattr(_torch.ops.npu, "prompt_flash_attention", None),
            )
            # 310P's PromptFlashAttention does NOT support a non-null pse_shift
            # (runtime tiling fails: "not support 310P when pse is not null"),
            # so we never pass one — q/k/v + atten_mask only.
            return _op(
                query,
                key,
                value,
                atten_mask=atten_mask,
                num_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                input_layout="BNSD",
                scale_value=scale_value,
                pre_tokens=65535,
                next_tokens=65535,
            )

        @staticmethod
        def symbolic(g, query, key, value, atten_mask, num_heads, num_kv_heads, scale_value):
            # Emit the node in the "npu" DOMAIN.  ATC's onnx plugin registers
            # NPUPromptFlashAttention under BOTH ai.onnx::11-16,19 AND npu::1.
            # The ai.onnx (default-domain) parser EXPANDS the node into a
            # subgraph and FORCES a real pse_shift anchor at input index 3 —
            # which 310P's kernel then rejects at runtime ("not support 310P
            # when pse is not null").  The npu::1 parser uses a tolerant
            # ParseParams that accepts q/k/v/atten_mask with NO pse_shift (this
            # is the path the golden reference export uses).  ATC permits the
            # npu-domain node alongside default-domain RoPE/RMSNorm as long as
            # opset_import stays a SINGLE entry (the convert script keeps only
            # the default ai.onnx opset and does NOT strip this node's domain).
            return g.op(
                "npu::NPUPromptFlashAttention",
                query,
                key,
                value,
                atten_mask,
                num_heads_i=num_heads,
                num_key_value_heads_i=num_kv_heads,
                scale_value_f=scale_value,
                input_layout_s="BNSD",
                pre_tokens_i=65535,
                next_tokens_i=65535,
            )

    def _flash_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=0.0,  # noqa: ARG001 — flash kernel has no train-time dropout here
        _fa=_NpuPromptFlashAttn,
        **kwargs,
    ):
        # query: (B, N, S, D); key/value: (B, Nkv, S, D) — GQA, *not* repeated.
        # Head counts MUST be Python ints (they become ONNX node *attributes*);
        # reading them from query.shape[1] yields a traced Tensor (a Gather op)
        # which i_() rejects.  Pull the static values from the module/config.
        num_heads = int(getattr(module, "num_heads", None) or module.config.num_attention_heads)
        num_kv_heads = int(getattr(module, "num_key_value_heads", None) or module.config.num_key_value_heads)
        seq_len = query.shape[2]

        if attention_mask is None:
            raise ValueError(
                "npu flash-attention patch requires an attention mask; PI05 VLM/AE always supply a causal mask"
            )
        # HF additive float mask (0 = attend, finfo.min = mask) -> bool mask
        # where True == masked, which is npu prompt-flash's convention.
        atten_mask = attention_mask < 0

        # PromptFlashAttention rejects a non-NULL atten_mask when the sequence
        # length is not 16-aligned ("attention mask must be NULL, when Qs,Kvs
        # is unAlign").  PI05's VLM prefix (e.g. 712 tokens) is unaligned, so
        # pad q/k/v + mask up to the next multiple of 16, then slice the real
        # rows back out.  Padded KEY columns are masked (True) so real queries
        # ignore them; padded QUERY rows are discarded after the kernel.
        align = 16
        pad = (align - seq_len % align) % align
        if pad:
            # q/k/v are (B, H, S, D): pad the seq dim (-2) at the end.
            query = torch.nn.functional.pad(query, (0, 0, 0, pad))
            key = torch.nn.functional.pad(key, (0, 0, 0, pad))
            value = torch.nn.functional.pad(value, (0, 0, 0, pad))
            # mask is (B, 1, Sq, Skv): pad both query rows and key columns;
            # padded entries are masked (True).
            atten_mask = torch.nn.functional.pad(atten_mask, (0, pad, 0, pad), value=True)

        attn_output = _fa.apply(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            atten_mask,
            num_heads,
            num_kv_heads,
            float(scaling),
        )

        if pad:
            # drop the padded query rows -> back to the real seq length.
            attn_output = attn_output[:, :, :seq_len, :]

        # BNSD (B, N, S, D) -> eager's contract (B, S, N, D)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, None

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

        mod.eager_attention_forward = _flash_attention_forward
        undo_log.append((mod, "eager_attention_forward", orig))
        LOGGER.info(
            "Patched %s.eager_attention_forward (torch_npu.npu_prompt_flash_attention -> NPUPromptFlashAttention)",
            mod_path,
        )

    return undo_log


# ---------------------------------------------------------------------------
# Post-export ONNX sanitization
# ---------------------------------------------------------------------------


def normalize_slice_for_atc(model) -> None:
    """Normalize Slice nodes to the form ATC's StridedSliceD can compile.

    Empirically ATC's ``StridedSliceD`` chokes on the combination
    ``axes=<positive int>`` + explicit ``steps=1`` input that the
    TorchScript ONNX exporter emits for ``x[..., a:b]`` (non-dynamo path,
    opset 14).  The dynamo exporter (opset 18) instead emits
    ``axes=[-1]`` and **omits** the steps input, and that form compiles
    fine — see the AE OM build for a working example.

    This pass rewrites every Slice node so it matches the AE form:

    1. Drop the 5th input (``steps``) when its initializer is all 1s.
    2. Convert positive ``axes`` initializer entries to negative
       (``axis - rank(data)``) using shape info from value_info / inputs
       / outputs.

    Both transformations are semantics-preserving.  Step 1 alone is
    usually enough to unblock ATC; step 2 is a belt-and-suspenders fix
    that also makes the graph diff cleanly against the AE export.
    """
    import os
    import tempfile

    import numpy as np
    import onnx
    from onnx import numpy_helper, shape_inference

    # ---- Step 1: build rank map -----------------------------------------
    # In-memory shape_inference fails on >2GB models ("Failed to serialize
    # proto").  Try the on-disk variant first; fall back to a lightweight
    # forward rank-propagation pass if even that fails.
    rank_map: dict[str, int] = {}

    def _ranks_from_value_info(vi_list):
        out = {}
        for vi in vi_list:
            tt = vi.type.tensor_type
            if tt.HasField("shape"):
                out[vi.name] = len(tt.shape.dim)
        return out

    # seed from declared inputs/outputs (always available)
    rank_map.update(_ranks_from_value_info(model.graph.input))
    rank_map.update(_ranks_from_value_info(model.graph.output))
    rank_map.update(_ranks_from_value_info(model.graph.value_info))
    # initializers carry shape directly
    for ini in model.graph.initializer:
        rank_map[ini.name] = len(ini.dims)

    try:
        with tempfile.TemporaryDirectory() as td:
            tmp_in = os.path.join(td, "in.onnx")
            tmp_out = os.path.join(td, "out.onnx")
            # Save without external data so the file is self-contained for
            # shape inference; tensors stay in raw_data already.
            onnx.save(
                model,
                tmp_in,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="in.onnx.data",
                size_threshold=1024,
            )
            # WARNING: onnx.save(save_as_external_data=True) MUTATES the
            # model in place — every large initializer's raw_data is
            # cleared and replaced with an external_data reference into
            # the temp directory.  When the TemporaryDirectory exits the
            # data files are deleted, leaving the caller's model proto
            # pointing at nothing (downstream onnx.save would write empty
            # tensors).  Re-load the bytes back into raw_data and strip
            # the external_data refs to restore self-containment.
            from onnx.external_data_helper import load_external_data_for_model

            load_external_data_for_model(model, td)
            for tensor in model.graph.initializer:
                if tensor.HasField("data_location"):
                    tensor.ClearField("data_location")
                del tensor.external_data[:]

            shape_inference.infer_shapes_path(tmp_in, tmp_out, strict_mode=False)
            inferred = onnx.load(tmp_out, load_external_data=False)
            rank_map.update(_ranks_from_value_info(inferred.graph.value_info))
            LOGGER.info(
                "normalize_slice_for_atc: shape inference (path) added %d ranks",
                len(inferred.graph.value_info),
            )
    except Exception as exc:
        LOGGER.warning(
            "normalize_slice_for_atc: infer_shapes_path failed (%s); falling back to forward rank propagation",
            exc,
        )

    # ---- Step 2: constant lookup ----------------------------------------
    init_map = {ini.name: ini for ini in model.graph.initializer}
    const_node_map: dict[str, Any] = {}
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.t.ByteSize() > 0:
                const_node_map[node.output[0]] = attr.t
                break

    def _const_array(name):
        ini = init_map.get(name) or const_node_map.get(name)
        if ini is None:
            return None
        try:
            return numpy_helper.to_array(ini)
        except Exception:
            return None

    # ---- Step 3: forward-propagate ranks (handles ops infer_shapes missed)
    # Most ops preserve input rank; a few mutate it.  We do a single pass
    # in topological order (ONNX graphs are stored topologically already).
    def _infer_node_output_rank(node):
        op = node.op_type
        in_ranks = [rank_map.get(n) for n in node.input]
        first = next((r for r in in_ranks if r is not None), None)

        if op == "Transpose":
            for attr in node.attribute:
                if attr.name == "perm":
                    return len(attr.ints)
            return first
        if op in ("Reshape", "Expand"):
            arr = _const_array(node.input[1]) if len(node.input) >= 2 else None
            return int(arr.size) if arr is not None else None
        if op in ("Unsqueeze",):
            axes_arr = _const_array(node.input[1]) if len(node.input) >= 2 else None
            n_axes = (
                int(axes_arr.size)
                if axes_arr is not None
                else sum(len(a.ints) for a in node.attribute if a.name == "axes")
            )
            return (first or 0) + n_axes if first is not None else None
        if op in ("Squeeze",):
            axes_arr = _const_array(node.input[1]) if len(node.input) >= 2 else None
            n_axes = (
                int(axes_arr.size)
                if axes_arr is not None
                else sum(len(a.ints) for a in node.attribute if a.name == "axes")
            )
            return max(0, (first or 0) - n_axes) if first is not None else None
        if op == "Gather":
            # rank = data_rank + indices_rank - 1
            d, i = in_ranks[0], in_ranks[1] if len(in_ranks) > 1 else None
            return (d + i - 1) if d is not None and i is not None else None
        if op == "Concat":
            return first
        if op == "MatMul":
            a, b = in_ranks[0], in_ranks[1] if len(in_ranks) > 1 else None
            if a is None or b is None:
                return None
            return max(a, b)
        if op == "Where":
            return max((r for r in in_ranks if r is not None), default=None)
        if op == "Shape":
            return 1
        if op in ("Constant",):
            for attr in node.attribute:
                if attr.name == "value":
                    return len(attr.t.dims)
            return None
        # Default: most elementwise / unary ops preserve rank of first input
        return first

    for node in model.graph.node:
        for out in node.output:
            if out and out not in rank_map:
                r = _infer_node_output_rank(node)
                if r is not None:
                    rank_map[out] = r

    # ---- Step 4: rewrite Slice nodes ------------------------------------
    dropped_steps = 0
    flipped_axes = 0
    total_slices = 0
    skipped_no_axes_const = 0
    skipped_no_rank = 0
    for node in model.graph.node:
        if node.op_type != "Slice":
            continue
        total_slices += 1

        # 1) drop trivial steps=[1,1,...]
        if len(node.input) >= 5 and node.input[4]:
            steps_arr = _const_array(node.input[4])
            if steps_arr is not None and np.all(steps_arr == 1):
                while len(node.input) > 4:
                    node.input.pop()
                dropped_steps += 1

        # 2) flip positive axes -> negative
        if len(node.input) >= 4 and node.input[3]:
            axes_name = node.input[3]
            axes_arr = _const_array(axes_name)
            data_rank = rank_map.get(node.input[0])
            if axes_arr is None:
                skipped_no_axes_const += 1
                continue
            if data_rank is None:
                skipped_no_rank += 1
                continue
            if np.any(axes_arr >= 0):
                new_axes = np.where(axes_arr >= 0, axes_arr - data_rank, axes_arr).astype(axes_arr.dtype)
                new_name = f"_atc_axes_{node.name}".replace("/", "_")
                model.graph.initializer.append(numpy_helper.from_array(new_axes, name=new_name))
                node.input[3] = new_name
                flipped_axes += 1

    LOGGER.info(
        "normalize_slice_for_atc: %d Slice node(s) | dropped %d steps | "
        "flipped %d axes | skipped %d (no axes const) + %d (no rank)",
        total_slices,
        dropped_steps,
        flipped_axes,
        skipped_no_axes_const,
        skipped_no_rank,
    )


def rewrite_slice_pairs_to_split(model) -> None:
    """Replace rotate_half-style Slice pairs with a single Split node.

    Pattern matched (per data input + axis):

        Slice(data, starts=[0],   ends=[K],  axes=[a])  -> first_half
        Slice(data, starts=[K],   ends=[N],  axes=[a])  -> second_half

    where N is the data's size on axis ``a`` (or "very large" sentinel
    like INT_MAX, which the exporter uses for ``x[..., K:]``).  Both
    Slices must have all-constant inputs and steps==1 (or absent).

    They are replaced with::

        Split(data, sizes=[K, N-K], axis=a) -> first_half, second_half

    which lowers to ``SplitVD`` on Ascend and bypasses the buggy
    ``StridedSliceD`` kernel entirely.

    No-op for any Slice that doesn't match the pattern.  Idempotent.
    """
    import os
    import tempfile

    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, shape_inference

    INT_SENTINEL = 2_147_483_640  # exporter writes 2^63-1 / INT_MAX-ish for "to end"

    # --- Resolve shapes (size on each axis), reusing the path-based trick ---
    shape_map: dict[str, list[int]] = {}

    def _shapes_from_value_info(vi_list):
        out = {}
        for vi in vi_list:
            tt = vi.type.tensor_type
            if tt.HasField("shape"):
                out[vi.name] = [d.dim_value if d.dim_value > 0 else -1 for d in tt.shape.dim]
        return out

    shape_map.update(_shapes_from_value_info(model.graph.input))
    shape_map.update(_shapes_from_value_info(model.graph.output))
    shape_map.update(_shapes_from_value_info(model.graph.value_info))
    for ini in model.graph.initializer:
        shape_map[ini.name] = list(ini.dims)

    try:
        with tempfile.TemporaryDirectory() as td:
            tmp_in = os.path.join(td, "in.onnx")
            tmp_out = os.path.join(td, "out.onnx")
            onnx.save(
                model,
                tmp_in,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="in.onnx.data",
                size_threshold=1024,
            )
            from onnx.external_data_helper import load_external_data_for_model

            load_external_data_for_model(model, td)
            for tensor in model.graph.initializer:
                if tensor.HasField("data_location"):
                    tensor.ClearField("data_location")
                del tensor.external_data[:]
            shape_inference.infer_shapes_path(tmp_in, tmp_out, strict_mode=False)
            inferred = onnx.load(tmp_out, load_external_data=False)
            shape_map.update(_shapes_from_value_info(inferred.graph.value_info))
    except Exception as exc:
        LOGGER.warning(
            "rewrite_slice_pairs_to_split: shape inference failed (%s); may miss some pairs",
            exc,
        )

    # --- Constant lookup (initializer + Constant nodes) ---
    init_map = {ini.name: ini for ini in model.graph.initializer}
    const_node_map: dict[str, Any] = {}
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.t.ByteSize() > 0:
                const_node_map[node.output[0]] = attr.t
                break

    def _const_array(name):
        ini = init_map.get(name) or const_node_map.get(name)
        if ini is None:
            return None
        try:
            return numpy_helper.to_array(ini)
        except Exception:
            return None

    # --- Collect candidate Slices (all-constant, single-axis, step=1) ---
    # Each entry: (idx_in_node_list, node, data, axis_int, start_int, end_int)
    candidates: list[tuple[int, Any, str, int, int, int]] = []
    skipped_inputs = 0
    skipped_non_const = 0
    skipped_steps = 0
    total_slice_seen = 0
    for idx, node in enumerate(model.graph.node):
        if node.op_type != "Slice":
            continue
        total_slice_seen += 1
        if len(node.input) < 3:
            skipped_inputs += 1
            continue
        starts = _const_array(node.input[1])
        ends = _const_array(node.input[2])
        axes = _const_array(node.input[3]) if len(node.input) >= 4 else None
        steps = _const_array(node.input[4]) if len(node.input) >= 5 else None
        if starts is None or ends is None or axes is None or starts.size != 1 or ends.size != 1 or axes.size != 1:
            skipped_non_const += 1
            continue
        if steps is not None and not np.all(steps == 1):
            skipped_steps += 1
            continue
        candidates.append(
            (
                idx,
                node,
                node.input[0],
                int(axes[0]),
                int(starts[0]),
                int(ends[0]),
            )
        )

    LOGGER.info(
        "rewrite_slice_pairs_to_split: %d Slice seen | %d candidates | "
        "skipped %d (few inputs) + %d (non-const) + %d (steps!=1)",
        total_slice_seen,
        len(candidates),
        skipped_inputs,
        skipped_non_const,
        skipped_steps,
    )

    # --- Group by (data, axis) and find rotate_half pairs ---
    groups: dict[tuple[str, int], list] = {}
    for c in candidates:
        groups.setdefault((c[2], c[3]), []).append(c)

    new_nodes: list[Any] = []
    new_initializers: list[Any] = []
    nodes_to_remove: set[int] = set()
    pairs_replaced = 0

    for (data, axis), entries in groups.items():
        if len(entries) < 2:
            continue
        data_shape = shape_map.get(data)
        if data_shape is None:
            continue
        # Normalize axis to non-negative for indexing the shape
        pos_axis = axis if axis >= 0 else axis + len(data_shape)
        if not (0 <= pos_axis < len(data_shape)):
            continue
        axis_size = data_shape[pos_axis]
        if axis_size <= 0:
            continue

        # Find a "first half" (start=0, end=K) and matching "second half"
        # (start=K, end=N or sentinel).
        firsts = [e for e in entries if e[4] == 0 and 0 < e[5] < axis_size]
        seconds_by_start = {}
        for e in entries:
            end_val = e[5]
            # Treat huge ends as axis_size
            if end_val >= INT_SENTINEL or end_val == axis_size or end_val == -1:
                seconds_by_start.setdefault(e[4], []).append(e)
            elif end_val < 0 and (axis_size + end_val) > e[4]:
                # negative end like -1 ... handle as axis_size + end
                seconds_by_start.setdefault(e[4], []).append(e)

        for first in firsts:
            K = first[5]
            partners = seconds_by_start.get(K)
            if not partners:
                continue
            second = partners[0]

            split_sizes = np.array([K, axis_size - K], dtype=np.int64)
            sizes_name = f"_atc_split_sizes_pair_{first[1].name}".replace("/", "_")
            new_initializers.append(numpy_helper.from_array(split_sizes, name=sizes_name))
            split_node = helper.make_node(
                "Split",
                inputs=[data, sizes_name],
                outputs=[first[1].output[0], second[1].output[0]],
                name=f"_atc_split_pair_{first[1].name}".replace("/", "_"),
                axis=axis,  # ONNX Split accepts negative axis since opset 13
            )
            # Insert at position of the first Slice so topology stays valid
            new_nodes.append((min(first[0], second[0]), split_node))
            nodes_to_remove.add(first[0])
            nodes_to_remove.add(second[0])
            pairs_replaced += 1

    if pairs_replaced == 0:
        LOGGER.info("rewrite_slice_pairs_to_split: no rotate_half pairs found")
        return

    # --- Apply changes: rebuild node list ---
    insertions: dict[int, list] = {}
    for pos, n in new_nodes:
        insertions.setdefault(pos, []).append(n)

    rebuilt = []
    for i, node in enumerate(model.graph.node):
        if i in insertions:
            rebuilt.extend(insertions[i])
        if i not in nodes_to_remove:
            rebuilt.append(node)

    del model.graph.node[:]
    model.graph.node.extend(rebuilt)
    model.graph.initializer.extend(new_initializers)

    LOGGER.info(
        "rewrite_slice_pairs_to_split: replaced %d Slice pair(s) with Split node(s)",
        pairs_replaced,
    )


def sanitize_nan_initializers(model) -> None:
    """Replace NaN values in float16/float32 initializers with zero.

    The torch dynamo ONNX exporter (``torch.onnx.export(dynamo=True)``)
    occasionally produces a handful of corrupted float16 weight values
    (e.g. raw bits ``0x7d00``) that are valid NaN bit patterns.  These
    NaN weights silently propagate through MatMul/Add nodes at inference
    time and corrupt all downstream activations.

    This function scans every initializer and zeroes out NaN elements.
    Typically only 1–8 values out of hundreds of millions are affected.
    """
    import numpy as np
    import onnx

    _DTYPE_MAP = {
        onnx.TensorProto.FLOAT16: np.float16,
        onnx.TensorProto.FLOAT: np.float32,
        onnx.TensorProto.DOUBLE: np.float64,
    }
    total_fixed = 0
    for tensor in model.graph.initializer:
        np_dtype = _DTYPE_MAP.get(tensor.data_type)
        if np_dtype is None:
            continue
        arr = np.frombuffer(tensor.raw_data, dtype=np_dtype)
        nan_mask = np.isnan(arr)
        n_nan = int(nan_mask.sum())
        if n_nan == 0:
            continue
        LOGGER.warning(
            "Initializer %s (%s, %s elems): %d NaN value(s) replaced with 0",
            tensor.name,
            np_dtype.__name__,
            arr.size,
            n_nan,
        )
        arr = arr.copy()
        arr[nan_mask] = 0
        tensor.raw_data = arr.tobytes()
        total_fixed += n_nan
    if total_fixed:
        LOGGER.info("Sanitized %d total NaN value(s) across initializers", total_fixed)


def downgrade_split_for_atc(model) -> None:
    """Convert opset-18 ``Split`` nodes to opset-13 compatible format.

    In opset 18 the ``Split`` operator uses a ``num_outputs`` **attribute**
    and has only one input (the data tensor).  Ascend ATC does not support
    this form and fails with::

        Current num_outputs not surpport

    ORT (with opset 18) requires either ``num_outputs`` or an explicit
    ``split`` input — so we cannot simply delete the attribute.

    This function rewrites every affected ``Split`` node in-place:

    1. Reads the split-axis dimension size from the graph's shape info.
    2. Creates an explicit ``split`` sizes initializer (e.g. ``[128, 128]``).
    3. Wires it as the 2nd input to the ``Split`` node.
    4. Removes the ``num_outputs`` attribute.

    This satisfies both ORT (has ``split`` input) and ATC (no ``num_outputs``).
    """
    import numpy as np
    from onnx import numpy_helper

    # --- Build a name → shape map from value_info, inputs, outputs ---
    shape_map: dict[str, list[int]] = {}
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        tt = vi.type.tensor_type
        if tt.HasField("shape"):
            dims = [d.dim_value if d.dim_value > 0 else -1 for d in tt.shape.dim]
            shape_map[vi.name] = dims

    fixed = 0
    for node in model.graph.node:
        if node.op_type != "Split":
            continue

        # Only touch nodes that carry the opset-18 num_outputs attribute
        num_outputs_val = None
        num_outputs_idx = None
        axis_val = 0
        for idx, attr in enumerate(node.attribute):
            if attr.name == "num_outputs":
                num_outputs_val = attr.i
                num_outputs_idx = idx
            elif attr.name == "axis":
                axis_val = attr.i

        if num_outputs_val is None:
            continue  # already opset-13 style

        if len(node.input) >= 2 and node.input[1]:
            continue  # already has a split-sizes input

        # Resolve the split-axis dimension size
        input_shape = shape_map.get(node.input[0])
        if input_shape is None or axis_val >= len(input_shape) or input_shape[axis_val] <= 0:
            LOGGER.warning(
                "Split node %s: cannot determine dim size on axis %d — skipping",
                node.name,
                axis_val,
            )
            continue

        dim_size = input_shape[axis_val]
        chunk = dim_size // num_outputs_val
        sizes = np.array([chunk] * num_outputs_val, dtype=np.int64)

        # Create initializer and wire as 2nd input
        split_name = f"_atc_split_sizes_{node.name}"
        model.graph.initializer.append(numpy_helper.from_array(sizes, name=split_name))
        if len(node.input) < 2:
            node.input.append(split_name)
        else:
            node.input[1] = split_name

        # Remove num_outputs attribute
        del node.attribute[num_outputs_idx]
        fixed += 1

    if fixed:
        LOGGER.info(
            "Downgraded %d Split node(s) from opset-18 to opset-13 format for ATC",
            fixed,
        )


def downgrade_ir_version(model, *, max_ir_version: int = 9) -> None:
    """Cap the ONNX IR version so older runtimes can load the model.

    ``torch.onnx.dynamo_export`` with recent ONNX packages may produce
    IR version 10, but ORT ≤ 1.18 only supports up to IR version 9.
    This simply clamps the field without changing any graph semantics.
    """
    if model.ir_version > max_ir_version:
        LOGGER.info(
            "Downgrading ONNX IR version %d → %d",
            model.ir_version,
            max_ir_version,
        )
        model.ir_version = max_ir_version


# ---------------------------------------------------------------------------
# Patch registry — add new entries here
# ---------------------------------------------------------------------------

_PATCH_REGISTRY: list[tuple[str, Any]] = [
    # (human-readable label, callable that returns undo_log)
    # NOTE: rotate_half is no longer patched — the new reshape-based
    # apply_rotary_pos_emb bypasses rotate_half entirely (no Slice/Split
    # in the RoPE subgraph), which is what lets us keep
    # TransdataTransposeTransdataFusionPass on without ATC mis-compiling.
    ("AdaRMSNorm (fp32→model_dtype)", _patch_gemma_ada_rmsnorm),
    ("apply_rotary_pos_emb (reshape-based, no Slice/Split)", _patch_gemma_rotary_pos_emb),
    ("eager_attention (value→query dtype)", _patch_gemma_eager_attention),
]


def _build_patch_registry(
    use_npu_ops: bool,
    *,
    softmax_in_model_dtype: bool = False,
    mqa_broadcast: bool = False,
    fast_gelu: bool = False,
) -> list[tuple[str, Any]]:
    """Return the active patch registry for the requested export mode.

    When ``use_npu_ops`` is set (i.e. exporting on an NPU device), the
    ORT-runnable reshape-based RoPE patch is swapped for the
    :func:`_patch_gemma_rotary_pos_emb_npu` variant that routes RoPE through
    the real ``torch_npu.npu_rotary_mul`` fused operator (emitted as an
    ``NPURotaryMul`` ONNX node).  Future NPU-affine optimizations
    (flash-attention, layernorm, qkv fusion) gate on this same flag.

    ``softmax_in_model_dtype`` / ``mqa_broadcast`` are action-expert-only
    attention optimisations threaded into :func:`_patch_gemma_eager_attention`;
    they default off so the VLM export (host-side fp32 prefix mask) is unchanged.

    ``fast_gelu`` routes gelu_pytorch_tanh through Ascend NPUFastGelu.  It is
    faster but numerically different from the tanh GELU used by PyTorch, so it
    defaults off and must be enabled explicitly after end-to-end validation.
    """
    rope_patch = (
        ("apply_rotary_pos_emb (torch_npu.npu_rotary_mul)", _patch_gemma_rotary_pos_emb_npu)
        if use_npu_ops
        else ("apply_rotary_pos_emb (reshape-based, no Slice/Split)", _patch_gemma_rotary_pos_emb)
    )
    rmsnorm_patch = (
        ("GemmaRMSNorm (torch_npu.npu_rms_norm)", _patch_gemma_ada_rmsnorm_npu)
        if use_npu_ops
        else ("AdaRMSNorm (fp32→model_dtype)", _patch_gemma_ada_rmsnorm)
    )
    # On NPU, the whole attention (score/mask/softmax/context) would ideally
    # collapse into a single npu_prompt_flash_attention kernel.  HOWEVER, flash
    # attention is DISABLED: it is fundamentally incompatible with 310P + ATC
    # (see "NPU 算子替换总结.md" — the default-domain PromptFlashAttention parser
    # forces a pse_shift input that 310P rejects, the npu domain cannot coexist
    # with default-domain standard ops under ATC's single-opset_import rule, and
    # a mask-less FA would corrupt PI05's always-padded prefix attention).  So on
    # NPU we keep the same ORT-runnable eager path (with the value→query dtype
    # fix) used off NPU; only RoPE + RMSNorm are routed to npu fused operators.
    attention_patch = (
        # ("flash attention (torch_npu.npu_prompt_flash_attention)", _patch_gemma_flash_attention_npu)
        # if use_npu_ops
        # else ("eager_attention (value→query dtype)", _patch_gemma_eager_attention)
        (
            "eager_attention (value→query dtype"
            + (", mqa_broadcast" if mqa_broadcast else "")
            + (", softmax_in_model_dtype" if softmax_in_model_dtype else "")
            + ")",
            functools.partial(
                _patch_gemma_eager_attention,
                softmax_in_model_dtype=softmax_in_model_dtype,
                mqa_broadcast=mqa_broadcast,
            ),
        )
    )
    # gelu_pytorch_tanh (Gemma text MLP + SigLIP vision MLP) decomposes into a
    # Mul/Add/Tanh elementwise chain at opset <= 18 — the single largest "glue"
    # cost in the VLM profile, unfused by ATC's GeluFusionPass.  On NPU, route
    # it to the fused NPUFastGelu kernel (default-domain registered at opset
    # 11-18, like RoPE/RMSNorm; NPUGeluV2 is npu-domain-only and thus unusable).
    # NOTE: FastGelu is a sigmoid-style approximation (NOT the exact tanh form),
    # so this introduces a small bounded error (max ≈2.1e-2) — validate action
    # accuracy end-to-end before relying on it.  Off NPU there is no fused-gelu
    # kernel for ORT, so keep the decomposition.
    registry = [
        rmsnorm_patch,
        rope_patch,
        attention_patch,
        # qkv fusion DISABLED: standalone it is a slight pessimisation (ATC
        # already fuses the q/k/v MatMuls; the extra Split adds launch cost on
        # the short-seq action expert).  It only pays off when PAIRED with flash
        # attention (which consumes the split q/k/v directly) — and FA is
        # disabled on 310P (see above), so qkv fusion has no upside here.
        # ("fused qkv projection", _patch_gemma_fused_qkv),
    ]
    if use_npu_ops and fast_gelu:
        registry.append(("gelu_pytorch_tanh (torch_npu.fast_gelu -> NPUFastGelu)", _patch_pytorch_gelu_tanh_npu))
    # NPUGeglu is the accuracy-preserving NPU default. An explicit fast_gelu
    # request takes precedence for every gelu_pytorch_tanh site, including Gemma.
    if use_npu_ops and not fast_gelu:
        registry.append(("GemmaMLP gelu(gate)*up (torch_npu.npu_geglu -> NPUGeglu)", _patch_gemma_geglu_npu))
    return registry


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@contextmanager
def ascend_onnx_export_patches(
    use_npu_ops: bool = False,
    *,
    softmax_in_model_dtype: bool = False,
    mqa_broadcast: bool = False,
    fast_gelu: bool = False,
):
    """Context manager that applies **all** registered Ascend patches.

    Patches are applied on ``__enter__`` and reverted on ``__exit__``,
    regardless of whether an exception occurred.

    Args:
        use_npu_ops: When ``True`` (export on an NPU device), route RoPE
            through the ``torch_npu.npu_rotary_mul`` fused operator (emitted
            as an ``NPURotaryMul`` ONNX node) instead of the ORT-runnable
            reshape-based form.  The resulting graph requires an ATC
            custom-op plugin and cannot be verified with ORT CPU.  Future
            NPU-affine optimizations gate on this same flag.
        softmax_in_model_dtype: Action-expert only.  Keep the attention score
            matrix and softmax in query/model dtype (no fp32 upcast Cast).
            Requires an additive mask sentinel representable by that dtype.
            Leave ``False`` for the VLM export.
        mqa_broadcast: Action-expert only.  Skip the ``repeat_kv`` Expand for
            multi-query layers and rely on matmul broadcasting instead.
        fast_gelu: Route gelu_pytorch_tanh to NPUFastGelu. Disabled by default
            because it is an approximation and can degrade PI05 action accuracy.

    On NPU, the Gemma text MLP gelu(gate)*up is fused into one NPUGeglu by
    default (GeGluV2, numerically exact tanh). When ``fast_gelu`` is true,
    NPUGeglu is suppressed and every gelu_pytorch_tanh site is routed to
    NPUFastGelu instead.

    Example::

        with ascend_onnx_export_patches():
            torch.onnx.export(model, ...)
    """
    all_undo: list[tuple[Any, str, Any]] = []
    applied: list[str] = []

    for label, patch_fn in _build_patch_registry(
        use_npu_ops,
        softmax_in_model_dtype=softmax_in_model_dtype,
        mqa_broadcast=mqa_broadcast,
        fast_gelu=fast_gelu,
    ):
        try:
            undo = patch_fn()
            all_undo.extend(undo)
            if undo:
                applied.append(label)
        except Exception as exc:
            LOGGER.warning("Failed to apply patch '%s': %s", label, exc)

    if applied:
        LOGGER.info("Ascend export patches active: %s", ", ".join(applied))
    else:
        LOGGER.info("No Ascend export patches applied")

    try:
        yield
    finally:
        # Restore in reverse order
        for mod, attr, orig in reversed(all_undo):
            setattr(mod, attr, orig)
        if all_undo:
            LOGGER.info("Reverted %d Ascend export patch(es)", len(all_undo))
