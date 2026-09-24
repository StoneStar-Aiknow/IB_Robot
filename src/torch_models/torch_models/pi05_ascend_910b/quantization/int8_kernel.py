#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
# ruff: noqa: N803, N806
"""INT8 GEMM (int8 x int8 -> int32) and per-token dynamic activation quant.

Backend: a Triton kernel on CUDA — ``torch._int_mm``'s CUDA path is unavailable
on this class of device (sm_120 / cu128 falls back to ``addmm_cuda`` which has
no Int support), while Triton's ``tl.dot(int8, int8, out_dtype=int32)`` hits
the INT8 tensor cores directly with no shape constraints (verified exact vs the
CPU reference for M=1 upward). CPU uses ``torch._int_mm`` (M > 16) or a plain
int32 matmul — bit-identical semantics, used by unit tests and CPU-only runs.
"""

import torch


def _torch_npu():
    try:
        import torch_npu  # type: ignore
    except Exception:  # pragma: no cover - optional backend
        return None
    return torch_npu


try:  # Triton ships with torch's CUDA builds; keep CPU-only installs working.
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _int8_mm_kernel(
        A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
    ):
        pid_m, pid_n = tl.program_id(0), tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), dtype=tl.int32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            a_mask = (rm[:, None] < M) & (rk[None, :] < K)
            b_mask = (rk[:, None] < K) & (rn[None, :] < N)
            a = tl.load(A + rm[:, None] * sam + rk[None, :] * sak, mask=a_mask, other=0)
            b = tl.load(B + rk[:, None] * sbk + rn[None, :] * sbn, mask=b_mask, other=0)
            acc += tl.dot(a, b, out_dtype=tl.int32)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(C + rm[:, None] * scm + rn[None, :] * scn, acc, mask=c_mask)

    @triton.jit
    def _int8_mm_dequant_kernel(
        A,
        B,
        C,
        AS,
        WS,
        M,
        N,
        K,
        sam,
        sak,
        sbk,
        sbn,
        scm,
        scn,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        # 与 _int8_mm_kernel 相同的整数主循环;epilogue 在寄存器里完成
        # int32 -> fp32 * act_scale[row] * weight_scale[col] -> out dtype,
        # 省去输出张量上的三遍逐元素后处理(int8 前向的主要开销)。
        pid_m, pid_n = tl.program_id(0), tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), dtype=tl.int32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            a_mask = (rm[:, None] < M) & (rk[None, :] < K)
            b_mask = (rk[:, None] < K) & (rn[None, :] < N)
            a = tl.load(A + rm[:, None] * sam + rk[None, :] * sak, mask=a_mask, other=0)
            b = tl.load(B + rk[:, None] * sbk + rn[None, :] * sbn, mask=b_mask, other=0)
            acc += tl.dot(a, b, out_dtype=tl.int32)
        a_s = tl.load(AS + rm, mask=rm < M, other=0.0).to(tl.float32)
        w_s = tl.load(WS + rn, mask=rn < N, other=0.0).to(tl.float32)
        y = acc.to(tl.float32) * a_s[:, None] * w_s[None, :]
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(C + rm[:, None] * scm + rn[None, :] * scn, y.to(C.dtype.element_ty), mask=c_mask)


def int8_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a [M,K] int8 @ b [K,N] int8 -> [M,N] int32`` (int32 accumulation).

    Accepts non-contiguous ``b`` (e.g. a ``qweight.t()`` view) — the kernel
    reads through strides.
    """
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise TypeError(f"int8_mm expects int8 inputs, got {a.dtype} x {b.dtype}")
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError(f"shape mismatch: [{M},{K}] @ [{K2},{N}]")
    if not a.is_cuda:
        if M > 16 and b.is_contiguous():
            return torch._int_mm(a, b)
        return a.int() @ b.int()
    if not _HAS_TRITON:  # pragma: no cover
        raise RuntimeError("int8_mm on CUDA requires triton")
    c = torch.empty(M, N, dtype=torch.int32, device=a.device)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _int8_mm_kernel[grid](a, b, c, M, N, K, *a.stride(), *b.stride(), *c.stride(), BM=64, BN=64, BK=64)
    return c


def int8_mm_dequant(
    a: torch.Tensor,
    b: torch.Tensor,
    act_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Fused ``(a @ b).to(fp32) * act_scale[:, None] * weight_scale[None, :]``
    cast to ``out_dtype`` — the dequant epilogue runs inside the kernel.
    ``act_scale``: fp32 ``[M]``; ``weight_scale``: fp32 ``[N]``."""
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise TypeError(f"int8_mm_dequant expects int8 inputs, got {a.dtype} x {b.dtype}")
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError(f"shape mismatch: [{M},{K}] @ [{K2},{N}]")
    if a.device.type == "npu":
        torch_npu = _torch_npu()
        if torch_npu is None:
            raise RuntimeError("int8_mm_dequant on NPU requires torch_npu")
        output_dtype = out_dtype if out_dtype in (torch.float16, torch.bfloat16) else torch.float16
        return torch_npu.npu_quant_matmul(
            a,
            b,
            weight_scale,
            pertoken_scale=act_scale.reshape(-1),
            output_dtype=output_dtype,
        ).to(out_dtype)
    if not a.is_cuda:
        y = (a.int() @ b.int()).to(torch.float32)
        return (y * act_scale.reshape(-1, 1) * weight_scale.reshape(1, -1)).to(out_dtype)
    if not _HAS_TRITON:  # pragma: no cover
        raise RuntimeError("int8_mm_dequant on CUDA requires triton")
    c = torch.empty(M, N, dtype=out_dtype, device=a.device)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _int8_mm_dequant_kernel[grid](
        a,
        b,
        c,
        act_scale,
        weight_scale,
        M,
        N,
        K,
        *a.stride(),
        *b.stride(),
        *c.stride(),
        BM=64,
        BN=64,
        BK=64,
    )
    return c


def quantize_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic symmetric per-token (per-row) int8 quantization.

    Returns ``(codes int8, scale fp32)`` with ``scale = absmax(row)/127`` on
    the last dim (shape ``[..., 1]``) and ``codes = round(x/scale)`` clamped to
    ``[-127, 127]``. Scales are computed in fp32 regardless of ``x``'s dtype.
    """
    x32 = x.to(torch.float32)
    scale = (x32.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-8)
    codes = torch.round(x32 / scale).clamp(-127, 127).to(torch.int8)
    return codes, scale


def dynamic_int8_mm_dequant(
    x: torch.Tensor,
    b: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dynamic per-token activation quantization followed by W8A8 matmul.

    ``x``: fp ``[M, K]``; ``b``: int8 ``[K, N]`` (usually ``qweight.t()``);
    ``weight_scale``: fp32 ``[N]``. On NPU this uses
    ``npu_dynamic_quant + npu_quant_matmul``. Other backends use the existing
    PyTorch/Triton quantize + int8 matmul path.
    """
    if x.dim() != 2 or b.dim() != 2:
        raise ValueError(f"dynamic_int8_mm_dequant expects 2D inputs, got {x.dim()}D x {b.dim()}D")
    M, K = x.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError(f"shape mismatch: [{M},{K}] @ [{K2},{N}]")
    if b.dtype != torch.int8:
        raise TypeError(f"dynamic_int8_mm_dequant expects int8 weight, got {b.dtype}")
    if weight_scale.shape != (N,):
        raise ValueError(f"weight_scale shape mismatch: expected [{N}], got {list(weight_scale.shape)}")

    if x.device.type == "npu":
        torch_npu = _torch_npu()
        if torch_npu is None:
            raise RuntimeError("dynamic_int8_mm_dequant on NPU requires torch_npu")
        quant_input = x if x.dtype in (torch.float16, torch.bfloat16) else x.to(torch.float16)
        codes, act_scale = torch_npu.npu_dynamic_quant(quant_input, quant_mode="pertoken")
        output_dtype = out_dtype if out_dtype in (torch.float16, torch.bfloat16) else torch.float16
        return torch_npu.npu_quant_matmul(
            codes,
            b,
            weight_scale,
            pertoken_scale=act_scale.reshape(-1),
            output_dtype=output_dtype,
        ).to(out_dtype)

    codes, act_scale = quantize_per_token(x)
    return int8_mm_dequant(codes, b, act_scale.reshape(-1), weight_scale, out_dtype=out_dtype)
