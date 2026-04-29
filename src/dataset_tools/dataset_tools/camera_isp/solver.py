"""ISP color-matching solver — pure numpy, no ROS / cv2.

Three public entry points, all returning V4L2 parameter recommendations
that the calibrator GUI then applies via ``ros2 param set``:

* :func:`auto_match_lab` — Lab P50 + Planckian-locus (Adobe-Camera-Raw style)
* :func:`manual_match_neutral` — pooled neutral-pixel constraint
* :func:`manual_match_patches` — color-checker weighted least-squares

All three honour ``device_caps`` (per-control min/max/step from
``v4l2-ctl --list-ctrls``) and force ``auto_white_balance=False`` whenever
they propose a manual ``white_balance`` (the two are mutually exclusive
on UVC: see ``usb_cam.cpp:340``).

The solver receives one (ref, cur) frame pair per call; the caller is
responsible for re-grabbing the live frame between iterations of
``auto_match_lab`` (so this module stays stateless and easily unit-testable).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .color_space import (
    bgr_to_lab,
    bgr_to_linear_rgb,
    bgr_to_xyz_chromaticity,
    mccamy_kelvin,
)
from .lut import lookup_kelvin


# --------------------------------------------------------------------------
# Linear-domain helpers (used to populate the *_sw debug fields the SW-ISP
# debug pane consumes; the legacy sRGB-domain numbers fed to lookup_kelvin /
# proposed are intentionally left untouched).
# --------------------------------------------------------------------------

def _linear_channel_means_bgr(bgr_uint8: np.ndarray) -> tuple[float, float, float]:
    """Return ``(B_lin, G_lin, R_lin)`` channel means in linear light.

    Pixels are sRGB-decoded before averaging, so the returned means follow
    physical linear-RGB semantics (channel multiplications are meaningful).
    Returns zeros if the input is empty.
    """
    if bgr_uint8.size == 0:
        return 0.0, 0.0, 0.0
    rgb_lin = bgr_to_linear_rgb(bgr_uint8)  # (..., 3) RGB order, [0, 1]
    flat = rgb_lin.reshape(-1, 3)
    mean_r = float(flat[:, 0].mean())
    mean_g = float(flat[:, 1].mean())
    mean_b = float(flat[:, 2].mean())
    return mean_b, mean_g, mean_r


def _linear_kr_kb_from_pool(bgr_uint8: np.ndarray) -> tuple[float, float] | None:
    """Compute linear-domain ``(kr, kb) = (G/R, G/B)`` from a neutral pool.

    Returns ``None`` if any channel mean is too small to give a stable ratio.
    """
    b, g, r = _linear_channel_means_bgr(bgr_uint8)
    if r < 1e-4 or b < 1e-4 or g < 1e-4:
        return None
    return g / r, g / b


def _solve_sw_wb_lsq(
    ref_bgr: np.ndarray,
    cur_bgr: np.ndarray,
    pairs: Sequence[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]],
) -> tuple[float, float, float, int] | None:
    """Linear-domain LSQ across *all* box pairs (neutral + plain).

    Solves per-channel ``ref_c = cur_c * (alpha * k_c)`` with G pinned by
    setting ``k_g = 1`` (so ``alpha`` is the linear exposure scale and
    ``k_r``/``k_b`` are residual chroma corrections). Each pair contributes
    three rows weighted by ``sqrt(min(N_ref, N_cur))``.

    Returns ``(alpha, kr, kb, total_px)`` or ``None`` if there isn't
    enough usable data. This matches the user's mental model:
    "every patch I picked should align after applying SW-ISP" — including
    coloured patches, not just neutral ones.
    """
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    total_px = 0
    for ref_box, cur_box in pairs:
        ref_px = _filter_saturated(_box_pixels(ref_bgr, ref_box))
        cur_px = _filter_saturated(_box_pixels(cur_bgr, cur_box))
        if ref_px.shape[0] < 5 or cur_px.shape[0] < 5:
            continue
        b_ref, g_ref, r_ref = _linear_channel_means_bgr(ref_px)
        b_cur, g_cur, r_cur = _linear_channel_means_bgr(cur_px)
        if min(g_cur, r_cur, b_cur) < 1e-5:
            continue
        w = float(np.sqrt(min(ref_px.shape[0], cur_px.shape[0])))
        # Columns: (alpha, beta_b, beta_r). Rows G, B, R.
        rows.append(np.array([g_cur, 0.0, 0.0]) * w);  rhs.append(g_ref * w)
        rows.append(np.array([0.0, b_cur, 0.0]) * w);  rhs.append(b_ref * w)
        rows.append(np.array([0.0, 0.0, r_cur]) * w);  rhs.append(r_ref * w)
        total_px += int(min(ref_px.shape[0], cur_px.shape[0]))
    if not rows:
        return None
    A = np.asarray(rows, dtype=np.float64)
    y = np.asarray(rhs, dtype=np.float64)
    try:
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    alpha, beta_b, beta_r = float(sol[0]), float(sol[1]), float(sol[2])
    if alpha <= 1e-6:
        return None
    return alpha, beta_r / alpha, beta_b / alpha, total_px


def _solve_sw_ccm_lsq(
    ref_bgr: np.ndarray,
    cur_bgr: np.ndarray,
    pairs: Sequence[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]],
    *,
    ridge_lambda: float = 0.0,
) -> tuple[np.ndarray, int, int] | None:
    """Linear-domain 3x3 CCM LSQ across all box pairs.

    Solves ``rgb_ref ≈ M @ rgb_cur`` for the full 9-parameter ``M`` (no
    diagonal restriction, no bias term). Each pair contributes 3 weighted
    equations (one per output channel).

    When ``ridge_lambda > 0`` the solve is regularised with shrinkage
    toward the identity matrix:

        argmin ‖Y - XMᵀ‖² + λ·s·‖M - I‖²

    where ``s = trace(XᵀWX)/k`` makes λ unit-free relative to the data
    magnitude. Without this, sparse / non-chromatic pair pools (e.g. 5
    patches dominated by one hue) produce wildly-overshooting matrices
    (observed ``diag=(1.26, 12.53, 1.40)`` on a blue-background scene)
    that clip to uint8 and posterise the SW pane into colour blocks.

    Returns ``(M, n_pairs_used, total_px)`` or ``None`` if the system is
    underdetermined (need ≥ 3 colour-distinct pairs for a stable solve;
    we accept ≥ 3 pairs and report rank in the diagnostic prints).
    The returned matrix already absorbs exposure — there is no separate
    ``alpha`` term.

    Future enhancement hooks (cross/squared products) will widen ``M`` to
    3xK and the design matrix accordingly.
    """
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    n_pairs = 0
    total_px = 0
    for ref_box, cur_box in pairs:
        ref_px = _filter_saturated(_box_pixels(ref_bgr, ref_box))
        cur_px = _filter_saturated(_box_pixels(cur_bgr, cur_box))
        if ref_px.shape[0] < 5 or cur_px.shape[0] < 5:
            continue
        b_ref, g_ref, r_ref = _linear_channel_means_bgr(ref_px)
        b_cur, g_cur, r_cur = _linear_channel_means_bgr(cur_px)
        if min(g_cur, r_cur, b_cur) < 1e-5:
            continue
        w = float(np.sqrt(min(ref_px.shape[0], cur_px.shape[0])))
        # Layout vec(M) row-major: [m_RR, m_RG, m_RB, m_GR, m_GG, m_GB, m_BR, m_BG, m_BB]
        z = np.zeros(9, dtype=np.float64)
        # Output R = m_RR*r + m_RG*g + m_RB*b
        row_r = z.copy(); row_r[0:3] = (r_cur, g_cur, b_cur); row_r *= w
        # Output G = m_GR*r + m_GG*g + m_GB*b
        row_g = z.copy(); row_g[3:6] = (r_cur, g_cur, b_cur); row_g *= w
        # Output B = m_BR*r + m_BG*g + m_BB*b
        row_b = z.copy(); row_b[6:9] = (r_cur, g_cur, b_cur); row_b *= w
        rows.extend((row_r, row_g, row_b))
        rhs.extend((r_ref * w, g_ref * w, b_ref * w))
        n_pairs += 1
        total_px += int(min(ref_px.shape[0], cur_px.shape[0]))

    if n_pairs < 3:
        return None
    A = np.asarray(rows, dtype=np.float64)
    y = np.asarray(rhs, dtype=np.float64)
    if ridge_lambda > 0.0:
        # Shrink toward identity. vec(I) under the row-major layout used
        # above is [1,0,0, 0,1,0, 0,0,1].
        AtA = A.T @ A
        scale = float(np.trace(AtA)) / 9.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        I_vec = np.zeros(9, dtype=np.float64)
        I_vec[0] = I_vec[4] = I_vec[8] = 1.0
        try:
            sol = np.linalg.solve(
                AtA + ridge_lambda * scale * np.eye(9),
                A.T @ y + ridge_lambda * scale * I_vec,
            )
        except np.linalg.LinAlgError:
            return None
        rank = 9  # ridged system is full-rank by construction
    else:
        try:
            sol, residuals, rank, _sv = np.linalg.lstsq(A, y, rcond=None)
        except np.linalg.LinAlgError:
            return None
    if rank < 9:
        # Underdetermined: not enough chromatic diversity in the patches.
        # Caller will fall back to diag mode.
        return None
    M = sol.reshape(3, 3)
    if not np.all(np.isfinite(M)):
        return None
    return M, n_pairs, total_px


# --------------------------------------------------------------------------
# RPCC2 (Root-Polynomial Color Correction, 2nd order — 6-d feature)
#
# Reference (sec. 3 of the user-cited 2025-2026 CCM survey):
#     ρ(R,G,B) = [R, G, B, √(RG), √(RB), √(GB)]
#
# Property — exposure scale invariance:
#     ρ(k·rgb) = k · ρ(rgb)   ∀ k > 0
# A single (3,6) M absorbs both per-channel chroma correction and any
# exposure scaling, so the SW-ISP debug pane works without separately
# tracking ``alpha`` / ``exp_scale``.
# --------------------------------------------------------------------------


def _rpcc2_features(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RPCC2 feature transform.

    Input shape ``(N, 3)`` (linear RGB, ≥ 0). Output shape ``(N, 6)``.
    Negative inputs are clipped to 0 before the sqrt to keep the feature
    space real-valued; the caller is expected to operate in linear light
    where negatives are physically nonsensical anyway.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[-1] != 3:
        raise ValueError(f"_rpcc2_features expects (N,3); got {rgb.shape}")
    rgb = np.clip(rgb, 0.0, None)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return np.stack(
        [r, g, b, np.sqrt(r * g), np.sqrt(r * b), np.sqrt(g * b)],
        axis=1,
    )


# Pair-feature matrix shape contract:
#   X  : (N, K)   — K=3 for linear, K=6 for RPCC2
#   Y  : (N, 3)   — target RGB per pair
#   w  : (N,)     — sqrt(min(N_ref,N_cur)) for each pair
# All three RPCC2 solvers below share this contract so the calibrator can
# build (X,Y,w) once and dispatch by variant.


def _build_pair_pool(
    ref_bgr: np.ndarray,
    cur_bgr: np.ndarray,
    pairs: Sequence[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    """Collect per-pair (rgb_cur, rgb_ref, weight) triples.

    Returns ``(rgb_cur_pool, rgb_ref_pool, weights, total_px)`` shaped
    ``((N,3), (N,3), (N,), int)`` or ``None`` if zero usable pairs.
    Filters identical to ``_solve_sw_ccm_lsq`` (saturation mask, ≥ 5 px,
    min channel mean ≥ 1e-5).
    """
    rgb_cur_rows: list[tuple[float, float, float]] = []
    rgb_ref_rows: list[tuple[float, float, float]] = []
    weights: list[float] = []
    total_px = 0
    for ref_box, cur_box in pairs:
        ref_px = _filter_saturated(_box_pixels(ref_bgr, ref_box))
        cur_px = _filter_saturated(_box_pixels(cur_bgr, cur_box))
        if ref_px.shape[0] < 5 or cur_px.shape[0] < 5:
            continue
        b_ref, g_ref, r_ref = _linear_channel_means_bgr(ref_px)
        b_cur, g_cur, r_cur = _linear_channel_means_bgr(cur_px)
        if min(g_cur, r_cur, b_cur) < 1e-5:
            continue
        w = float(np.sqrt(min(ref_px.shape[0], cur_px.shape[0])))
        rgb_cur_rows.append((r_cur, g_cur, b_cur))
        rgb_ref_rows.append((r_ref, g_ref, b_ref))
        weights.append(w)
        total_px += int(min(ref_px.shape[0], cur_px.shape[0]))
    if not rgb_cur_rows:
        return None
    return (
        np.asarray(rgb_cur_rows, dtype=np.float64),
        np.asarray(rgb_ref_rows, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        total_px,
    )


def _solve_sw_rpcc2_lsq(
    X: np.ndarray, Y: np.ndarray, w: np.ndarray
) -> np.ndarray | None:
    """Plain weighted least-squares for RPCC2.

    Returns a ``(3, 6)`` matrix M such that ``Y ≈ X @ M.T``, or ``None``
    if rank-deficient or any non-finite. Requires ``N ≥ 6``.
    """
    if X.shape[0] < 6 or X.shape[1] != 6 or Y.shape[1] != 3:
        return None
    Xw = X * w[:, None]
    Yw = Y * w[:, None]
    try:
        sol, _resid, rank, _sv = np.linalg.lstsq(Xw, Yw, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 6:
        return None
    M = sol.T  # (3, 6)
    if not np.all(np.isfinite(M)):
        return None
    return M


def _solve_sw_rpcc2_ridge(
    X: np.ndarray, Y: np.ndarray, w: np.ndarray, lam: float
) -> np.ndarray | None:
    """Tikhonov ridge regression for RPCC2 with shrinkage toward identity.

    ``M.T = (XᵀWX + λ·s·I)⁻¹ (XᵀWY + λ·s·I_RPCC)`` with W = diag(w²)
    and ``s = trace(XᵀWX)/k`` so that ``λ`` is unit-free relative to the
    data magnitude. ``I_RPCC`` is the (6,3) target matrix whose top 3
    rows are the 3x3 identity — this shrinks the linear coefficients
    toward 1 and the cross-product coefficients toward 0 when data is
    sparse, instead of collapsing toward the zero matrix (which would
    paint the SW pane black).

    Without trace-relative scaling λ=1e-3 against linear-light RGB ∈[0,1]²
    is effectively zero and the ridge degenerates to plain LSQ for sparse
    pair pools.
    """
    if X.shape[0] < 6 or X.shape[1] != 6 or Y.shape[1] != 3:
        return None
    if not np.isfinite(lam) or lam < 0:
        return None
    Xw = X * w[:, None]
    Yw = Y * w[:, None]
    XtX = Xw.T @ Xw
    k = XtX.shape[0]
    scale = float(np.trace(XtX)) / max(k, 1)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    A = XtX + lam * scale * np.eye(k, dtype=np.float64)
    # Identity target: linear part of M -> I, cross-product part -> 0.
    I_target = np.zeros((k, 3), dtype=np.float64)
    I_target[:3, :] = np.eye(3)
    rhs = Xw.T @ Yw + lam * scale * I_target
    try:
        sol = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        return None
    M = sol.T
    if not np.all(np.isfinite(M)):
        return None
    return M


def _srgb_encode_arr(rgb_lin: np.ndarray) -> np.ndarray:
    """Vectorised sRGB OETF (linear → sRGB-encoded) for the ALS loss step."""
    rgb_lin = np.clip(rgb_lin, 0.0, 1.0)
    a = 0.055
    out = np.where(
        rgb_lin <= 0.0031308,
        rgb_lin * 12.92,
        (1.0 + a) * np.power(rgb_lin, 1.0 / 2.4) - a,
    )
    return np.clip(out, 0.0, 1.0)


def _rgb_to_lab_d65(rgb_lin: np.ndarray) -> np.ndarray:
    """Linear RGB (sRGB primaries, D65) → CIE LAB. Input ``(N,3)``."""
    # sRGB → XYZ (D65), Bradford-adapted matrix from Lindbloom.
    M_rgb_to_xyz = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = np.clip(rgb_lin, 0.0, 1.0) @ M_rgb_to_xyz.T
    # D65 reference white.
    xn, yn, zn = 0.95047, 1.0, 1.08883
    fx = _lab_f(xyz[:, 0] / xn)
    fy = _lab_f(xyz[:, 1] / yn)
    fz = _lab_f(xyz[:, 2] / zn)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=1)


def _lab_f(t: np.ndarray) -> np.ndarray:
    delta = 6.0 / 29.0
    return np.where(t > delta ** 3, np.cbrt(t), t / (3.0 * delta * delta) + 4.0 / 29.0)


def _solve_sw_rpcc2_als_lab(
    X: np.ndarray,
    Y: np.ndarray,
    w: np.ndarray,
    lam: float,
    *,
    max_iter: int = 20,
    tol: float = 1e-5,
    alpha: float = 0.05,
) -> tuple[np.ndarray, dict] | None:
    """ALS + RPCC2 with perceptual (LAB) IRLS reweighting.

    Initial ``M_0`` from ``_solve_sw_rpcc2_ridge``. At each iteration:

    1. Predict ``Y_hat = X @ M.T``.
    2. ΔE76 between ``Y_hat`` and ``Y`` in CIE LAB (D65, sRGB primaries).
    3. Reweight pairs by ``w' = w · 1/(1 + α·ΔE²)``.
    4. Re-solve the same ridge system with the new weights.

    Returns ``(M, info)`` where ``info`` contains ``{"iters": int,
    "delta_e_median_init": float, "delta_e_median_final": float}``.
    Returns ``None`` if even the ridge initialiser fails.
    """
    M = _solve_sw_rpcc2_ridge(X, Y, w, lam)
    if M is None:
        return None

    def _delta_e_pairs(M_now: np.ndarray) -> np.ndarray:
        Y_hat = X @ M_now.T
        # NB: LAB conversion expects [0,1] linear RGB; clamp predictions.
        lab_pred = _rgb_to_lab_d65(np.clip(Y_hat, 0.0, 1.0))
        lab_targ = _rgb_to_lab_d65(np.clip(Y, 0.0, 1.0))
        return np.linalg.norm(lab_pred - lab_targ, axis=1)

    de0 = _delta_e_pairs(M)
    de_med_init = float(np.median(de0))
    iters_done = 0
    for it in range(1, max_iter + 1):
        de = _delta_e_pairs(M)
        w_new = w * 1.0 / (1.0 + alpha * de * de)
        M_new = _solve_sw_rpcc2_ridge(X, Y, w_new, lam)
        if M_new is None:
            break
        diff = float(np.linalg.norm(M_new - M))
        M = M_new
        iters_done = it
        if diff < tol:
            break
    de_final = _delta_e_pairs(M)
    info = {
        "iters": iters_done,
        "delta_e_median_init": de_med_init,
        "delta_e_median_final": float(np.median(de_final)),
    }
    return M, info


# --------------------------------------------------------------------------
# Variant dispatcher — single entry point for the calibrator key 0/1/2/3.
# Computes ALL four variants from a single pair pool so the GUI can switch
# instantly without re-running the solver.
# --------------------------------------------------------------------------

# Default ridge λ. Caller may override via SW_CCM_RIDGE_LAMBDA env var
# (read once at module import, not per-call, to avoid surprising the user
# mid-session).
import os as _os

# Default ridge λ — interpreted as a fraction of trace(XᵀWX)/k inside
# ``_solve_sw_rpcc2_ridge``, so λ=1e-1 means "~10% of the data scale".
# Sparse pair pools (≤8 patches) need genuine regularisation; with linear
# 0..1 RGB and the un-scaled ridge in earlier iterations λ=1e-3 was
# observed to leave coefficients in the −300…+30 range (severe overfit).
_RIDGE_LAMBDA_DEFAULT = 1e-1
# Plain LSQ (``rpcc2``) needs at least this many pairs to be non-degenerate.
# At feat_dim=6 the system has 6×3=18 equations; demand a 2× safety margin
# so that the LSQ result generalises beyond the training patches.
_RPCC2_LSQ_MIN_PAIRS = 12
try:
    _env_lam = _os.environ.get("SW_CCM_RIDGE_LAMBDA")
    if _env_lam is not None:
        _RIDGE_LAMBDA_DEFAULT = float(_env_lam)
except (TypeError, ValueError):
    pass
try:
    _env_min = _os.environ.get("SW_CCM_RPCC2_MIN_PAIRS")
    if _env_min is not None:
        _RPCC2_LSQ_MIN_PAIRS = max(6, int(_env_min))
except (TypeError, ValueError):
    pass


def _compute_ccm_variants(
    ref_bgr: np.ndarray,
    cur_bgr: np.ndarray,
    pairs: Sequence[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]],
    *,
    ridge_lambda: float | None = None,
) -> dict[str, dict | None]:
    """Compute all four SW-CCM variants from one pair pool.

    Returns a dict with keys ``linear`` / ``rpcc2`` / ``rpcc2_ridge`` /
    ``rpcc2_als``. Each entry is either ``None`` (variant not solvable
    given the current pair count) or a dict::

        {
            "M":               np.ndarray,        # (3,3) for linear, (3,6) for RPCC2
            "feat_dim":        3 | 6,
            "n_pairs":         int,
            "lambda":          float | None,
            "iters":           int  | None,
            "delta_e_median":  float | None,      # final ΔE on training pairs
        }

    The caller (``manual_match_neutral`` / ``manual_match_patches``) then
    stores this dict under ``debug["ccm_variants"]``.
    """
    if ridge_lambda is None:
        ridge_lambda = _RIDGE_LAMBDA_DEFAULT
    out: dict[str, dict | None] = {
        "linear": None,
        "rpcc2": None,
        "rpcc2_ridge": None,
        "rpcc2_als": None,
    }

    # --- Variant 0: legacy linear 3x3 (re-use the existing helper, but
    # apply the same identity-shrinkage ridge as the RPCC2 variants so a
    # sparse / low-chroma pair pool degrades to M~I instead of an
    # extreme-gain matrix that posterises the SW pane). ---
    lin = _solve_sw_ccm_lsq(ref_bgr, cur_bgr, pairs, ridge_lambda=ridge_lambda)
    if lin is not None:
        M_lin, n_lin, _px = lin
        out["linear"] = {
            "M": M_lin,
            "feat_dim": 3,
            "n_pairs": n_lin,
            "lambda": float(ridge_lambda) if ridge_lambda > 0 else None,
            "iters": None,
            "delta_e_median": None,
        }

    # --- Build the (X_rpcc, Y, w) pool once for variants 1/2/3 ---
    pool = _build_pair_pool(ref_bgr, cur_bgr, pairs)
    if pool is None:
        return out
    rgb_cur, rgb_ref, w, _total_px = pool
    if rgb_cur.shape[0] < 6:
        # Not enough pairs for a 6-D RPCC fit; only `linear` is populated.
        return out
    X = _rpcc2_features(rgb_cur)
    Y = rgb_ref

    def _delta_e_median(M: np.ndarray) -> float:
        Y_hat = X @ M.T
        de = np.linalg.norm(
            _rgb_to_lab_d65(np.clip(Y_hat, 0.0, 1.0))
            - _rgb_to_lab_d65(np.clip(Y, 0.0, 1.0)),
            axis=1,
        )
        return float(np.median(de))

    n_rpcc = int(rgb_cur.shape[0])

    # --- Variant 1: RPCC2 plain LSQ (only when pair pool is large enough) ---
    # Below the 2x feat_dim threshold the system is exactly- or near-
    # exactly-determined; LSQ then perfectly interpolates the training
    # patches (ΔE≈0) but produces wildly out-of-range matrix entries that
    # generalise terribly. Skipping is preferable to misleading the user.
    if n_rpcc >= _RPCC2_LSQ_MIN_PAIRS:
        M_lsq = _solve_sw_rpcc2_lsq(X, Y, w)
        if M_lsq is not None:
            out["rpcc2"] = {
                "M": M_lsq,
                "feat_dim": 6,
                "n_pairs": n_rpcc,
                "lambda": None,
                "iters": None,
                "delta_e_median": _delta_e_median(M_lsq),
            }

    # --- Variant 2: RPCC2 + ridge ---
    M_rid = _solve_sw_rpcc2_ridge(X, Y, w, ridge_lambda)
    if M_rid is not None:
        out["rpcc2_ridge"] = {
            "M": M_rid,
            "feat_dim": 6,
            "n_pairs": n_rpcc,
            "lambda": float(ridge_lambda),
            "iters": None,
            "delta_e_median": _delta_e_median(M_rid),
        }

    # --- Variant 3: ALS-LAB (initialised from ridge) ---
    als = _solve_sw_rpcc2_als_lab(X, Y, w, ridge_lambda)
    if als is not None:
        M_als, info = als
        out["rpcc2_als"] = {
            "M": M_als,
            "feat_dim": 6,
            "n_pairs": n_rpcc,
            "lambda": float(ridge_lambda),
            "iters": int(info.get("iters", 0)),
            "delta_e_median": float(info.get("delta_e_median_final", 0.0)),
        }

    return out


def _log_ccm_variants(variants: Mapping[str, dict | None], *, prefix: str = "") -> None:
    """One-line console summary for each successfully solved variant."""
    tag = f"{prefix} " if prefix else ""
    for name, entry in variants.items():
        if entry is None:
            print(f"[{tag}SW-CCM:{name}] unavailable")
            continue
        de = entry.get("delta_e_median")
        de_str = f"  ΔE_med={de:.3f}" if de is not None else ""
        lam = entry.get("lambda")
        lam_str = f"  λ={lam:.1e}" if lam is not None else ""
        it = entry.get("iters")
        it_str = f"  iters={it}" if it is not None else ""
        print(
            f"[{tag}SW-CCM:{name}] feat={entry['feat_dim']}  "
            f"pairs={entry['n_pairs']}{lam_str}{it_str}{de_str}"
        )


# --------------------------------------------------------------------------
# ColorChecker reference path — same 4-variant solver but the reference
# RGB values come from a known-good chart (e.g. X-Rite Classic 24, see
# ``colorchecker24.py``) instead of being measured from a reference image.
# Used by the ``camera_isp_calibrator --colorchecker`` entry mode.
# --------------------------------------------------------------------------


def _build_pair_pool_from_reference(
    ref_rgb_uint8: np.ndarray,
    cur_bgr: np.ndarray,
    cur_boxes: Sequence[tuple[int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    """Build (X_rgb, Y_rgb, w, total_px) using known reference RGB values.

    Counterpart of :func:`_build_pair_pool` that does **not** read any
    reference image — instead, ``ref_rgb_uint8[i]`` is taken as the
    target sRGB triple for ``cur_boxes[i]``. Both sides are converted to
    linear light so the result is drop-in compatible with the RPCC2
    solver pipeline.

    Filters: each ``cur_box`` must contribute ≥ 5 non-saturated pixels;
    the ref triple must have a non-degenerate channel mean (≥ 1e-5
    after sRGB decode); skip entries that do not pass either filter.

    Returns ``(rgb_cur, rgb_ref, weights, total_px)`` shaped
    ``((N,3), (N,3), (N,), int)`` or ``None`` if the resulting pool is
    empty.
    """
    if ref_rgb_uint8.ndim != 2 or ref_rgb_uint8.shape[-1] != 3:
        raise ValueError(
            f"_build_pair_pool_from_reference: ref_rgb_uint8 must be (N,3); "
            f"got {ref_rgb_uint8.shape}"
        )
    if ref_rgb_uint8.dtype != np.uint8:
        raise TypeError(
            "_build_pair_pool_from_reference: ref_rgb_uint8 must be uint8"
        )
    if len(cur_boxes) != ref_rgb_uint8.shape[0]:
        raise ValueError(
            f"_build_pair_pool_from_reference: got {len(cur_boxes)} boxes "
            f"but {ref_rgb_uint8.shape[0]} reference triples"
        )

    # Reference: sRGB uint8 -> linear RGB float64 (N,3).
    ref_lin = bgr_to_linear_rgb(ref_rgb_uint8[:, ::-1])  # RGB -> BGR -> linear-RGB

    rgb_cur_rows: list[tuple[float, float, float]] = []
    rgb_ref_rows: list[tuple[float, float, float]] = []
    weights: list[float] = []
    total_px = 0

    for i, cur_box in enumerate(cur_boxes):
        cur_px = _filter_saturated(_box_pixels(cur_bgr, cur_box))
        if cur_px.shape[0] < 5:
            continue
        b_cur, g_cur, r_cur = _linear_channel_means_bgr(cur_px)
        if min(g_cur, r_cur, b_cur) < 1e-5:
            continue
        r_ref, g_ref, b_ref = (
            float(ref_lin[i, 0]), float(ref_lin[i, 1]), float(ref_lin[i, 2]),
        )
        if min(r_ref, g_ref, b_ref) < 1e-5:
            # Avoid the degenerate Y=0 row (would not constrain anything).
            continue
        # Weighting: ref RGB is a constant triple (no measurement noise),
        # so the only stochastic side is ``cur``. Use sqrt(N_cur) like the
        # image-ref path to keep λ behaviour comparable across modes.
        w = float(np.sqrt(cur_px.shape[0]))
        rgb_cur_rows.append((r_cur, g_cur, b_cur))
        rgb_ref_rows.append((r_ref, g_ref, b_ref))
        weights.append(w)
        total_px += int(cur_px.shape[0])

    if not rgb_cur_rows:
        return None
    return (
        np.asarray(rgb_cur_rows, dtype=np.float64),
        np.asarray(rgb_ref_rows, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        total_px,
    )


def compute_ccm_variants_from_reference_rgb(
    ref_rgb_uint8: np.ndarray,
    cur_bgr: np.ndarray,
    cur_boxes: Sequence[tuple[int, int, int, int]],
    *,
    ridge_lambda: float | None = None,
) -> dict[str, dict | None]:
    """Public counterpart of :func:`_compute_ccm_variants` for ColorChecker mode.

    Identical return contract — ``{linear, rpcc2, rpcc2_ridge, rpcc2_als}``
    each mapped to ``None`` or a dict with ``M / feat_dim / n_pairs /
    lambda / iters / delta_e_median``. The calibrator can drop this dict
    straight into ``debug["ccm_variants"]`` and the existing
    ``_update_sw_isp_from_dbg`` / 0/1/2/3 toggle logic works unchanged.
    """
    if ridge_lambda is None:
        ridge_lambda = _RIDGE_LAMBDA_DEFAULT
    out: dict[str, dict | None] = {
        "linear": None,
        "rpcc2": None,
        "rpcc2_ridge": None,
        "rpcc2_als": None,
    }

    pool = _build_pair_pool_from_reference(ref_rgb_uint8, cur_bgr, cur_boxes)
    if pool is None:
        return out
    rgb_cur, rgb_ref, w, _total_px = pool
    n_pairs = int(rgb_cur.shape[0])
    if n_pairs < 3:
        return out

    # --- Variant 0: linear 3x3 with identity-shrinkage ridge ------------
    # Build the same (X_lin, Y_lin) system the image-ref path uses but
    # via direct tensor algebra (no need to fabricate pseudo-boxes).
    Xl = rgb_cur * w[:, None]
    Yl = rgb_ref * w[:, None]
    XltXl = Xl.T @ Xl  # (3,3)
    s_lin = float(np.trace(XltXl)) / 3.0
    if not np.isfinite(s_lin) or s_lin <= 0.0:
        s_lin = 1.0
    A_lin = XltXl + ridge_lambda * s_lin * np.eye(3, dtype=np.float64)
    rhs_lin = Xl.T @ Yl + ridge_lambda * s_lin * np.eye(3, dtype=np.float64)
    try:
        M_lin_T = np.linalg.solve(A_lin, rhs_lin)
        M_lin = M_lin_T.T
        if np.all(np.isfinite(M_lin)):
            out["linear"] = {
                "M": M_lin,
                "feat_dim": 3,
                "n_pairs": n_pairs,
                "lambda": float(ridge_lambda),
                "iters": None,
                "delta_e_median": None,
            }
    except np.linalg.LinAlgError:
        pass

    if n_pairs < 6:
        return out

    X = _rpcc2_features(rgb_cur)
    Y = rgb_ref

    def _delta_e_median(M: np.ndarray) -> float:
        Y_hat = X @ M.T
        de = np.linalg.norm(
            _rgb_to_lab_d65(np.clip(Y_hat, 0.0, 1.0))
            - _rgb_to_lab_d65(np.clip(Y, 0.0, 1.0)),
            axis=1,
        )
        return float(np.median(de))

    if n_pairs >= _RPCC2_LSQ_MIN_PAIRS:
        M_lsq = _solve_sw_rpcc2_lsq(X, Y, w)
        if M_lsq is not None:
            out["rpcc2"] = {
                "M": M_lsq,
                "feat_dim": 6,
                "n_pairs": n_pairs,
                "lambda": None,
                "iters": None,
                "delta_e_median": _delta_e_median(M_lsq),
            }

    M_rid = _solve_sw_rpcc2_ridge(X, Y, w, ridge_lambda)
    if M_rid is not None:
        out["rpcc2_ridge"] = {
            "M": M_rid,
            "feat_dim": 6,
            "n_pairs": n_pairs,
            "lambda": float(ridge_lambda),
            "iters": None,
            "delta_e_median": _delta_e_median(M_rid),
        }

    als = _solve_sw_rpcc2_als_lab(X, Y, w, ridge_lambda)
    if als is not None:
        M_als, info = als
        out["rpcc2_als"] = {
            "M": M_als,
            "feat_dim": 6,
            "n_pairs": n_pairs,
            "lambda": float(ridge_lambda),
            "iters": int(info.get("iters", 0)),
            "delta_e_median": float(info.get("delta_e_median_final", 0.0)),
        }

    return out


def manual_match_colorchecker(
    ref_rgb_uint8: np.ndarray,
    cur_bgr: np.ndarray,
    cur_boxes: Sequence[tuple[int, int, int, int]],
    current_params: Mapping[str, int | bool] | None = None,
    device_caps: Mapping[str, Mapping[str, int]] | None = None,
) -> tuple[dict, dict]:
    """ColorChecker entry point used by the calibrator.

    Returns ``(proposed, debug)`` with ``proposed = {}`` because the
    color-checker workflow is **SW-only** by design — the calibrator
    explicitly does not write hardware in this mode. ``debug`` contains
    the same ``ccm_variants`` shape that ``_update_sw_isp_from_dbg``
    consumes, plus a ``warnings`` list and a ``_sw_ccm_pairs`` counter
    to keep HUD wiring identical.
    """
    debug: dict = {
        "kr_sw": 1.0, "kb_sw": 1.0, "exp_scale_sw": 1.0,
        "_sw_neutral_n": 0, "_sw_ref_saturated": False,
        "ccm_sw": None, "_sw_ccm_pairs": 0,
        "warnings": [],
        "ccm_variants": None,
        "mode": "colorchecker",
    }
    proposed: dict = {}
    _ = current_params, device_caps  # accepted for signature parity, unused

    if not cur_boxes:
        debug["warnings"].append("no_boxes")
        return proposed, debug

    variants = compute_ccm_variants_from_reference_rgb(
        ref_rgb_uint8, cur_bgr, cur_boxes,
    )
    debug["ccm_variants"] = variants
    # Carry the linear-domain CCM (3,3) into ccm_sw for backward
    # compatibility (the legacy SW-ISP HUD path reads ccm_sw).
    lin = variants.get("linear")
    if lin is not None:
        debug["ccm_sw"] = lin["M"]
        debug["_sw_ccm_pairs"] = int(lin.get("n_pairs", 0))
    _log_ccm_variants(variants, prefix="cc24")
    if all(v is None for v in variants.values()):
        debug["warnings"].append("all_variants_unsolvable")
    return proposed, debug


# --------------------------------------------------------------------------
# Tunables (documented in 临时/camera_isp_plan.md §Phase 2)
# --------------------------------------------------------------------------

_SAT_LO, _SAT_HI = 5, 250          # uint8 — saturation mask thresholds
_EXP_GAMMA = 0.7                    # sub-linear damping for exposure scale
_EXP_SCALE_MIN, _EXP_SCALE_MAX = 0.3, 3.0  # outlier guard
_AUTO_MAX_ITERS = 4
_AUTO_DL_TOL = 2.0                  # ΔL* convergence threshold (Lab units)
_AUTO_DK_TOL = 50.0                 # |Δkelvin| convergence threshold
_KELVIN_DEFAULT_MIN, _KELVIN_DEFAULT_MAX = 2000, 10000
_EXPOSURE_DEFAULT_MIN, _EXPOSURE_DEFAULT_MAX = 1, 20000


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _saturated_mask(*frames: np.ndarray) -> np.ndarray:
    """Return a 2D bool mask: True = pixel kept, False = saturated/clipped.

    A pixel is rejected if **any** channel in **any** of the supplied frames
    is ≤ ``_SAT_LO`` or ≥ ``_SAT_HI``. All frames must share shape (H, W, 3).
    """
    if not frames:
        raise ValueError("_saturated_mask requires at least one frame")
    h, w, _ = frames[0].shape
    keep = np.ones((h, w), dtype=bool)
    for f in frames:
        if f.shape != (h, w, 3):
            raise ValueError(f"frame shape mismatch: {f.shape} vs ({h},{w},3)")
        ok = ((f > _SAT_LO) & (f < _SAT_HI)).all(axis=-1)
        keep &= ok
    return keep


def _p50_lab(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return [L_P50, a_P50, b_P50] over masked pixels. NaN if mask all-False."""
    if not mask.any():
        return np.array([np.nan, np.nan, np.nan])
    pixels = bgr[mask]
    lab = bgr_to_lab(pixels)
    return np.median(lab, axis=0)


def _p50_chromaticity(bgr: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Return (x_P50, y_P50) over masked pixels. (NaN, NaN) if degenerate."""
    if not mask.any():
        return float("nan"), float("nan")
    pixels = bgr[mask]
    x, y = bgr_to_xyz_chromaticity(pixels)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return float("nan"), float("nan")
    return float(np.median(x[finite])), float(np.median(y[finite]))


def _clip_int(value: float, key: str, caps: Mapping[str, Mapping[str, int]] | None,
              fallback: tuple[int, int]) -> int:
    """Clip ``value`` to ``caps[key]`` range, falling back to ``fallback``."""
    if caps and key in caps:
        lo = int(caps[key].get("min", fallback[0]))
        hi = int(caps[key].get("max", fallback[1]))
    else:
        lo, hi = fallback
    return int(np.clip(round(value), lo, hi))


def _hits_rail(value: int, key: str, caps: Mapping[str, Mapping[str, int]] | None,
               fallback: tuple[int, int]) -> bool:
    if caps and key in caps:
        lo = int(caps[key].get("min", fallback[0]))
        hi = int(caps[key].get("max", fallback[1]))
    else:
        lo, hi = fallback
    return value <= lo or value >= hi


# --------------------------------------------------------------------------
# Auto: Lab + Planckian locus (one iteration; caller loops)
# --------------------------------------------------------------------------


@dataclass
class AutoStepResult:
    proposed: dict
    converged: bool
    debug: dict


def auto_match_lab(
    ref_bgr: np.ndarray,
    cur_bgr: np.ndarray,
    current_params: Mapping[str, int | bool],
    device_caps: Mapping[str, Mapping[str, int]] | None = None,
) -> AutoStepResult:
    """Compute one Auto-mode step (exposure + WB) toward the reference.

    Args:
        ref_bgr: reference frame (H, W, 3) uint8 BGR.
        cur_bgr: current live frame (H, W, 3) uint8 BGR. Must match ref shape.
        current_params: dict with at least ``exposure`` (int) and
            ``white_balance`` (int kelvin). ``auto_white_balance`` is read if
            present (a True value is treated as "unknown current kelvin" —
            the proposal will use the device default).
        device_caps: optional ``{key: {"min": int, "max": int}}`` from
            v4l2-ctl. Falls back to conservative built-in ranges.

    Returns:
        ``AutoStepResult`` with:
          ``proposed`` — dict ready to merge into the V4L2 param set
            (always includes ``auto_white_balance: False`` when ``white_balance``
            is proposed).
          ``converged`` — True if both ΔL* and Δkelvin are below thresholds
            (caller can stop iterating).
          ``debug`` — diagnostic floats: ``exp_scale``, ``L_ref``, ``L_cur``,
            ``T_ref``, ``T_cur``, ``num_pixels``, optional ``_warning``.
    """
    if ref_bgr.shape != cur_bgr.shape:
        raise ValueError(f"ref/cur shape mismatch: {ref_bgr.shape} vs {cur_bgr.shape}")
    if ref_bgr.dtype != np.uint8 or cur_bgr.dtype != np.uint8:
        raise TypeError("auto_match_lab expects uint8 BGR frames")

    mask = _saturated_mask(ref_bgr, cur_bgr)
    n_pix = int(mask.sum())

    proposed: dict = {}
    debug: dict = {"num_pixels": n_pix}
    warnings: list[str] = []

    if n_pix < 100:
        # Degenerate: too few non-saturated pixels (e.g. all-black frame).
        debug["_warning"] = "insufficient_pixels"
        return AutoStepResult(proposed={}, converged=True, debug=debug)

    # ------------------------------------------------------------------
    # Stage A: Exposure (L* match)
    # ------------------------------------------------------------------
    lab_ref = _p50_lab(ref_bgr, mask)
    lab_cur = _p50_lab(cur_bgr, mask)
    L_ref, L_cur = float(lab_ref[0]), float(lab_cur[0])
    debug["L_ref"], debug["L_cur"] = L_ref, L_cur
    dL = L_ref - L_cur

    if L_cur > 0.5:  # avoid div-by-near-zero on near-black
        # In CIE Lab, L* is roughly proportional to Y^(1/3); we map via Y.
        # Y is roughly proportional to exposure for a linear sensor in the
        # mid-tones, so request exposure scale = (Y_ref / Y_cur)^_EXP_GAMMA.
        Y_ref = ((L_ref + 16.0) / 116.0) ** 3
        Y_cur = ((L_cur + 16.0) / 116.0) ** 3
        if Y_cur > 1e-6:
            raw_scale = Y_ref / Y_cur
            exp_scale = float(raw_scale ** _EXP_GAMMA)
        else:
            raw_scale, exp_scale = 1.0, 1.0
    else:
        raw_scale, exp_scale = 1.0, 1.0

    if not (_EXP_SCALE_MIN <= raw_scale <= _EXP_SCALE_MAX):
        warnings.append("scene_difference")
    debug["exp_scale"] = exp_scale
    debug["raw_exp_scale"] = float(raw_scale)

    cur_exposure = int(current_params.get("exposure", 312))
    new_exposure = _clip_int(
        cur_exposure * exp_scale, "exposure", device_caps,
        (_EXPOSURE_DEFAULT_MIN, _EXPOSURE_DEFAULT_MAX),
    )
    proposed["exposure"] = new_exposure

    # ------------------------------------------------------------------
    # Stage B: White balance (chromaticity → Planckian → kelvin delta)
    # ------------------------------------------------------------------
    x_ref, y_ref = _p50_chromaticity(ref_bgr, mask)
    x_cur, y_cur = _p50_chromaticity(cur_bgr, mask)
    debug["xy_ref"] = (x_ref, y_ref)
    debug["xy_cur"] = (x_cur, y_cur)

    T_ref = mccamy_kelvin(x_ref, y_ref) if np.isfinite(x_ref + y_ref) else float("nan")
    T_cur = mccamy_kelvin(x_cur, y_cur) if np.isfinite(x_cur + y_cur) else float("nan")
    debug["T_ref"] = T_ref
    debug["T_cur"] = T_cur

    cur_kelvin = int(current_params.get("white_balance", 4600))
    if np.isfinite(T_ref) and np.isfinite(T_cur):
        # Delta-form: trust the *delta* in McCamy projections, not the absolute.
        target_kelvin_raw = cur_kelvin + (T_ref - T_cur)
        new_kelvin = _clip_int(
            target_kelvin_raw, "white_balance", device_caps,
            (_KELVIN_DEFAULT_MIN, _KELVIN_DEFAULT_MAX),
        )
        proposed["white_balance"] = new_kelvin
        debug["target_kelvin_raw"] = float(target_kelvin_raw)
        if _hits_rail(new_kelvin, "white_balance", device_caps,
                      (_KELVIN_DEFAULT_MIN, _KELVIN_DEFAULT_MAX)):
            warnings.append("kelvin_rail")
        dK = new_kelvin - cur_kelvin
    else:
        warnings.append("chromaticity_degenerate")
        dK = 0.0

    # ------------------------------------------------------------------
    # SW-ISP debug fields (linear-domain channel gains the debug pane
    # can apply to the live frame as a "what would the right answer
    # look like, with no camera in the loop?" preview). These are
    # *not* fed back into proposed / hardware: that path stays on the
    # legacy sRGB-domain LUT lookup so existing test contracts hold.
    # ------------------------------------------------------------------
    b_ref, g_ref, r_ref = _linear_channel_means_bgr(ref_bgr.reshape(-1, 3))
    b_cur, g_cur, r_cur = _linear_channel_means_bgr(cur_bgr.reshape(-1, 3))
    if g_cur > 1e-4 and r_cur > 1e-4 and b_cur > 1e-4:
        # Per-channel ratio: how much do we need to multiply each cur
        # channel to land on the ref channel mean (in linear light)?
        # We pin G to 1 by absorbing its scale into exp_scale_sw; kr_sw
        # and kb_sw are residual chroma corrections.
        debug["exp_scale_sw"] = g_ref / g_cur if g_ref > 0 else 1.0
        debug["kr_sw"] = (r_ref / r_cur) / debug["exp_scale_sw"] if r_cur > 0 else 1.0
        debug["kb_sw"] = (b_ref / b_cur) / debug["exp_scale_sw"] if b_cur > 0 else 1.0
    else:
        debug["exp_scale_sw"] = 1.0
        debug["kr_sw"] = 1.0
        debug["kb_sw"] = 1.0
    # auto_match_lab works on whole frames (no box pairs), so a CCM solve
    # would be ill-conditioned — leave ccm_sw at None and let the SW pane
    # fall back to diag mode for Auto results.
    debug["ccm_sw"] = None
    debug["_sw_ccm_pairs"] = 0

    # ------------------------------------------------------------------
    # Convergence + warnings
    # ------------------------------------------------------------------
    converged = (abs(dL) < _AUTO_DL_TOL) and (abs(dK) < _AUTO_DK_TOL)
    if warnings:
        debug["_warning"] = ",".join(warnings)
    return AutoStepResult(proposed=proposed, converged=converged, debug=debug)


# --------------------------------------------------------------------------
# Manual + neutral
# --------------------------------------------------------------------------


def _box_pixels(bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Return (N, 3) BGR pixels inside a (x0, y0, x1, y1) box, clipped to image."""
    h, w, _ = bgr.shape
    x0, y0, x1, y1 = box
    x0 = max(0, min(int(x0), w))
    x1 = max(0, min(int(x1), w))
    y0 = max(0, min(int(y0), h))
    y1 = max(0, min(int(y1), h))
    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 3), dtype=bgr.dtype)
    return bgr[y0:y1, x0:x1].reshape(-1, 3)


def _filter_saturated(pixels: np.ndarray) -> np.ndarray:
    if pixels.size == 0:
        return pixels
    keep = ((pixels > _SAT_LO) & (pixels < _SAT_HI)).all(axis=-1)
    return pixels[keep]


def manual_match_neutral(
    ref_bgr: np.ndarray,
    cur_bgr: np.ndarray,
    neutral_box_pairs: Sequence[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]],
    plain_box_pairs: Sequence[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]],
    current_params: Mapping[str, int | bool],
    device_caps: Mapping[str, Mapping[str, int]] | None = None,
) -> tuple[dict, dict]:
    """Pooled-gray manual mode.

    Each box pair is ``(ref_box, cur_box)`` in (x0, y0, x1, y1) pixel coords.
    Neutral boxes carry the WB constraint; *all* boxes (neutral + plain)
    contribute to the exposure least-squares.

    If ``neutral_box_pairs`` is empty, the caller should dispatch to
    :func:`manual_match_patches` instead — we still handle the empty case
    gracefully (returns proposed={} with a warning).
    """
    debug: dict = {}
    proposed: dict = {}
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # WB: pooled neutral pixels → (kr, kb) → LUT → kelvin
    # ------------------------------------------------------------------
    if neutral_box_pairs:
        cur_neutral_all = []
        for _ref_box, cur_box in neutral_box_pairs:
            px = _filter_saturated(_box_pixels(cur_bgr, cur_box))
            if px.size:
                cur_neutral_all.append(px)
        if cur_neutral_all:
            pool = np.concatenate(cur_neutral_all, axis=0)
            mean_b = float(pool[:, 0].mean())
            mean_g = float(pool[:, 1].mean())
            mean_r = float(pool[:, 2].mean())
            if mean_r > 1e-3 and mean_b > 1e-3 and mean_g > 1e-3:
                kr = mean_g / mean_r
                kb = mean_g / mean_b
                kelvin = lookup_kelvin(kr, kb)
                debug["kr"], debug["kb"] = kr, kb
                debug["lut_kelvin"] = kelvin
                cur_kelvin = int(current_params.get("white_balance", 4600))
                # Manual neutral mode: LUT gives an *absolute* kelvin
                # estimate (because we measured a true neutral patch), so
                # there's no delta-form fallback — just clip and apply.
                new_kelvin = _clip_int(
                    kelvin, "white_balance", device_caps,
                    (_KELVIN_DEFAULT_MIN, _KELVIN_DEFAULT_MAX),
                )
                proposed["white_balance"] = new_kelvin
                if _hits_rail(new_kelvin, "white_balance", device_caps,
                              (_KELVIN_DEFAULT_MIN, _KELVIN_DEFAULT_MAX)):
                    warnings.append("kelvin_rail")
                debug["delta_kelvin"] = new_kelvin - cur_kelvin
            else:
                warnings.append("neutral_too_dark")
        else:
            warnings.append("neutral_all_saturated")
    else:
        warnings.append("no_neutral_box")

    # ------------------------------------------------------------------
    # Exposure: weighted LS over G channel of ALL boxes
    # ------------------------------------------------------------------
    all_pairs = list(neutral_box_pairs) + list(plain_box_pairs)
    weights = []
    ratios = []
    for ref_box, cur_box in all_pairs:
        ref_px = _filter_saturated(_box_pixels(ref_bgr, ref_box))
        cur_px = _filter_saturated(_box_pixels(cur_bgr, cur_box))
        if ref_px.shape[0] < 5 or cur_px.shape[0] < 5:
            continue
        ref_g = float(ref_px[:, 1].mean())
        cur_g = float(cur_px[:, 1].mean())
        if cur_g < 1.0:
            continue
        ratios.append(ref_g / cur_g)
        weights.append(np.sqrt(min(ref_px.shape[0], cur_px.shape[0])))

    if ratios:
        ratios_arr = np.asarray(ratios, dtype=np.float64)
        weights_arr = np.asarray(weights, dtype=np.float64)
        # Weighted least-squares for a single scalar: closed-form.
        exp_scale = float(np.sum(weights_arr * ratios_arr) / np.sum(weights_arr))
        if not (_EXP_SCALE_MIN <= exp_scale <= _EXP_SCALE_MAX):
            warnings.append("scene_difference")
        debug["exp_scale"] = exp_scale
        cur_exposure = int(current_params.get("exposure", 312))
        proposed["exposure"] = _clip_int(
            cur_exposure * exp_scale, "exposure", device_caps,
            (_EXPOSURE_DEFAULT_MIN, _EXPOSURE_DEFAULT_MAX),
        )
    else:
        warnings.append("no_exposure_constraint")

    # ------------------------------------------------------------------
    # SW-ISP debug fields (linear domain — independent of LUT / hardware
    # path). The debug pane consumes kr_sw / kb_sw / exp_scale_sw to
    # render "what the right answer should look like, in pixels".
    #
    # Use a unified linear-domain LSQ across *all* box pairs (neutral
    # AND plain). This matches the user's mental model: "every patch I
    # picked — including the blue and orange ones — should align after
    # SW-ISP". Per-channel rows: cur_c * (alpha * k_c) ≈ ref_c, with
    # k_g pinned to 1.
    # ------------------------------------------------------------------
    debug["kr_sw"] = 1.0
    debug["kb_sw"] = 1.0
    debug["exp_scale_sw"] = 1.0
    debug["_sw_neutral_n"] = 0
    debug["_sw_ref_saturated"] = False  # legacy field; LSQ tolerates partial sat
    debug["ccm_sw"] = None              # 3x3 linear-domain CCM (when solvable)
    debug["_sw_ccm_pairs"] = 0
    all_pairs_for_sw = list(neutral_box_pairs) + list(plain_box_pairs)

    # Always solve the diag fallback first so kr_sw / kb_sw / exp_scale_sw
    # remain populated even if the CCM solve is rank-deficient.
    sw_sol = _solve_sw_wb_lsq(ref_bgr, cur_bgr, all_pairs_for_sw)
    if sw_sol is not None:
        alpha_sw, kr_sw, kb_sw, total_px = sw_sol
        debug["exp_scale_sw"] = alpha_sw
        debug["kr_sw"] = kr_sw
        debug["kb_sw"] = kb_sw
        debug["_sw_neutral_n"] = total_px
        print(
            f"[solver SW-WB LSQ] {len(all_pairs_for_sw)} pair(s), "
            f"{total_px} px → alpha={alpha_sw:.4f}  "
            f"kr_sw={kr_sw:.4f}  kb_sw={kb_sw:.4f}"
        )
    else:
        print(
            "[solver SW-WB LSQ] no usable pairs (all saturated/empty) — "
            "kr/kb/exp left at identity"
        )

    # CCM solve: needs ≥ 3 chromatic-distinct pairs. Falls back silently
    # to None (caller uses diag) when underdetermined.
    ccm_sol = _solve_sw_ccm_lsq(ref_bgr, cur_bgr, all_pairs_for_sw)
    if ccm_sol is not None:
        M, n_pairs, ccm_px = ccm_sol
        debug["ccm_sw"] = M
        debug["_sw_ccm_pairs"] = n_pairs
        print(
            f"[solver SW-CCM LSQ] {n_pairs} pair(s), {ccm_px} px → 3x3 M:\n"
            f"  R<- [{M[0,0]:+.3f} {M[0,1]:+.3f} {M[0,2]:+.3f}]\n"
            f"  G<- [{M[1,0]:+.3f} {M[1,1]:+.3f} {M[1,2]:+.3f}]\n"
            f"  B<- [{M[2,0]:+.3f} {M[2,1]:+.3f} {M[2,2]:+.3f}]"
        )
    else:
        print(
            f"[solver SW-CCM LSQ] not solvable (need ≥ 3 chromatic-distinct "
            f"pairs; got {len(all_pairs_for_sw)}) — diag fallback in effect"
        )

    # All four variants (key 0/1/2/3 in calibrator GUI). The legacy
    # ``ccm_sw`` field above is kept as the linear-3x3 default for
    # backward compatibility.
    variants = _compute_ccm_variants(ref_bgr, cur_bgr, all_pairs_for_sw)
    debug["ccm_variants"] = variants
    _log_ccm_variants(variants, prefix="neutral")

    if warnings:
        debug["_warning"] = ",".join(warnings)
    return proposed, debug


# --------------------------------------------------------------------------
# Manual + REF (no neutral required; user-supplied ROI pairs only)
# --------------------------------------------------------------------------


def manual_match_ref(
    ref_bgr: np.ndarray,
    cur_bgr: np.ndarray,
    box_pairs: Sequence[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]],
    current_params: Mapping[str, int | bool],
    device_caps: Mapping[str, Mapping[str, int]] | None = None,
    *,
    sat_con_band: float = 0.6,
) -> tuple[dict, dict]:
    """REF-mode manual match — no neutral labelling required.

    Each ``(ref_box, cur_box)`` is "this region of cur should look like
    this region of ref". The user's explicit correspondences carry all
    the info, so we skip the cc24 "this is gray" gate and solve four
    hardware controls plus the SW-ISP debug pane directly:

    * ``exposure``      — G-channel weighted LSQ over all pairs.
    * ``white_balance`` — kr/kb from the linear-domain LSQ → kelvin LUT.
    * ``saturation``    — chroma_mag_ref/chroma_mag_cur × current,
      clipped to default ± ``sat_con_band`` (default ±60%, wider than
      AUTO's ±30% because the user picked the patches deliberately).
    * ``contrast``      — y_std_ref / y_std_cur × current, ditto.

    Stats are pooled over the ROI pixel union (all pairs concatenated)
    so a few well-chosen patches drive the proposal.
    """
    # Local imports to avoid pulling cv2 into module-load path twice.
    from .hw_stages import (
        CtrlCaps,
        compute_chroma_stats,
        compute_y_stats,
        propose_sat_con,
    )

    debug: dict = {}
    proposed: dict = {}
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # 1. Exposure — G-channel weighted LSQ across all pairs.
    # ------------------------------------------------------------------
    weights: list[float] = []
    ratios: list[float] = []
    for ref_box, cur_box in box_pairs:
        ref_px = _filter_saturated(_box_pixels(ref_bgr, ref_box))
        cur_px = _filter_saturated(_box_pixels(cur_bgr, cur_box))
        if ref_px.shape[0] < 5 or cur_px.shape[0] < 5:
            continue
        ref_g = float(ref_px[:, 1].mean())
        cur_g = float(cur_px[:, 1].mean())
        if cur_g < 1.0:
            continue
        ratios.append(ref_g / cur_g)
        weights.append(np.sqrt(min(ref_px.shape[0], cur_px.shape[0])))

    if ratios:
        ratios_arr = np.asarray(ratios, dtype=np.float64)
        weights_arr = np.asarray(weights, dtype=np.float64)
        exp_scale = float(np.sum(weights_arr * ratios_arr) / np.sum(weights_arr))
        if not (_EXP_SCALE_MIN <= exp_scale <= _EXP_SCALE_MAX):
            warnings.append("scene_difference")
        debug["exp_scale"] = exp_scale
        cur_exposure = int(current_params.get("exposure", 312))
        proposed["exposure"] = _clip_int(
            cur_exposure * exp_scale, "exposure", device_caps,
            (_EXPOSURE_DEFAULT_MIN, _EXPOSURE_DEFAULT_MAX),
        )
    else:
        warnings.append("no_exposure_constraint")

    # ------------------------------------------------------------------
    # 2. WB — kelvin DELTA from per-pair channel ratios.
    #
    # For *any* shared patch (not just neutral), R_ref/R_cur cancels the
    # patch reflectance and equals the per-channel illuminant/WB-gain
    # ratio between the two frames. Pooling across pairs gives a clean
    # measurement that's INVARIANT to patch chromaticity (as long as the
    # patches are not all saturated to one channel).
    #
    # Convert to a kelvin DELTA via the LUT at the (1, 1) anchor — this
    # avoids the trap of feeding "absolute" (kr_sw, kb_sw) into the LUT,
    # which conflates patch chromaticity with illuminant chromaticity.
    # The previous full-frame augmentation also corrupted this because
    # framing differences (different background coverage) bled in.
    # ------------------------------------------------------------------
    debug["kr_sw"] = 1.0
    debug["kb_sw"] = 1.0
    debug["exp_scale_sw"] = 1.0
    debug["_sw_neutral_n"] = 0
    debug["_sw_ref_saturated"] = False
    debug["ccm_sw"] = None
    debug["_sw_ccm_pairs"] = 0

    sw_sol = _solve_sw_wb_lsq(ref_bgr, cur_bgr, box_pairs)
    if sw_sol is not None:
        alpha_sw, kr_sw, kb_sw, total_px = sw_sol
        debug["exp_scale_sw"] = alpha_sw
        debug["kr_sw"] = kr_sw
        debug["kb_sw"] = kb_sw
        debug["_sw_neutral_n"] = total_px
        # Anchor: kelvin of a perfectly neutral (rg=bg=1) patch ≈ camera's
        # canonical "white". Then the OBSERVED kelvin under the inverse
        # correction is what cur should pretend the illuminant is, so
        # delta = K_obs_inverse - K_anchor moves cur's WB the right way.
        # Inverting kr/kb is needed for sign: kr>1 means "cur lacks R" →
        # the illuminant cur should pretend is COOLER (higher K) → so we
        # query the LUT at (1/kr, 1/kb).
        K_anchor = lookup_kelvin(1.0, 1.0)
        K_obs = lookup_kelvin(1.0 / max(kr_sw, 1e-3), 1.0 / max(kb_sw, 1e-3))
        delta_K = K_obs - K_anchor
        debug["kr"] = kr_sw
        debug["kb"] = kb_sw
        debug["lut_kelvin_anchor"] = K_anchor
        debug["lut_kelvin_obs"] = K_obs
        cur_kelvin = int(current_params.get("white_balance", 4600))
        new_kelvin = _clip_int(
            cur_kelvin + delta_K, "white_balance", device_caps,
            (_KELVIN_DEFAULT_MIN, _KELVIN_DEFAULT_MAX),
        )
        proposed["white_balance"] = new_kelvin
        if _hits_rail(new_kelvin, "white_balance", device_caps,
                      (_KELVIN_DEFAULT_MIN, _KELVIN_DEFAULT_MAX)):
            warnings.append("kelvin_rail")
        debug["delta_kelvin"] = new_kelvin - cur_kelvin
        print(
            f"[solver REF WB] {len(box_pairs)} pair(s), {total_px} px → "
            f"kr={kr_sw:.4f} kb={kb_sw:.4f}  "
            f"K_anchor={K_anchor:.0f} K_obs={K_obs:.0f}  "
            f"ΔK={delta_K:+.0f} → {cur_kelvin}+ΔK→{new_kelvin}K"
        )
    else:
        warnings.append("wb_underconstrained")

    # ------------------------------------------------------------------
    # 3. Saturation / contrast — adaptive per-pair / full-frame.
    #
    # Dispatch rule:
    #   * ≥ 2 usable pairs → per-pair relative ratios (most accurate;
    #     patch reflectance cancels out for sat, cross-pair Y-mean spread
    #     for con).
    #   * 0 or 1 pair      → full-frame relative ratio (same as AUTO).
    #
    # Both modes are RELATIVE: new = current × (ref/cur). The full-frame
    # form assumes ref and cur are the same scene, in which case
    # scene chroma / luma terms cancel in the ratio.
    # ------------------------------------------------------------------

    # Headroom guard: cap any boosting ratio so no currently-unclipped
    # channel is newly pushed to 255 or 0.  Model: camera applies a
    # linear stretch around neutral 128 (ch' = 128 + (ch-128)*ratio).
    def _hc_cap(bgr: np.ndarray, kind: str) -> float:
        """Max ratio not introducing NEW 255/0 clips in *bgr*."""
        if bgr is None or bgr.size == 0:
            return float("inf")
        a = bgr.reshape(-1, 3).astype(np.float32)
        cap = float("inf")
        if kind == "sat":
            for c in range(3):
                ch = a[:, c]
                hi = ch[ch < 254.5]
                lo = ch[ch > 0.5]
                if hi.size > 0:
                    p99 = float(np.percentile(hi, 99))
                    if p99 > 128.0:
                        cap = min(cap, 127.0 / (p99 - 128.0))
                if lo.size > 0:
                    p01 = float(np.percentile(lo, 1))
                    if p01 < 128.0:
                        cap = min(cap, 128.0 / (128.0 - p01))
        else:  # "con"
            y = 0.114 * a[:, 0] + 0.587 * a[:, 1] + 0.299 * a[:, 2]
            hi = y[y < 250.0]
            lo = y[y > 5.0]
            if hi.size > 0:
                p99 = float(np.percentile(hi, 99))
                if p99 > 128.0:
                    cap = min(cap, 127.0 / (p99 - 128.0))
            if lo.size > 0:
                p01 = float(np.percentile(lo, 1))
                if p01 < 128.0:
                    cap = min(cap, 128.0 / (128.0 - p01))
        return cap

    # Caps: prefer device_caps, fall back to UVC-typical defaults.
    def _caps_for(key: str, fb_min: int, fb_max: int, fb_def: int) -> CtrlCaps:
        if device_caps and key in device_caps:
            d = device_caps[key]
            return CtrlCaps(
                minimum=int(d.get("min", fb_min)),
                maximum=int(d.get("max", fb_max)),
                default=int(d.get("default", fb_def)),
            )
        return CtrlCaps(fb_min, fb_max, fb_def)

    sat_caps = _caps_for("saturation", 0, 100, 64)
    con_caps = _caps_for("contrast", 0, 100, 32)
    cur_sat = int(current_params.get("saturation", sat_caps.default))
    cur_con = int(current_params.get("contrast", con_caps.default))

    sat_lo = max(sat_caps.minimum, int(round(sat_caps.default * (1.0 - sat_con_band))))
    sat_hi = min(sat_caps.maximum, int(round(sat_caps.default * (1.0 + sat_con_band))))
    con_lo = max(con_caps.minimum, int(round(con_caps.default * (1.0 - sat_con_band))))
    con_hi = min(con_caps.maximum, int(round(con_caps.default * (1.0 + sat_con_band))))

    # --- saturation: per-pair chroma ratio ---
    sat_ratios: list[float] = []
    sat_wts: list[float] = []
    y_means_ref: list[float] = []
    y_means_cur: list[float] = []

    _CHROMA_MIN = 2.0  # skip near-neutral patches for saturation

    for ref_box, cur_box in box_pairs:
        ref_px = _box_pixels(ref_bgr, ref_box)
        cur_px = _box_pixels(cur_bgr, cur_box)
        if ref_px.shape[0] < 5 or cur_px.shape[0] < 5:
            continue
        ref_px3 = ref_px.reshape(1, -1, 3)
        cur_px3 = cur_px.reshape(1, -1, 3)
        ref_ch = compute_chroma_stats(ref_px3).chroma_mag_median
        cur_ch = compute_chroma_stats(cur_px3).chroma_mag_median
        if cur_ch >= _CHROMA_MIN:
            sat_ratios.append(float(ref_ch / max(cur_ch, 1e-3)))
            sat_wts.append(float(cur_ch))  # higher-chroma patches are more reliable
        ref_ys = compute_y_stats(ref_px3)
        cur_ys = compute_y_stats(cur_px3)
        if ref_ys.valid_pixel_count > 0 and cur_ys.valid_pixel_count > 0:
            y_means_ref.append(ref_ys.y_mean_excl_clip)
            y_means_cur.append(cur_ys.y_mean_excl_clip)

    if len(sat_ratios) >= 2:
        # Per-pair path: ≥ 2 chroma-bearing pairs cancel patch reflectance
        # and give a clean sat ratio.
        wts_arr = np.asarray(sat_wts, dtype=np.float64)
        rat_arr = np.asarray(sat_ratios, dtype=np.float64)
        sat_ratio = float(np.sum(wts_arr * rat_arr) / np.sum(wts_arr))
        if sat_ratio > 1.0:
            sat_ratio = min(sat_ratio, _hc_cap(cur_bgr, "sat"))
        new_sat = int(round(cur_sat * sat_ratio))
        new_sat = max(sat_lo, min(sat_hi, new_sat))
        debug["sat_ratio"] = sat_ratio
        debug["sat_source"] = "per_pair"
        print(
            f"[solver REF sat] per-pair({len(sat_ratios)}) "
            f"ratio={sat_ratio:.3f} → {cur_sat}→{new_sat}"
        )
    else:
        # 0 or 1 usable pair → full-frame fallback (same as AUTO uses).
        # Single-pair chroma ratio is too noisy to trust; aggregate is
        # more stable (assumes ref and cur are the same scene → scene
        # chroma cancels in the ratio, leaving sat_ref/sat_cur).
        ref_c = compute_chroma_stats(ref_bgr)
        cur_c = compute_chroma_stats(cur_bgr)
        if cur_c.chroma_mag_median >= _CHROMA_MIN:
            sat_ratio = float(ref_c.chroma_mag_median / max(cur_c.chroma_mag_median, 1e-3))
            if sat_ratio > 1.0:
                sat_ratio = min(sat_ratio, _hc_cap(cur_bgr, "sat"))
            new_sat = int(round(cur_sat * sat_ratio))
            new_sat = max(sat_lo, min(sat_hi, new_sat))
            debug["sat_ratio"] = sat_ratio
            debug["sat_source"] = (
                "full_frame_no_pairs" if not sat_ratios else "full_frame_single_pair"
            )
            warnings.append("sat_from_full_frame")
            print(
                f"[solver REF sat] full-frame ({debug['sat_source']}) "
                f"ratio={sat_ratio:.3f} → {cur_sat}→{new_sat}"
            )
        else:
            new_sat = cur_sat
            debug["sat_ratio"] = None
            debug["sat_source"] = "kept_current"
            warnings.append("sat_kept_current")

    proposed["saturation"] = new_sat

    # --- contrast: cross-pair Y-mean spread ratio ---
    con_ratio: float | None = None
    if len(y_means_ref) >= 2:
        spread_ref = float(np.std(y_means_ref))
        spread_cur = float(np.std(y_means_cur))
        if spread_cur >= 5.0:
            con_ratio = spread_ref / max(spread_cur, 1e-3)
            if con_ratio > 1.0:
                con_ratio = min(con_ratio, _hc_cap(cur_bgr, "con"))
            new_con = int(round(cur_con * con_ratio))
            new_con = max(con_lo, min(con_hi, new_con))
            debug["con_ratio"] = con_ratio
            debug["con_y_spread_ref"] = spread_ref
            debug["con_y_spread_cur"] = spread_cur
            debug["con_source"] = "cross_pair_spread"
            print(
                f"[solver REF con] cross-pair({len(y_means_ref)}) "
                f"spread {spread_cur:.1f}→{spread_ref:.1f} "
                f"ratio={con_ratio:.3f} → {cur_con}→{new_con}"
            )
        else:
            # Patches all at similar brightness; fall through to full-frame.
            con_ratio = None
            warnings.append("con_low_y_spread")
    if con_ratio is None:
        # Fall back to full-frame y_std ratio.
        ref_y = compute_y_stats(ref_bgr)
        cur_y = compute_y_stats(cur_bgr)
        if cur_y.y_std_excl_clip >= 0.02:
            con_ratio = float(ref_y.y_std_excl_clip / max(cur_y.y_std_excl_clip, 1e-6))
            if con_ratio > 1.0:
                con_ratio = min(con_ratio, _hc_cap(cur_bgr, "con"))
            new_con = int(round(cur_con * con_ratio))
            new_con = max(con_lo, min(con_hi, new_con))
            debug["con_ratio"] = con_ratio
            debug["con_source"] = "full_frame_fallback"
            if "con_low_y_spread" in warnings:
                warnings.append("con_from_full_frame")
            print(
                f"[solver REF con] full-frame fallback "
                f"y_std {cur_y.y_std_excl_clip:.4f}→{ref_y.y_std_excl_clip:.4f} "
                f"ratio={con_ratio:.3f} → {cur_con}→{new_con}"
            )
        else:
            new_con = cur_con
            debug["con_ratio"] = None
            debug["con_source"] = "kept_current"
            warnings.append("con_kept_current")

    proposed["contrast"] = new_con
    if "sat_con_partial" in warnings:
        pass  # legacy field no longer needed


    # ------------------------------------------------------------------
    # 4. CCM solve for SW-ISP debug pane (best-effort; falls back to diag).
    # Augment the user's pairs with a synthetic full-frame pair so the
    # CCM also fits the global colour cast — the right pane should look
    # like ref overall, not just on the picked patches.
    # ------------------------------------------------------------------
    h_ref, w_ref = ref_bgr.shape[:2]
    h_cur, w_cur = cur_bgr.shape[:2]
    full_pair = ((0, 0, w_ref, h_ref), (0, 0, w_cur, h_cur))
    ccm_pairs: list = [full_pair]
    ccm_pairs.extend(box_pairs)
    ccm_sol = _solve_sw_ccm_lsq(ref_bgr, cur_bgr, ccm_pairs)
    if ccm_sol is not None:
        M, n_pairs, ccm_px = ccm_sol
        debug["ccm_sw"] = M
        debug["_sw_ccm_pairs"] = n_pairs
        print(
            f"[solver SW-CCM LSQ] full+{n_pairs - 1} pair(s), {ccm_px} px → 3x3 M:\n"
            f"  R<- [{M[0,0]:+.3f} {M[0,1]:+.3f} {M[0,2]:+.3f}]\n"
            f"  G<- [{M[1,0]:+.3f} {M[1,1]:+.3f} {M[1,2]:+.3f}]\n"
            f"  B<- [{M[2,0]:+.3f} {M[2,1]:+.3f} {M[2,2]:+.3f}]"
        )
    variants = _compute_ccm_variants(ref_bgr, cur_bgr, ccm_pairs)
    debug["ccm_variants"] = variants
    _log_ccm_variants(variants, prefix="ref")

    if warnings:
        debug["_warning"] = ",".join(warnings)
    return proposed, debug


# --------------------------------------------------------------------------
# Manual + plain (patch least-squares)
# --------------------------------------------------------------------------


def manual_match_patches(
    ref_bgr: np.ndarray,
    cur_bgr: np.ndarray,
    box_pairs: Sequence[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]],
    current_params: Mapping[str, int | bool],
    device_caps: Mapping[str, Mapping[str, int]] | None = None,
) -> tuple[dict, dict]:
    """Color-checker style weighted LS for (exp_scale, kr, kb).

    Per box pair: take saturation-masked mean BGR for ref and cur sides.
    Build a 3K-row system: per channel ``cur_c * exp_scale * k_c = ref_c``
    with ``k_g = 1`` pinned. Solve for ``(exp_scale, exp_scale * kr,
    exp_scale * kb)`` then deproject. Weight rows by sqrt(N_pixels) per pair.
    """
    debug: dict = {}
    proposed: dict = {}
    warnings: list[str] = []

    if len(box_pairs) < 2:
        warnings.append("too_few_patches")
        debug["_warning"] = ",".join(warnings)
        return proposed, debug

    # Collect per-pair channel means and weights.
    rows: list[np.ndarray] = []  # rhs / lhs rows, channel-stacked
    rhs: list[float] = []
    weights_row: list[float] = []
    n_used = 0
    for ref_box, cur_box in box_pairs:
        ref_px = _filter_saturated(_box_pixels(ref_bgr, ref_box))
        cur_px = _filter_saturated(_box_pixels(cur_bgr, cur_box))
        if ref_px.shape[0] < 5 or cur_px.shape[0] < 5:
            continue
        ref_mean = ref_px.astype(np.float64).mean(axis=0)  # B, G, R
        cur_mean = cur_px.astype(np.float64).mean(axis=0)
        if (cur_mean < 1.0).any():
            continue
        n_used += 1
        w = float(np.sqrt(min(ref_px.shape[0], cur_px.shape[0])))
        # Unknowns u = (alpha, beta, gamma) where alpha = exp_scale,
        # beta = exp_scale * kr, gamma = exp_scale * kb. (k_g pinned to 1.)
        # Channel order in BGR pixel: idx 0=B, 1=G, 2=R.
        # G  : alpha   * cur_g = ref_g    → row [cur_g, 0,     0]
        # R  : beta    * cur_r = ref_r    → row [0,     0,     cur_r]    (uses gamma column? — we want kr, paired with R)
        # B  : gamma   * cur_b = ref_b    → row [0,     cur_b, 0]        (kb paired with B)
        # We choose column order (alpha, beta_b, beta_r) = (exp_scale,
        # exp_scale*kb, exp_scale*kr). G uses alpha; B uses beta_b; R uses
        # beta_r.
        rows.append(np.array([cur_mean[1], 0.0, 0.0]) * w)
        rhs.append(ref_mean[1] * w); weights_row.append(w)
        rows.append(np.array([0.0, cur_mean[0], 0.0]) * w)
        rhs.append(ref_mean[0] * w); weights_row.append(w)
        rows.append(np.array([0.0, 0.0, cur_mean[2]]) * w)
        rhs.append(ref_mean[2] * w); weights_row.append(w)

    if n_used < 2:
        warnings.append("too_few_valid_patches")
        debug["_warning"] = ",".join(warnings)
        return proposed, debug

    A = np.asarray(rows, dtype=np.float64)
    y = np.asarray(rhs, dtype=np.float64)
    # Condition number on the un-weighted matrix shape — the column scales
    # roughly cancel since each column corresponds to one channel.
    try:
        cond = float(np.linalg.cond(A))
    except np.linalg.LinAlgError:
        cond = float("inf")
    debug["matrix_cond"] = cond
    if cond > 100.0:
        warnings.append("ill_conditioned")

    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    alpha, beta_b, beta_r = float(sol[0]), float(sol[1]), float(sol[2])
    if alpha <= 1e-6:
        warnings.append("degenerate_exp_scale")
        debug["_warning"] = ",".join(warnings)
        return proposed, debug
    kr = beta_r / alpha
    kb = beta_b / alpha
    debug["exp_scale"] = alpha
    debug["kr"], debug["kb"] = kr, kb

    if not (_EXP_SCALE_MIN <= alpha <= _EXP_SCALE_MAX):
        warnings.append("scene_difference")

    cur_exposure = int(current_params.get("exposure", 312))
    proposed["exposure"] = _clip_int(
        cur_exposure * alpha, "exposure", device_caps,
        (_EXPOSURE_DEFAULT_MIN, _EXPOSURE_DEFAULT_MAX),
    )

    if kr > 0 and kb > 0:
        kelvin = lookup_kelvin(kr, kb)
        new_kelvin = _clip_int(
            kelvin, "white_balance", device_caps,
            (_KELVIN_DEFAULT_MIN, _KELVIN_DEFAULT_MAX),
        )
        proposed["white_balance"] = new_kelvin
        debug["lut_kelvin"] = kelvin
        if _hits_rail(new_kelvin, "white_balance", device_caps,
                      (_KELVIN_DEFAULT_MIN, _KELVIN_DEFAULT_MAX)):
            warnings.append("kelvin_rail")
    else:
        warnings.append("invalid_wb_ratios")

    # ------------------------------------------------------------------
    # SW-ISP debug fields (shared linear LSQ helpers — diag + CCM).
    # ------------------------------------------------------------------
    debug["kr_sw"] = 1.0
    debug["kb_sw"] = 1.0
    debug["exp_scale_sw"] = 1.0
    debug["_sw_neutral_n"] = 0
    debug["_sw_ref_saturated"] = False
    debug["ccm_sw"] = None
    debug["_sw_ccm_pairs"] = 0
    pairs_list = list(box_pairs)
    sw_sol = _solve_sw_wb_lsq(ref_bgr, cur_bgr, pairs_list)
    if sw_sol is not None:
        alpha_sw, kr_sw, kb_sw, total_px = sw_sol
        debug["exp_scale_sw"] = alpha_sw
        debug["kr_sw"] = kr_sw
        debug["kb_sw"] = kb_sw
        debug["_sw_neutral_n"] = total_px
        print(
            f"[solver SW-WB LSQ patches] {len(pairs_list)} pair(s), "
            f"{total_px} px → alpha={alpha_sw:.4f}  "
            f"kr_sw={kr_sw:.4f}  kb_sw={kb_sw:.4f}"
        )
    ccm_sol = _solve_sw_ccm_lsq(ref_bgr, cur_bgr, pairs_list)
    if ccm_sol is not None:
        M, n_pairs, ccm_px = ccm_sol
        debug["ccm_sw"] = M
        debug["_sw_ccm_pairs"] = n_pairs
        print(
            f"[solver SW-CCM LSQ patches] {n_pairs} pair(s), {ccm_px} px → 3x3 M solved"
        )

    # All four variants for the GUI 0/1/2/3 toggle.
    variants = _compute_ccm_variants(ref_bgr, cur_bgr, pairs_list)
    debug["ccm_variants"] = variants
    _log_ccm_variants(variants, prefix="patches")

    if warnings:
        debug["_warning"] = ",".join(warnings)
    return proposed, debug


# --------------------------------------------------------------------------
# Stage 4 of the hardware calibration pipeline: kelvin-only solver.
#
# Extracted from auto_match_lab so the new 4-stage state machine in
# ``camera_isp_calibrator.py`` can ask for *just* a white-balance
# proposal (after Stage 1-3 have already settled exposure / gain /
# saturation / contrast). Sharing the same chromaticity → McCamy →
# delta-kelvin path keeps the new pipeline numerically equivalent to
# the legacy Auto on the WB axis.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class KelvinResult:
    """Result of :func:`solve_kelvin_only`.

    Attributes
    ----------
    new_kelvin:
        Proposed ``white_balance`` value, already clipped to caps.
    delta:
        ``new_kelvin - current_kelvin`` for HUD reporting.
    converged:
        True when ``|delta| < _AUTO_DK_TOL`` (caller may skip another
        iteration).
    rationale:
        ``"degenerate"`` when chromaticity could not be extracted,
        ``"rail"`` when the proposal hit caps, ``"ok"`` otherwise.
    debug:
        Diagnostic floats for the HUD (``T_ref``, ``T_cur``,
        ``num_pixels``, ``xy_ref``, ``xy_cur``).
    """

    new_kelvin: int
    delta: float
    converged: bool
    rationale: str
    debug: dict


def solve_kelvin_only(
    ref_bgr: np.ndarray,
    cur_bgr: np.ndarray,
    current_kelvin: int,
    device_caps: Mapping[str, Mapping[str, int]] | None = None,
) -> KelvinResult:
    """Compute a white-balance proposal in isolation (no exposure, no CCM).

    Pipeline (matches the WB stage of :func:`auto_match_lab`):
      1. Build a saturation mask common to ref & cur.
      2. Extract median CIE xy-chromaticity over masked pixels of each frame.
      3. Convert to correlated colour temperature (McCamy 1992).
      4. Apply the *delta* in CCT to the current kelvin to avoid trusting
         the absolute (which is only as accurate as gray-world allows).
      5. Clip to caps with the conservative ``_KELVIN_DEFAULT_*`` fallback.

    Parameters
    ----------
    ref_bgr, cur_bgr:
        ``(H, W, 3)`` uint8 BGR frames sharing the same shape.
    current_kelvin:
        The hardware's current ``white_balance`` value (kelvin), used as
        the anchor for the delta.
    device_caps:
        Optional ``{"white_balance": {"min": int, "max": int}}``
        from v4l2-ctl.

    Returns
    -------
    KelvinResult
        Always returns; never raises on degenerate input. Callers must
        check ``rationale`` before trusting ``new_kelvin``.
    """
    if ref_bgr.shape != cur_bgr.shape:
        raise ValueError(
            f"ref/cur shape mismatch: {ref_bgr.shape} vs {cur_bgr.shape}"
        )
    if ref_bgr.dtype != np.uint8 or cur_bgr.dtype != np.uint8:
        raise TypeError("solve_kelvin_only expects uint8 BGR frames")

    mask = _saturated_mask(ref_bgr, cur_bgr)
    n_pix = int(mask.sum())
    debug: dict = {"num_pixels": n_pix}

    if n_pix < 100:
        return KelvinResult(
            new_kelvin=int(current_kelvin),
            delta=0.0,
            converged=True,
            rationale="degenerate",
            debug={**debug, "_warning": "insufficient_pixels"},
        )

    x_ref, y_ref = _p50_chromaticity(ref_bgr, mask)
    x_cur, y_cur = _p50_chromaticity(cur_bgr, mask)
    debug["xy_ref"] = (x_ref, y_ref)
    debug["xy_cur"] = (x_cur, y_cur)

    if not (np.isfinite(x_ref + y_ref) and np.isfinite(x_cur + y_cur)):
        return KelvinResult(
            new_kelvin=int(current_kelvin),
            delta=0.0,
            converged=True,
            rationale="degenerate",
            debug={**debug, "_warning": "chromaticity_degenerate"},
        )

    T_ref = mccamy_kelvin(x_ref, y_ref)
    T_cur = mccamy_kelvin(x_cur, y_cur)
    debug["T_ref"] = float(T_ref)
    debug["T_cur"] = float(T_cur)

    if not (np.isfinite(T_ref) and np.isfinite(T_cur)):
        return KelvinResult(
            new_kelvin=int(current_kelvin),
            delta=0.0,
            converged=True,
            rationale="degenerate",
            debug={**debug, "_warning": "mccamy_degenerate"},
        )

    target_raw = float(current_kelvin) + float(T_ref - T_cur)
    new_kelvin = _clip_int(
        target_raw, "white_balance", device_caps,
        (_KELVIN_DEFAULT_MIN, _KELVIN_DEFAULT_MAX),
    )
    delta = float(new_kelvin - int(current_kelvin))
    rationale = "rail" if _hits_rail(
        new_kelvin, "white_balance", device_caps,
        (_KELVIN_DEFAULT_MIN, _KELVIN_DEFAULT_MAX),
    ) else "ok"
    return KelvinResult(
        new_kelvin=int(new_kelvin),
        delta=delta,
        converged=abs(delta) < _AUTO_DK_TOL,
        rationale=rationale,
        debug=debug,
    )
