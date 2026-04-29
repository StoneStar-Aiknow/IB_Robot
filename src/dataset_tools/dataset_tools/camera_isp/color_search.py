"""Unified K / Contrast / Saturation search for the camera ISP.

This module is the **new** color-calibration path described in
``临时/camera_isp_unified_color_search_plan.md`` (v4). It runs in
parallel with the legacy 4-stage pipeline (``hw_pipeline.py``) and the
legacy solvers (``solver.py``) — the legacy code is preserved as-is so
the hard-won exposure stage is never disturbed.

Three modes share the same search driver, differing only in the cost
function injected:

* **24-card mode** — cost = Σ ΔE2000(patch_cur, patch_truth).
* **AUTO ref mode** — single-side cluster (cur Lab) + Hungarian
  assignment to a one-shot pre-clustered ref signature; cost is the
  uncertainty-softened ΔE2000 plus an L\\* quantile term.
* **m / ROI mode** — cost = Σ ΔE2000(roi_cur_mean, roi_ref_mean) plus
  a quadratic regulariser anchored to the previous solution.

Everything here is **pure numpy + scipy** with side-effects pushed
behind two small protocols (``HwWriter``, ``FrameGrabber``) so the unit
tests can drive the search with a mock camera. The module never calls
``cv2``, never touches ROS, and never reads/writes hardware directly.

Design notes (open for iteration):

* Cost functions are plain callables ``cost(frame_bgr) -> float``;
  swap them per mode without touching the search driver. The driver
  has zero knowledge of "modes".
* Cluster signatures are a list of ``ClusterPoint`` dataclasses, kept
  intentionally small and JSON-serialisable so they survive future
  refactors (e.g. moving the ref-signature cache to disk).
* The search strategy is encapsulated in ``SearchConfig``; replacing
  the direct 3-D grid with coordinate descent or Nelder-Mead is a
  one-knob change.
* When the search fails to improve over the seed, the driver returns
  ``fallback_used=True`` and emits the seed unchanged — guaranteeing
  the new path never regresses past the legacy initial estimator.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from .color_space import bgr_to_lab


# ---------------------------------------------------------------------------
# Bridge protocols — calibrator / driver / tests provide these.
# ---------------------------------------------------------------------------


class HwWriter(Protocol):
    """Synchronous V4L2 writer. Must block until the device has the value."""

    def write(self, params: Mapping[str, int]) -> None: ...


class FrameGrabber(Protocol):
    """Live frame source. Must always return a fresh BGR uint8 frame."""

    def grab(self) -> np.ndarray | None: ...


# ---------------------------------------------------------------------------
# Configuration objects (all dataclasses — easy to override per call site).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SettleConfig:
    """How to wait + average after a hardware write.

    Defaults are tuned for a 30 fps stream + V4L2 control writes that
    take effect within ~50 ms. The fresh-frame gate in the GUI adapter
    already guarantees the first grabbed frame is post-write, so we
    only need a small drop+capture window for noise averaging.
    """

    delay_ms: int = 80           # was 200; V4L2 settles in <50ms in practice,
                                 # and the fresh-frame gate already enforces
                                 # post-write freshness on the calibrator side
    n_drop: int = 1              # was 2; gate already filters stale frame
    n_capture: int = 2           # was 5; trim-mean of 2 frames still cuts noise
    trim: float = 0.0            # 2 frames -> trim disabled (would drop both)


@dataclass(frozen=True)
class SearchConfig:
    """K/C/Sat search around an initial seed.

    Two strategies share this config:

    * ``"coord"`` (default) — coordinate descent: line search along K,
      then C, then S, for ``coord_passes`` rounds. Total evals
      ≈ ``coord_passes * (n_K + n_C + n_S)`` plus the optional refine.
      Much faster than the 3-D grid (~5-6× fewer evals) and good
      enough when axes are roughly orthogonal in cost.
    * ``"grid"`` — original 3-D grid scan, kept as a fallback for
      pathological non-convex surfaces.

    Saturation reach was widened from the original ±12 (step 4 × 7 pts)
    to ±40 (step 8 × 11 pts) — users reported the previous range was
    too narrow given the device's 0..255 saturation cap.
    """

    # K range: step 400K × 9 points = ±1600K reach around the seed
    # (~3800K to 7100K when seeded at ~5400K).
    step_K: int = 400
    step_C: int = 4
    step_S: int = 8
    n_K: int = 9
    n_C: int = 7
    n_S: int = 11
    # Cap kept generous so coord descent's extra passes still fit, and
    # the rare "grid" caller still gets reasonable coverage.
    max_evals: int = 300
    improve_tol: float = 0.01      # relative J improvement to keep searching
    final_refine: bool = True       # 3×3×3 around argmin
    # New: search strategy + how many coord-descent passes to run.
    strategy: str = "coord"         # "coord" | "grid"
    coord_passes: int = 2


@dataclass(frozen=True)
class ClusterConfig:
    """Lab K-Means signature parameters."""

    k_min: int = 4
    k_max: int = 16
    downsample_max: int = 200
    edge_pad: int = 4
    L_min: float = 5.0
    L_max: float = 95.0
    seed: int = 0
    small_cluster_frac: float = 0.01


# ---------------------------------------------------------------------------
# Lightweight value types.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterPoint:
    """One Lab cluster: centroid + weight (normalised) + intra std (ΔE)."""

    centroid: tuple[float, float, float]   # (L*, a*, b*)
    weight: float
    intra_std: float                        # in ΔE-equivalent units


@dataclass
class KCS:
    """A candidate triple. Mutable so we can clip into device caps in place."""

    K: int
    C: int
    Sat: int

    def as_params(self) -> dict[str, int]:
        return {"white_balance": int(self.K),
                "contrast": int(self.C),
                "saturation": int(self.Sat)}

    def copy(self) -> "KCS":
        return KCS(self.K, self.C, self.Sat)


@dataclass
class SearchTraceEntry:
    """One evaluated point — captured for HUD / diagnosis."""

    kcs: KCS
    J: float
    note: str = ""
    # Optional human-readable metric attached by the cost factory
    # (mean ΔE2000 for cc24/m, SWD-(a*,b*) for AUTO). HUD reads this;
    # ``J`` itself stays the raw cost driving the search.
    metric_value: float = float("nan")
    metric_label: str = ""


@dataclass
class SearchResult:
    best: KCS
    seed: KCS
    seed_J: float
    best_J: float
    trace: list[SearchTraceEntry] = field(default_factory=list)
    fallback_used: bool = False
    n_evals: int = 0
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Pure helpers — pixel masking and clustering.
# ---------------------------------------------------------------------------


def mask_pixels(
    bgr: np.ndarray,
    cfg: ClusterConfig = ClusterConfig(),
) -> np.ndarray:
    """Return a 2-D bool mask: True = pixel kept.

    Drops pixels that are too dark / too bright in L\\* (clip noise) and
    a thin border of ``cfg.edge_pad`` to suppress chromatic aberration
    edges. Operates on the full frame; downsampling is the caller's
    concern.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"expected (H,W,3) BGR, got {bgr.shape}")
    h, w = bgr.shape[:2]
    L = bgr_to_lab(bgr)[..., 0]
    keep = (L > cfg.L_min) & (L < cfg.L_max)
    if cfg.edge_pad > 0 and h > 2 * cfg.edge_pad and w > 2 * cfg.edge_pad:
        border = np.zeros_like(keep)
        border[cfg.edge_pad:h - cfg.edge_pad,
               cfg.edge_pad:w - cfg.edge_pad] = True
        keep &= border
    return keep


def _downsample(bgr: np.ndarray, max_side: int) -> np.ndarray:
    """Stride-based downsample without cv2. Preserves uint8 dtype."""
    h, w = bgr.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return bgr
    step = int(np.ceil(side / max_side))
    return bgr[::step, ::step]


def _select_k(lab_pixels: np.ndarray, cfg: ClusterConfig) -> int:
    """Cheap k selection in ``[k_min, k_max]`` via a within-cluster
    variance elbow heuristic. Good enough as a first cut; replace
    with silhouette/BIC later by swapping this function.
    """
    n = lab_pixels.shape[0]
    if n <= cfg.k_min:
        return max(1, n)
    # Heuristic: k ≈ sqrt(n)/8 clipped to [k_min, k_max].
    k = int(round(np.sqrt(n) / 8.0))
    return int(np.clip(k, cfg.k_min, cfg.k_max))


def kmeans_signature_lab(
    bgr: np.ndarray,
    mask: np.ndarray | None = None,
    cfg: ClusterConfig = ClusterConfig(),
) -> list[ClusterPoint]:
    """Build a Lab K-Means signature from an image.

    Returns a list of :class:`ClusterPoint`. Empty if no usable pixels.
    Uses ``scipy.cluster.vq.kmeans2`` to avoid an sklearn dependency.
    """
    from scipy.cluster.vq import kmeans2

    img = _downsample(bgr, cfg.downsample_max)
    if mask is None:
        m = mask_pixels(img, cfg)
    else:
        # Caller-supplied mask aligned with the *original* frame; downsample
        # it the same way before applying.
        if mask.shape != bgr.shape[:2]:
            raise ValueError("mask shape mismatch with bgr")
        h, w = bgr.shape[:2]
        side = max(h, w)
        if side <= cfg.downsample_max:
            m = mask
        else:
            step = int(np.ceil(side / cfg.downsample_max))
            m = mask[::step, ::step]

    lab = bgr_to_lab(img).reshape(-1, 3)
    flat_mask = m.reshape(-1)
    pts = lab[flat_mask]
    if pts.shape[0] < cfg.k_min:
        return []

    k = _select_k(pts, cfg)
    rng = np.random.default_rng(cfg.seed)
    # kmeans2 with 'points' init + fixed RNG via numpy seed.
    np.random.seed(cfg.seed)
    centroids, labels = kmeans2(pts, k, iter=20, minit="++", seed=cfg.seed)

    out: list[ClusterPoint] = []
    n_total = pts.shape[0]
    for j in range(k):
        sel = labels == j
        n_j = int(sel.sum())
        if n_j == 0:
            continue
        cluster = pts[sel]
        centroid = centroids[j]
        # intra-cluster std in ΔE-equivalent (Euclidean Lab) units.
        if n_j >= 2:
            d = np.linalg.norm(cluster - centroid, axis=1)
            intra_std = float(d.std())
        else:
            intra_std = 0.0
        weight = n_j / n_total
        out.append(ClusterPoint(
            centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
            weight=float(weight),
            intra_std=intra_std,
        ))

    # Merge tiny clusters (<small_cluster_frac) into the nearest big one
    # but keep their weight on the absorbing cluster — preserves total
    # mass and prevents noise spikes from steering Hungarian assignment.
    if cfg.small_cluster_frac > 0 and len(out) > 1:
        out = _merge_small_clusters(out, cfg.small_cluster_frac)

    # Determinism: sort by L* for reproducible test fixtures.
    out.sort(key=lambda c: c.centroid[0])
    return out


def _merge_small_clusters(
    points: list[ClusterPoint], min_frac: float,
) -> list[ClusterPoint]:
    big = [p for p in points if p.weight >= min_frac]
    small = [p for p in points if p.weight < min_frac]
    if not big:
        # Nothing big enough — keep originals (caller will see weak signal).
        return points
    for s in small:
        # Find nearest big cluster in Lab Euclidean.
        s_c = np.asarray(s.centroid)
        dists = [float(np.linalg.norm(np.asarray(b.centroid) - s_c)) for b in big]
        idx = int(np.argmin(dists))
        b = big[idx]
        new_w = b.weight + s.weight
        big[idx] = ClusterPoint(
            centroid=b.centroid, weight=new_w, intra_std=b.intra_std,
        )
    return big


# ---------------------------------------------------------------------------
# ΔE2000 (CIEDE2000) — vectorised over (N,3) Lab arrays.
# ---------------------------------------------------------------------------


def delta_e2000(
    lab1: np.ndarray, lab2: np.ndarray, kL: float = 1.0,
    kC: float = 1.0, kH: float = 1.0,
) -> np.ndarray:
    """Compute CIEDE2000 between two ``(...,3)`` Lab arrays.

    Standard formulation (Sharma et al. 2005). Inputs broadcast.
    """
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar = 0.5 * (C1 + C2)

    G = 0.5 * (1.0 - np.sqrt(C_bar**7 / (C_bar**7 + 25.0**7 + 1e-30)))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    # If either chroma is zero, hue difference contributes nothing.
    zero_chroma = (C1p * C2p) == 0
    dhp = np.where(zero_chroma, 0.0, dhp)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)

    Lp_bar = 0.5 * (L1 + L2)
    Cp_bar = 0.5 * (C1p + C2p)

    h_sum = h1p + h2p
    h_diff = np.abs(h1p - h2p)
    Hp_bar = np.where(
        zero_chroma, h_sum,
        np.where(h_diff <= 180.0, 0.5 * h_sum,
                 np.where(h_sum < 360.0, 0.5 * (h_sum + 360.0),
                          0.5 * (h_sum - 360.0))),
    )

    T = (1.0
         - 0.17 * np.cos(np.radians(Hp_bar - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * Hp_bar))
         + 0.32 * np.cos(np.radians(3.0 * Hp_bar + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * Hp_bar - 63.0)))

    d_theta = 30.0 * np.exp(-(((Hp_bar - 275.0) / 25.0) ** 2))
    R_C = 2.0 * np.sqrt(Cp_bar**7 / (Cp_bar**7 + 25.0**7 + 1e-30))
    S_L = 1.0 + (0.015 * (Lp_bar - 50.0) ** 2) / np.sqrt(20.0 + (Lp_bar - 50.0) ** 2)
    S_C = 1.0 + 0.045 * Cp_bar
    S_H = 1.0 + 0.015 * Cp_bar * T
    R_T = -np.sin(np.radians(2.0 * d_theta)) * R_C

    dE = np.sqrt(
        (dLp / (kL * S_L)) ** 2
        + (dCp / (kC * S_C)) ** 2
        + (dHp / (kH * S_H)) ** 2
        + R_T * (dCp / (kC * S_C)) * (dHp / (kH * S_H))
    )
    return dE


def delta_e2000_pair(c1: tuple[float, float, float],
                     c2: tuple[float, float, float]) -> float:
    """Scalar convenience wrapper."""
    return float(delta_e2000(np.asarray(c1, dtype=np.float64),
                             np.asarray(c2, dtype=np.float64)))


# ---------------------------------------------------------------------------
# Hungarian-based nearest-neighbour signature matching.
# ---------------------------------------------------------------------------


def nn_match_signatures(
    P: Sequence[ClusterPoint],
    Q: Sequence[ClusterPoint],
    *,
    soften_with_intra_std: bool = True,
) -> float:
    """Match cur signature *P* to ref signature *Q* and return total cost.

    Uses ``scipy.optimize.linear_sum_assignment`` on a ΔE2000 cost
    matrix. When ``len(P) > len(Q)`` we duplicate Q columns so multiple
    cur clusters can map to the same ref cluster (and vice versa).

    Returns 0.0 when either signature is empty (caller is responsible
    for guarding that — silent zero would mask bugs).
    """
    from scipy.optimize import linear_sum_assignment

    if not P or not Q:
        raise ValueError("nn_match_signatures: empty signature")

    p_lab = np.asarray([c.centroid for c in P], dtype=np.float64)
    q_lab = np.asarray([c.centroid for c in Q], dtype=np.float64)
    p_w = np.asarray([c.weight for c in P], dtype=np.float64)
    p_s = np.asarray([c.intra_std for c in P], dtype=np.float64)
    q_s = np.asarray([c.intra_std for c in Q], dtype=np.float64)

    n, m = len(P), len(Q)
    if n <= m:
        # ΔE2000 cost matrix (n × m).
        cost = np.zeros((n, m), dtype=np.float64)
        for i in range(n):
            cost[i] = delta_e2000(np.broadcast_to(p_lab[i], (m, 3)), q_lab)
        if soften_with_intra_std:
            soft = (p_s[:, None] + q_s[None, :])
            cost = np.maximum(0.0, cost - soft)
        # Pad to square so unmatched ref columns cost 0 (we don't care
        # about ref clusters not represented in cur).
        if n < m:
            pad = np.zeros((m - n, m), dtype=np.float64)
            cost = np.vstack([cost, pad])
            p_w = np.concatenate([p_w, np.zeros(m - n)])
        row_idx, col_idx = linear_sum_assignment(cost)
        # Sum only over the real (non-padded) cur rows.
        total = 0.0
        for r, c in zip(row_idx, col_idx):
            if r < n:
                total += p_w[r] * cost[r, c]
        return float(total)
    else:
        # Duplicate Q columns to allow many-to-one matches.
        # Build cost (n × n) where columns 0..m-1 are real, m..n-1 cycle.
        cols = [j % m for j in range(n)]
        cost = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            qs = q_lab[cols]
            qss = q_s[cols]
            d = delta_e2000(np.broadcast_to(p_lab[i], (n, 3)), qs)
            if soften_with_intra_std:
                d = np.maximum(0.0, d - (p_s[i] + qss))
            cost[i] = d
        row_idx, col_idx = linear_sum_assignment(cost)
        total = 0.0
        for r, c in zip(row_idx, col_idx):
            total += p_w[r] * cost[r, c]
        return float(total)


def quantile_distance_L(
    bgr_cur: np.ndarray, bgr_ref: np.ndarray,
    quantiles: tuple[float, ...] = (5.0, 25.0, 50.0, 75.0, 95.0),
) -> float:
    """L\\* quantile L1 distance — stable proxy for contrast match."""
    L_cur = bgr_to_lab(bgr_cur)[..., 0].reshape(-1)
    L_ref = bgr_to_lab(bgr_ref)[..., 0].reshape(-1)
    qc = np.percentile(L_cur, quantiles)
    qr = np.percentile(L_ref, quantiles)
    return float(np.mean(np.abs(qc - qr)))


# ---------------------------------------------------------------------------
# Cost functions (one per mode). Each returns ``cost(frame_bgr) -> float``.
# ---------------------------------------------------------------------------


def cost_24card(
    cur_patch_means_bgr: Callable[[np.ndarray], np.ndarray],
    truth_lab: np.ndarray,
    weights: np.ndarray | None = None,
) -> Callable[[np.ndarray], float]:
    """Return cost(frame) = Σ w_i ΔE2000(patch_cur, patch_truth).

    Parameters
    ----------
    cur_patch_means_bgr:
        Callable that, given a live frame, returns the (24, 3) BGR uint8
        per-patch means. The caller owns patch detection; this module
        stays agnostic.
    truth_lab:
        (24, 3) Lab truth values.
    weights:
        Optional (24,) weights; defaults to 1.0 each.
    """
    truth_lab = np.asarray(truth_lab, dtype=np.float64)
    if truth_lab.shape != (24, 3):
        raise ValueError(f"truth_lab must be (24,3), got {truth_lab.shape}")
    w = np.ones(24, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)

    def cost(frame_bgr: np.ndarray) -> float:
        patches_bgr = cur_patch_means_bgr(frame_bgr)
        if patches_bgr.shape != (24, 3):
            raise ValueError(f"patch means must be (24,3), got {patches_bgr.shape}")
        # bgr_to_lab expects uint8 (...,3); shape (24,3) → (1,24,3) is also fine.
        cur_lab = bgr_to_lab(patches_bgr.astype(np.uint8))
        de = delta_e2000(cur_lab, truth_lab)
        # Mean ΔE over patches that have weight > 0 (ignore padded slots).
        active = w > 0
        n_active = int(active.sum())
        mean_de = float(np.mean(de[active])) if n_active else float("nan")
        cost.last_metrics = {"label": "dE", "value": mean_de, "n": n_active}
        return float(np.sum(w * de))
    cost.last_metrics = {"label": "dE", "value": float("nan"), "n": 0}
    return cost


def cost_ref_cluster(
    ref_signature: list[ClusterPoint],
    ref_bgr: np.ndarray,
    *,
    cluster_cfg: ClusterConfig = ClusterConfig(),
    lambda_L: float = 0.3,
) -> Callable[[np.ndarray], float]:
    """Return cost(frame) for AUTO ref mode.

    Re-clusters ``frame`` each call, matches to the (one-shot)
    ``ref_signature`` via Hungarian, and adds an L\\* quantile term.
    """
    def cost(frame_bgr: np.ndarray) -> float:
        cur_sig = kmeans_signature_lab(frame_bgr, cfg=cluster_cfg)
        if not cur_sig:
            cost.last_metrics = {"label": "dE", "value": float("nan"), "n": 0}
            return float("inf")
        de_total = nn_match_signatures(cur_sig, ref_signature)
        d_L = quantile_distance_L(frame_bgr, ref_bgr)
        # de_total is a weighted sum over P-clusters with weights summing
        # to 1 → already a mean ΔE per chroma cluster.
        cost.last_metrics = {"label": "dE", "value": float(de_total), "n": len(cur_sig)}
        return float(de_total + lambda_L * d_L)
    cost.last_metrics = {"label": "dE", "value": float("nan"), "n": 0}
    return cost


def cost_manual_roi(
    roi_pairs: Sequence[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]],
    ref_bgr: np.ndarray,
    *,
    prev_kcs: KCS | None = None,
    sigma_K: float = 1000.0,
    sigma_C: float = 20.0,
    sigma_S: float = 20.0,
    lam_reg: float = 0.0,
    lam_global: float = 0.5,
    weights: Sequence[float] | None = None,
    swd_cfg: "PaletteSwdConfig | None" = None,
) -> Callable[[np.ndarray, KCS], float]:
    """Return cost(frame, kcs) = Σ ΔE2000(roi_cur, roi_ref) + λ_g · SWD_global + λ R(kcs).

    Three terms:

    * **ROI data term** — the original strict ΔE2000 on user-selected
      patch pairs. Drives the search to the user's ground-truth colors.
    * **Global SWD term** (``lam_global > 0``) — chroma-weighted Sliced
      Wasserstein on (a\\*, b\\*) of the *whole* live frame against the
      *whole* ref frame. Prevents the optimum from drifting the
      background away from the reference palette while individual ROI
      patches still match (the m-mode under-constrained-background
      failure mode). Disable with ``lam_global=0.0``.
    * **Regulariser** — quadratic penalty anchored to ``prev_kcs``
      (when ``lam_reg > 0``).

    The candidate (K,C,Sat) is needed for the regulariser, so the
    returned callable takes both frame and the kcs that produced it —
    distinct from the AUTO/24-card cost signatures. ``search_KCS``
    detects the arity at runtime.
    """
    weights_arr = (np.ones(len(roi_pairs), dtype=np.float64)
                   if weights is None else np.asarray(weights, dtype=np.float64))

    def _roi_mean_bgr(
        img: np.ndarray,
        box: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        h, w = img.shape[:2]
        x0, y0, x1, y1 = box
        x0 = max(0, min(int(x0), w))
        x1 = max(0, min(int(x1), w))
        y0 = max(0, min(int(y0), h))
        y1 = max(0, min(int(y1), h))
        if x1 <= x0 or y1 <= y0:
            return None
        return img[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)

    def _roi_mean_lab(
        img: np.ndarray,
        box: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        mean_bgr = _roi_mean_bgr(img, box)
        if mean_bgr is None:
            return None
        return bgr_to_lab(mean_bgr.astype(np.uint8)[None, None, :])[0, 0]

    # Pre-compute ref ROI Lab means (ref doesn't move during search).
    ref_lab_list: list[np.ndarray] = []
    for ref_box, _ in roi_pairs:
        ref_lab = _roi_mean_lab(ref_bgr, ref_box)
        if ref_lab is None:
            return lambda frame_bgr, kcs: float("inf")
        ref_lab_list.append(ref_lab)
    ref_lab_means = (np.stack(ref_lab_list)
                     if ref_lab_list else np.empty((0, 3)))

    # Build the global SWD cost lazily; reuse its ref samples across calls.
    _global_cost: Callable[[np.ndarray], float] | None = None
    if lam_global > 0.0:
        _global_cost = cost_palette_swd(
            ref_bgr,
            swd_cfg=swd_cfg if swd_cfg is not None else PaletteSwdConfig(),
        )

    def cost(frame_bgr: np.ndarray, kcs: KCS) -> float:
        if not roi_pairs:
            cost.last_metrics = {"label": "dE", "value": float("nan"), "n": 0}
            return float("inf")
        cur_lab_list: list[np.ndarray] = []
        for _, cur_box in roi_pairs:
            cur_lab = _roi_mean_lab(frame_bgr, cur_box)
            if cur_lab is None:
                cost.last_metrics = {"label": "dE", "value": float("nan"),
                                     "n": 0}
                return float("inf")
            cur_lab_list.append(cur_lab)
        cur_lab_means = np.stack(cur_lab_list)
        de = delta_e2000(cur_lab_means, ref_lab_means)
        data_term = float(np.sum(weights_arr * de))
        global_term = 0.0
        if _global_cost is not None:
            g = _global_cost(frame_bgr)
            if np.isfinite(g):
                global_term = lam_global * g
        reg = 0.0
        if lam_reg > 0.0 and prev_kcs is not None:
            reg = lam_reg * (
                ((kcs.K - prev_kcs.K) / sigma_K) ** 2
                + ((kcs.C - prev_kcs.C) / sigma_C) ** 2
                + ((kcs.Sat - prev_kcs.Sat) / sigma_S) ** 2
            )
        # Human metric = mean ΔE2000 across the user-picked ROI pairs;
        # this is the term the user actually cares about. Global SWD
        # and regulariser are *guidance* and stay off the HUD.
        mean_de = float(np.mean(de))
        cost.last_metrics = {"label": "dE", "value": mean_de,
                             "n": len(roi_pairs)}
        return data_term + global_term + reg
    cost.last_metrics = {"label": "dE", "value": float("nan"),
                         "n": len(roi_pairs)}
    return cost


# ---------------------------------------------------------------------------
# AUTO mode v2 — Sliced Wasserstein on chroma-weighted (a*, b*).
# ---------------------------------------------------------------------------
#
# Why this replaces ``cost_ref_cluster`` for AUTO mode:
#
# * **No clustering** → no cluster-identity drift across evals → cost
#   is a smooth function of (K, C, Sat).
# * **No correspondence** → ref and live can have different scene
#   composition (different angles / occlusions); we only require the
#   *color palette* to look the same.
# * **(a*, b*) only** → orthogonal to L*, so exposure/contrast scene
#   noise doesn't pollute the white-balance/saturation signal. L* is
#   already well-handled by the legacy 4-stage exposure pipeline.
# * **Chroma weighting** → bright colored patches dominate over neutral
#   wall/floor pixels, so the metric tracks the colorchecker / colored
#   objects (the actual color signal) instead of the dominant
#   background.
# * **Sliced Wasserstein** is a standard, well-grounded distribution
#   distance (Rabin et al. 2011, Bonneel et al. 2015) widely used in
#   color transfer; cheap to compute (project onto N random 1-D
#   directions, do sorted L1 per direction, average).


def _ab_samples(
    bgr: np.ndarray,
    *,
    cluster_cfg: ClusterConfig,
    n_samples: int,
    chroma_weighted: bool,
    chroma_floor: float,
    rng: np.random.Generator,
    downsample_max_override: int | None = None,
) -> tuple[np.ndarray, float]:
    """Sample (a*, b*) pixels from *bgr*, optionally chroma-weighted.

    Parameters
    ----------
    downsample_max_override
        When given, overrides ``cluster_cfg.downsample_max`` for the
        per-side cap. Lets SWD use a larger working resolution
        (~half-size, ~76 k px) than the kmeans-tuned default (80 px).

    Returns
    -------
    ab : (M, 2) array
        Sampled (a*, b*) values. Empty array if no usable pixels.
    chroma_mean : float
        Mean of √(a²+b²) over the *kept* (above-floor) pixels — used
        only for diagnostics.
    """
    ds_cap = (downsample_max_override
              if downsample_max_override is not None
              else cluster_cfg.downsample_max)
    img = _downsample(bgr, ds_cap)
    keep = mask_pixels(img, cluster_cfg)
    lab = bgr_to_lab(img).reshape(-1, 3)
    flat_keep = keep.reshape(-1)
    pts = lab[flat_keep]
    if pts.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64), 0.0

    ab = pts[:, 1:].astype(np.float64)
    chroma = np.hypot(ab[:, 0], ab[:, 1])
    # Drop near-neutral pixels (chroma below floor) — they're mostly
    # camera noise around grey and confuse the distribution.
    above = chroma >= chroma_floor
    if above.sum() < max(50, ab.shape[0] // 200):
        # Almost no chromatic content → fall back to all kept pixels so
        # the metric still has *something* to compare.
        ab_use = ab
        chroma_use = chroma
    else:
        ab_use = ab[above]
        chroma_use = chroma[above]

    n = ab_use.shape[0]
    if n <= n_samples:
        return ab_use, float(chroma_use.mean()) if n else 0.0

    if chroma_weighted:
        # Importance-sample by chroma so saturated pixels are over-
        # represented relative to a uniform sample.
        w = chroma_use
        w = w / w.sum()
        idx = rng.choice(n, size=n_samples, replace=False, p=w)
    else:
        idx = rng.choice(n, size=n_samples, replace=False)
    return ab_use[idx], float(chroma_use.mean())


def sliced_wasserstein_2d(
    P: np.ndarray, Q: np.ndarray,
    *,
    n_dirs: int = 32,
    n_quantiles: int = 256,
    rng: np.random.Generator | None = None,
) -> float:
    """Sliced Wasserstein-1 distance between two 2-D point clouds.

    Computes the mean over ``n_dirs`` random unit directions of the
    1-D Wasserstein-1 distance (a.k.a. sorted L1) between projections
    of P and Q. Sample sizes need not match — both projections are
    re-sampled to ``n_quantiles`` evenly spaced quantiles.
    """
    if P.size == 0 or Q.size == 0:
        return float("inf")
    if rng is None:
        rng = np.random.default_rng(0)
    angles = rng.uniform(0.0, np.pi, size=n_dirs)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (n_dirs, 2)
    qs = np.linspace(0.0, 1.0, n_quantiles)

    P_proj = P @ dirs.T  # (M, n_dirs)
    Q_proj = Q @ dirs.T  # (N, n_dirs)
    # Per-direction sorted quantiles.
    P_q = np.quantile(P_proj, qs, axis=0)  # (n_quantiles, n_dirs)
    Q_q = np.quantile(Q_proj, qs, axis=0)
    return float(np.mean(np.abs(P_q - Q_q)))


@dataclass(frozen=True)
class PaletteSwdConfig:
    """AUTO-mode SWD cost knobs.

    Tuned for 640×480 live frames matched against 800×602 (or larger)
    reference posters. ``downsample_max_swd`` overrides the kmeans-era
    ``ClusterConfig.downsample_max=80`` so SWD operates on ~half-size
    (≈76 k px), which gives a smoother, lower-variance loss surface.
    """
    n_samples: int = 16000          # per-frame ab samples (4× previous)
    n_dirs: int = 64                # SWD random directions (2× previous)
    n_quantiles: int = 256          # quantile resolution per direction
    chroma_weighted: bool = True    # importance-sample by chroma
    chroma_floor: float = 4.0       # ΔE units below this counts as neutral
    rng_seed: int = 0
    downsample_max_swd: int = 320   # per-side cap (was 80 via cluster_cfg)


def cost_palette_swd(
    ref_bgr: np.ndarray,
    *,
    cluster_cfg: ClusterConfig = ClusterConfig(),
    swd_cfg: PaletteSwdConfig = PaletteSwdConfig(),
) -> Callable[[np.ndarray], float]:
    """Return cost(frame) = SWD( frame_ab_samples, ref_ab_samples ).

    The reference samples are computed once and frozen for the lifetime
    of the cost function — there's no per-eval reference work, just the
    live sampling and a fast SWD reduction.

    Returns ``inf`` only when the frame is unusable (all-saturated or
    mask drops everything); otherwise the metric is finite and smooth
    in (K, C, Sat).
    """
    rng_ref = np.random.default_rng(swd_cfg.rng_seed)
    ref_ab, _ = _ab_samples(
        ref_bgr,
        cluster_cfg=cluster_cfg,
        n_samples=swd_cfg.n_samples,
        chroma_weighted=swd_cfg.chroma_weighted,
        chroma_floor=swd_cfg.chroma_floor,
        rng=rng_ref,
        downsample_max_override=swd_cfg.downsample_max_swd,
    )
    if ref_ab.shape[0] == 0:
        # Pre-compute step decided the ref has no usable color content;
        # caller should have caught this via kmeans_signature_lab() but
        # we guard anyway.
        def _bad(_frame: np.ndarray) -> float:
            return float("inf")
        _bad.last_metrics = {"label": "ab", "value": float("nan"), "n": 0}
        return _bad

    # Re-sample fresh per call only conceptually; in practice we use a
    # *fixed* seed for the live sampler so the cost function is a
    # deterministic function of (params -> frame). Grid + refine search
    # depends on this: a per-frame seed (e.g. derived from frame.sum())
    # makes equal-param re-evaluations return slightly different J,
    # which corrupts coarse↔refine comparisons.
    def cost(frame_bgr: np.ndarray) -> float:
        rng_live = np.random.default_rng(swd_cfg.rng_seed + 1)
        live_ab, _ = _ab_samples(
            frame_bgr,
            cluster_cfg=cluster_cfg,
            n_samples=swd_cfg.n_samples,
            chroma_weighted=swd_cfg.chroma_weighted,
            chroma_floor=swd_cfg.chroma_floor,
            rng=rng_live,
            downsample_max_override=swd_cfg.downsample_max_swd,
        )
        if live_ab.shape[0] == 0:
            cost.last_metrics = {"label": "ab", "value": float("nan"), "n": 0}
            return float("inf")
        swd = sliced_wasserstein_2d(
            live_ab, ref_ab,
            n_dirs=swd_cfg.n_dirs,
            n_quantiles=swd_cfg.n_quantiles,
            rng=np.random.default_rng(swd_cfg.rng_seed),
        )
        # Human metric: SWD itself, in Lab (a*, b*) units. ~ΔE-equivalent
        # for chroma; values ~2 are imperceptible, ~5 are subtle, >10
        # are obvious.
        cost.last_metrics = {"label": "ab", "value": float(swd), "n": 0}
        return swd

    cost.last_metrics = {"label": "ab", "value": float("nan"), "n": 0}
    return cost


# ---------------------------------------------------------------------------
# Settle-aware frame capture.
# ---------------------------------------------------------------------------


def frame_capture(
    grabber: FrameGrabber,
    settle: SettleConfig = SettleConfig(),
    *,
    sleeper: Callable[[float], None] = time.sleep,
) -> np.ndarray | None:
    """Sleep, drop n_drop, capture n_capture, trimmed-mean.

    ``sleeper`` is injectable so unit tests don't actually sleep.
    """
    sleeper(settle.delay_ms / 1000.0)
    for _ in range(settle.n_drop):
        if grabber.grab() is None:
            return None
    frames: list[np.ndarray] = []
    for _ in range(max(1, settle.n_capture)):
        f = grabber.grab()
        if f is None:
            continue
        frames.append(f)
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    stack = np.stack(frames, axis=0).astype(np.float32)
    n = stack.shape[0]
    k = int(np.floor(n * settle.trim))
    if k > 0 and n - 2 * k >= 1:
        sorted_stack = np.sort(stack, axis=0)
        kept = sorted_stack[k:n - k]
        avg = kept.mean(axis=0)
    else:
        avg = stack.mean(axis=0)
    return np.clip(avg, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Search driver — direct 3-D grid (per plan §7.1, user answer #4).
# ---------------------------------------------------------------------------


def _clip_kcs(
    kcs: KCS,
    device_caps: Mapping[str, Mapping[str, int]] | None,
) -> KCS:
    def clip(val: int, key: str, fb: tuple[int, int]) -> int:
        if device_caps and key in device_caps:
            lo = int(device_caps[key].get("min", fb[0]))
            hi = int(device_caps[key].get("max", fb[1]))
        else:
            lo, hi = fb
        return int(np.clip(val, lo, hi))
    return KCS(
        K=clip(kcs.K, "white_balance", (2000, 10000)),
        C=clip(kcs.C, "contrast", (0, 100)),
        Sat=clip(kcs.Sat, "saturation", (0, 100)),
    )


def _enumerate_grid(seed: KCS, search: SearchConfig) -> list[KCS]:
    """Enumerate the 3-D grid centered on *seed*, ordered by Manhattan
    distance from the seed so early termination still keeps the seed
    neighbourhood well-covered.
    """
    Ks = [seed.K + d * search.step_K
          for d in range(-(search.n_K // 2), search.n_K // 2 + 1)]
    Cs = [seed.C + d * search.step_C
          for d in range(-(search.n_C // 2), search.n_C // 2 + 1)]
    Ss = [seed.Sat + d * search.step_S
          for d in range(-(search.n_S // 2), search.n_S // 2 + 1)]
    grid = [KCS(K, C, S) for K in Ks for C in Cs for S in Ss]
    grid.sort(key=lambda kcs: (
        abs(kcs.K - seed.K) // max(1, search.step_K)
        + abs(kcs.C - seed.C) // max(1, search.step_C)
        + abs(kcs.Sat - seed.Sat) // max(1, search.step_S)
    ))
    return grid


def search_KCS(
    seed: KCS,
    cost_fn: Callable[..., float],
    writer: HwWriter,
    grabber: FrameGrabber,
    *,
    settle: SettleConfig = SettleConfig(),
    search: SearchConfig = SearchConfig(),
    device_caps: Mapping[str, Mapping[str, int]] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    on_trace: Callable[[SearchTraceEntry], None] | None = None,
) -> SearchResult:
    """Run a direct 3-D grid search around *seed* and return the best K/C/Sat.

    The driver is mode-agnostic. *cost_fn* is either a single-arg
    callable ``f(frame) -> float`` (24-card / AUTO) or a two-arg
    callable ``f(frame, kcs) -> float`` (m / ROI mode where the
    regulariser depends on the candidate).

    Fallback contract (plan §7.3): if no grid point beats the seed, the
    seed is returned unchanged with ``fallback_used=True``. The legacy
    initial estimator is therefore the lower bound on quality.
    """
    t0 = time.monotonic()
    seed = _clip_kcs(seed, device_caps)

    def _read_metrics() -> tuple[float, str]:
        m = getattr(cost_fn, "last_metrics", None)
        if not isinstance(m, dict):
            return float("nan"), ""
        return float(m.get("value", float("nan"))), str(m.get("label", ""))

    def _eval(kcs: KCS) -> tuple[float, str, float, str]:
        writer.write(kcs.as_params())
        frame = frame_capture(grabber, settle, sleeper=sleeper)
        if frame is None:
            return float("inf"), "no_frame", float("nan"), ""
        try:
            # cost_fn arity detection: try 2-arg first, fall back to 1.
            try:
                J = cost_fn(frame, kcs)
            except TypeError:
                J = cost_fn(frame)
        except Exception as exc:  # pragma: no cover — diagnostic hook
            return float("inf"), f"cost_error:{exc}", float("nan"), ""
        mv, ml = _read_metrics()
        return float(J), "ok", mv, ml

    trace: list[SearchTraceEntry] = []
    n_evals_box = {"n": 0}

    def _record(kcs: KCS, J: float, note: str, mv: float, ml: str) -> SearchTraceEntry:
        entry = SearchTraceEntry(kcs.copy(), J, note,
                                 metric_value=mv, metric_label=ml)
        trace.append(entry)
        n_evals_box["n"] += 1
        if on_trace is not None:
            on_trace(entry)
        return entry

    seed_J, note, seed_mv, seed_ml = _eval(seed)
    _record(seed, seed_J, f"seed:{note}", seed_mv, seed_ml)

    best = seed.copy()
    best_J = seed_J

    # Track which (K, C, Sat) triples we've already evaluated so coord
    # descent and refine don't waste budget on duplicates.
    evaluated: dict[tuple[int, int, int], float] = {
        (seed.K, seed.C, seed.Sat): seed_J,
    }

    def _eval_unique(kcs: KCS, prefix: str) -> float:
        """Evaluate (or look up) *kcs*, return its J. Updates best."""
        nonlocal best, best_J
        kcs = _clip_kcs(kcs, device_caps)
        key = (kcs.K, kcs.C, kcs.Sat)
        if key in evaluated:
            return evaluated[key]
        if n_evals_box["n"] >= search.max_evals:
            return float("inf")
        J, note, mv, ml = _eval(kcs)
        _record(kcs, J, f"{prefix}:{note}", mv, ml)
        evaluated[key] = J
        if J < best_J:
            best_J = J
            best = kcs.copy()
        return J

    if search.strategy == "coord":
        axes = (
            ("K",   search.step_K, search.n_K),
            ("Sat", search.step_S, search.n_S),
            ("C",   search.step_C, search.n_C),
        )
        for pass_idx in range(max(1, search.coord_passes)):
            improved_this_pass = False
            for axis_name, step, n_pts in axes:
                if n_evals_box["n"] >= search.max_evals:
                    break
                # Symmetric line around current best, excluding 0 (==best).
                offsets = [d * step for d in range(-(n_pts // 2), n_pts // 2 + 1)
                           if d != 0]
                # Evaluate from inside out so a small move breaks ties first.
                offsets.sort(key=lambda v: abs(v))
                axis_best_J = best_J
                for off in offsets:
                    if n_evals_box["n"] >= search.max_evals:
                        break
                    cand = best.copy()
                    if axis_name == "K":
                        cand.K += off
                    elif axis_name == "C":
                        cand.C += off
                    else:
                        cand.Sat += off
                    _eval_unique(cand, f"coord{pass_idx}:{axis_name}")
                if best_J < axis_best_J - search.improve_tol * max(1.0, axis_best_J):
                    improved_this_pass = True
            if not improved_this_pass:
                break
    else:
        # Original 3-D grid scan.
        grid = _enumerate_grid(seed, search)
        for cand in grid:
            if n_evals_box["n"] >= search.max_evals:
                break
            _eval_unique(cand, "grid")

    # Optional 3×3×3 final refine around argmin.
    if search.final_refine and best_J < seed_J:
        refine_search = SearchConfig(
            step_K=max(1, search.step_K // 4),
            step_C=max(1, search.step_C // 2),
            step_S=max(1, search.step_S // 2),
            n_K=3, n_C=3, n_S=3,
            max_evals=27,
            improve_tol=search.improve_tol,
            final_refine=False,
            strategy="grid",
        )
        refine_grid = _enumerate_grid(best, refine_search)
        refine_budget = n_evals_box["n"] + 27
        for cand in refine_grid:
            if n_evals_box["n"] >= refine_budget:
                break
            cand = _clip_kcs(cand, device_caps)
            key = (cand.K, cand.C, cand.Sat)
            if key in evaluated:
                continue
            J, note, mv, ml = _eval(cand)
            _record(cand, J, f"refine:{note}", mv, ml)
            evaluated[key] = J
            if J < best_J:
                best_J = J
                best = cand.copy()

    fallback_used = best_J >= seed_J
    if fallback_used:
        # Rewrite seed so the device leaves the search session at the
        # known-good legacy value, not at the last (worse) candidate.
        writer.write(seed.as_params())
        best = seed.copy()
        best_J = seed_J

    elapsed = time.monotonic() - t0
    return SearchResult(
        best=best, seed=seed, seed_J=seed_J, best_J=best_J,
        trace=trace, fallback_used=fallback_used,
        n_evals=n_evals_box["n"], elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Offline tables — load once, share across modes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OfflineTables:
    kelvin_curve: dict      # K → {"rg": float, "bg": float}
    contrast_curve: dict    # C → L* IQR ratio relative to default
    sat_curve: dict         # Sat → C* median ratio relative to default
    settle: SettleConfig
    search: SearchConfig

    @staticmethod
    def from_dict(data: Mapping[str, object]) -> "OfflineTables":
        settle_d = dict(data.get("settle", {}) or {})  # type: ignore[arg-type]
        search_d = dict(data.get("search", {}) or {})  # type: ignore[arg-type]
        return OfflineTables(
            kelvin_curve=dict(data.get("kelvin_curve", {}) or {}),  # type: ignore[arg-type]
            contrast_curve=dict(data.get("contrast_curve", {}) or {}),  # type: ignore[arg-type]
            sat_curve=dict(data.get("sat_curve", {}) or {}),  # type: ignore[arg-type]
            settle=SettleConfig(
                delay_ms=int(settle_d.get("delay_ms", SettleConfig.delay_ms)),
                n_drop=int(settle_d.get("n_drop", SettleConfig.n_drop)),
                n_capture=int(settle_d.get("n_capture", SettleConfig.n_capture)),
                trim=float(settle_d.get("trim", SettleConfig.trim)),
            ),
            search=SearchConfig(
                step_K=int(search_d.get("step_K", SearchConfig.step_K)),
                step_C=int(search_d.get("step_C", SearchConfig.step_C)),
                step_S=int(search_d.get("step_S", SearchConfig.step_S)),
                n_K=int(search_d.get("n_K", SearchConfig.n_K)),
                n_C=int(search_d.get("n_C", SearchConfig.n_C)),
                n_S=int(search_d.get("n_S", SearchConfig.n_S)),
                max_evals=int(search_d.get("max_evals", SearchConfig.max_evals)),
                improve_tol=float(search_d.get("improve_tol", SearchConfig.improve_tol)),
                final_refine=bool(search_d.get("final_refine", SearchConfig.final_refine)),
                strategy=str(search_d.get("strategy", SearchConfig.strategy)),
                coord_passes=int(search_d.get("coord_passes", SearchConfig.coord_passes)),
            ),
        )

    @staticmethod
    def load(path: str | os.PathLike[str]) -> "OfflineTables":
        with open(path, "r", encoding="utf-8") as f:
            return OfflineTables.from_dict(json.load(f))


def default_offline_tables() -> OfflineTables:
    """Conservative defaults used when no JSON is supplied."""
    return OfflineTables.from_dict({})
