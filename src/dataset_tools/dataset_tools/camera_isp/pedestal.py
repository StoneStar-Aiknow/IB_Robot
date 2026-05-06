"""Black-level (pedestal) estimation for the camera ISP calibrator.

USB-cam ``brightness`` is a *constant* offset added to every pixel
in the 8-bit output domain — **not** scene brightness. Treating it
as a free 0..255 parameter (the way K/C/Sat are tuned) lets the
calibrator drift it positive and wash the image grey, which
silently destroys downstream training data.

This module is the single source of truth for the signed-Δ
black-level offset. It exposes three estimators:

* :func:`estimate_pedestal_offset_auto` — used by AUTO/cc24 modes
  and as the safe default when no user nomination is available.
* :func:`estimate_pedestal_offset_ref_mode` — AUTO/ref-mode variant
  that compares live vs. ref blacks and *never* returns positive
  Δ (a washed reference yields Δ=0 plus a warning hint).
* :func:`estimate_pedestal_offset_manual` — used by ``m`` mode when
  the user explicitly nominates a black-reference patch on REF.
  Allows a small positive Δ (capped via :class:`PedestalConfig`).

All three return a :class:`DarkLevelEstimate` carrying the proposed
Δ, sample-set diagnostics, and a ``warn`` string the calibrator
surfaces verbatim. The single place that turns Δ into a device
register value is :func:`apply_pedestal_offset`.

Pure numpy, no cv2, no ROS — same shape as ``color_search.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .color_space import bgr_to_lab


# ---------------------------------------------------------------------------
# Configuration & lightweight value types.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PedestalConfig:
    """Tunables for the pedestal estimator.

    The defaults are deliberately conservative (large frac threshold,
    narrow Δ caps) — pedestal mistakes are *much* more expensive than
    a no-op skip, because a mis-applied pedestal ruins every frame
    captured afterwards.
    """

    # --- Pixel acceptance (operates on raw 8-bit Y).
    Y_DARK_MAX_RAW: int = 32
    """Pixel kept only if its raw 8-bit luma ≤ this threshold."""

    CHROMA_NEUTRAL_MAX: float = 6.0
    """Pixel kept only if sqrt(a*² + b*²) ≤ this (Lab units)."""

    MIN_DARK_PIXEL_FRAC: float = 0.005
    """Estimator skips if fewer than this fraction of pixels qualify."""

    # --- Robust statistic over the kept dark pool.
    PCTL_LOW: float = 5.0
    PCTL_HIGH: float = 25.0

    # --- Auto mode: ``brightness`` may only subtract.
    AUTO_DELTA_MIN: int = -20
    AUTO_DELTA_MAX: int = 0

    # --- Manual mode (user confirmed a black patch): tiny + allowed.
    MANUAL_DELTA_MIN: int = -20
    MANUAL_DELTA_MAX: int = 5

    # --- Sanity warning: user pick vs. auto-detected darkest neutral.
    PICK_VS_AUTO_WARN_GAP: int = 8


@dataclass(frozen=True)
class DarkLevelEstimate:
    """Result of one estimator call. Always finite — never NaN."""

    delta: int
    confidence: float
    n_dark_pixels: int
    dark_pixel_frac: float
    measured_y: float
    used: str
    warn: str = ""


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _raw_y_from_bgr(bgr_uint8: np.ndarray) -> np.ndarray:
    """Return the raw 8-bit luma plane in float64, same H×W as input.

    We use Rec.601 (BT.601) Y because that is what UVC cameras
    typically integrate against. The pedestal Δ lives in 8-bit code
    values, so we deliberately stay in raw uint8 space (NOT L* in
    [0, 100]).
    """
    if bgr_uint8.ndim != 3 or bgr_uint8.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3) BGR, got {bgr_uint8.shape}")
    f = bgr_uint8.astype(np.float64)
    # BGR order
    return 0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]


def _select_neutral_dark_mask(
    bgr_uint8: np.ndarray, cfg: PedestalConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y_raw, keep_mask)``.

    Both arrays share the input H×W. Pixels passing ``keep_mask`` are
    near-neutral and dim — i.e. plausible black-evidence pixels.
    """
    y = _raw_y_from_bgr(bgr_uint8)
    lab = bgr_to_lab(bgr_uint8)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    keep = (y <= cfg.Y_DARK_MAX_RAW) & (chroma <= cfg.CHROMA_NEUTRAL_MAX)
    return y, keep


def _delta_from_pool(pool_y: np.ndarray, cfg: PedestalConfig) -> tuple[float, float]:
    """Return ``(measured_y, delta_raw)`` for a non-empty dark pool.

    The robust statistic is the mean of the inter-quantile slice
    [PCTL_LOW..PCTL_HIGH] — drops the absolute-min outliers (single
    hot/dead pixels) without losing the dark cluster's centre.
    """
    lo = np.percentile(pool_y, cfg.PCTL_LOW)
    hi = np.percentile(pool_y, cfg.PCTL_HIGH)
    sliced = pool_y[(pool_y >= lo) & (pool_y <= hi)]
    if sliced.size == 0:
        sliced = pool_y
    measured_y = float(sliced.mean())
    delta_raw = -measured_y  # caller clamps + rounds
    return measured_y, delta_raw


def _clamp_delta(delta_raw: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(delta_raw))))


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def estimate_pedestal_offset_auto(
    bgr_uint8: np.ndarray,
    *,
    cfg: PedestalConfig = PedestalConfig(),
) -> DarkLevelEstimate:
    """Auto-mode estimator: scan the whole frame for neutral darks.

    Returns ``delta`` clamped to ``[AUTO_DELTA_MIN, AUTO_DELTA_MAX]``
    (i.e. ≤ 0 — never positive). When the frame has insufficient
    dark-neutral evidence the estimator returns ``delta=0`` and a
    ``used="skipped:..."`` reason — this is the safe no-op path.
    """
    y, keep = _select_neutral_dark_mask(bgr_uint8, cfg)
    total = int(y.size)
    n_dark = int(keep.sum())
    frac = n_dark / max(1, total)

    if frac < cfg.MIN_DARK_PIXEL_FRAC:
        return DarkLevelEstimate(
            delta=0,
            confidence=0.0,
            n_dark_pixels=n_dark,
            dark_pixel_frac=frac,
            measured_y=float("nan"),
            used="skipped:no_dark_evidence",
        )

    pool_y = y[keep]
    measured_y, delta_raw = _delta_from_pool(pool_y, cfg)
    delta = _clamp_delta(delta_raw, cfg.AUTO_DELTA_MIN, cfg.AUTO_DELTA_MAX)
    confidence = float(min(1.0, frac / 0.02))
    return DarkLevelEstimate(
        delta=delta,
        confidence=confidence,
        n_dark_pixels=n_dark,
        dark_pixel_frac=frac,
        measured_y=measured_y,
        used="auto",
    )


def estimate_pedestal_offset_ref_mode(
    live_bgr: np.ndarray,
    ref_bgr: np.ndarray,
    *,
    cfg: PedestalConfig = PedestalConfig(),
) -> DarkLevelEstimate:
    """AUTO/ref-mode estimator: match live blacks to ref blacks.

    Both frames are scanned with the auto estimator. The proposed
    Δ tries to push the live frame's darkest neutral region to the
    same Y as the ref's. If the ref's darkest neutrals are *higher*
    than the live's (i.e. ref looks washed) we return ``delta=0``
    plus a warn string — auto mode is forbidden from pushing Δ
    positive without explicit user confirmation.
    """
    auto_live = estimate_pedestal_offset_auto(live_bgr, cfg=cfg)
    auto_ref = estimate_pedestal_offset_auto(ref_bgr, cfg=cfg)

    # If either side has no usable evidence, fall back to the
    # live-only estimator. This already returns delta=0 when needed.
    if auto_live.used.startswith("skipped") or auto_ref.used.startswith("skipped"):
        warn = ""
        if auto_ref.used.startswith("skipped"):
            warn = "ref has no dark-neutral evidence; pedestal kept at Δ=0"
        return DarkLevelEstimate(
            delta=auto_live.delta,
            confidence=auto_live.confidence,
            n_dark_pixels=auto_live.n_dark_pixels,
            dark_pixel_frac=auto_live.dark_pixel_frac,
            measured_y=auto_live.measured_y,
            used=f"ref_mode_fallback:{auto_live.used}",
            warn=warn,
        )

    # delta_raw = ref_y - live_y so live blacks shift toward ref blacks.
    delta_raw = float(auto_ref.measured_y - auto_live.measured_y)
    delta = _clamp_delta(delta_raw, cfg.AUTO_DELTA_MIN, cfg.AUTO_DELTA_MAX)
    used = "ref_mode"
    warn = ""
    if delta_raw > cfg.AUTO_DELTA_MAX:
        # Reference looked washed; auto-cap blocked positive movement.
        used = "ref_mode:auto_cap_blocked_positive"
        warn = (
            f"reference appears washed (ref Y_dark="
            f"{auto_ref.measured_y:.0f}); pedestal kept at Δ=0. "
            "Switch to m-mode to confirm a black patch if you want "
            "to match it."
        )
    return DarkLevelEstimate(
        delta=delta,
        confidence=min(auto_live.confidence, auto_ref.confidence),
        n_dark_pixels=auto_live.n_dark_pixels,
        dark_pixel_frac=auto_live.dark_pixel_frac,
        measured_y=auto_live.measured_y,
        used=used,
        warn=warn,
    )


def estimate_pedestal_offset_manual(
    ref_bgr: np.ndarray,
    live_bgr: np.ndarray,
    ref_box_xyxy: tuple[int, int, int, int],
    live_box_xyxy: tuple[int, int, int, int],
    *,
    cfg: PedestalConfig = PedestalConfig(),
) -> DarkLevelEstimate:
    """Manual estimator: match live blacks to ref blacks at user-nominated patches.

    The user nominates a black-reference box on BOTH the ref and
    live images. ``delta = ref_black_Y - live_black_Y`` pushes the
    live camera's black toward the reference's black level. Works in
    both directions:

    * Negative delta (live too bright): darken a camera with sensor
      pedestal so its black matches the ref.
    * Positive delta (ref brighter than live, up to
      ``MANUAL_DELTA_MAX``): raise camera brightness slightly when
      the reference has a higher baked-in black floor than the live.

    Also sanity-checks the ref pick against the auto estimator on
    the ref image; emits a non-blocking ``warn`` string when they
    diverge by more than ``PICK_VS_AUTO_WARN_GAP``.
    """
    # --- Measure ref black at user's ref box. ----------------------
    rh, rw = ref_bgr.shape[:2]
    x0, y0, x1, y1 = ref_box_xyxy
    x0 = max(0, min(int(x0), rw))
    x1 = max(0, min(int(x1), rw))
    y0 = max(0, min(int(y0), rh))
    y1 = max(0, min(int(y1), rh))
    if x1 <= x0 or y1 <= y0:
        return DarkLevelEstimate(
            delta=0, confidence=0.0,
            n_dark_pixels=0, dark_pixel_frac=0.0,
            measured_y=float("nan"),
            used="skipped:empty_ref_box",
        )
    ref_crop = ref_bgr[y0:y1, x0:x1]
    ref_y = float(_raw_y_from_bgr(ref_crop).mean())

    # --- Measure live black at user's live box. --------------------
    lh, lw = live_bgr.shape[:2]
    lx0, ly0, lx1, ly1 = live_box_xyxy
    lx0 = max(0, min(int(lx0), lw))
    lx1 = max(0, min(int(lx1), lw))
    ly0 = max(0, min(int(ly0), lh))
    ly1 = max(0, min(int(ly1), lh))
    if lx1 <= lx0 or ly1 <= ly0:
        return DarkLevelEstimate(
            delta=0, confidence=0.0,
            n_dark_pixels=0, dark_pixel_frac=0.0,
            measured_y=ref_y,
            used="skipped:empty_live_box",
        )
    live_crop = live_bgr[ly0:ly1, lx0:lx1]
    live_y = float(_raw_y_from_bgr(live_crop).mean())

    # --- delta = ref_black - live_black  (match live to ref). ------
    delta_raw = ref_y - live_y
    delta = _clamp_delta(delta_raw, cfg.MANUAL_DELTA_MIN, cfg.MANUAL_DELTA_MAX)

    # Sanity: compare ref pick to what auto finds on the ref.
    auto = estimate_pedestal_offset_auto(ref_bgr, cfg=cfg)
    warn = ""
    if (not auto.used.startswith("skipped")
            and abs(ref_y - auto.measured_y) > cfg.PICK_VS_AUTO_WARN_GAP):
        warn = (
            f"ref pick reads Y={ref_y:.0f} but the darkest "
            f"neutral region on the ref is Y={auto.measured_y:.0f}; "
            "pedestal estimate may be off. Continuing with your pick."
        )

    n_dark = int(ref_crop.size // 3)
    return DarkLevelEstimate(
        delta=delta,
        confidence=1.0,
        n_dark_pixels=n_dark,
        dark_pixel_frac=n_dark / max(1, rh * rw),
        measured_y=ref_y,
        used=f"manual_pick:ref_y={ref_y:.1f},live_y={live_y:.1f}",
        warn=warn,
    )


# ---------------------------------------------------------------------------
# Δ → device register mapping.
# ---------------------------------------------------------------------------


def apply_pedestal_offset(
    delta: int,
    brightness_default: int,
    brightness_min: int,
    brightness_max: int,
) -> int:
    """Convert a signed Δ into a hardware-clamped register value.

    The device range (typically 0..255) still has authority — Δ is
    clamped *after* the per-mode soft cap by the constraint that the
    final register stays inside ``[brightness_min, brightness_max]``.
    """
    raw = int(brightness_default) + int(delta)
    return int(max(brightness_min, min(brightness_max, raw)))


def darkest_pair_index(
    pairs: Sequence[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]],
    ref_bgr: np.ndarray,
) -> int | None:
    """Return the index of the pair whose REF box has the lowest mean Y.

    Used by the ``m``-mode wizard to pre-suggest "which existing
    pair looks most like a black patch". Returns ``None`` for an
    empty pairs list.
    """
    if not pairs:
        return None
    h, w = ref_bgr.shape[:2]
    best_y = float("inf")
    best_idx: int | None = None
    for i, (ref_box, _) in enumerate(pairs):
        x0, y0, x1, y1 = ref_box
        x0 = max(0, min(int(x0), w))
        x1 = max(0, min(int(x1), w))
        y0 = max(0, min(int(y0), h))
        y1 = max(0, min(int(y1), h))
        if x1 <= x0 or y1 <= y0:
            continue
        y_mean = float(_raw_y_from_bgr(ref_bgr[y0:y1, x0:x1]).mean())
        if y_mean < best_y:
            best_y = y_mean
            best_idx = i
    return best_idx
