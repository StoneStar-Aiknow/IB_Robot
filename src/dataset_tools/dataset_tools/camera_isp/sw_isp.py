"""Software ISP — minimal post-processing applied in the calibrator's debug pane.

The calibrator's third pane (gated by the 'd' key, off by default) renders the
live frame after applying these gains *in linear light*. This lets the user
sanity-check whether the solver's recommended channel gains and exposure
scalar — when applied to actual pixels with no camera involved — recover the
reference look. If the SW pane aligns and the hardware path does not, the
discrepancy is in the camera control chain (range, quantisation, LUT
mismatch), not in the solver maths.

Pipeline (BGR uint8 -> BGR uint8):

    1. sRGB-decode to linear [0, 1].
    2. Multiply blue channel by ``kb``, red by ``kr``, all by ``exp_scale``.
    3. Clip to [0, 1] and re-encode via sRGB OETF.
    4. Cast back to uint8.

``gamma`` and ``ccm`` are reserved future hooks; the present implementation
deliberately ignores them so the linear-RGB invariants we test against stay
trivial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _srgb_decode(rgb_norm: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(
        rgb_norm <= 0.04045,
        rgb_norm / 12.92,
        ((rgb_norm + a) / (1 + a)) ** 2.4,
    )


def _srgb_encode(rgb_lin: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(
        rgb_lin <= 0.0031308,
        rgb_lin * 12.92,
        (1 + a) * np.power(np.maximum(rgb_lin, 0.0), 1.0 / 2.4) - a,
    )


@dataclass
class SwIspParams:
    """Linear-domain post-process gains.

    Two mutually-exclusive modes (CCM takes precedence when set):

    * **Diagonal mode** (``ccm`` is None): apply ``exp_scale`` then per-channel
      gains ``kr`` (red) and ``kb`` (blue) in linear light. Cheap and
      well-conditioned but only fixes white-balance + exposure.

    * **CCM mode** (``ccm`` is a 3x3 array): apply ``rgb_lin' = ccm @ rgb_lin``
      directly. The CCM is assumed to *already absorb* exposure and any
      per-channel gains (the solver folds them in), so ``kr`` / ``kb`` /
      ``exp_scale`` are ignored when ``ccm`` is set. Future hooks (cross
      products, squared terms) will extend this matrix to 3xK.

    Attributes:
        kr: red-channel multiplicative gain in linear light. 1.0 is identity.
        kb: blue-channel multiplicative gain in linear light. 1.0 is identity.
        exp_scale: global luminance scalar in linear light. 1.0 is identity.
        gamma: reserved hook (currently ignored).
        ccm:   colour-correction matrix in linear RGB. Two accepted shapes:
               ``(3,3)`` (legacy linear ``rgb_out = M @ rgb``) or
               ``(3,6)`` (RPCC2 root-polynomial: ``rgb_out = M @ ρ`` with
               ``ρ = [R, G, B, √(RG), √(RB), √(GB)]``). ``None`` selects
               diagonal-gain mode.
    """

    kr: float = 1.0
    kb: float = 1.0
    exp_scale: float = 1.0
    gamma: Optional[float] = None
    ccm: Optional[np.ndarray] = None


def apply_sw_isp(bgr_uint8: np.ndarray, params: SwIspParams) -> np.ndarray:
    """Apply linear-domain ISP (diag gains or CCM) to a BGR uint8 frame.

    The transform is performed in linear light (sRGB-decoded) so channel
    multiplications are physically meaningful; the result is re-encoded via
    the sRGB OETF before being cast back to uint8.

    Args:
        bgr_uint8: HxWx3 uint8 BGR (OpenCV convention).
        params:    :class:`SwIspParams`.

    Returns:
        HxWx3 uint8 BGR with the same shape and dtype as the input.
    """
    if bgr_uint8.dtype != np.uint8:
        raise TypeError("apply_sw_isp expects uint8 input")
    if bgr_uint8.ndim != 3 or bgr_uint8.shape[-1] != 3:
        raise ValueError(
            f"apply_sw_isp expects HxWx3 BGR; got shape {bgr_uint8.shape}"
        )

    ccm = params.ccm
    if ccm is not None:
        ccm = np.asarray(ccm, dtype=np.float64)
        # Two accepted shapes:
        #   (3, 3) — legacy linear CCM:           rgb_out = M @ rgb
        #   (3, 6) — RPCC2 (root polynomial 2):   rgb_out = M @ ρ(rgb)
        #            where ρ = [R, G, B, √(RG), √(RB), √(GB)]
        # All other shapes are rejected.
        if ccm.shape not in ((3, 3), (3, 6)):
            raise ValueError(
                f"SwIspParams.ccm must be (3,3) or (3,6); got shape {ccm.shape}"
            )
        if not np.all(np.isfinite(ccm)):
            raise ValueError("SwIspParams.ccm contains non-finite values")

    # Identity short-circuit — preserves bit-exact input.
    is_diag_identity = (
        params.kr == 1.0
        and params.kb == 1.0
        and params.exp_scale == 1.0
    )
    if ccm is None:
        is_ccm_identity = False
    elif ccm.shape == (3, 3):
        is_ccm_identity = bool(np.allclose(ccm, np.eye(3), atol=1e-12))
    else:  # (3, 6)
        # RPCC identity: M = [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]]
        rpcc_identity = np.zeros((3, 6), dtype=np.float64)
        rpcc_identity[:, :3] = np.eye(3)
        is_ccm_identity = bool(np.allclose(ccm, rpcc_identity, atol=1e-12))
    if (ccm is None and is_diag_identity) or is_ccm_identity:
        return bgr_uint8.copy()

    # BGR uint8 -> RGB float [0,1] -> linear RGB.
    rgb_norm = bgr_uint8[..., ::-1].astype(np.float64) / 255.0
    rgb_lin = _srgb_decode(rgb_norm)

    if ccm is not None:
        flat = rgb_lin.reshape(-1, 3)
        if ccm.shape == (3, 3):
            # Linear: rgb_out_row = rgb_in_row @ M.T.
            out_flat = flat @ ccm.T
        else:
            # RPCC2: build the 6-D feature in linear light then mat-mul.
            # Negative inputs (numerical noise from sRGB decode at 0) are
            # clipped so that sqrt stays real.
            f = np.clip(flat, 0.0, None)
            r, g, b = f[:, 0], f[:, 1], f[:, 2]
            feat = np.stack(
                [r, g, b, np.sqrt(r * g), np.sqrt(r * b), np.sqrt(g * b)],
                axis=1,
            )
            out_flat = feat @ ccm.T
        rgb_lin = out_flat.reshape(rgb_lin.shape)
    else:
        # Diagonal mode: per-channel gains. Pixel layout is (..., R, G, B).
        rgb_lin[..., 0] *= float(params.kr)
        rgb_lin[..., 2] *= float(params.kb)
        rgb_lin *= float(params.exp_scale)

    rgb_lin = np.clip(rgb_lin, 0.0, 1.0)
    rgb_norm_out = _srgb_encode(rgb_lin)
    rgb_norm_out = np.clip(rgb_norm_out, 0.0, 1.0)

    # Back to BGR uint8.
    bgr_out = (rgb_norm_out[..., ::-1] * 255.0 + 0.5).astype(np.uint8)
    return bgr_out
