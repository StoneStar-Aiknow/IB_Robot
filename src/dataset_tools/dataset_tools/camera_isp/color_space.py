"""Color space conversions used by the ISP solver.

Pure numpy. All inputs assumed sRGB-encoded BGR uint8 (OpenCV convention).
"""

from __future__ import annotations

import numpy as np

# sRGB (D65) → CIE XYZ matrix. Source: IEC 61966-2-1.
_M_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)

# D65 reference white.
_XN, _YN, _ZN = 0.95047, 1.00000, 1.08883


def _srgb_decode(rgb_norm: np.ndarray) -> np.ndarray:
    """Inverse sRGB gamma. Input/output both in [0, 1]."""
    a = 0.055
    out = np.where(
        rgb_norm <= 0.04045,
        rgb_norm / 12.92,
        ((rgb_norm + a) / (1 + a)) ** 2.4,
    )
    return out


def bgr_to_linear_rgb(bgr_uint8: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 array (...,3) to linear-light RGB float64 in [0,1]."""
    if bgr_uint8.dtype != np.uint8:
        raise TypeError("bgr_to_linear_rgb expects uint8 input")
    rgb = bgr_uint8[..., ::-1].astype(np.float64) / 255.0
    return _srgb_decode(rgb)


def bgr_to_xyz(bgr_uint8: np.ndarray) -> np.ndarray:
    """BGR uint8 → CIE XYZ float64 (D65, Y in [0, 1])."""
    rgb_lin = bgr_to_linear_rgb(bgr_uint8)
    flat = rgb_lin.reshape(-1, 3)
    xyz = flat @ _M_RGB_TO_XYZ.T
    return xyz.reshape(rgb_lin.shape)


def _f_lab(t: np.ndarray) -> np.ndarray:
    delta = 6.0 / 29.0
    return np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4.0 / 29.0)


def bgr_to_lab(bgr_uint8: np.ndarray) -> np.ndarray:
    """BGR uint8 → CIE L*a*b* float64. L* in [0, 100], a*/b* roughly [-128, 127]."""
    xyz = bgr_to_xyz(bgr_uint8)
    fx = _f_lab(xyz[..., 0] / _XN)
    fy = _f_lab(xyz[..., 1] / _YN)
    fz = _f_lab(xyz[..., 2] / _ZN)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def bgr_to_xyz_chromaticity(bgr_uint8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """BGR uint8 → (x, y) CIE 1931 chromaticity coordinates.

    Returned arrays have one fewer axis than input (the channel axis is collapsed).
    Pixels with X+Y+Z ≈ 0 yield NaN (filter before averaging).
    """
    xyz = bgr_to_xyz(bgr_uint8)
    s = xyz[..., 0] + xyz[..., 1] + xyz[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(s > 1e-9, xyz[..., 0] / s, np.nan)
        y = np.where(s > 1e-9, xyz[..., 1] / s, np.nan)
    return x, y


def mccamy_kelvin(x: float, y: float) -> float:
    """McCamy's approximation: chromaticity (x, y) → correlated colour temperature.

    Accurate to ±5 K in 2856–6500 K, ±50 K outside.
    Reference: McCamy, "Correlated Color Temperature as an Explicit Function
    of Chromaticity Coordinates", Color Research & Application, 1992.
    """
    denom = 0.1858 - y
    if abs(denom) < 1e-9:
        return float("nan")
    n = (x - 0.3320) / denom
    return 449.0 * n**3 + 3525.0 * n**2 + 6823.3 * n + 5520.33
