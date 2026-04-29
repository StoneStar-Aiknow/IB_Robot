"""Bradford CAT lookup table: (R/G, B/G) → kelvin.

Loaded once per session from ``lut_data.npz`` (committed binary asset, generated
offline by ``scripts/build_bradford_lut.py``). Bilinear lookup at runtime.

The table maps the chromaticity ratio of a perfect-white reflector as captured
by an sRGB-D65-calibrated camera under a Planckian illuminant of temperature T,
back to T. This lets the solver convert observed gray-pixel R/G, B/G ratios to
a kelvin recommendation for ``white_balance_temperature``.
"""

from __future__ import annotations

from importlib import resources
from typing import Tuple

import numpy as np

_TABLE: np.ndarray | None = None
_RG_RANGE: Tuple[float, float] = (0.5, 2.0)
_BG_RANGE: Tuple[float, float] = (0.5, 2.0)


def _load_table() -> np.ndarray:
    global _TABLE, _RG_RANGE, _BG_RANGE
    if _TABLE is not None:
        return _TABLE
    with resources.files("dataset_tools.camera_isp").joinpath("lut_data.npz").open("rb") as fh:
        data = np.load(fh, allow_pickle=False)
        _TABLE = np.ascontiguousarray(data["T_grid"], dtype=np.float32)
        _RG_RANGE = (float(data["rg_min"]), float(data["rg_max"]))
        _BG_RANGE = (float(data["bg_min"]), float(data["bg_max"]))
    return _TABLE


def lookup_kelvin(rg: float, bg: float) -> float:
    """Bilinear lookup of CCT from a (R/G, B/G) ratio pair.

    Args:
        rg: observed R/G of a neutral patch (clipped to LUT range).
        bg: observed B/G of a neutral patch (clipped to LUT range).

    Returns:
        Estimated correlated colour temperature in kelvin.
    """
    table = _load_table()
    rg_min, rg_max = _RG_RANGE
    bg_min, bg_max = _BG_RANGE
    n_rg, n_bg = table.shape

    rg_c = float(np.clip(rg, rg_min, rg_max))
    bg_c = float(np.clip(bg, bg_min, bg_max))

    fi = (rg_c - rg_min) / (rg_max - rg_min) * (n_rg - 1)
    fj = (bg_c - bg_min) / (bg_max - bg_min) * (n_bg - 1)
    i0 = int(np.floor(fi))
    j0 = int(np.floor(fj))
    i1 = min(i0 + 1, n_rg - 1)
    j1 = min(j0 + 1, n_bg - 1)
    di = fi - i0
    dj = fj - j0

    t00 = table[i0, j0]
    t01 = table[i0, j1]
    t10 = table[i1, j0]
    t11 = table[i1, j1]
    return float(
        (1 - di) * (1 - dj) * t00
        + (1 - di) * dj * t01
        + di * (1 - dj) * t10
        + di * dj * t11
    )
