#!/usr/bin/env python3
"""Generate Bradford CAT LUT mapping (R/G, B/G) → kelvin.

Run once; commit the resulting ``lut_data.npz`` alongside the package
(``src/dataset_tools/dataset_tools/camera_isp/lut_data.npz``). Idempotent:
deterministic output for a given source.

Co-located with its only consumer (``dataset_tools.camera_isp.lut``) so that
the asset, the loader, and the generator stay together.

Usage:
    python -m dataset_tools.camera_isp.build_lut [output_path]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata


# sRGB (D65) → CIE XYZ matrix.
_M_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_M_XYZ_TO_RGB = np.linalg.inv(_M_RGB_TO_XYZ)


def planckian_xy(T: float) -> tuple[float, float]:
    """CIE Planckian locus chromaticity coordinates for blackbody temperature T."""
    if 1667 <= T <= 4000:
        x = (
            -0.2661239e9 / T**3
            - 0.2343580e6 / T**2
            + 0.8776956e3 / T
            + 0.179910
        )
    else:  # 4000 < T <= 25000
        x = (
            -3.0258469e9 / T**3
            + 2.1070379e6 / T**2
            + 0.2226347e3 / T
            + 0.240390
        )
    if 1667 <= T <= 2222:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif 2222 < T <= 4000:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    return float(x), float(y)


def build_lut(grid: int = 100, t_min: int = 2500, t_max: int = 8000):
    """Walk Planckian locus, compute (R/G, B/G) per kelvin, invert onto grid."""
    samples = []
    for T in np.arange(t_min, t_max + 1, 25):
        x, y = planckian_xy(T)
        Y = 1.0
        X = x * Y / y
        Z = (1 - x - y) * Y / y
        rgb = _M_XYZ_TO_RGB @ np.array([X, Y, Z])
        rgb = np.maximum(rgb, 1e-6)
        rg = rgb[0] / rgb[1]
        bg = rgb[2] / rgb[1]
        samples.append((rg, bg, float(T)))

    samples_arr = np.array(samples)

    rg_min, rg_max = 0.5, 2.0
    bg_min, bg_max = 0.5, 2.0
    rg_axis = np.linspace(rg_min, rg_max, grid)
    bg_axis = np.linspace(bg_min, bg_max, grid)
    rg_grid, bg_grid = np.meshgrid(rg_axis, bg_axis, indexing="ij")

    T_grid = griddata(
        samples_arr[:, :2], samples_arr[:, 2], (rg_grid, bg_grid), method="linear"
    )
    nan_mask = np.isnan(T_grid)
    if nan_mask.any():
        T_nearest = griddata(
            samples_arr[:, :2], samples_arr[:, 2], (rg_grid, bg_grid), method="nearest"
        )
        T_grid[nan_mask] = T_nearest[nan_mask]

    return T_grid.astype(np.float32), (rg_min, rg_max), (bg_min, bg_max)


def main(out_path: Path) -> None:
    T_grid, (rg_min, rg_max), (bg_min, bg_max) = build_lut()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        T_grid=T_grid,
        rg_min=np.float32(rg_min),
        rg_max=np.float32(rg_max),
        bg_min=np.float32(bg_min),
        bg_max=np.float32(bg_max),
    )
    print(f"wrote {out_path} shape={T_grid.shape} range T=[{T_grid.min():.0f},{T_grid.max():.0f}]K")


if __name__ == "__main__":
    # Default writes next to this module — i.e. into the package itself,
    # producing the committed binary asset consumed by ``lut.py``.
    default = Path(__file__).resolve().parent / "lut_data.npz"
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    main(out)
