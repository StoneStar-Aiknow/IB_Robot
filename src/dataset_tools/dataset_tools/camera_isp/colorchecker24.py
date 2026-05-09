"""ColorChecker Classic 24 reference values.

Source: Pascale 2006 sRGB (D65 / 2-deg), 8-bit. Same numbers used by
OpenCV's mcc module, Imatest, and LibRaw. Stored as ``(24, 3) uint8``
in **RGB** channel order (NOT BGR — convert at the call site).

Layout: 4 rows x 6 cols, top-left = 1, scanned row-major to 24.

Public API
----------
``COLORCHECKER24_SRGB``  : ``np.ndarray`` shape ``(24, 3)`` dtype ``uint8``
``COLORCHECKER24_NAMES`` : ``list[str]`` length 24
``make_checker_thumbnail(h, w, *, with_index=True, highlight=None)``:
    Render a BGR uint8 thumbnail of the 24-patch board. Used by the
    calibrator's left pane in colorchecker mode so the user always knows
    which patch corresponds to which index.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "COLORCHECKER24_SRGB",
    "COLORCHECKER24_NAMES",
    "make_checker_thumbnail",
]


# (R, G, B) sRGB 8-bit — Pascale 2006 reference values.
COLORCHECKER24_SRGB: np.ndarray = np.asarray(
    [
        (115,  82,  68),  #  1 Dark Skin
        (194, 150, 130),  #  2 Light Skin
        ( 98, 122, 157),  #  3 Blue Sky
        ( 87, 108,  67),  #  4 Foliage
        (133, 128, 177),  #  5 Blue Flower
        (103, 189, 170),  #  6 Bluish Green
        (214, 126,  44),  #  7 Orange
        ( 80,  91, 166),  #  8 Purplish Blue
        (193,  90,  99),  #  9 Moderate Red
        ( 94,  60, 108),  # 10 Purple
        (157, 188,  64),  # 11 Yellow Green
        (224, 163,  46),  # 12 Orange Yellow
        ( 56,  61, 150),  # 13 Blue
        ( 70, 148,  73),  # 14 Green
        (175,  54,  60),  # 15 Red
        (231, 199,  31),  # 16 Yellow
        (187,  86, 149),  # 17 Magenta
        (  8, 133, 161),  # 18 Cyan
        (243, 243, 242),  # 19 White (.05*)
        (200, 200, 200),  # 20 Neutral 8
        (160, 160, 160),  # 21 Neutral 6.5
        (122, 122, 121),  # 22 Neutral 5
        ( 85,  85,  85),  # 23 Neutral 3.5
        ( 52,  52,  52),  # 24 Black (1.50)
    ],
    dtype=np.uint8,
)
COLORCHECKER24_SRGB.flags.writeable = False

COLORCHECKER24_NAMES: list[str] = [
    "Dark Skin", "Light Skin", "Blue Sky", "Foliage",
    "Blue Flower", "Bluish Green", "Orange", "Purplish Blue",
    "Moderate Red", "Purple", "Yellow Green", "Orange Yellow",
    "Blue", "Green", "Red", "Yellow",
    "Magenta", "Cyan", "White (.05*)", "Neutral 8",
    "Neutral 6.5", "Neutral 5", "Neutral 3.5", "Black (1.50)",
]

_GRID_ROWS, _GRID_COLS = 4, 6
assert len(COLORCHECKER24_NAMES) == _GRID_ROWS * _GRID_COLS == 24


def make_checker_thumbnail(
    h: int,
    w: int,
    *,
    with_index: bool = True,
    highlight: int | None = None,
) -> np.ndarray:
    """Render a (h, w, 3) BGR uint8 thumbnail of the 24-patch board.

    Args:
        h, w: target image size.
        with_index: overlay 1..24 numbers on each patch.
        highlight: 1-based patch index to highlight (red rectangle).

    Returns:
        BGR uint8 array suitable for display in OpenCV windows.
    """
    if h <= 0 or w <= 0:
        raise ValueError(f"make_checker_thumbnail: bad size ({h}, {w})")
    img = np.full((h, w, 3), 30, dtype=np.uint8)  # dark gray background

    # Inner area with small margin so labels are readable.
    margin_y = max(8, h // 40)
    margin_x = max(8, w // 60)
    bar_h = max(20, h // 20)  # bottom caption bar reserved for the title
    inner_y0 = margin_y
    inner_y1 = h - margin_y - bar_h
    inner_x0 = margin_x
    inner_x1 = w - margin_x

    grid_h = inner_y1 - inner_y0
    grid_w = inner_x1 - inner_x0
    if grid_h <= 0 or grid_w <= 0:
        return img

    cell_h = grid_h // _GRID_ROWS
    cell_w = grid_w // _GRID_COLS
    pad = max(2, min(cell_h, cell_w) // 20)

    for idx in range(24):
        r, c = divmod(idx, _GRID_COLS)
        y0 = inner_y0 + r * cell_h
        x0 = inner_x0 + c * cell_w
        y1 = y0 + cell_h
        x1 = x0 + cell_w
        # Paint patch (sRGB stored, draw as BGR).
        rgb = COLORCHECKER24_SRGB[idx]
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        img[y0 + pad : y1 - pad, x0 + pad : x1 - pad] = bgr

    if with_index or highlight is not None:
        # Lazy cv2 import — colorchecker24 stays usable on headless tests.
        try:
            import cv2  # type: ignore
        except Exception:  # noqa: BLE001
            cv2 = None  # type: ignore[assignment]
        if cv2 is not None:
            for idx in range(24):
                r, c = divmod(idx, _GRID_COLS)
                y0 = inner_y0 + r * cell_h
                x0 = inner_x0 + c * cell_w
                if with_index:
                    # Auto-pick text color for contrast.
                    rgb = COLORCHECKER24_SRGB[idx]
                    luma = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                    txt_color = (0, 0, 0) if luma > 140 else (255, 255, 255)
                    cv2.putText(
                        img, f"{idx + 1}",
                        (x0 + pad + 4, y0 + pad + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, txt_color, 1,
                        cv2.LINE_AA,
                    )
                if highlight is not None and (idx + 1) == highlight:
                    cv2.rectangle(
                        img,
                        (x0 + pad - 2, y0 + pad - 2),
                        (x0 + cell_w - pad + 2, y0 + cell_h - pad + 2),
                        (0, 0, 255), 3,
                    )
            # Title bar.
            title = "ColorChecker 24"
            if highlight is not None and 1 <= highlight <= 24:
                title = (
                    f"#{highlight} {COLORCHECKER24_NAMES[highlight - 1]}  "
                    f"RGB={tuple(int(v) for v in COLORCHECKER24_SRGB[highlight - 1])}"
                )
            cv2.putText(
                img, title,
                (margin_x, h - margin_y // 2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
                cv2.LINE_AA,
            )
    return img
