"""The two-hand gesture model (HaGRIDv2 YOLOv10n, 320 cut-head om) and its CPU post-processing. **No ROS.**

Model I/O (motion-template-hand-HANDOFF.md, 11-18): input ``images`` float32
[1,3,320,320], RGB / 255; output [1,38,2100]: rows 0-3 are x1,y1,x2,y2 in the
320 input, rows 4-37 the 34 class scores after sigmoid. No NMS (one-to-one head).
Only the cut-head om: a ``*_e2e.om`` carries TopK on the AICPU and segfaults on
load on the .5 board, so ``GestureOm`` refuses that name.

``post_fast`` is copied from ``motion-template-hand/post_cut.py`` (numpy only;
checked there bit-for-bit against the e2e post-processing).

Classes kept (names.json ids): holy 2, xsign 7, hand_heart 8 + hand_heart2 9 =
heart; everything else is ``none``.

Every call must come from the thread that built the session (ACL contexts are per thread).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

INPUT_SIZE = 320
OUTPUT_SHAPE = (1, 38, 2100)
NUM_CLASSES = 34
CLASS_IDS = {"heart": (8, 9), "holy": (2,), "xsign": (7,)}
NONE = "none"
CLASSES = (*CLASS_IDS, NONE)


class BadOutput(ValueError):
    """The model returned something that is not a finite [1,38,2100] tensor."""


def post_fast(y, conf=0.25, max_det=300):
    """[1,38,N] -> [k,6] (x1,y1,x2,y2 in the model input, score, class id), score >= conf, best first."""
    t = y.reshape(y.shape[-2], y.shape[-1])
    s = t[4:]
    ms = s.max(0)
    a = np.flatnonzero(ms >= conf)
    if a.size > max_det:
        a = a[np.argsort(-ms[a], kind="stable")[:max_det]]
    sub = s[:, a]
    c, j = np.nonzero(sub >= conf)
    sc = sub[c, j]
    o = np.argsort(-sc, kind="stable")[:max_det]
    return np.concatenate([t[:4, a[j[o]]].T, sc[o, None], c[o, None].astype(np.float32)], 1)


def best_gesture(dets: np.ndarray, conf: float) -> tuple[str, float, np.ndarray | None]:
    """The best-scoring kept class among the detections: (name, score, xyxy in model input) or (none, 0, None)."""
    best: tuple[str, float, np.ndarray | None] = (NONE, 0.0, None)
    if dets.size == 0:
        return best
    cls = dets[:, 5].astype(np.int64)
    for name, ids in CLASS_IDS.items():
        rows = np.flatnonzero(np.isin(cls, ids) & (dets[:, 4] >= conf))
        if rows.size:
            r = rows[int(np.argmax(dets[rows, 4]))]
            score = float(dets[r, 4])
            if score > best[1]:
                best = (name, score, dets[r, :4].astype(np.float64))
    return best


def to_source(xyxy, x0: int, y0: int, side: int) -> list[float]:
    """Model-input box -> source pixels for the square (x0, y0, side) that was resized to INPUT_SIZE."""
    r = INPUT_SIZE / float(side)
    b = np.asarray(xyxy, dtype=np.float64) / r + (x0, y0, x0, y0)
    return [float(v) for v in b]


def preprocess(square_bgr: np.ndarray) -> np.ndarray:
    """Square BGR uint8 -> [1,3,320,320] float32 RGB / 255 (a square needs no letterbox padding)."""
    import cv2

    if square_bgr.shape[0] != INPUT_SIZE or square_bgr.shape[1] != INPUT_SIZE:
        square_bgr = cv2.resize(square_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    x = square_bgr[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)
    x *= np.float32(1.0 / 255.0)
    return np.ascontiguousarray(x)


def resolve_path(path) -> Path:
    """A relative path is under ``$WORKSPACE``, as the hand ``model_dir`` is."""
    p = Path(str(path)).expanduser()
    if p.is_absolute():
        return p
    workspace = os.environ.get("WORKSPACE")
    if not workspace:
        raise ValueError(f"gesture model path {str(path)!r} is relative but WORKSPACE is unset")
    return Path(workspace) / p


def _open_session(device: int, path: str):
    from ais_bench.infer.interface import InferSession

    return InferSession(device, path)


class GestureOm:
    """One ais_bench ``InferSession`` on the gesture om, built in (and only used from) the calling thread."""

    def __init__(self, path, device: int = 0):
        self.path = resolve_path(path)
        if self.path.name.endswith("_e2e.om"):
            raise ValueError(f"{self.path.name}: the e2e om segfaults on load on the 310P board; use the cut-head om")
        if not self.path.is_file():
            raise FileNotFoundError(f"gesture model {self.path} missing")
        self.session = _open_session(int(device), str(self.path))

    def infer(self, x: np.ndarray) -> np.ndarray:
        out = self.session.infer([x], mode="static")
        y = np.asarray(out[0])
        if y.shape != OUTPUT_SHAPE:
            raise BadOutput(f"output shape {y.shape}, expected {OUTPUT_SHAPE}")
        if not np.isfinite(y).all():
            raise BadOutput("output has NaN/inf")
        return y

    def close(self) -> None:
        session, self.session = getattr(self, "session", None), None
        if session is not None:
            try:
                session.free_resource()
            except Exception:  # noqa: BLE001  shutdown path: nothing to recover
                pass
