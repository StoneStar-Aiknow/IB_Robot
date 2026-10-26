"""Per-frame debug record of the hand pipeline, for comparing the two hands offline. **No ROS in here.**

``dump_dir`` set: every processed frame becomes one JSON line in
``dump_dir/<start time>-<pid>/frames.jsonl`` with what the HandFrame drops:
the crop rectangle, every hand the pipeline found (keypoints with z in crop
pixels, world keypoints, the landmark model's raw handedness and presence,
ROI, the classifier's label / score, the keypoint rule) and which side each
one was assigned to. ``crops`` also writes a small JPEG around each found hand
(``crops/<frame_id>_<k>.jpg``, every ``crop_every``-th frame); those can show a
face, so they stay on the board and are deleted after the analysis.

A writer thread does the disk work; the queue is bounded and a full queue drops
the record (counted in ``dropped``), so the NPU thread never waits on the disk.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np

CROP_MAX_PX = 160
CROP_MARGIN = 0.4


def _r(a, nd=4):
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def hand_crop(img: np.ndarray, lm_px: np.ndarray):
    """Small BGR crop around the hand's keypoints (source pixels), its origin and scale; None if empty."""
    h, w = img.shape[:2]
    lo, hi = np.nanmin(lm_px, axis=0), np.nanmax(lm_px, axis=0)
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        return None
    side = float(max(hi - lo)) * (1.0 + 2.0 * CROP_MARGIN)
    c = 0.5 * (lo + hi)
    x0, y0 = int(max(0, c[0] - side / 2)), int(max(0, c[1] - side / 2))
    x1, y1 = int(min(w, c[0] + side / 2)), int(min(h, c[1] + side / 2))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    crop = np.ascontiguousarray(img[y0:y1, x0:x1])
    return crop, (x0, y0), min(1.0, CROP_MAX_PX / max(x1 - x0, y1 - y0))


class FrameDump:
    def __init__(self, root, crops: bool = False, crop_every: int = 2, max_queue: int = 256):
        self.dir = Path(root) / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
        self.crop_dir = self.dir / "crops"
        self.crop_dir.mkdir(parents=True, exist_ok=False)
        self.crops = bool(crops)
        self.crop_every = max(1, int(crop_every))
        self.dropped = 0
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._f = open(self.dir / "frames.jsonl", "a", buffering=1 << 16)
        self._t = threading.Thread(target=self._write, name="hand-dump", daemon=True)
        self._t.start()

    def add(self, **kw) -> None:
        try:
            self._add(**kw)
        except Exception:  # noqa: BLE001  debug output must never stop the stream
            self.dropped += 1

    def _add(self, *, stamp_ns: int, frame_id: int, valid: bool, box, crop_rect, found, pick, img) -> None:
        rec = {"t": time.time(), "stamp_ns": int(stamp_ns), "frame_id": int(frame_id), "valid": bool(valid)}
        crops = []
        if valid and found is not None:
            side_of = {k: s for s, k in (pick or {}).items() if k is not None}
            rec["box"] = _r(box, 1)
            rec["crop_rect"] = [int(v) for v in crop_rect]
            hands = []
            for k, f in enumerate(found):
                hands.append(
                    {
                        "side": side_of.get(k),
                        "lm": _r(f["lm"], 2),  # source pixels
                        "lm3": _r(f["lm3"], 3),  # crop pixels, z = network z * roi / 224
                        "wlm": _r(f["wlm"], 5),
                        "handed": round(float(f["handed"]), 4),
                        "presence": round(float(f["presence"]), 4),
                        "roi": _r(f["roi"], 3),
                        "gesture": f["gesture"],
                        "gesture_score": round(float(f["gesture_score"]), 4),
                        "rule": f["rule"],
                    }
                )
                if self.crops and img is not None and frame_id % self.crop_every == 0:
                    c = hand_crop(img, np.asarray(f["lm"], dtype=np.float64))
                    if c is not None:
                        crops.append((f"{frame_id}_{k}.jpg", *c))
                        hands[-1]["crop"] = {"file": crops[-1][0], "origin": list(c[1]), "scale": round(c[2], 4)}
            rec["hands"] = hands
        try:
            self._q.put_nowait((rec, crops))
        except queue.Full:
            self.dropped += 1

    def _write(self):
        import cv2

        while True:
            item = self._q.get()
            if item is None:
                break
            rec, crops = item
            try:
                for name, crop, _, scale in crops:
                    if scale < 1.0:
                        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(self.crop_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                rec["dump_dropped"] = self.dropped
                self._f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            except Exception:  # noqa: BLE001  debug output must never stop the stream
                self.dropped += 1

    def close(self) -> None:
        self._q.put(None)
        self._t.join(timeout=3.0)
        self._f.close()
