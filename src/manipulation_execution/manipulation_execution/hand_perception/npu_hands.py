"""Hand keypoints and gesture labels on the 310P NPU. **No ROS in here.**

Nothing third-party lives in this package. The hand pipeline module
(``mp_hand_pipeline.py``) and the keypoint helpers (``hand_keypoints.py``) are
loaded at run time from ``extern_dir``, next to the model files, the same way
the ``.om`` models and ``ais_bench`` are. ``load_extern`` is the only door.

    hands = NpuHands(model_dir, extern_dir)
    hands.reset()                               # drop the tracked hands
    out = hands.infer(bgr_crop, person_box_in_crop)
    # -> list[dict]: lm (21,2) crop pixels, wlm (21,3) metres,
    #    handed_score, gesture / gesture_score, rule

Gesture label: ``gesture_embedder.om`` followed by the 8-way classifier head
(BN -> ReLU -> dense -> softmax) in numpy, weights from
``canned_gesture_classifier.npz``. The embedder gets the pipeline's keypoints
in crop pixels; handedness and world_hand are fed zeros.

Every call must come from the thread that built the object (ACL contexts are
per thread).
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

LABELS = ("None", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou")
DET_OM = "hd_npu.om"
LM_OM = "hand_landmarks_detector.om"
EMB_OM = "gesture_embedder.om"
HEAD = "canned_gesture_classifier.npz"
MODEL_FILES = (DET_OM, LM_OM, EMB_OM, HEAD)
PIPELINE_PY = "mp_hand_pipeline.py"
KEYPOINTS_PY = "hand_keypoints.py"
HAND_CORE_PY = "hand_core.py"  # loaded by x2.hand_follow
CALIB_MEDIAPIPE_JSON = "data/dexhand_calib_mediapipe.json"
DEFAULT_EXTERN_DIR = "/root/x2-dexhand-run/extern"
EXTERN_FILES = (PIPELINE_PY, KEYPOINTS_PY)


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_extern(extern_dir, filename: str):
    """Import ``extern_dir/filename`` as a private module (not on sys.path)."""
    path = Path(extern_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing (hand modules are deployed beside the models, not in git)")
    name = f"_hand_extern_{path.stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def relabel(label: str, score: float, rule: str) -> tuple[str, float]:
    """``rule_labels``: two keypoint rules win over the classifier, ILoveYou reads as rock."""
    if rule == "thumb_index_middle":
        return "gun", max(float(score), 1.0)
    if rule == "six":
        return "six", max(float(score), 1.0)
    if label == "ILoveYou":
        return "rock", float(score)
    return label, float(score)


class NpuHands:
    def __init__(self, model_dir, extern_dir, device: int = 0, rule_labels: bool = False):
        from ais_bench.infer.interface import InferSession

        d = Path(model_dir)
        self.paths = {name: d / name for name in MODEL_FILES}
        missing = [str(p) for p in self.paths.values() if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"model files missing: {missing}")
        pipeline = load_extern(extern_dir, PIPELINE_PY)
        self.rule_labels = bool(rule_labels)
        backend = pipeline.AclBackend(str(self.paths[DET_OM]), str(self.paths[LM_OM]), device)
        self.pipe = pipeline.HandPipeline(backend, track=True)
        self.emb = InferSession(device, str(self.paths[EMB_OM]))
        self._emb_names = [i.name for i in self.emb.get_inputs()]
        if sorted(self._emb_names) != ["hand", "handedness", "world_hand"]:
            raise ValueError(f"{EMB_OM} inputs are {self._emb_names}, expected hand / handedness / world_hand")
        w = np.load(self.paths[HEAD])
        self._scale, self._shift = w["bn_mul"].astype(np.float64), w["bn_add"].astype(np.float64)
        self._dense, self._bias = w["W"].astype(np.float64), w["b"].astype(np.float64)
        if self._dense.shape != (128, len(LABELS)):
            raise ValueError(f"{HEAD} W is {self._dense.shape}, expected (128, {len(LABELS)})")
        self._zeros = {"handedness": np.zeros((1, 1), np.float32), "world_hand": np.zeros((1, 21, 3), np.float32)}

    @property
    def description(self) -> str:
        return f"NPU hand pipeline + {EMB_OM} + numpy head, rule_labels={self.rule_labels}"

    def reset(self) -> None:
        self.pipe.reset()

    def _classify(self, lm_px: np.ndarray) -> tuple[str, float]:
        feed = {"hand": np.ascontiguousarray(lm_px[None], dtype=np.float32), **self._zeros}
        emb = np.asarray(self.emb.infer([feed[n] for n in self._emb_names])[0], dtype=np.float64).reshape(-1)
        logits = np.maximum(emb * self._scale + self._shift, 0.0) @ self._dense + self._bias
        p = np.exp(logits - logits.max())
        p /= p.sum()
        k = int(np.argmax(p))
        return LABELS[k], float(p[k])

    def infer(self, bgr: np.ndarray, person_box) -> list[dict]:
        hands, _ = self.pipe(bgr, person_box)
        out = []
        for h in hands:
            lm = np.asarray(h["lm"], dtype=np.float64)  # (21,3) crop pixels
            label, score = self._classify(lm)
            rule = str(h.get("gesture") or "")
            if self.rule_labels:
                label, score = relabel(label, score, rule)
            handed = float(h["handed"])
            out.append(
                {
                    "lm": lm[:, :2].copy(),
                    "wlm": np.asarray(h["wlm"], dtype=np.float64),
                    "handed_score": max(handed, 1.0 - handed),
                    "gesture": label,
                    "gesture_score": score,
                    "rule": rule,
                }
            )
        return out

    def close(self) -> None:
        for s in [*getattr(self.pipe.be, "s", {}).values(), self.emb]:
            free = getattr(s, "free_resource", None)
            if free is not None:
                with contextlib.suppress(Exception):
                    free()
