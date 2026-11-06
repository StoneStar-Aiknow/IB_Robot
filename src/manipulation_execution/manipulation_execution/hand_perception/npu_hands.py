"""Hand keypoints and gesture labels on the 310P NPU. **No ROS in here.**

The hand pipeline module (``mp_hand_pipeline.py``) and the keypoint helpers
(``hand_keypoints.py``) ship with this package, in ``x2/_vendor/``. A
non-empty ``extern_dir`` loads them from that directory instead (a field
override). ``load_extern`` is the only door. The ``.om`` models are not in git:
``model_dir`` holds them (see ``model_paths``).

Two backends run them. ``ais_bench`` (the default) is the code from before and
takes a bundle or a flat directory, so configurations that do not name a backend
run as they did. ``inference_service`` is that package's ACL session in this
process, roles ``det`` / ``lm`` / ``embedder`` of the bundle's manifest; it needs
a bundle.

    hands = NpuHands(model_dir)
    hands.reset()                               # drop the tracked hands
    out = hands.infer(bgr_crop, person_box_in_crop)
    # -> list[dict]: lm (21,2) crop pixels, wlm (21,3) metres,
    #    handed_score, gesture / gesture_score, rule

Gesture label: ``gesture_embedder.om`` followed by the 8-way classifier head
(BN -> ReLU -> dense -> softmax) in numpy, weights from
``canned_gesture_classifier.npz``. The embedder gets the pipeline's keypoints
in crop pixels; handedness and world_hand are fed zeros (the model ignores
them). The embedder only reads right hands: a left hand (landmark model's
handedness < 0.5) is mirrored in x first (``as_right_hand``). Without it a
person's left hand gets the right label at 0.5-0.6, under the 0.7 entry gate
(10-24 run, replayed: Victory 0.01 -> 0.93 of frames over the gate).

Every call must come from the thread that built the object (ACL contexts are
per thread).
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

BACKENDS = ("inference_service", "ais_bench")
DEFAULT_BACKEND = "ais_bench"  # configurations without ``backend`` (the flat model_dir ones) keep working
DEPLOYMENT = "ascend_310p"
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
DEFAULT_EXTERN_DIR = ""  # empty: the copies in this package
DEFAULT_MODEL_DIR = "models/hand_mediapipe_310p"
BUNDLE_ARTIFACTS = "artifacts/ascend_310p"
EXTERN_FILES = (PIPELINE_PY, KEYPOINTS_PY)
_VENDOR_DIR = Path(__file__).parent.parent / "imitation_retargeting" / "x2" / "_vendor"
_BUNDLED = {
    name: _VENDOR_DIR / name for name in (PIPELINE_PY, KEYPOINTS_PY, HAND_CORE_PY, CALIB_MEDIAPIPE_JSON)
}


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extern_path(extern_dir, filename: str) -> Path:
    """``extern_dir/filename``; empty ``extern_dir``: the copy in this package."""
    return Path(extern_dir) / filename if extern_dir else _BUNDLED[filename]


def load_extern(extern_dir, filename: str):
    """Import the hand module (``extern_path``) as a private module (not on sys.path)."""
    path = extern_path(extern_dir, filename)
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing (extern_dir={str(extern_dir)!r})")
    name = f"_hand_extern_{path.stem}"
    loaded = sys.modules.get(name)
    if loaded is not None and Path(loaded.__file__) == path:
        return loaded
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_model_dir(model_dir) -> Path:
    """A relative ``model_dir`` is under ``$WORKSPACE``, as ``bundle_path`` is (``perception_runtime_config``)."""
    d = Path(model_dir).expanduser()
    if d.is_absolute():
        return d
    workspace = os.environ.get("WORKSPACE")
    if not workspace:
        raise ValueError(f"model_dir {model_dir!r} is relative but WORKSPACE is unset")
    return Path(workspace) / d


def model_paths(model_dir) -> dict[str, Path]:
    """Model file name -> path: in the bundle's artifact directory, or directly in ``model_dir``."""
    d = resolve_model_dir(model_dir)
    if (d / BUNDLE_ARTIFACTS).is_dir():
        d = d / BUNDLE_ARTIFACTS
    return {name: d / name for name in MODEL_FILES}


def relabel(label: str, score: float, rule: str) -> tuple[str, float]:
    """``rule_labels``: two keypoint rules win over the classifier, ILoveYou reads as rock."""
    if rule == "thumb_index_middle":
        return "gun", max(float(score), 1.0)
    if rule == "six":
        return "six", max(float(score), 1.0)
    if label == "ILoveYou":
        return "rock", float(score)
    return label, float(score)


def as_right_hand(lm: np.ndarray, handed: float) -> np.ndarray:
    """Keypoints the embedder can read: a left hand (handed < 0.5) mirrored in x about its own mean."""
    if handed >= 0.5:
        return lm
    out = np.array(lm, dtype=np.float64, copy=True)
    out[:, 0] = 2.0 * out[:, 0].mean() - out[:, 0]
    return out


class ServiceBackend:
    """``HandPipeline``'s backend (``infer("det" | "lm", x)``) and the embedder, on inference_service's ACL session."""

    def __init__(self, model_dir, device: int):
        from inference_manifest import load_inference_manifest
        from inference_service.backends.ascend.acl_runtime import AclRuntimeManager
        from inference_service.backends.types import RuntimeContext
        from inference_service.model_sessions.ascend import AscendOmModelSession
        from inference_service.unified_runtime import ExecutionContext, ModelRequest

        validated = load_inference_manifest(resolve_model_dir(model_dir), DEPLOYMENT)
        self._session = AscendOmModelSession(device, runtime_manager=AclRuntimeManager())
        self._session.load(RuntimeContext(validated_manifest=validated, runtime_profile=validated.runtime_profile))
        self._request, self._context = ModelRequest(), ExecutionContext("hand_perception")
        self._bindings = validated.deployment.bindings

    def _run(self, role: str, inputs: dict) -> list[np.ndarray]:
        out = self._session.execute_role(role, inputs, self._request, self._context)
        return [np.asarray(out[b.semantic]) for b in self._bindings[role].outputs]

    def infer(self, name: str, x: np.ndarray):
        return tuple(self._run(name, {self._bindings[name].inputs[0].semantic: np.ascontiguousarray(x)}))

    def embed(self, feed: dict[str, np.ndarray]) -> np.ndarray:
        inputs = {b.semantic: feed[b.semantic.rpartition(".")[2]] for b in self._bindings["embedder"].inputs}
        return self._run("embedder", inputs)[0]

    def close(self) -> None:
        self._session.close()


class AisBenchBackend:
    """The pipeline's own ``AclBackend`` and an ``InferSession`` for the embedder."""

    def __init__(self, pipeline, paths: dict[str, Path], device: int):
        from ais_bench.infer.interface import InferSession

        self._pipeline_backend = pipeline.AclBackend(str(paths[DET_OM]), str(paths[LM_OM]), device)
        self.emb = InferSession(device, str(paths[EMB_OM]))
        self._emb_names = [i.name for i in self.emb.get_inputs()]
        if sorted(self._emb_names) != ["hand", "handedness", "world_hand"]:
            raise ValueError(f"{EMB_OM} inputs are {self._emb_names}, expected hand / handedness / world_hand")

    def infer(self, name: str, x: np.ndarray):
        return self._pipeline_backend.infer(name, x)

    def embed(self, feed: dict[str, np.ndarray]) -> np.ndarray:
        return self.emb.infer([feed[n] for n in self._emb_names])[0]

    def close(self) -> None:
        for s in [*self._pipeline_backend.s.values(), self.emb]:
            with contextlib.suppress(Exception):
                s.free_resource()


class NpuHands:
    def __init__(
        self,
        model_dir,
        extern_dir=DEFAULT_EXTERN_DIR,
        device: int = 0,
        rule_labels: bool = False,
        backend: str = DEFAULT_BACKEND,
    ):
        if backend not in BACKENDS:
            raise ValueError(f"backend {backend!r} is not one of {BACKENDS}")
        self.paths = model_paths(model_dir)
        missing = [str(p) for p in self.paths.values() if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"model files missing: {missing}")
        pipeline = load_extern(extern_dir, PIPELINE_PY)
        self.rule_labels = bool(rule_labels)
        self.backend_name = backend
        if backend == "inference_service":
            self.npu = ServiceBackend(model_dir, device)
        else:
            self.npu = AisBenchBackend(pipeline, self.paths, device)
        self.pipe = pipeline.HandPipeline(self.npu, track=True)
        w = np.load(self.paths[HEAD])
        self._scale, self._shift = w["bn_mul"].astype(np.float64), w["bn_add"].astype(np.float64)
        self._dense, self._bias = w["W"].astype(np.float64), w["b"].astype(np.float64)
        if self._dense.shape != (128, len(LABELS)):
            raise ValueError(f"{HEAD} W is {self._dense.shape}, expected (128, {len(LABELS)})")
        self._zeros = {"handedness": np.zeros((1, 1), np.float32), "world_hand": np.zeros((1, 21, 3), np.float32)}

    @property
    def description(self) -> str:
        return f"NPU hand pipeline + {EMB_OM} + numpy head ({self.backend_name}), rule_labels={self.rule_labels}"

    def reset(self) -> None:
        self.pipe.reset()

    def _classify(self, lm_px: np.ndarray) -> tuple[str, float]:
        feed = {"hand": np.ascontiguousarray(lm_px[None], dtype=np.float32), **self._zeros}
        emb = np.asarray(self.npu.embed(feed), dtype=np.float64).reshape(-1)
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
            handed = float(h["handed"])
            label, score = self._classify(as_right_hand(lm, handed))
            rule = str(h.get("gesture") or "")
            if self.rule_labels:
                label, score = relabel(label, score, rule)
            out.append(
                {
                    "lm": lm[:, :2].copy(),
                    "wlm": np.asarray(h["wlm"], dtype=np.float64),
                    "handed_score": max(handed, 1.0 - handed),
                    "gesture": label,
                    "gesture_score": score,
                    "rule": rule,
                    # raw pipeline values, for the debug record (dump.py)
                    "lm3": lm.copy(),
                    "handed": handed,
                    "presence": float(h.get("presence", np.nan)),
                    "roi": np.asarray(h.get("roi", ()), dtype=np.float64),
                }
            )
        return out

    def close(self) -> None:
        self.npu.close()
