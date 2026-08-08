"""Canonical GraspGen point-cloud preprocessing and grasp decoding."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from inference_manifest import (
    GRASPGEN_CONFIDENCE_SEMANTIC,
    GRASPGEN_NPOINTS,
    GRASPGEN_NSAMPLES,
    GRASPGEN_POINT_CLOUD_SEMANTIC,
    GRASPGEN_POSE_SEMANTIC,
    GRASPGEN_RADII,
)
from inference_service.generic_runtime import NamedTensorResult

from .graspgen_geometry import positive_float, positive_int, remove_point_cloud_outliers
from .perception_adapter import AdapterIdentity, PerceptionAdapter

GRASPGEN_PREPROCESSING = "object-points-outlier-filtered-centered-kappa-scaled-v1"
GRASPGEN_POSTPROCESSING = "grasp-pose-matrix-confidence-sorted-v1"


@dataclass(frozen=True)
class GraspCandidatePose:
    """One decoded grasp: a 4x4 camera-frame transform and its discriminator score."""

    pose_matrix: np.ndarray
    confidence: float


@dataclass(frozen=True)
class GraspGenConfig:
    """The model-specific constants the session and the packager both read.

    These used to live in a fabricated LeRobot ``config.json``. GraspGen is not a
    LeRobot policy, so they live in ``assets/adapter.json`` next to the identity the
    adapter already validates.
    """

    kappa: float
    diffusion_steps: int
    grasp_batch_size: int
    point_count: int
    npoints: tuple[int, int]
    radii: tuple[float, float]
    nsamples: tuple[int, int]

    @classmethod
    def from_assets(cls, assets: Mapping[str, object]) -> GraspGenConfig:
        geometry = assets.get("geometry") or {}
        if not isinstance(geometry, Mapping):
            raise ValueError("GraspGen adapter.json geometry must be an object")
        npoints_raw = list(geometry.get("npoints") or GRASPGEN_NPOINTS)
        radii_raw = list(geometry.get("radii") or GRASPGEN_RADII)
        nsamples_raw = list(geometry.get("nsamples") or GRASPGEN_NSAMPLES)
        if len(npoints_raw) < 2 or len(radii_raw) < 2 or len(nsamples_raw) < 2:
            raise ValueError("GraspGen geometry must contain two sampled stages")
        return cls(
            kappa=positive_float(assets.get("kappa", 2.02217), "kappa"),
            diffusion_steps=positive_int(assets.get("diffusion_steps", 10), "diffusion_steps"),
            grasp_batch_size=positive_int(assets.get("grasp_batch_size", 1000), "grasp_batch_size"),
            point_count=positive_int(assets.get("point_count", 2048), "point_count"),
            npoints=tuple(
                positive_int(value, f"geometry.npoints[{index}]") for index, value in enumerate(npoints_raw[:2])
            ),
            radii=tuple(positive_float(value, f"geometry.radii[{index}]") for index, value in enumerate(radii_raw[:2])),
            nsamples=tuple(
                positive_int(value, f"geometry.nsamples[{index}]") for index, value in enumerate(nsamples_raw[:2])
            ),
        )


class GraspGenAdapter(PerceptionAdapter):
    identity = AdapterIdentity(
        family="graspgen",
        preprocessing=GRASPGEN_PREPROCESSING,
        postprocessing=GRASPGEN_POSTPROCESSING,
        supported_deployments=frozenset({"ascend_310p", "ascend_310b"}),
    )

    compiled_abi_finalized = True

    def __init__(self, config: GraspGenConfig) -> None:
        self.config = config

    @classmethod
    def from_bundle(cls, bundle_root: str | Path, _identity=None, *, model=None, deployment=None) -> GraspGenAdapter:
        del deployment
        root = Path(bundle_root)
        assets = json.loads((root / "assets" / "adapter.json").read_text(encoding="utf-8"))
        expected = {
            "family": cls.identity.family,
            "preprocessing": cls.identity.preprocessing,
            "postprocessing": cls.identity.postprocessing,
        }
        if any(assets.get(name) != value for name, value in expected.items()):
            raise ValueError(f"GraspGen adapter identity mismatch: expected {expected}, got {assets}")
        return cls(GraspGenConfig.from_assets(assets))

    def preprocess(self, points_xyz: np.ndarray) -> dict[str, np.ndarray]:
        return self.prepare(points_xyz)[0]

    def prepare(self, points_xyz: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Clean, cap and centre an object point cloud into the model's own frame.

        The centroid is returned alongside the tensor because the network only ever sees
        the centred cloud and ``postprocess`` has to put the decoded grasps back into the
        camera frame. Handing it to the caller rather than storing it on the adapter keeps
        one adapter instance usable for concurrent requests.
        """
        points = np.asarray(points_xyz, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"GraspGen object points must have shape [N, 3], got {points.shape}")
        finite = points[np.isfinite(points).all(axis=1)]
        if len(finite) == 0:
            raise ValueError("GraspGen object point cloud contains no finite points")
        kept, _ = remove_point_cloud_outliers(finite)
        if len(kept) == 0:
            kept = finite
        if len(kept) > self.config.point_count:
            # A fixed seed keeps one point cloud mapping to one grasp set across runs;
            # the downsample is a capacity limit, not a source of intended randomness.
            rng = np.random.default_rng(0)
            kept = kept[np.sort(rng.choice(len(kept), self.config.point_count, replace=False))]
        center = kept.mean(axis=0, dtype=np.float32)
        scaled = np.ascontiguousarray((kept - center) * np.float32(self.config.kappa), dtype=np.float32)
        return {GRASPGEN_POINT_CLOUD_SEMANTIC: scaled}, center

    def postprocess(
        self,
        result: NamedTensorResult,
        *,
        object_center: np.ndarray | None = None,
        max_grasps: int = 0,
        min_confidence: float = 0.0,
    ) -> list[GraspCandidatePose]:
        poses = self._required(result, GRASPGEN_POSE_SEMANTIC)
        confidence = self._required(result, GRASPGEN_CONFIDENCE_SEMANTIC).reshape(-1)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise RuntimeError(f"GraspGen poses must have shape [N, 4, 4], got {poses.shape}")
        if confidence.shape[0] != poses.shape[0]:
            raise RuntimeError(f"GraspGen returned {poses.shape[0]} poses but {confidence.shape[0]} confidences")
        if not np.isfinite(poses).all() or not np.isfinite(confidence).all():
            raise RuntimeError("GraspGen returned non-finite grasp poses or confidences")
        if object_center is not None:
            poses = poses.copy()
            poses[:, :3, 3] += np.asarray(object_center, dtype=np.float32).reshape(3)
        order = np.argsort(-confidence, kind="stable")
        candidates = [
            GraspCandidatePose(np.ascontiguousarray(poses[index], dtype=np.float32), float(confidence[index]))
            for index in order
            if float(confidence[index]) >= min_confidence
        ]
        return candidates[:max_grasps] if max_grasps > 0 else candidates

    @staticmethod
    def _required(result: NamedTensorResult, semantic: str) -> np.ndarray:
        try:
            return np.asarray(result.outputs[semantic], dtype=np.float32)
        except KeyError as exc:
            raise RuntimeError(f"GraspGen runtime result is missing {semantic!r}") from exc
