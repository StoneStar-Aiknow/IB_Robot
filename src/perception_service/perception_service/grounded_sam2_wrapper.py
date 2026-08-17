"""Wrapper for Grounding-DINO + SAM2 inference pipeline."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from .grounding_dino_config import GROUNDING_DINO_SWINT_OGC_CONFIG
from .model_utils import DEFAULT_MODEL_DIR, resolve_model_path, resolve_text_encoder

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("perception_service requires NumPy and OpenCV; run `./scripts/setup.sh`.") from exc

try:
    import torch
    from torchvision.ops import box_convert
except ModuleNotFoundError:
    torch = None
    box_convert = None

logger = logging.getLogger(__name__)


# Keys the official upstream SwinT-OGC release checkpoint carries that the
# inference model does not own. Calibrated against the real checkpoint
# (groundingdino_swint_ogc.pth loaded via load_state_dict(strict=False) with
# the audited wheel + GROUNDING_DINO_SWINT_OGC_CONFIG):
#   - label_enc.weight: training-time denoising (DN) label embedding; the
#     inference model has no label_enc submodule (dn_number=0), so this key
#     always appears in unexpected_keys when loading the official checkpoint.
#   - bert.embeddings.position_ids: legacy buffer removed in newer transformers.
#
# No missing keys are expected with the official checkpoint (measured empty);
# keep _ALLOWED_MISSING_KEY_PREFIXES strict on purpose.
_ALLOWED_UNEXPECTED_KEYS = frozenset({"label_enc.weight"})
_ALLOWED_UNEXPECTED_KEY_PREFIXES = ("bert.",)
_ALLOWED_MISSING_KEY_PREFIXES = ()


def _verify_grounding_dino_checkpoint_compatible(incompatible) -> None:
    """Reject Grounding-DINO checkpoint keys that fall outside the allow-list.

    ``strict=False`` is required to load the upstream Grounding-DINO SwintOGC
    checkpoint because the official release carries training-time weights
    (e.g. ``label_enc.weight``) that the inference model does not own, and
    newer transformers removes legacy buffers (e.g. ``position_ids``).
    Silently dropping the returned ``missing_keys`` / ``unexpected_keys`` would
    let a stale, mismatched, or wrong-architecture checkpoint boot into a model
    with random weights and surface as an unexplained accuracy regression at
    inference time. Audit the known-compatible keys here and fail fast on any
    other structural drift.
    """

    def _filter_missing(keys, prefixes):
        return [key for key in keys if not key.startswith(prefixes)]

    def _filter_unexpected(keys, exact, prefixes):
        return [key for key in keys if key not in exact and not key.startswith(prefixes)]

    missing = _filter_missing(getattr(incompatible, "missing_keys", []) or [], _ALLOWED_MISSING_KEY_PREFIXES)
    unexpected = _filter_unexpected(
        getattr(incompatible, "unexpected_keys", []) or [],
        _ALLOWED_UNEXPECTED_KEYS,
        _ALLOWED_UNEXPECTED_KEY_PREFIXES,
    )
    if missing or unexpected:
        raise RuntimeError(
            "Grounding-DINO checkpoint is incompatible with the configured model: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _enable_grounding_dino_autocast() -> None:
    """Enable the CUDA precision supported by Grounding-DINO's custom ops."""

    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()


def volume_centroid_hull(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Volume centroid of the convex hull enclosing ``points``.

    Decomposes the hull into tetrahedra (one per triangular face joined to an
    interior anchor) and integrates the volume-weighted centroid using the
    divergence theorem. Returns ``(centroid_xyz, volume_m3)``. Falls back to the
    surface mean when the points are degenerate (fewer than 4, coplanar, or
    ``scipy`` unavailable).
    """
    if points.shape[0] < 4:
        return points.mean(axis=0) if points.shape[0] > 0 else np.zeros(3), 0.0
    try:
        from scipy.spatial import ConvexHull
    except ModuleNotFoundError:
        return points.mean(axis=0), 0.0

    try:
        hull = ConvexHull(points)
    except Exception:  # QhullError on coplanar/degenerate input
        return points.mean(axis=0), 0.0

    anchor = points[hull.vertices].mean(axis=0)
    total_v = 0.0
    total_vc = np.zeros(3)
    for simplex in hull.simplices:
        a, b, c = points[simplex] - anchor
        # Use the absolute geometric volume of each tetrahedron as weight.
        # scipy's hull simplices do NOT guarantee consistent outward-facing
        # normals, so the signed volume can flip sign per face; using abs
        # yields the correct non-overlapping tetrahedron partition for an
        # interior anchor and avoids the centroid drifting outside the hull.
        v = abs(float(np.dot(a, np.cross(b, c)) / 6.0))
        tet_centroid = (anchor + points[simplex].sum(axis=0)) / 4.0
        total_v += v
        total_vc += v * tet_centroid

    if total_v < 1e-12:
        return points.mean(axis=0), 0.0
    return total_vc / total_v, total_v


@dataclass
class Detection:
    label: str
    confidence: float
    bbox_xyxy: np.ndarray
    mask: np.ndarray
    centroid_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Volume centroid of the convex hull enclosing the object's 3D points.
    # Unlike centroid_xyz (visible-surface mean), this approximates the
    # filled-volume center by integrating the convex hull, which is closer to
    # the physical center of mass for convex objects.
    volume_centroid_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    volume_m3: float = 0.0
    point_count: int = 0


class GroundedSAM2Wrapper:
    def __init__(
        self,
        device: str = "cuda",
        sam_checkpoint: str = "grounded_sam2_swint_ogc/assets/sam2.1_hiera_tiny.pt",
        sam_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
        gdino_checkpoint: str = "grounded_sam2_swint_ogc/assets/groundingdino_swint_ogc.pth",
        gdino_text_encoder: str = "grounded_sam2_swint_ogc/assets/bert-base-uncased",
        model_dir: str | None = None,
    ):
        if torch is None or box_convert is None:
            raise ModuleNotFoundError(
                "The Torch Grounding-DINO/SAM2 backend dependencies are missing. Install them with "
                "`./scripts/setup.sh` or select `inference_backend:=ascend_local`."
            )
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = device

        base = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR

        sam_ckpt = resolve_model_path(sam_checkpoint, base)
        sam_cfg = sam_config
        gdino_ckpt = resolve_model_path(gdino_checkpoint, base)
        text_encoder_type = resolve_text_encoder(gdino_text_encoder, base)

        for p, label in [
            (sam_ckpt, "SAM2 checkpoint"),
            (gdino_ckpt, "GDINO checkpoint"),
        ]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"{label} not found: {p}\nRun: ./scripts/download_perception_models.sh")

        text_encoder_path = Path(text_encoder_type)
        if text_encoder_path.is_absolute() and not text_encoder_path.exists():
            raise FileNotFoundError(
                f"GDINO text encoder not found: {text_encoder_path}\nRun: ./scripts/download_perception_models.sh"
            )

        logger.info("Loading SAM2 config %s with checkpoint %s", sam_cfg, sam_ckpt)
        sam2_model = build_sam2(sam_cfg, str(sam_ckpt), device=self.device)
        self.sam_predictor = SAM2ImagePredictor(sam2_model)

        logger.info("Loading Grounding-DINO from %s", gdino_ckpt)
        self.grounding_model = self._load_grounding_model(gdino_ckpt, text_encoder_type, self.device)

        if self.device == "cuda":
            _enable_grounding_dino_autocast()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        logger.info("GroundedSAM2Wrapper ready on %s", self.device)

    @staticmethod
    def _load_grounding_model(
        checkpoint_path: Path,
        text_encoder_type: str,
        device: str,
    ):
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict

        args = SLConfig(GROUNDING_DINO_SWINT_OGC_CONFIG.copy())
        args.device = device
        args.text_encoder_type = text_encoder_type

        try:
            model = build_model(args)
        except OSError as exc:
            raise RuntimeError(
                "Grounding-DINO text encoder assets could not be loaded. Run: ./scripts/download_perception_models.sh"
            ) from exc

        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        incompatible = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        _verify_grounding_dino_checkpoint_compatible(incompatible)
        model.eval()
        return model

    def detect_and_segment(
        self,
        image_bgr: np.ndarray,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> list[Detection]:
        from groundingdino.util.inference import predict as gdino_predict

        if not text_prompt.endswith("."):
            text_prompt = text_prompt + "."

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        image_tensor = self._preprocess_gdino_image(image_rgb)

        self.sam_predictor.set_image(image_rgb)

        boxes, confidences, labels = gdino_predict(
            model=self.grounding_model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )

        if len(boxes) == 0:
            return []

        boxes_scaled = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        masks, scores, _ = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        if masks.ndim == 4:
            masks = masks[:, 0, :, :]

        detections = []
        for i in range(len(labels)):
            det = Detection(
                label=labels[i],
                confidence=float(confidences[i]),
                bbox_xyxy=input_boxes[i].astype(np.float32),
                mask=masks[i].astype(np.uint8),
            )
            detections.append(det)

        return detections

    @staticmethod
    def _preprocess_gdino_image(image_rgb: np.ndarray) -> torch.Tensor:
        from groundingdino.datasets import transforms as gdino_transforms
        from PIL import Image

        transform = gdino_transforms.Compose(
            [
                gdino_transforms.RandomResize([800], max_size=1333),
                gdino_transforms.ToTensor(),
                gdino_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_pil = Image.fromarray(image_rgb)
        image_tensor, _ = transform(image_pil, None)
        return image_tensor

    @staticmethod
    def compute_3d_centroids(
        detections: list[Detection],
        depth_image: np.ndarray,
        camera_intrinsics: np.ndarray,
        depth_scale: float = 1000.0,
        depth_trunc: float = 3.0,
    ) -> list[Detection]:
        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[1, 1]
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[1, 2]

        for det in detections:
            mask = det.mask > 0
            if mask.sum() == 0:
                det.centroid_xyz = np.array([0.0, 0.0, 0.0])
                det.point_count = 0
                continue

            depth_m = depth_image.astype(np.float64) / depth_scale
            depth_m[depth_m > depth_trunc] = 0
            if mask.shape != depth_image.shape[:2]:
                mask = (
                    cv2.resize(
                        mask.astype(np.uint8),
                        (depth_image.shape[1], depth_image.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    > 0
                )

            depth_masked = depth_m * mask

            valid = depth_masked > 0
            det.point_count = int(valid.sum())

            if det.point_count == 0:
                det.centroid_xyz = np.array([0.0, 0.0, 0.0])
                continue

            ys, xs = np.where(valid)
            zs = depth_masked[valid]

            # Reject depth outliers before centroiding. At close range the mask may
            # include gripper fingers or background pixels whose depth differs from
            # the target by hundreds of mm; keeping them corrupts the centroid.
            median_z = float(np.median(zs))
            mad_z = float(np.median(np.abs(zs - median_z)))
            # 1.4826*MAD approximates a std; clamp with an absolute floor so a very
            # tight cluster does not reject every point due to sensor quantization.
            z_tol = max(3.0 * 1.4826 * mad_z, 0.02)
            inliers = np.abs(zs - median_z) <= z_tol
            if inliers.sum() >= max(10, int(0.2 * zs.size)):
                xs, ys, zs = xs[inliers], ys[inliers], zs[inliers]

            det.point_count = int(zs.size)

            # Back-project each pixel to 3D first, THEN average. Averaging pixel
            # coordinates and depth independently (mean_x * mean_z) is only valid
            # when depth is uniform across the mask and diverges badly otherwise.
            xs3d = (xs - cx) * zs / fx
            ys3d = (ys - cy) * zs / fy

            x3d = float(np.mean(xs3d))
            y3d = float(np.mean(ys3d))
            z3d = float(np.mean(zs))

            det.centroid_xyz = np.array([x3d, y3d, z3d], dtype=np.float64)

            # Convex-hull volume centroid: integrates the filled volume enclosing
            # the object's 3D points. Closer to the physical center of mass than
            # the visible-surface mean above, especially for convex objects.
            pts3d = np.stack([xs3d, ys3d, zs], axis=1)
            vol_c, vol = volume_centroid_hull(pts3d)
            det.volume_centroid_xyz = vol_c
            det.volume_m3 = vol

        return detections
