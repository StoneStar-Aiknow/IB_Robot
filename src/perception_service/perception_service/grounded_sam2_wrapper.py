"""Wrapper for Grounding-DINO + SAM2 inference pipeline."""

import inspect
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import cv2
    import numpy as np
    import torch
    from torchvision.ops import box_convert
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Grounding-DINO/SAM2 perception dependencies are missing. Install them with "
        "`./scripts/setup.sh --with-perception` or "
        "`python3 -m pip install -r requirements/perception.txt`."
    ) from exc

logger = logging.getLogger(__name__)


def _find_workspace_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "src").is_dir() and (parent / ".git").exists():
            return parent
        if (parent / "models").is_dir():
            return parent
    return Path(".").resolve()


_WORKSPACE_ROOT = _find_workspace_root()
_DEFAULT_MODEL_DIR = _WORKSPACE_ROOT / "models" / "perception"
_GDINO_CONFIG_DIR = Path(__file__).resolve().parent / "config" / "gdino"


def _resolve_model_path(path_str: str, model_dir: Path) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    candidate = model_dir / p
    if candidate.exists():
        return candidate
    workspace_models = Path(os.environ.get("IB_ROBOT_WORKSPACE", ".")).resolve() / "models" / "perception" / p
    if workspace_models.exists():
        return workspace_models
    return candidate


def _resolve_text_encoder_type(text_encoder: str, model_dir: Path) -> str:
    p = Path(text_encoder).expanduser()
    if p.is_absolute():
        return str(p)

    candidate = model_dir / p
    if candidate.exists():
        return str(candidate)

    workspace_models = Path(os.environ.get("IB_ROBOT_WORKSPACE", ".")).resolve() / "models" / "perception" / p
    if workspace_models.exists():
        return str(workspace_models)

    return text_encoder


def _patch_transformers_bert_for_groundingdino() -> None:
    from transformers import BertModel

    if getattr(BertModel, "_ibrobot_groundingdino_compat", False):
        return

    sig = inspect.signature(BertModel.get_extended_attention_mask)
    params = list(sig.parameters.values())
    if len(params) >= 4 and params[3].name == "dtype":
        original_get_extended_attention_mask = BertModel.get_extended_attention_mask

        def get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
            if isinstance(device, torch.dtype):
                dtype = device
            return original_get_extended_attention_mask(self, attention_mask, input_shape, dtype=dtype)

        BertModel.get_extended_attention_mask = get_extended_attention_mask

    if not hasattr(BertModel, "get_head_mask"):

        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is None:
                return [None] * num_hidden_layers

            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            if head_mask.dim() != 5:
                raise ValueError(f"head_mask.dim != 5, instead {head_mask.dim()}")

            head_mask = head_mask.to(dtype=self.dtype)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
            return head_mask

        BertModel.get_head_mask = get_head_mask

    BertModel._ibrobot_groundingdino_compat = True


@dataclass
class Detection:
    label: str
    confidence: float
    bbox_xyxy: np.ndarray
    mask: np.ndarray
    centroid_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    point_count: int = 0


class GroundedSAM2Wrapper:
    def __init__(
        self,
        device: str = "cuda",
        sam_checkpoint: str = "sam2.1_hiera_tiny.pt",
        sam_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
        gdino_config: str = "GroundingDINO_SwinT_OGC.py",
        gdino_checkpoint: str = "groundingdino_swint_ogc.pth",
        gdino_text_encoder: str = "bert-base-uncased",
        model_dir: str | None = None,
    ):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = device if torch.cuda.is_available() else "cpu"

        base = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR

        sam_ckpt = _resolve_model_path(sam_checkpoint, base)
        sam_cfg = sam_config
        gdino_cfg = _GDINO_CONFIG_DIR / gdino_config
        gdino_ckpt = _resolve_model_path(gdino_checkpoint, base)
        text_encoder_type = _resolve_text_encoder_type(gdino_text_encoder, base)

        for p, label in [
            (sam_ckpt, "SAM2 checkpoint"),
            (gdino_cfg, "GDINO config"),
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
        self.grounding_model = self._load_grounding_model(gdino_cfg, gdino_ckpt, text_encoder_type, self.device)

        if self.device == "cuda":
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        logger.info("GroundedSAM2Wrapper ready on %s", self.device)

    @staticmethod
    def _load_grounding_model(
        config_path: Path,
        checkpoint_path: Path,
        text_encoder_type: str,
        device: str,
    ):
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict

        _patch_transformers_bert_for_groundingdino()

        args = SLConfig.fromfile(str(config_path))
        args.device = device
        args.text_encoder_type = text_encoder_type

        try:
            model = build_model(args)
        except OSError as exc:
            raise RuntimeError(
                "Grounding-DINO text encoder assets could not be loaded. Run: ./scripts/download_perception_models.sh"
            ) from exc

        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
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

            depth_masked = depth_m * mask

            valid = depth_masked > 0
            det.point_count = int(valid.sum())

            if det.point_count == 0:
                det.centroid_xyz = np.array([0.0, 0.0, 0.0])
                continue

            ys, xs = np.where(valid)
            zs = depth_masked[valid]

            mean_x = np.mean(xs)
            mean_y = np.mean(ys)
            mean_z = np.mean(zs)

            x3d = (mean_x - cx) * mean_z / fx
            y3d = (mean_y - cy) * mean_z / fy
            z3d = mean_z

            det.centroid_xyz = np.array([x3d, y3d, z3d], dtype=np.float64)

        return detections
