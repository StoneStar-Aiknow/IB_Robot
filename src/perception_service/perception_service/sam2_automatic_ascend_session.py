"""Host-orchestrated SAM2 automatic mask generation on Ascend.

The compiled encoder and decoder OMs are the same artifacts used by the
box-prompt deployment: the decoder accepts ``pointnums=2`` prompts.  For
automatic mask generation each grid point is a single foreground prompt;
the second point slot is padded with ``label=-1`` (``not_a_point``), which
the prompt encoder replaces with its learned non-point embedding.  This
keeps the compiled graph identical while letting the host drive the
decoder with one-point prompts.

The session runs the encoder once, then batches grid points through the
decoder, applies IoU/stability filtering and NMS, and returns the same
``masks``/``boxes``/``scores``/``stability_scores`` outputs as the
Torch automatic-mask pipeline.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import cv2
import numpy as np

from inference_service.backends.errors import BackendInferenceError, BackendLoadError
from inference_service.model_sessions import AscendOmModelSession
from inference_service.unified_runtime import ExecutionContext, ModelRequest

_MASK_THRESHOLD = 0.0
_STABILITY_OFFSET = 1.0
_DEFAULT_POINTS_PER_SIDE = 32
_DEFAULT_PRED_IOU_THRESH = 0.72
_DEFAULT_STABILITY_THRESH = 0.90
_BOX_NMS_IOU = 0.7
_CANVAS_SIZE = 1024
_PIXEL_OFFSET = 0.5


class SAM2AutomaticAscendSession(AscendOmModelSession):
    """Drive encoder + decoder OMs to produce automatic masks on Ascend."""

    allowed_runtime_options = AscendOmModelSession.allowed_runtime_options

    def _load(self, context, rollback) -> None:
        deployment = context.deployment
        if tuple(deployment.execution) != ("encoder", "decoder"):
            raise BackendLoadError(
                f"SAM2 automatic Ascend requires execution ['encoder', 'decoder'], got {list(deployment.execution)}",
                code="invalid_execution_plan",
            )
        if deployment.device_links:
            raise BackendLoadError(
                "SAM2 automatic Ascend must not declare device links; the host orchestrates encoder outputs",
                code="invalid_device_link",
            )
        super()._load(context, rollback)
        adapter_path = context.validated_manifest.bundle_root / "assets" / "adapter.json"
        try:
            adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BackendLoadError(f"cannot load SAM2 adapter config {adapter_path}: {exc}") from exc
        self._points_per_side = int(adapter.get("points_per_side", _DEFAULT_POINTS_PER_SIDE))
        self._pred_iou_thresh = float(adapter.get("pred_iou_thresh", _DEFAULT_PRED_IOU_THRESH))
        self._stability_thresh = float(adapter.get("stability_score_thresh", _DEFAULT_STABILITY_THRESH))
        self._decoder_batch = self._decoder_batch_size(deployment)

    def _execute(self, request: ModelRequest, context: ExecutionContext) -> Mapping[str, object]:
        context.check("backend")
        image_rgb = np.asarray(request.inputs["observation.image"], dtype=np.uint8)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise BackendInferenceError("SAM2 automatic requires an RGB uint8 HxWx3 image", code="input_shape_mismatch")
        height, width = image_rgb.shape[:2]

        canvas, scale, pad_h, pad_w = self._letterbox(image_rgb)
        encoder_inputs = {"host.sam2.image": canvas[None].astype(np.float32)}
        enc_out = self._run_role(0, "encoder", encoder_inputs)
        image_embed = np.asarray(enc_out["internal.image_embed"], dtype=np.float32)
        high_res_0 = np.asarray(enc_out["internal.high_res_feats_0"], dtype=np.float32)
        high_res_1 = np.asarray(enc_out["internal.high_res_feats_1"], dtype=np.float32)

        points = self._point_grid(self._points_per_side)
        masks_logits: list[np.ndarray] = []
        ious: list[float] = []
        for batch in self._batches(points, self._decoder_batch):
            logits, iou = self._run_decoder(image_embed, high_res_0, high_res_1, batch)
            for index in range(len(logits)):
                masks_logits.append(logits[index, 0])
                ious.append(float(iou[index, 0]))

        return self._postprocess(masks_logits, ious, height, width, scale, pad_h, pad_w)

    def _letterbox(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        height, width = image.shape[:2]
        scale = float(_CANVAS_SIZE) / max(height, width)
        new_h = max(1, int(round(height * scale)))
        new_w = max(1, int(round(width * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((_CANVAS_SIZE, _CANVAS_SIZE, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        normalized = canvas.astype(np.float32) / np.float32(255.0)
        normalized = (normalized - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        return np.ascontiguousarray(normalized.transpose(2, 0, 1)), scale, new_h, new_w

    def _point_grid(self, side: int) -> np.ndarray:
        offset = 1.0 / (2 * side)
        coords = np.linspace(offset, 1 - offset, side)
        xs, ys = np.meshgrid(coords, coords)
        return np.stack([xs.ravel(), ys.ravel()], axis=-1)

    def _batches(self, points: np.ndarray, batch: int):
        for start in range(0, len(points), batch):
            yield points[start : start + batch]

    def _run_decoder(self, embed, hr0, hr1, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        count = len(points)
        point_coords = np.zeros((count, 2, 2), dtype=np.float32)
        point_labels = np.zeros((count, 2), dtype=np.int8)
        for index, (gx, gy) in enumerate(points):
            point_coords[index, 0] = [gx * _CANVAS_SIZE, gy * _CANVAS_SIZE]
            point_labels[index, 0] = 1
            point_coords[index, 1] = [gx * _CANVAS_SIZE, gy * _CANVAS_SIZE]
            point_labels[index, 1] = -1
        values = {
            "internal.image_embed": np.broadcast_to(embed, (count, *embed.shape[1:])).copy(),
            "internal.high_res_feats_0": np.broadcast_to(hr0, (count, *hr0.shape[1:])).copy(),
            "internal.high_res_feats_1": np.broadcast_to(hr1, (count, *hr1.shape[1:])).copy(),
            "host.sam2.point_coords": np.ascontiguousarray(point_coords),
            "host.sam2.point_labels": np.ascontiguousarray(point_labels),
            "host.sam2.mask_input": np.zeros((count, 1, 256, 256), dtype=np.float32),
            "host.sam2.has_mask_input": np.zeros((count,), dtype=np.int8),
        }
        out = self._run_role(1, "decoder", values)
        logits = np.asarray(out["host.sam2.mask_logits"], dtype=np.float32)
        iou = np.asarray(out["host.sam2.iou_predictions"], dtype=np.float32)
        return logits, iou

    def _postprocess(
        self,
        masks_logits: list[np.ndarray],
        ious: list[float],
        height: int,
        width: int,
        scale: float,
        new_h: int,
        new_w: int,
    ) -> dict[str, object]:
        if not masks_logits:
            return self._empty_outputs(height, width)
        filtered = self._filter(masks_logits, ious)
        if not filtered:
            return self._empty_outputs(height, width)
        nms = self._nms(filtered)
        if not nms:
            return self._empty_outputs(height, width)
        masks, boxes, scores, stability = self._build(nms, height, width, scale, new_h, new_w)
        return {
            "masks": np.ascontiguousarray(masks, dtype=np.uint8),
            "boxes": np.ascontiguousarray(boxes, dtype=np.float32),
            "scores": np.ascontiguousarray(scores, dtype=np.float32),
            "stability_scores": np.ascontiguousarray(stability, dtype=np.float32),
        }

    def _filter(self, masks_logits, ious):
        kept = []
        for logits, iou in zip(masks_logits, ious, strict=True):
            if self._pred_iou_thresh > 0 and iou <= self._pred_iou_thresh:
                continue
            stability = self._stability(logits)
            if self._stability_thresh > 0 and stability < self._stability_thresh:
                continue
            kept.append((logits, float(iou), float(stability)))
        return kept

    def _stability(self, logits: np.ndarray) -> float:
        upper = float(np.sum(logits > _STABILITY_OFFSET))
        lower = float(np.sum(logits > -_STABILITY_OFFSET))
        return upper / lower if lower > 0 else 1.0

    def _nms(self, items):
        binary = [logits > _MASK_THRESHOLD for logits, _, _ in items]
        areas = [float(np.count_nonzero(m)) for m in binary]
        order = sorted(range(len(items)), key=lambda i: -items[i][1])
        keep = []
        suppressed = [False] * len(items)
        for idx in order:
            if suppressed[idx]:
                continue
            keep.append(idx)
            for jdx in order:
                if jdx == idx or suppressed[jdx]:
                    continue
                inter = float(np.count_nonzero(binary[idx] & binary[jdx]))
                iou = inter / max(1.0, min(areas[idx], areas[jdx]))
                if iou > _BOX_NMS_IOU:
                    suppressed[jdx] = True
        return [items[i] for i in keep]

    def _build(self, items, height, width, scale, new_h, new_w):
        masks_out, boxes_out, scores_out, stability_out = [], [], [], []
        for logits, iou, stability in items:
            mask_canvas = (logits > _MASK_THRESHOLD).astype(np.uint8)
            resized = cv2.resize(mask_canvas, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(resized, (width, height), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue
            x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
            masks_out.append(mask)
            boxes_out.append([x1, y1, x2 + 1, y2 + 1])
            scores_out.append(iou)
            stability_out.append(stability)
        if not masks_out:
            return [], [], [], []
        return np.stack(masks_out), np.array(boxes_out, dtype=np.float32), np.array(scores_out), np.array(stability_out)

    def _empty_outputs(self, height: int, width: int) -> dict[str, object]:
        return {
            "masks": np.empty((0, height, width), dtype=np.uint8),
            "boxes": np.empty((0, 4), dtype=np.float32),
            "scores": np.empty((0,), dtype=np.float32),
            "stability_scores": np.empty((0,), dtype=np.float32),
        }

    @staticmethod
    def _decoder_batch_size(deployment) -> int:
        binding = deployment.bindings["decoder"].inputs
        point_coords = next((b for b in binding if b.semantic == "host.sam2.point_coords"), None)
        if point_coords is None or len(point_coords.shape) < 1 or point_coords.shape[0] < 1:
            raise BackendInferenceError(
                "SAM2 decoder OM must declare a concrete point_coords batch",
                code="invalid_input_bindings",
            )
        return int(point_coords.shape[0])


__all__ = ["SAM2AutomaticAscendSession"]
