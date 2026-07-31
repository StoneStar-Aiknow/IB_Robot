"""Deterministic representative-view selection and optional captioning."""

from dataclasses import dataclass

import numpy as np

from .database import CaptionRecord


@dataclass(frozen=True)
class RepresentativeView:
    object_id: str
    stamp_ns: int
    confidence: float
    mask_area: int
    sharpness: float
    image_bgr: np.ndarray
    mask: np.ndarray
    bbox_xyxy: np.ndarray

    @property
    def rank(self) -> tuple:
        return (self.confidence, self.mask_area, self.sharpness, -self.stamp_ns)


class RepresentativeViewStore:
    def __init__(self):
        self._views: dict[str, RepresentativeView] = {}

    @staticmethod
    def create(
        object_id: str,
        stamp_ns: int,
        confidence: float,
        image_bgr: np.ndarray,
        mask: np.ndarray,
        bbox_xyxy: np.ndarray,
    ) -> RepresentativeView:
        import cv2

        mask = np.asarray(mask) > 0
        bbox = np.asarray(bbox_xyxy, dtype=np.int32)
        x1, y1, x2, y2 = bbox.tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_bgr.shape[1], x2), min(image_bgr.shape[0], y2)
        if x2 <= x1 or y2 <= y1 or mask.shape != image_bgr.shape[:2]:
            raise ValueError("representative crop has invalid mask or bounding box")
        gray = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray.size else 0.0
        return RepresentativeView(
            object_id=object_id,
            stamp_ns=stamp_ns,
            confidence=confidence,
            mask_area=int(np.count_nonzero(mask)),
            sharpness=sharpness,
            image_bgr=image_bgr.copy(),
            mask=mask.astype(np.uint8),
            bbox_xyxy=np.asarray([x1, y1, x2, y2], dtype=np.int32),
        )

    def consider(self, view: RepresentativeView) -> bool:
        current = self._views.get(view.object_id)
        if current is not None and current.rank >= view.rank:
            return False
        self._views[view.object_id] = view
        return True

    def get(self, object_id: str) -> RepresentativeView | None:
        return self._views.get(object_id)


class OptionalCaptioner:
    """Call an injected VLMClient lazily and convert all failures to records."""

    def __init__(self, client, model_identity: str):
        self.client = client
        self.model_identity = model_identity

    def caption(self, object_id: str, view: RepresentativeView, prompt: str) -> CaptionRecord:
        try:
            import cv2

            x1, y1, x2, y2 = view.bbox_xyxy.tolist()
            crop = view.image_bgr[y1:y2, x1:x2].copy()
            crop_mask = view.mask[y1:y2, x1:x2] > 0
            crop[~crop_mask] = 127
            encoded, payload = cv2.imencode(".jpg", crop)
            if not encoded:
                raise RuntimeError("failed to encode representative crop")
            result = self.client.chat(prompt, image=payload.tobytes(), clear_history=True)
            if result.get("status") != "ok":
                raise RuntimeError(str(result.get("error") or result.get("content") or "caption request failed"))
            text = str(result.get("content", "")).strip()
            if not text:
                raise RuntimeError("caption model returned empty text")
            return CaptionRecord(object_id, text, self.model_identity, view.stamp_ns)
        except Exception as exc:
            return CaptionRecord(
                object_id=object_id,
                caption="",
                model_identity=self.model_identity,
                created_ns=view.stamp_ns,
                success=False,
                message=str(exc),
            )
