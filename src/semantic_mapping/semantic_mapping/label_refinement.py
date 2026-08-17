"""Optional cloud-VLM refinement for ambiguous semantic object labels."""

import json
import re
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .association import SemanticTrack, has_manual_label
from .representative_view import RepresentativeView

_LABEL_PATTERN = re.compile(r"^[a-z0-9][a-z0-9 _-]{0,63}$")


@dataclass(frozen=True)
class LabelRefinementResult:
    label: str
    confidence: float
    candidate_match: bool
    model_identity: str
    candidates: tuple[tuple[str, float], ...]
    created_ns: int


class LabelRefinementRejected(ValueError):
    """An otherwise valid cloud response was rejected by local policy."""

    def __init__(self, reason: str, *, label: str, confidence: float, candidate_match: bool):
        super().__init__(reason)
        self.label = label
        self.confidence = confidence
        self.candidate_match = candidate_match


def should_refine_label(
    label: str,
    confidence: float,
    excluded_labels,
    trigger_below_confidence: float,
    *,
    inconsistent: bool = False,
) -> bool:
    """Refine labels that are unknown or impossible in the configured scene."""
    normalized = label.strip().casefold()
    excluded = {str(value).strip().casefold() for value in excluded_labels}
    return inconsistent or normalized == "unlabeled" or normalized in excluded or confidence < trigger_below_confidence


def parse_refinement_response(
    content: str,
    *,
    model_identity: str,
    candidates: tuple[tuple[str, float], ...],
    min_confidence: float,
    excluded_labels=(),
    created_ns: int | None = None,
) -> LabelRefinementResult:
    """Parse the strict JSON contract returned by the cloud model."""
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("label refinement response must be a JSON object") from exc
    if not isinstance(value, dict) or set(value) != {"label", "confidence"}:
        raise ValueError("label refinement response must contain only label and confidence")
    label = value["label"]
    confidence = value["confidence"]
    if not isinstance(label, str) or not _LABEL_PATTERN.fullmatch(label.strip().casefold()):
        raise ValueError("refined label must be a short lowercase object category")
    if isinstance(confidence, bool) or not isinstance(confidence, int | float) or not 0.0 <= confidence <= 1.0:
        raise ValueError("refined confidence must be in [0.0, 1.0]")
    normalized = label.strip().casefold()
    candidate_labels = {candidate.strip().casefold() for candidate, _score in candidates}
    candidate_match = normalized in candidate_labels
    excluded = {str(value).strip().casefold() for value in excluded_labels}
    if normalized in excluded:
        raise LabelRefinementRejected(
            f"cloud label {normalized!r} is excluded by scene labels {sorted(excluded)}",
            label=normalized,
            confidence=float(confidence),
            candidate_match=candidate_match,
        )
    if confidence < min_confidence:
        raise LabelRefinementRejected(
            f"cloud label {normalized!r} confidence {float(confidence):.3f} is below threshold {min_confidence:.3f}",
            label=normalized,
            confidence=float(confidence),
            candidate_match=candidate_match,
        )
    return LabelRefinementResult(
        normalized,
        float(confidence),
        candidate_match,
        model_identity,
        candidates,
        time.time_ns() if created_ns is None else created_ns,
    )


class CloudLabelRefiner:
    """Send one masked representative crop to an injected multimodal client."""

    def __init__(
        self, client, *, model: str, model_identity: str, prompt: str, min_confidence: float, excluded_labels=()
    ):
        self.client = client
        self.model = model
        self.model_identity = model_identity
        self.prompt = prompt
        self.min_confidence = min_confidence
        self.excluded_labels = tuple(excluded_labels)

    def refine(
        self, view: RepresentativeView, candidates: tuple[tuple[str, float], ...], *, max_retries: int = 2
    ) -> LabelRefinementResult:
        review_image = self._review_image(view)
        encoded, payload = cv2.imencode(".jpg", review_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not encoded:
            raise RuntimeError("failed to encode representative crop")
        candidate_text = ", ".join(f"{label} ({score:.3f})" for label, score in candidates) or "none"
        excluded_text = ", ".join(sorted(self.excluded_labels)) or "none"
        request = (
            f"{self.prompt}\n"
            f"Labels forbidden in this scene: {excluded_text}. Never return a forbidden label.\n"
            f"RAM++ candidates: {candidate_text}\n"
            'Return only JSON: {"label":"object category","confidence":0.0}'
        )
        image_bytes = payload.tobytes()
        last_error = None
        for attempt in range(max_retries + 1):
            result = self.client.chat(request, image=image_bytes, model=self.model or None, clear_history=True)
            if result.get("status") == "ok":
                return parse_refinement_response(
                    str(result.get("content", "")),
                    model_identity=self.model_identity,
                    candidates=candidates,
                    min_confidence=self.min_confidence,
                    excluded_labels=self.excluded_labels,
                )
            last_error = str(result.get("error") or result.get("content") or "label refinement failed")
            if attempt < max_retries and "timeout" in last_error.lower():
                time.sleep(2.0)
                continue
            break
        raise RuntimeError(last_error or "label refinement failed")

    @staticmethod
    def _review_image(view: RepresentativeView) -> object:
        x1, y1, x2, y2 = view.bbox_xyxy.tolist()
        context = view.image_bgr.copy()
        target_mask = view.mask > 0
        green = context.copy()
        green[target_mask] = (0, 255, 0)
        context = cv2.addWeighted(context, 0.75, green, 0.25, 0.0)
        cv2.rectangle(context, (x1, y1), (x2, y2), (0, 0, 255), 3)

        crop = view.image_bgr[y1:y2, x1:x2].copy()
        crop_mask = target_mask[y1:y2, x1:x2]
        crop[~crop_mask] = 127
        panel_width = max(192, context.shape[1] // 2)
        scale = min(panel_width / crop.shape[1], context.shape[0] / crop.shape[0])
        crop = cv2.resize(
            crop,
            (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
            interpolation=cv2.INTER_NEAREST,
        )
        panel = 127 * np.ones((context.shape[0], panel_width, 3), dtype=np.uint8)
        y_offset = (panel.shape[0] - crop.shape[0]) // 2
        x_offset = (panel.shape[1] - crop.shape[1]) // 2
        panel[y_offset : y_offset + crop.shape[0], x_offset : x_offset + crop.shape[1]] = crop
        return np.concatenate((context, panel), axis=1)


def apply_refinement(track: SemanticTrack, result: LabelRefinementResult) -> None:
    """Apply an accepted refinement while retaining its auditable provenance."""
    if has_manual_label(track):
        raise ValueError("manual track labels cannot be replaced by automatic refinement")
    previous_label = track.label
    track.label = result.label
    track.canonical_label = result.label
    track.confidence = result.confidence
    track.object_version += 1
    track.attributes.pop("label_refinement_last_rejection", None)
    track.attributes["label_refinement"] = {
        "source": "cloud_vlm",
        "previous_label": previous_label,
        "model_identity": result.model_identity,
        "confidence": result.confidence,
        "candidate_match": result.candidate_match,
        "candidates": [{"label": label, "score": score} for label, score in result.candidates],
        "created_ns": result.created_ns,
    }


def record_refinement_rejection(
    track: SemanticTrack,
    *,
    candidates: tuple[tuple[str, float], ...],
    model_identity: str,
    error: Exception,
) -> None:
    """Persist a safe diagnostic for the most recent rejected or failed request."""
    record = {
        "source": "cloud_vlm",
        "ram_label": track.label,
        "ram_confidence": track.confidence,
        "model_identity": model_identity,
        "reason": str(error),
        "candidates": [{"label": label, "score": score} for label, score in candidates],
        "created_ns": time.time_ns(),
    }
    if isinstance(error, LabelRefinementRejected):
        record.update(
            {
                "cloud_label": error.label,
                "cloud_confidence": error.confidence,
                "candidate_match": error.candidate_match,
                "failure_kind": "policy_rejected",
            }
        )
    else:
        record["failure_kind"] = "request_or_contract_error"
    track.attributes["label_refinement_last_rejection"] = record
    track.object_version += 1
