"""Parse object-level grounding results from VLM responses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from embodied_common.json_utils import parse_confidence


@dataclass
class GroundedObject:
    label: str
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    attributes: dict[str, Any] = field(default_factory=dict)
    source: str = "vlm_bbox"


def _bbox_from_payload(raw_bbox: Any) -> tuple[float, float, float, float] | None:
    if isinstance(raw_bbox, dict):
        if all(key in raw_bbox for key in ("x1", "y1", "x2", "y2")):
            return (
                float(raw_bbox["x1"]),
                float(raw_bbox["y1"]),
                float(raw_bbox["x2"]),
                float(raw_bbox["y2"]),
            )
        if all(key in raw_bbox for key in ("x", "y", "width", "height")):
            x = float(raw_bbox["x"])
            y = float(raw_bbox["y"])
            return (x, y, x + float(raw_bbox["width"]), y + float(raw_bbox["height"]))
    if isinstance(raw_bbox, list | tuple) and len(raw_bbox) == 4:
        return tuple(float(value) for value in raw_bbox)  # type: ignore[return-value]
    return None


def _clamp_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    image_width: int | None,
    image_height: int | None,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox_xyxy
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if image_width is not None and image_height is not None and image_width > 0 and image_height > 0:
        min_coord = min(x1, y1, x2, y2)
        max_coord = max(x1, y1, x2, y2)
        # Some VLMs return bbox coordinates in a 0-1000 image grid despite being asked for pixels.
        if min_coord >= 0.0 and max_coord > max(image_width, image_height) and max_coord <= 1000.0:
            x1 = x1 / 1000.0 * image_width
            x2 = x2 / 1000.0 * image_width
            y1 = y1 / 1000.0 * image_height
            y2 = y2 / 1000.0 * image_height
    if image_width is not None and image_width > 0:
        x1 = max(0.0, min(x1, float(image_width - 1)))
        x2 = max(0.0, min(x2, float(image_width - 1)))
    if image_height is not None and image_height > 0:
        y1 = max(0.0, min(y1, float(image_height - 1)))
        y2 = max(0.0, min(y2, float(image_height - 1)))
    ix1, iy1, ix2, iy2 = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def parse_grounded_objects(
    payload: dict[str, Any],
    image_width: int | None = None,
    image_height: int | None = None,
    min_confidence: float = 0.0,
) -> list[GroundedObject]:
    """Parse optional ``objects`` from a scene-understanding JSON payload."""

    raw_objects = payload.get("objects", [])
    if not isinstance(raw_objects, list):
        return []

    objects: list[GroundedObject] = []
    for index, item in enumerate(raw_objects):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("name") or item.get("object") or "").strip()
        if not label:
            continue
        raw_bbox = item.get("bbox_2d", item.get("bbox", item.get("box")))
        bbox = _bbox_from_payload(raw_bbox)
        if bbox is None:
            continue
        clamped = _clamp_bbox(bbox, image_width=image_width, image_height=image_height)
        if clamped is None:
            continue
        confidence = parse_confidence(item.get("confidence", 1.0))
        if confidence < min_confidence:
            continue
        attributes = item.get("attributes", {})
        if not isinstance(attributes, dict):
            attributes = {"raw_attributes": attributes}
        if "view_name" in item and "view_name" not in attributes:
            attributes["view_name"] = item["view_name"]
        attributes.setdefault("object_index", index)
        objects.append(
            GroundedObject(
                label=label,
                bbox_xyxy=clamped,
                confidence=confidence,
                attributes=attributes,
                source=str(item.get("source", "vlm_bbox")),
            )
        )
    return objects


def attributes_to_json(attributes: dict[str, Any]) -> str:
    return json.dumps(attributes, ensure_ascii=False, sort_keys=True)
