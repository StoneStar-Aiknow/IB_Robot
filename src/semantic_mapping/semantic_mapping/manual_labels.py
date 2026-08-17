"""Export representative track views and apply reviewed semantic labels."""

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np

from .association import normalize_embedding
from .database import SemanticMapDatabase

_LABEL_PATTERN = re.compile(r"^[a-z0-9][a-z0-9 _-]{0,63}$")
REJECT_LABEL = "unlabel"


def export_review_bundle(
    database_path: str | Path,
    diagnostics_dir: str | Path,
    output_dir: str | Path,
    *,
    min_observations: int = 3,
) -> dict:
    """Write one largest-mask representative image and review record per track."""
    if min_observations < 1:
        raise ValueError("min_observations must be positive")
    root = Path(diagnostics_dir).expanduser()
    output = Path(output_dir).expanduser()
    frame_dir = root / "frames"
    rgb_dir = root / "rgb"
    if not frame_dir.is_dir() or not rgb_dir.is_dir():
        raise FileNotFoundError("diagnostics must contain frames/ and rgb/ directories")
    database = SemanticMapDatabase(database_path, read_only=True, diagnostic=True)
    try:
        tracks = {track.object_id: track for track in database.load() if track.observation_count >= min_observations}
    finally:
        database.close()
    representatives = {}
    for frame_path in sorted(frame_dir.glob("*.json")):
        frame = json.loads(frame_path.read_text(encoding="utf-8"))
        for mask in frame.get("masks", {}).values():
            object_id = str(mask.get("object_id", ""))
            if object_id not in tracks:
                continue
            candidate = (int(mask.get("area", 0)), frame_path.stem, mask.get("bbox", []))
            if len(candidate[2]) == 4 and candidate > representatives.get(object_id, (-1, "", [])):
                representatives[object_id] = candidate
    image_dir = output / "tracks"
    image_dir.mkdir(parents=True, exist_ok=True)
    for stale_image in image_dir.glob("*.jpg"):
        stale_image.unlink()
    records = []
    for object_id, track in sorted(tracks.items()):
        image_name = ""
        representative = representatives.get(object_id)
        if representative is not None:
            _area, frame_id, bbox = representative
            image = cv2.imread(str(rgb_dir / f"{frame_id}.jpg"))
            if image is not None:
                x1, y1, x2, y2 = (int(value) for value in bbox)
                padding = 24
                x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
                x2, y2 = min(image.shape[1], x2 + padding), min(image.shape[0], y2 + padding)
                if x2 > x1 and y2 > y1:
                    image_name = f"tracks/{object_id}.jpg"
                    cv2.imwrite(str(output / image_name), image[y1:y2, x1:x2])
        manual = track.attributes.get("manual_label", {})
        records.append(
            {
                "object_id": object_id,
                "current_label": track.label,
                "observation_count": track.observation_count,
                "representative_image": image_name,
                "manual_label": manual.get("label", "") if isinstance(manual, dict) else "",
                "actionable": bool(manual.get("actionable", False)) if isinstance(manual, dict) else False,
            }
        )
    records.sort(key=lambda row: (-row["observation_count"], row["object_id"]))
    _write_contact_sheet(output, records)
    result = {"schema_version": 1, "tracks": records}
    (output / "review.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return {"tracks": len(records), "representative_images": sum(bool(row["representative_image"]) for row in records)}


def _write_contact_sheet(output: Path, records: list[dict]) -> None:
    tile_width, tile_height, columns = 300, 220, 4
    rows = max(1, (len(records) + columns - 1) // columns)
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 245, dtype=np.uint8)
    for index, record in enumerate(records):
        row, column = divmod(index, columns)
        x, y = column * tile_width, row * tile_height
        image = cv2.imread(str(output / record["representative_image"])) if record["representative_image"] else None
        if image is not None:
            scale = min((tile_width - 8) / image.shape[1], 164 / image.shape[0])
            resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))))
            offset_x = x + (tile_width - resized.shape[1]) // 2
            sheet[y : y + resized.shape[0], offset_x : offset_x + resized.shape[1]] = resized
        cv2.putText(
            sheet,
            f"{record['object_id'][:8]} obs={record['observation_count']}",
            (x + 6, y + 184),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            str(record["current_label"])[:34],
            (x + 6, y + 207),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output / "contact_sheet.jpg"), sheet)


def apply_reviewed_labels(database_path: str | Path, review_path: str | Path) -> dict:
    """Persist reviewed labels as immutable track overrides."""
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if not isinstance(review, dict) or not isinstance(review.get("tracks"), list):
        raise ValueError("review file must contain a tracks list")
    database = SemanticMapDatabase(database_path)
    try:
        tracks = {track.object_id: track for track in database.load()}
        applied = 0
        removed = 0
        for record in review["tracks"]:
            if not isinstance(record, dict) or not isinstance(record.get("object_id"), str):
                raise ValueError("each review record must contain an object_id")
            label = str(record.get("manual_label", "")).strip().casefold()
            if not label:
                continue
            if not _LABEL_PATTERN.fullmatch(label) or not isinstance(record.get("actionable"), bool):
                raise ValueError("manual labels must be short categories with a boolean actionable flag")
            track = tracks.get(record["object_id"])
            if track is None:
                raise ValueError(f"review references unknown object {record['object_id']}")
            if label == REJECT_LABEL:
                previous_manual = track.attributes.get("manual_label", {})
                automatic_label = (
                    str(previous_manual.get("automatic_label", "")) if isinstance(previous_manual, dict) else ""
                ) or track.label
                database.connection.execute(
                    "DELETE FROM semantic_objects WHERE object_id = ?",
                    (track.object_id,),
                )
                database.connection.execute(
                    "DELETE FROM object_geometry WHERE object_id = ?",
                    (track.object_id,),
                )
                database.connection.commit()
                removed += 1
                print(f"rejected manual label '{automatic_label}' track {track.object_id}: removed from map")
                continue
            previous_manual = track.attributes.get("manual_label", {})
            automatic_label = (
                str(previous_manual.get("automatic_label", "")) if isinstance(previous_manual, dict) else track.label
            ) or track.label
            track.label = label
            track.canonical_label = label
            track.confidence = 1.0
            track.object_version += 1
            track.attributes["manual_label"] = {
                "label": label,
                "actionable": record["actionable"],
                "automatic_label": automatic_label,
                "source": "human",
                "applied_ns": time.time_ns(),
            }
            track.attributes["semantic_actionable"] = record["actionable"]
            database.upsert(track)
            applied += 1
        return {"applied": applied, "removed": removed}
    finally:
        database.close()


def _merge_sum_evidence(group, key: str) -> dict:
    merged: dict = {}
    for item in group:
        for name, value in item.attributes.get(key, {}).items():
            merged[name] = merged.get(name, 0) + value
    return merged


def _merge_candidate_evidence(group) -> dict:
    merged: dict = {}
    for item in group:
        for name, value in item.attributes.get("label_candidate_evidence", {}).items():
            current = merged.get(name, {})
            merged[name] = {
                "count": int(current.get("count", 0)) + int(value.get("count", 0)),
                "score_sum": float(current.get("score_sum", 0.0)) + float(value.get("score_sum", 0.0)),
                "max_score": max(float(current.get("max_score", 0.0)), float(value.get("max_score", 0.0))),
            }
    return merged


def consolidate_manual_tracks(database_path: str | Path) -> dict:
    """Merge manually labeled fragment tracks into one canonical object per label.

    Unique-item constraint: one manual label denotes exactly one physical object, so
    every fragment track sharing that label is re-pointed onto the canonical track with
    the most observations. Observations and geometry artifacts move to the canonical
    object id, forming one persistent point cloud; all fragment automatic labels are
    kept as association aliases so future observations still bind to the same id.
    """
    database = SemanticMapDatabase(database_path)
    try:
        groups: dict[str, list] = {}
        for track in database.load():
            manual = track.attributes.get("manual_label")
            if isinstance(manual, dict) and str(manual.get("label", "")).strip():
                groups.setdefault(str(manual["label"]).casefold(), []).append(track)
        details = []
        merged_total = 0
        for label in sorted(groups):
            group = sorted(groups[label], key=lambda item: (-item.observation_count, item.object_id))
            if len(group) < 2:
                continue
            canonical, fragments = group[0], group[1:]
            weights = np.asarray([item.observation_count for item in group], dtype=np.float64)
            canonical.position = sum(w * item.position for w, item in zip(weights, group, strict=False)) / weights.sum()
            canonical.size = sum(w * item.size for w, item in zip(weights, group, strict=False)) / weights.sum()
            embeddings = [item.embedding for item in group if item.embedding is not None]
            if embeddings:
                embed_weights = np.asarray(
                    [item.observation_count for item in group if item.embedding is not None], dtype=np.float64
                )
                stacked = np.stack([np.asarray(e, dtype=np.float64) for e in embeddings])
                canonical.embedding = normalize_embedding((stacked * embed_weights[:, None]).sum(axis=0))
            canonical.point_count = max(item.point_count for item in group)
            canonical.first_seen_ns = min(item.first_seen_ns for item in group)
            canonical.last_seen_ns = max(item.last_seen_ns for item in group)
            canonical.observation_count = int(weights.sum())
            canonical.object_version = max(item.object_version for item in group) + 1
            attributes = dict(canonical.attributes)
            attributes["label_evidence"] = _merge_sum_evidence(group, "label_evidence")
            attributes["label_score_evidence"] = _merge_sum_evidence(group, "label_score_evidence")
            max_confidence: dict[str, float] = {}
            for item in group:
                for name, value in item.attributes.get("label_max_confidence", {}).items():
                    max_confidence[name] = max(max_confidence.get(name, 0.0), float(value))
            attributes["label_max_confidence"] = max_confidence
            attributes["label_candidate_evidence"] = _merge_candidate_evidence(group)
            automatic_labels = []
            for item in group:
                item_manual = item.attributes.get("manual_label", {})
                automatic = (
                    str(item_manual.get("automatic_label", "")).strip().casefold()
                    if isinstance(item_manual, dict)
                    else ""
                )
                if automatic and automatic not in automatic_labels:
                    automatic_labels.append(automatic)
            manual = dict(attributes.get("manual_label", {}))
            manual["automatic_labels"] = automatic_labels
            manual["merged_object_ids"] = sorted(item.object_id for item in fragments)
            attributes["manual_label"] = manual
            canonical.attributes = attributes
            connection = database.connection
            geometry_version = canonical.object_version
            for fragment in fragments:
                connection.execute(
                    "UPDATE semantic_observations SET object_id = ? WHERE object_id = ?",
                    (canonical.object_id, fragment.object_id),
                )
                geometry_rows = connection.execute(
                    "SELECT object_version, artifact_type FROM object_geometry WHERE object_id = ?",
                    (fragment.object_id,),
                ).fetchall()
                for row in geometry_rows:
                    geometry_version += 1
                    connection.execute(
                        "UPDATE object_geometry SET object_id = ?, object_version = ? "
                        "WHERE object_id = ? AND object_version = ? AND artifact_type = ?",
                        (canonical.object_id, geometry_version, fragment.object_id, row[0], row[1]),
                    )
                connection.execute("DELETE FROM semantic_objects WHERE object_id = ?", (fragment.object_id,))
            database.upsert(canonical)
            connection.commit()
            merged_total += len(fragments)
            details.append(
                {
                    "label": label,
                    "canonical_object_id": canonical.object_id,
                    "merged_object_ids": manual["merged_object_ids"],
                    "automatic_labels": automatic_labels,
                    "observations": canonical.observation_count,
                }
            )
        return {"labels_consolidated": len(details), "tracks_merged": merged_total, "details": details}
    finally:
        database.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("database_path")
    export.add_argument("diagnostics_dir")
    export.add_argument("output_dir")
    export.add_argument("--min-observations", type=int, default=3)
    apply = subparsers.add_parser("apply")
    apply.add_argument("database_path")
    apply.add_argument("review_path")
    consolidate = subparsers.add_parser("consolidate")
    consolidate.add_argument("database_path")
    args = parser.parse_args()
    if args.command == "export":
        print(
            json.dumps(
                export_review_bundle(
                    args.database_path,
                    args.diagnostics_dir,
                    args.output_dir,
                    min_observations=args.min_observations,
                ),
                indent=2,
            )
        )
    elif args.command == "apply":
        print(json.dumps({"applied": apply_reviewed_labels(args.database_path, args.review_path)}))
    else:
        print(json.dumps(consolidate_manual_tracks(args.database_path), indent=2))


if __name__ == "__main__":
    main()
