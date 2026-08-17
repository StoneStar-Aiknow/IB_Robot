"""Render final track labels back onto offline mapping diagnostic frames."""

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def _latest_run_id(connection: sqlite3.Connection) -> str:
    row = connection.execute("SELECT run_id FROM mapping_runs ORDER BY started_ns DESC LIMIT 1").fetchone()
    if row is None:
        raise ValueError("semantic map has no mapping runs")
    return str(row[0])


def _load_results(database_path: Path, run_id: str | None) -> tuple[str, dict[int, list[dict]]]:
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        selected_run = run_id or _latest_run_id(connection)
        rows = connection.execute(
            """
            SELECT observation_id, source_stamp_ns, semantic_observations.object_id,
                   semantic_observations.label AS observed_label,
                   semantic_observations.confidence AS observed_confidence,
                   semantic_objects.label AS final_label,
                   semantic_objects.confidence AS final_confidence,
                   semantic_objects.attributes_json
            FROM semantic_observations
            JOIN semantic_objects USING (object_id)
            WHERE mapping_run_id = ?
            ORDER BY source_stamp_ns, observation_id
            """,
            (selected_run,),
        ).fetchall()
    finally:
        connection.close()

    by_stamp: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        attributes = json.loads(row["attributes_json"])
        refinement = attributes.get("label_refinement", {})
        source = "cloud" if refinement.get("source") == "cloud_vlm" else "aligned"
        by_stamp[int(row["source_stamp_ns"])].append(
            {
                "object_id": str(row["object_id"]),
                "observed_label": str(row["observed_label"]),
                "observed_confidence": float(row["observed_confidence"]),
                "final_label": str(row["final_label"]),
                "final_confidence": float(row["final_confidence"]),
                "source": source,
            }
        )
    return selected_run, dict(by_stamp)


def _draw_label(image: np.ndarray, bbox: list[float], text: str, color: tuple[int, int, int], row: int) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
    y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    text_x = min(x1, max(0, width - text_width - 6))
    preferred_y = y1 - 5 - row * (text_height + baseline + 5)
    text_y = preferred_y if preferred_y - text_height - baseline >= 0 else min(height - baseline - 3, y1 + 18)
    cv2.rectangle(
        image,
        (text_x, text_y - text_height - 4),
        (min(width - 1, text_x + text_width + 6), min(height - 1, text_y + baseline + 2)),
        (20, 20, 20),
        cv2.FILLED,
    )
    cv2.putText(image, text, (text_x + 3, text_y), font, scale, color, thickness, cv2.LINE_AA)


def render_final_labels(
    diagnostics_dir: str | Path,
    database_path: str | Path,
    *,
    run_id: str | None = None,
) -> dict:
    """Render final labels for one mapping run and return a summary."""
    root = Path(diagnostics_dir).expanduser()
    database = Path(database_path).expanduser()
    if not database.is_file():
        raise FileNotFoundError(f"semantic map database does not exist: {database}")
    frame_dir = root / "frames"
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"diagnostic frame directory does not exist: {frame_dir}")

    selected_run, observations = _load_results(database, run_id)
    output_rgb = root / "final_labels"
    output_siglip = root / "final_labels_siglip"
    output_rgb.mkdir(parents=True, exist_ok=True)
    output_siglip.mkdir(parents=True, exist_ok=True)

    colors = {"cloud": (60, 220, 60), "aligned": (0, 190, 255), "unchanged": (255, 210, 70)}
    summary = {
        "run_id": selected_run,
        "frames_rendered": 0,
        "frames_with_tracks": 0,
        "annotations": 0,
        "cloud_refined_annotations": 0,
        "aligned_annotations": 0,
        "unchanged_annotations": 0,
        "unmatched_observations": 0,
    }
    for frame_path in sorted(frame_dir.glob("*.json")):
        frame = json.loads(frame_path.read_text(encoding="utf-8"))
        stamp_ns = int(frame["stamp_ns"])
        rgb_path = root / "rgb" / f"{frame_path.stem}.jpg"
        if not rgb_path.is_file():
            continue
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            continue
        siglip_path = root / "siglip2" / f"{frame_path.stem}.jpg"
        siglip = cv2.imread(str(siglip_path)) if siglip_path.is_file() else rgb.copy()
        if siglip is None:
            siglip = rgb.copy()

        frame_observations = observations.get(stamp_ns, [])
        accepted_masks = [int(index) for index in frame.get("accepted_masks", [])]
        masks = frame.get("masks", {})
        if frame_observations:
            summary["frames_with_tracks"] += 1
        for row, (mask_index, observation) in enumerate(zip(accepted_masks, frame_observations, strict=False)):
            mask = masks.get(str(mask_index))
            if not isinstance(mask, dict) or "bbox" not in mask:
                summary["unmatched_observations"] += 1
                continue
            changed = observation["observed_label"].casefold() != observation["final_label"].casefold()
            source = observation["source"] if changed else "unchanged"
            color = colors[source]
            text = (
                f"M{mask_index} {observation['observed_label']}:{observation['observed_confidence']:.2f}"
                f" -> {observation['final_label']}:{observation['final_confidence']:.2f}"
                f" [{source}] #{observation['object_id'][:8]}"
            )
            _draw_label(rgb, mask["bbox"], text, color, row)
            _draw_label(siglip, mask["bbox"], text, color, row)
            summary["annotations"] += 1
            summary["cloud_refined_annotations" if source == "cloud" else f"{source}_annotations"] += 1
        if len(frame_observations) > len(accepted_masks):
            summary["unmatched_observations"] += len(frame_observations) - len(accepted_masks)
        cv2.imwrite(str(output_rgb / f"{frame_path.stem}.jpg"), rgb)
        cv2.imwrite(str(output_siglip / f"{frame_path.stem}.jpg"), siglip)
        summary["frames_rendered"] += 1

    (root / "final_labels_index.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostics_dir")
    parser.add_argument("database_path")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    summary = render_final_labels(args.diagnostics_dir, args.database_path, run_id=args.run_id)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
