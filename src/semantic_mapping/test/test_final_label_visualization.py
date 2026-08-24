import json
import sqlite3

import cv2
import numpy as np

from semantic_mapping.final_label_visualization import render_final_labels


def test_render_final_labels_maps_observations_to_accepted_masks(tmp_path):
    diagnostics = tmp_path / "diagnostics"
    for name in ("frames", "rgb", "siglip2"):
        (diagnostics / name).mkdir(parents=True)
    stem = "0001_123"
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    cv2.imwrite(str(diagnostics / "rgb" / f"{stem}.jpg"), image)
    cv2.imwrite(str(diagnostics / "siglip2" / f"{stem}.jpg"), image)
    (diagnostics / "frames" / f"{stem}.json").write_text(
        json.dumps(
            {
                "stamp_ns": 123,
                "accepted_masks": [2],
                "masks": {"2": {"bbox": [10, 10, 80, 60], "area": 2400, "object_id": "object-12345678"}},
            }
        ),
        encoding="utf-8",
    )

    database = tmp_path / "map.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE mapping_runs (run_id TEXT, started_ns INTEGER);
        CREATE TABLE semantic_objects (
            object_id TEXT, label TEXT, confidence REAL, attributes_json TEXT
        );
        CREATE TABLE semantic_observations (
            observation_id INTEGER, source_stamp_ns INTEGER, object_id TEXT,
            label TEXT, confidence REAL, mapping_run_id TEXT
        );
        INSERT INTO mapping_runs VALUES ('run-1', 1);
        INSERT INTO semantic_objects VALUES (
            'object-12345678', 'banana', 0.93,
            '{"manual_label":{"label":"banana","actionable":true},"label_refinement":{"source":"cloud_vlm"}}'
        );
        INSERT INTO semantic_objects VALUES (
            'object-unlabeled', 'haystack', 0.9, '{}'
        );
        INSERT INTO semantic_observations VALUES (1, 123, 'object-12345678', 'hay', 0.69, 'run-1');
        INSERT INTO semantic_observations VALUES (2, 123, 'object-unlabeled', 'straw', 0.75, 'run-1');
        """
    )
    connection.commit()
    connection.close()

    summary = render_final_labels(diagnostics, database)

    assert summary["frames_rendered"] == 1
    assert summary["cloud_refined_annotations"] == 1
    assert summary["unmatched_observations"] == 0
    assert (diagnostics / "final_labels" / f"{stem}.jpg").is_file()
    assert (diagnostics / "final_labels_siglip" / f"{stem}.jpg").is_file()
