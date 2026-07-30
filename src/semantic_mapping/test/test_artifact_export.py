import json

import numpy as np

from semantic_mapping.artifact_export import SemanticArtifactExporter, sha256_path
from semantic_mapping.association import SemanticTrack
from semantic_mapping.database import SemanticMapDatabase, SemanticMapManifest


def _identities():
    common = {"preprocessing_contract": "pre-v1", "output_semantics": "output-v1"}
    return {
        "sam2": {"logical_model_revision": "sam2@v1", **common},
        "ram_plus": {"logical_model_revision": "ram-plus@v1", **common},
        "siglip2_image": {
            "logical_model_revision": "siglip2@v1",
            **common,
            "embedding": {
                "embedding_space_id": "siglip2-space:v1",
                "dimension": 2,
                "normalization": "l2",
                "image_preprocessing": "image-v1",
                "text_preprocessing": "text-v1",
            },
        },
    }


def _manifest():
    return SemanticMapManifest(
        global_frame="map",
        geometry_map_id="map-id",
        geometry_map_hash="map-hash",
        localization_session_id="session",
        calibration_id="calibration",
        urdf_hash="urdf",
        coordinate_convention="ros-rep-103-map-enu",
        semantic_identities=_identities(),
    )


def test_export_writes_manifest_and_versioned_object_geometry_only(tmp_path):
    database = SemanticMapDatabase(str(tmp_path / "semantic.sqlite3"), _manifest())
    database.upsert(
        SemanticTrack(
            object_id="object-id",
            label="cup",
            confidence=0.8,
            position=np.zeros(3),
            size=np.ones(3),
            point_count=2,
            first_seen_ns=1,
            last_seen_ns=10,
            object_version=3,
        )
    )
    exporter = SemanticArtifactExporter(tmp_path / "artifacts", database)
    manifest_path = exporter.export_manifest(_manifest())
    points = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    geometry_path = exporter.export_geometry("object-id", 3, points, 10)
    row = database.connection.execute("SELECT * FROM object_geometry").fetchone()
    database.close()

    assert json.loads(manifest_path.read_text())["geometry_map_hash"] == "map-hash"
    assert geometry_path.name == "object-id.v3.pcd"
    assert row["artifact_hash"] == sha256_path(geometry_path)
    assert row["point_count"] == 2
    assert not (tmp_path / "artifacts" / "map.pgm").exists()
