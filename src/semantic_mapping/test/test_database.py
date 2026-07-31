import json
import sqlite3

import numpy as np
import pytest

from semantic_mapping.association import FROZEN, SemanticObservation, SemanticTrack
from semantic_mapping.database import (
    CaptionRecord,
    DatabaseCompatibilityError,
    MappingRunRecord,
    ObjectGeometryRecord,
    SemanticMapDatabase,
    SemanticMapManifest,
    inspect_database,
)
from semantic_mapping.migrate_database import migrate_prototype_database


def _identity(name, *, embedding=False):
    value = {
        "logical_model_revision": f"{name}@v1",
        "preprocessing_contract": f"{name}-pre-v1",
        "output_semantics": f"{name}-output-v1",
    }
    if embedding:
        value["embedding"] = {
            "embedding_space_id": "siglip2-base-patch16-224:v1",
            "dimension": 2,
            "normalization": "l2",
            "image_preprocessing": "siglip2-image-224-v1",
            "text_preprocessing": "siglip2-tokenizer-64-v1",
        }
    return value


def _identities():
    return {
        "sam2": _identity("sam2"),
        "ram_plus": _identity("ram-plus"),
        "siglip2_image": _identity("siglip2", embedding=True),
    }


def _manifest(**overrides):
    values = {
        "global_frame": "map",
        "geometry_map_id": "warehouse-a",
        "geometry_map_hash": "map-sha256",
        "localization_session_id": "session-1",
        "calibration_id": "d435-calibration-v1",
        "urdf_hash": "urdf-sha256",
        "coordinate_convention": "ros-rep-103-map-enu",
        "semantic_identities": _identities(),
    }
    values.update(overrides)
    return SemanticMapManifest(**values)


def _track():
    return SemanticTrack(
        object_id="stable-id",
        canonical_label="cup",
        label="Cup",
        confidence=0.9,
        position=np.array([1.0, 2.0, 3.0]),
        size=np.array([0.1, 0.2, 0.3]),
        point_count=42,
        first_seen_ns=10,
        last_seen_ns=20,
        observation_count=3,
        embedding=np.array([0.6, 0.8], dtype=np.float32),
        map_version="map-sha256",
        session_id="session-1",
        object_version=4,
        model_versions={"siglip2": "siglip-hash"},
        lifecycle_evidence={"identity_confirmed": True},
        attributes={"source_frame": "camera"},
        semantic_identities=_identities(),
        deployment_provenance={"siglip2_image": {"backend": "cuda"}},
    )


def _observation():
    return SemanticObservation(
        label="Cup",
        canonical_label="cup",
        confidence=0.85,
        position=np.array([1.0, 2.0, 3.0]),
        size=np.array([0.1, 0.2, 0.3]),
        point_count=42,
        stamp_ns=20,
        source_frame="d435_color_optical_frame",
        map_version="map-sha256",
        session_id="session-1",
        model_versions={"sam2": "sam-hash", "siglip2": "siglip-hash"},
        semantic_identities=_identities(),
        deployment_provenance={"sam2": {"backend": "cuda"}},
        mapping_run_id="run-1",
    )


def test_versioned_database_round_trip_preserves_identity_and_provenance(tmp_path):
    path = tmp_path / "semantic.sqlite3"
    database = SemanticMapDatabase(str(path), _manifest())
    database.create_mapping_run(
        MappingRunRecord(
            "run-1",
            7,
            {"sam2": "sam", "ram_plus": "ram", "siglip2_image": "siglip"},
            _identities(),
            "active",
            1,
            1,
        )
    )
    database.upsert(_track(), _observation())
    database.upsert_caption(CaptionRecord("stable-id", "a cup", "vlm-hash", 30))
    database.upsert_geometry(ObjectGeometryRecord("stable-id", 4, "pointcloud", "objects/id.pcd", "pcd-hash", 42, 30))

    loaded = database.load()
    caption = database.get_caption("stable-id")
    table_names = {
        row[0] for row in database.connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    database.close()

    assert inspect_database(path) == "versioned"
    assert {
        "semantic_manifest",
        "semantic_objects",
        "semantic_observations",
        "mapping_runs",
        "semantic_captions",
        "object_geometry",
    } <= (table_names)
    assert loaded[0].object_id == "stable-id"
    assert loaded[0].object_version == 4
    assert np.allclose(loaded[0].embedding, [0.6, 0.8])
    assert loaded[0].model_versions == {"siglip2": "siglip-hash"}
    assert loaded[0].semantic_identities == _identities()
    assert loaded[0].deployment_provenance["siglip2_image"]["backend"] == "cuda"
    assert caption.caption == "a cup"


def test_manifest_mismatch_fails_closed_but_diagnostic_read_only_opens(tmp_path):
    path = tmp_path / "semantic.sqlite3"
    SemanticMapDatabase(str(path), _manifest()).close()
    incompatible = _manifest(geometry_map_hash="another-map")

    with pytest.raises(DatabaseCompatibilityError, match="geometry_map_hash"):
        SemanticMapDatabase(str(path), incompatible)
    diagnostic = SemanticMapDatabase(str(path), incompatible, read_only=True, diagnostic=True)

    assert diagnostic.compatibility_errors == ["geometry_map_hash"]
    with pytest.raises(PermissionError, match="read-only"):
        diagnostic.upsert(_track())
    diagnostic.close()


def test_manifest_localization_session_mismatch_fails_closed(tmp_path):
    path = tmp_path / "semantic.sqlite3"
    SemanticMapDatabase(str(path), _manifest()).close()

    with pytest.raises(DatabaseCompatibilityError, match="localization_session_id"):
        SemanticMapDatabase(str(path), _manifest(localization_session_id="session-2"))


def _create_prototype(path):
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE semantic_objects (
            object_id TEXT PRIMARY KEY, label TEXT NOT NULL, confidence REAL NOT NULL,
            position_json TEXT NOT NULL, size_json TEXT NOT NULL, point_count INTEGER NOT NULL,
            first_seen_ns INTEGER NOT NULL, last_seen_ns INTEGER NOT NULL,
            observation_count INTEGER NOT NULL, embedding BLOB, embedding_size INTEGER NOT NULL,
            active INTEGER NOT NULL, attributes_json TEXT NOT NULL
        )
        """
    )
    embedding = np.array([0.6, 0.8], dtype=np.float32)
    connection.execute(
        "INSERT INTO semantic_objects VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "legacy-id",
            "Cup",
            0.7,
            json.dumps([1.0, 2.0, 3.0]),
            json.dumps([0.1, 0.2, 0.3]),
            20,
            1,
            2,
            2,
            embedding.tobytes(),
            embedding.size,
            1,
            "{}",
        ),
    )
    connection.commit()
    connection.close()


def test_prototype_requires_explicit_copy_migration_and_preserves_source(tmp_path):
    source = tmp_path / "prototype.sqlite3"
    destination = tmp_path / "versioned.sqlite3"
    _create_prototype(source)
    source_bytes = source.read_bytes()

    with pytest.raises(DatabaseCompatibilityError, match="explicit migration"):
        SemanticMapDatabase(str(source), _manifest())
    with pytest.raises(PermissionError, match="confirmation"):
        migrate_prototype_database(source, destination, _manifest(), confirmed=False)
    count = migrate_prototype_database(source, destination, _manifest(), confirmed=True)

    database = SemanticMapDatabase(str(destination), _manifest())
    loaded = database.load()
    database.close()
    assert count == 1
    assert source.read_bytes() == source_bytes
    assert loaded[0].state == FROZEN
    assert loaded[0].map_version == "map-sha256"
    assert loaded[0].attributes["prototype_active"] is True


def test_manifest_rejects_missing_required_model_identity(tmp_path):
    with pytest.raises(ValueError, match="ram_plus, siglip2_image"):
        _manifest(semantic_identities={"sam2": _identity("sam2")})


def test_manifest_compatibility_ignores_deployment_provenance(tmp_path):
    path = tmp_path / "semantic.sqlite3"
    SemanticMapDatabase(str(path), _manifest()).close()

    compatible = SemanticMapDatabase(str(path), _manifest())
    assert compatible.compatibility_errors == []
    compatible.close()


def test_optional_gdino_is_not_required_for_manifest_or_database_open(tmp_path):
    path = tmp_path / "semantic.sqlite3"
    manifest = _manifest()
    SemanticMapDatabase(str(path), manifest).close()

    reopened = SemanticMapDatabase(str(path), manifest)
    assert "grounding_dino" not in reopened.manifest.semantic_identities
    reopened.close()


def test_mapping_run_failure_reason_is_persisted(tmp_path):
    database = SemanticMapDatabase(str(tmp_path / "semantic.sqlite3"), _manifest())
    database.create_mapping_run(
        MappingRunRecord(
            "run-1",
            7,
            {"sam2": "sam", "ram_plus": "ram", "siglip2_image": "siglip"},
            _identities(),
            "active",
            1,
            1,
        )
    )
    database.update_mapping_run_status("run-1", "failed", 2, ended_ns=2, reason="identity changed")

    run = database.get_mapping_run("run-1")
    assert run.status == "failed"
    assert run.ended_ns == 2
    assert run.status_reason == "identity changed"
    database.close()
