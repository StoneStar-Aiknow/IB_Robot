from types import SimpleNamespace

import numpy as np
import pytest

from semantic_mapping.association import SemanticTrack
from semantic_mapping.database import DatabaseCompatibilityError, SemanticMapDatabase, SemanticMapManifest
from semantic_mapping.semantic_mapping_node import SemanticMappingNode
from semantic_mapping.slam_readiness import evaluate_slam_readiness


def _identities(sam_revision="sam2@v1"):
    common = {"preprocessing_contract": "pre-v1", "output_semantics": "output-v1"}
    return {
        "sam2": {"logical_model_revision": sam_revision, **common},
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


def _manifest(**overrides):
    values = {
        "global_frame": "map",
        "geometry_map_id": "map-id",
        "geometry_map_hash": "map-hash",
        "localization_session_id": "session",
        "calibration_id": "calibration",
        "urdf_hash": "urdf",
        "coordinate_convention": "ros-rep-103-map-enu",
        "semantic_identities": _identities(),
    }
    values.update(overrides)
    return SemanticMapManifest(**values)


def test_restart_loads_persisted_tracks_with_version_and_state(tmp_path):
    path = tmp_path / "semantic.sqlite3"
    database = SemanticMapDatabase(str(path), _manifest())
    database.upsert(
        SemanticTrack(
            object_id="persisted",
            label="cup",
            confidence=0.8,
            position=np.zeros(3),
            size=np.ones(3),
            point_count=10,
            first_seen_ns=1,
            last_seen_ns=2,
            object_version=4,
            state="stale",
        )
    )
    database.close()

    restarted = SemanticMapDatabase(str(path), _manifest())
    tracks = restarted.load()
    restarted.close()

    assert tracks[0].object_id == "persisted"
    assert tracks[0].object_version == 4
    assert tracks[0].state == "stale"


@pytest.mark.parametrize(
    "manifest",
    [
        _manifest(geometry_map_hash="different-map"),
        _manifest(semantic_identities=_identities("sam2@v2")),
    ],
)
def test_restart_rejects_map_or_model_mismatch(tmp_path, manifest):
    path = tmp_path / "semantic.sqlite3"
    SemanticMapDatabase(str(path), _manifest()).close()

    with pytest.raises(DatabaseCompatibilityError, match="identity mismatch"):
        SemanticMapDatabase(str(path), manifest)


def test_relocalization_loss_revokes_readiness_without_deleting_objects():
    readiness = evaluate_slam_readiness(
        expected_map_hash="map-hash",
        active_map_hash="map-hash",
        localization_ready=False,
        authoritative_map_odom=True,
        cloud_map_ready=True,
        timestamped_tf_ready=True,
    )

    assert not readiness.ready
    assert "localization" in readiness.reason


def test_online_pin_mismatch_closes_admission_and_ends_persisted_run():
    updates = []
    node = SimpleNamespace(
        _run=SimpleNamespace(run_id="run-1"),
        _run_admission_open=True,
        _database=SimpleNamespace(update_mapping_run_status=lambda *args, **kwargs: updates.append((args, kwargs))),
        get_logger=lambda: SimpleNamespace(error=lambda _message: None),
    )

    SemanticMappingNode._stop_pinned_run(node, "semantic identity changed")

    assert not node._run_admission_open
    assert updates[0][0][0:2] == ("run-1", "failed")
    assert updates[0][1]["ended_ns"] == updates[0][0][2]
    assert updates[0][1]["reason"] == "semantic identity changed"
