import json

import cv2
import numpy as np

from semantic_mapping.association import (
    SemanticObservation,
    SemanticTrack,
    SemanticTracker,
    is_manually_actionable,
)
from semantic_mapping.database import SemanticMapDatabase, SemanticMapManifest
from semantic_mapping.manual_labels import apply_reviewed_labels, consolidate_manual_tracks, export_review_bundle


def _manifest():
    return SemanticMapManifest(
        global_frame="map",
        geometry_map_id="lab-map",
        geometry_map_hash="map-hash",
        localization_session_id="session",
        calibration_id="calibration",
        urdf_hash="urdf",
        coordinate_convention="ros-rep-103-map-enu",
        semantic_identities={},
        settings={"mapping_backend": "embedded"},
    )


def _track():
    return SemanticTrack(
        object_id="stable-track",
        label="box",
        canonical_label="box",
        confidence=0.8,
        position=np.array([1.0, 2.0, 0.0]),
        size=np.array([0.2, 0.2, 0.2]),
        point_count=100,
        first_seen_ns=1,
        last_seen_ns=2,
        observation_count=3,
        map_version="map-hash",
        session_id="session",
    )


def _fragment_track():
    return SemanticTrack(
        object_id="fragment-track",
        label="paper",
        canonical_label="paper",
        confidence=0.7,
        position=np.array([1.1, 2.05, 0.0]),
        size=np.array([0.2, 0.2, 0.2]),
        point_count=60,
        first_seen_ns=4,
        last_seen_ns=5,
        observation_count=2,
        map_version="map-hash",
        session_id="session",
    )


def _observation(label: str, stamp_ns: int):
    return SemanticObservation(
        label=label,
        confidence=0.8,
        position=np.array([1.0, 2.0, 0.0]),
        size=np.array([0.2, 0.2, 0.2]),
        point_count=50,
        stamp_ns=stamp_ns,
        map_version="map-hash",
        session_id="session",
    )


def test_consolidate_merges_same_label_fragments_into_canonical_track(tmp_path):
    database_path = tmp_path / "semantic.sqlite3"
    database = SemanticMapDatabase(database_path, _manifest())
    database.upsert(_track(), _observation("box", 1))
    database.upsert(_fragment_track(), _observation("paper", 4))
    database.close()

    review = {
        "schema_version": 1,
        "tracks": [
            {"object_id": "stable-track", "manual_label": "box", "actionable": True},
            {"object_id": "fragment-track", "manual_label": "box", "actionable": True},
        ],
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    assert apply_reviewed_labels(database_path, review_path) == {"applied": 2, "removed": 0}

    result = consolidate_manual_tracks(database_path)
    assert result["tracks_merged"] == 1
    assert result["details"][0]["canonical_object_id"] == "stable-track"
    assert result["details"][0]["automatic_labels"] == ["box", "paper"]

    database = SemanticMapDatabase(database_path)
    tracks = database.load()
    observation_ids = {
        row[0] for row in database.connection.execute("SELECT DISTINCT object_id FROM semantic_observations").fetchall()
    }
    database.close()
    assert [track.object_id for track in tracks] == ["stable-track"]
    canonical = tracks[0]
    assert canonical.observation_count == 5
    assert canonical.first_seen_ns == 1
    assert canonical.last_seen_ns == 5
    assert canonical.attributes["manual_label"]["merged_object_ids"] == ["fragment-track"]
    assert set(canonical.attributes["manual_label"]["automatic_labels"]) == {"box", "paper"}
    assert observation_ids == {"stable-track"}

    tracker = SemanticTracker()
    tracker.add_track(canonical)
    updated = tracker.update(_observation("paper", 9))
    assert updated.object_id == "stable-track"
    assert updated.label == "box"
    assert is_manually_actionable(updated)


def test_review_bundle_exports_representative_and_applies_immutable_label(tmp_path):
    database_path = tmp_path / "semantic.sqlite3"
    database = SemanticMapDatabase(database_path, _manifest())
    database.upsert(_track())
    database.close()
    diagnostics = tmp_path / "diagnostics"
    (diagnostics / "frames").mkdir(parents=True)
    (diagnostics / "rgb").mkdir()
    cv2.imwrite(str(diagnostics / "rgb" / "0001_2.jpg"), np.full((80, 100, 3), 127, dtype=np.uint8))
    (diagnostics / "frames" / "0001_2.json").write_text(
        json.dumps({"masks": {"0": {"object_id": "stable-track", "area": 40, "bbox": [10, 20, 50, 60]}}}),
        encoding="utf-8",
    )
    review_dir = tmp_path / "review"

    assert export_review_bundle(database_path, diagnostics, review_dir) == {"tracks": 1, "representative_images": 1}
    review_path = review_dir / "review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["tracks"][0]["manual_label"] = "yellow bag"
    review["tracks"][0]["actionable"] = True
    review_path.write_text(json.dumps(review), encoding="utf-8")

    assert apply_reviewed_labels(database_path, review_path) == {"applied": 1, "removed": 0}
    database = SemanticMapDatabase(database_path)
    track = database.load()[0]
    database.close()
    assert track.label == "yellow bag"
    assert is_manually_actionable(track)
    assert (review_dir / "tracks" / "stable-track.jpg").is_file()

    tracker = SemanticTracker()
    tracker.add_track(track)
    updated = tracker.update(
        SemanticObservation(
            label="box",
            confidence=0.95,
            position=np.array([1.0, 2.0, 0.0]),
            size=np.array([0.2, 0.2, 0.2]),
            point_count=100,
            stamp_ns=3,
            map_version="map-hash",
            session_id="session",
        )
    )
    assert updated.label == "yellow bag"


def test_unlabel_rejects_track_from_map(tmp_path):
    database_path = tmp_path / "semantic.sqlite3"
    database = SemanticMapDatabase(database_path, _manifest())
    database.upsert(_track(), _observation("box", 1))
    database.close()

    review = {
        "schema_version": 1,
        "tracks": [{"object_id": "stable-track", "manual_label": "unlabel", "actionable": False}],
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")

    assert apply_reviewed_labels(database_path, review_path) == {"applied": 0, "removed": 1}
    database = SemanticMapDatabase(database_path)
    remaining = database.load()
    observations = database.connection.execute("SELECT count(*) FROM semantic_observations").fetchone()[0]
    database.close()
    assert remaining == []
    assert observations == 0


def test_manual_track_rejects_far_embedding_only_but_learns_near_alias():
    tracker = SemanticTracker()
    track = _track()
    track.label = "yellow bag"
    track.canonical_label = "yellow bag"
    track.embedding = np.full(8, 0.5, dtype=np.float32)
    track.attributes["manual_label"] = {"label": "yellow bag", "actionable": True, "automatic_labels": ["box"]}
    tracker.add_track(track)

    def observation(label: str, position, stamp_ns: int):
        return SemanticObservation(
            label=label,
            confidence=0.9,
            position=np.asarray(position, dtype=np.float64),
            size=np.array([0.2, 0.2, 0.2]),
            point_count=50,
            stamp_ns=stamp_ns,
            map_version="map-hash",
            session_id="session",
            embedding=np.ones(8, dtype=np.float32),
        )

    stray = tracker.update(observation("plane", [1.0, 2.3, 0.0], 9))
    assert stray.object_id != "stable-track"
    tracker.tracks.pop(stray.object_id)

    near = tracker.update(observation("plane", [1.0, 2.05, 0.0], 10))
    assert near.object_id == "stable-track"
    assert near.label == "yellow bag"
    assert "plane" in near.attributes["manual_label"]["automatic_labels"]

    aliased = tracker.update(observation("plane", [1.0, 2.4, 0.0], 11))
    assert aliased.object_id == "stable-track"
    assert aliased.label == "yellow bag"
