import json
from types import SimpleNamespace

import numpy as np
import pytest
from tf2_ros import TransformException

from semantic_mapping.artifact_export import SemanticArtifactExporter
from semantic_mapping.association import SemanticObservation, SemanticTracker
from semantic_mapping.database import SemanticMapDatabase, SemanticMapManifest
from semantic_mapping.geometry import project_masked_depth
from semantic_mapping.offline_bag import OfflineBagSource, OfflineTopicContract


def _stamp(nanoseconds):
    return SimpleNamespace(sec=nanoseconds // 1_000_000_000, nanosec=nanoseconds % 1_000_000_000)


def _message(stamp_ns):
    return SimpleNamespace(header=SimpleNamespace(stamp=_stamp(stamp_ns), frame_id="camera"))


class _Reader:
    def __init__(self, messages):
        self.messages_by_topic = messages

    def topic_types(self):
        return {
            "/rgb": "sensor_msgs/msg/Image",
            "/depth": "sensor_msgs/msg/Image",
            "/info": "sensor_msgs/msg/CameraInfo",
            "/tf": "tf2_msgs/msg/TFMessage",
            "/tf_static": "tf2_msgs/msg/TFMessage",
        }

    def messages(self, topics):
        for topic in sorted(topics):
            for message in self.messages_by_topic.get(topic, []):
                yield topic, message, 0


class _Buffer:
    def __init__(self, ready):
        self.ready = ready

    def lookup_transform(self, target, source, stamp):
        if not self.ready:
            raise TransformException("missing TF")
        return SimpleNamespace()


def _manifest():
    return SemanticMapManifest(
        global_frame="map",
        geometry_map_id="fixture",
        geometry_map_hash="fixture-hash",
        localization_session_id="offline-fixture",
        calibration_id="fixture-calibration",
        urdf_hash="fixture-urdf",
        coordinate_convention="ros-rep-103-map-enu",
        semantic_identities=_identities(),
    )


def _run_valid_fixture(output_dir):
    database = SemanticMapDatabase(str(output_dir / "semantic.sqlite3"), _manifest())
    tracker = SemanticTracker()
    depth = np.full((3, 3), 1000, dtype=np.uint16)
    mask = np.ones((3, 3), dtype=np.uint8)
    intrinsics = np.asarray([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]])
    geometry = project_masked_depth(mask, depth, intrinsics, 1000.0, 4.0, 4)
    observation = SemanticObservation(
        label="cup",
        confidence=0.9,
        position=geometry.centroid,
        size=geometry.size,
        point_count=len(geometry.points),
        stamp_ns=1_000_000_000,
        map_version="fixture-hash",
        session_id="offline-fixture",
        source_frame="camera",
        semantic_identities=_identities(),
    )
    track = tracker.update(observation)
    track.object_id = "deterministic-object"
    tracker.tracks = {track.object_id: track}
    database.upsert(track, observation)
    exporter = SemanticArtifactExporter(output_dir / "artifacts", database)
    manifest_path = exporter.export_manifest(_manifest())
    geometry_path = exporter.export_geometry(track.object_id, track.object_version, geometry.points, track.last_seen_ns)
    database.close()
    return manifest_path.read_bytes(), geometry_path.read_bytes()


def test_offline_contract_rejects_missing_depth_and_historical_tf():
    topics = OfflineTopicContract("/rgb", "/depth", "/info")
    source = OfflineBagSource(
        _Reader({"/rgb": [_message(1)], "/info": [_message(1)]}),
        topics,
        global_frame="map",
        sync_slop_ns=1,
    )
    with pytest.raises(ValueError, match="aligned-depth stream"):
        list(source.frames(_Buffer(True)))

    source = OfflineBagSource(
        _Reader({"/rgb": [_message(1)], "/depth": [_message(1)], "/info": [_message(1)]}),
        topics,
        global_frame="map",
        sync_slop_ns=1,
    )
    assert list(source.frames(_Buffer(False))) == []
    assert source.diagnostics.rejected_tf == 1


def test_valid_fixture_fuses_exports_and_reruns_deterministically(tmp_path):
    first_manifest, first_geometry = _run_valid_fixture(tmp_path / "first")
    second_manifest, second_geometry = _run_valid_fixture(tmp_path / "second")

    assert json.loads(first_manifest)["geometry_map_hash"] == "fixture-hash"
    assert first_manifest == second_manifest
    assert first_geometry == second_geometry
    assert not (tmp_path / "first" / "artifacts" / "map.pgm").exists()


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
