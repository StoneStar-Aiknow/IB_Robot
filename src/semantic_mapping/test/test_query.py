import numpy as np

from semantic_mapping.association import SemanticTrack
from semantic_mapping.query import ObjectQuery, query_tracks


def _track(object_id, position, embedding, *, state="observed", confidence=0.8, last_seen=90):
    return SemanticTrack(
        object_id=object_id,
        canonical_label="cup",
        label="Cup",
        confidence=confidence,
        position=np.asarray(position, dtype=np.float64),
        size=np.asarray([0.2, 0.2, 0.2]),
        point_count=10,
        first_seen_ns=1,
        last_seen_ns=last_seen,
        embedding=np.asarray(embedding, dtype=np.float32),
        state=state,
    )


def test_query_filters_structured_fields_and_region_intersection():
    tracks = [
        _track("near", [1.05, 0.0, 0.0], [1.0, 0.0]),
        _track("far", [3.0, 0.0, 0.0], [1.0, 0.0]),
        _track("stale", [1.0, 0.0, 0.0], [1.0, 0.0], state="stale"),
    ]
    query = ObjectQuery(
        canonical_label="cup",
        region_center=np.asarray([1.0, 0.0, 0.0]),
        region_radius_m=0.1,
        max_age_ns=20,
    )

    results = query_tracks(tracks, query, now_ns=100)

    assert [item.track.object_id for item in results] == ["near"]


def test_semantic_ranking_is_deterministic_and_embeddings_remain_on_tracks():
    tracks = [
        _track("second", [0.0, 0.0, 0.0], [0.8, 0.2], confidence=0.9),
        _track("first", [0.0, 0.0, 0.0], [1.0, 0.0], confidence=0.7),
        _track("none", [0.0, 0.0, 0.0], [0.0, 1.0]),
    ]

    results = query_tracks(
        tracks,
        ObjectQuery(max_results=2),
        now_ns=100,
        query_embedding=np.asarray([1.0, 0.0]),
    )

    assert [item.track.object_id for item in results] == ["first", "second"]
    assert results[0].semantic_score == 1.0


def test_caption_fallback_is_optional_lower_priority_and_caption_free_lookup_still_works():
    embedded = _track("embedded", [0.0, 0.0, 0.0], [1.0, 0.0])
    captioned = _track("captioned", [0.0, 0.0, 0.0], [1.0, 0.0])
    captioned.embedding = None

    results = query_tracks(
        [captioned, embedded],
        ObjectQuery(query_text="red cup"),
        now_ns=100,
        query_embedding=np.asarray([1.0, 0.0]),
        captions={"captioned": "a red cup on a table"},
    )
    structured = query_tracks([captioned], ObjectQuery(), now_ns=100)

    assert [item.track.object_id for item in results] == ["embedded", "captioned"]
    assert results[1].score_source == "caption"
    assert structured[0].track.object_id == "captioned"
