"""Structured filtering and deterministic semantic-object ranking."""

from dataclasses import dataclass

import numpy as np

from .association import SemanticTrack, cosine_similarity


@dataclass(frozen=True)
class ObjectQuery:
    object_ids: frozenset[str] = frozenset()
    canonical_label: str = ""
    states: frozenset[str] = frozenset()
    include_inactive: bool = False
    min_confidence: float = 0.0
    max_age_ns: int = 0
    region_center: np.ndarray | None = None
    region_radius_m: float = 0.0
    max_results: int = 0
    query_text: str = ""
    caption_fallback: bool = True


@dataclass(frozen=True)
class RankedTrack:
    track: SemanticTrack
    semantic_score: float | None
    age_ns: int
    score_source: str = "structured"


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(left.casefold().split())
    right_tokens = set(right.casefold().split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def query_tracks(
    tracks: list[SemanticTrack],
    query: ObjectQuery,
    *,
    now_ns: int,
    query_embedding: np.ndarray | None = None,
    captions: dict[str, str] | None = None,
) -> list[RankedTrack]:
    if not 0.0 <= query.min_confidence <= 1.0:
        raise ValueError("minimum confidence must be between zero and one")
    if query.max_age_ns < 0 or query.region_radius_m < 0.0 or query.max_results < 0:
        raise ValueError("age, region radius, and result limit must be non-negative")
    center = None if query.region_center is None else np.asarray(query.region_center, dtype=np.float64)
    if center is not None and center.shape != (3,):
        raise ValueError("query region center must contain three coordinates")

    results = []
    for track in tracks:
        age_ns = max(0, now_ns - track.last_seen_ns)
        if query.object_ids and track.object_id not in query.object_ids:
            continue
        if query.canonical_label and track.canonical_label.casefold() != query.canonical_label.casefold():
            continue
        if query.states and track.state not in query.states:
            continue
        if not query.include_inactive and not track.active:
            continue
        if track.confidence < query.min_confidence:
            continue
        if query.max_age_ns and age_ns > query.max_age_ns:
            continue
        if center is not None and query.region_radius_m > 0.0:
            extent_radius = float(np.linalg.norm(track.size) / 2.0)
            if np.linalg.norm(track.position - center) > query.region_radius_m + extent_radius:
                continue
        score = cosine_similarity(track.embedding, query_embedding) if query_embedding is not None else None
        score_source = "embedding" if score is not None else "structured"
        if query_embedding is not None and score is None:
            caption = (captions or {}).get(track.object_id, "")
            if not query.caption_fallback or not caption:
                continue
            score = _token_overlap(query.query_text, caption) - 2.0
            score_source = "caption"
        results.append(RankedTrack(track, score, age_ns, score_source))

    results.sort(
        key=lambda item: (
            -(item.semantic_score if item.semantic_score is not None else 0.0),
            -item.track.confidence,
            item.age_ns,
            item.track.object_id,
        )
    )
    return results[: query.max_results or None]
