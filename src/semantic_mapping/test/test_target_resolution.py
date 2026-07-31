import numpy as np

from semantic_mapping.association import SemanticTrack
from semantic_mapping.target_resolution import resolve_target


def _track(state="observed"):
    return SemanticTrack(
        object_id="object",
        label="cup",
        confidence=0.9,
        position=np.asarray([2.0, 0.0, 0.5]),
        size=np.asarray([0.2, 0.2, 0.2]),
        point_count=10,
        first_seen_ns=1,
        last_seen_ns=2,
        state=state,
    )


def test_resolver_returns_distinct_stand_off_pose_facing_object():
    resolution = resolve_target(
        _track(),
        np.zeros(3),
        0.8,
        lambda candidate: (True, ""),
    )

    assert resolution.ready
    assert not np.allclose(resolution.staging.position, resolution.object.position)
    assert np.linalg.norm(resolution.staging.position[:2] - resolution.object.position[:2]) == 0.8


def test_resolver_checks_candidates_and_rejects_invalid_lifecycle_states():
    checked = []

    def checker(candidate):
        checked.append(candidate)
        return (len(checked) == 2, "blocked")

    assert resolve_target(_track(), np.zeros(3), 0.8, checker).ready
    assert len(checked) == 2
    for state in ("stale", "missing", "lost"):
        checked.clear()
        result = resolve_target(_track(state), np.zeros(3), 0.8, checker)
        assert not result.ready
        assert checked == []
