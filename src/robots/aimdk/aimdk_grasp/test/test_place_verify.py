"""Place verification judges the candidate the place aimed at, not the detector's first guess."""

import time
from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp import place_verify
from aimdk_grasp.place_verify import PlaceVerifier

SIZE = 100


def observation():
    intrinsics = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
    return SimpleNamespace(intrinsics=intrinsics, depth_m=np.ones((SIZE, SIZE)), base_from_camera=np.eye(4), stamp_ns=1)


def plan():
    # An opening of +-0.1 m around the optical axis, one metre in front: pixels 40..60.
    opening = np.array([[-0.1, -0.1, 1.0], [0.1, -0.1, 1.0], [0.1, 0.1, 1.0], [-0.1, 0.1, 1.0]])
    return SimpleNamespace(container=SimpleNamespace(opening=opening, normal=np.array([0.0, 0, 1]), rim_height_m=0.95))


def mask(rows, cols):
    out = np.zeros((SIZE, SIZE), bool)
    out[rows[0] : rows[1], cols[0] : cols[1]] = True
    return out


INSIDE, OUTSIDE = mask((45, 55), (45, 55)), mask((80, 90), (80, 90))


@pytest.fixture
def detected(monkeypatch):
    calls = {}

    def run(*masks):
        def detect_masks(services, observation, query, *, cancelled, settings):
            calls.update(query=query, settings=settings)
            yield from masks

        monkeypatch.setattr(place_verify, "detect_masks", detect_masks)
        verdict = PlaceVerifier("services", settings="pick settings").verify(
            observation(),
            plan(),
            SimpleNamespace(target_query="an apple", radius_m=0.05),
            deadline=time.monotonic() + 10,
            cancelled=lambda: False,
        )
        return verdict, calls

    return run


def test_an_object_ranked_first_outside_the_container_does_not_decide(detected):
    """Live 2026-11-07: the apple was in the bowl, but another object outside was detected first."""
    verdict, calls = detected(OUTSIDE, INSIDE)
    assert verdict.success and verdict.report["chosen"] == 1
    assert [c["inside_fraction"] for c in verdict.report["candidates"]] == [0.0, 1.0]
    assert calls == {"query": "an apple", "settings": "pick settings"}


def test_the_object_nearest_the_opening_outside_it_is_placed_outside(detected):
    verdict, _ = detected(OUTSIDE)
    assert (verdict.success, verdict.reason_code) == (False, "PLACED_OUTSIDE")
    assert verdict.report["inside_fraction"] == 0.0 and len(verdict.report["candidates"]) == 1


def test_a_partly_inside_object_needs_most_of_its_points_inside(detected):
    straddling = mask((45, 55), (55, 65))  # half over the rim
    verdict, _ = detected(straddling)
    assert (verdict.success, verdict.reason_code) == (False, "PLACED_OUTSIDE")
    assert 0.4 <= verdict.report["inside_fraction"] <= 0.6


def test_nothing_measurable_is_unseen(detected):
    verdict, _ = detected(mask((0, 2), (0, 2)))  # 4 points, below the minimum
    assert (verdict.success, verdict.reason_code) == (False, "PLACE_UNSEEN")
    assert verdict.report == {"candidates": []}


def test_a_verdict_is_recorded_with_its_image_and_candidate_masks(monkeypatch, tmp_path, detected):
    """CR6/CL2 (2026-11-08): false verdicts could not be checked offline without the image and the masks."""
    import pickle

    from aimdk_grasp import inputs_record

    monkeypatch.setenv(inputs_record.INPUTS_DIR_ENV, str(tmp_path))
    threads = []
    original = place_verify.save_record
    monkeypatch.setattr(place_verify, "save_record", lambda *args: threads.append(original(*args)))
    verdict, _ = detected(OUTSIDE, INSIDE)
    for thread in threads:
        thread.join(timeout=10)
    (saved,) = tmp_path.glob("verify_*.pkl")
    record = pickle.loads(saved.read_bytes())
    assert verdict.success  # unchanged by the record
    assert record["kind"] == "verify" and record["target_query"] == "an apple" and record["held_radius_m"] == 0.05
    shapes, packed = zip(*record["masks"], strict=True)
    masks = [np.unpackbits(bits)[: SIZE * SIZE].reshape(shape).astype(bool) for shape, bits in record["masks"]]
    assert shapes == ((SIZE, SIZE),) * 2 and (masks[0] == OUTSIDE).all() and (masks[1] == INSIDE).all()
    assert record["verdict"] == dict(success=True, reason_code="", report=verdict.report)
    assert record["observation"]["depth_m"].shape == (SIZE, SIZE) and record["container"]["rim_height_m"] == 0.95
