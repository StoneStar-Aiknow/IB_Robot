"""Picking the round's photo: the ranks, the deadline, the person box, and the time it takes.

The images are synthetic colour blocks (no people). A sharp frame is the
blocks as drawn; a blurred one is the same blocks through a Gaussian blur, so
which one is sharper is known by construction.
"""

import cv2
import numpy as np
import pytest

from manipulation_execution.mirror_photo.scoring import (
    Candidate,
    jpeg_size,
    laplacian_sharpness,
    looks_like_jpeg,
    ranks,
    select,
)
from manipulation_execution.mirror_photo.triggers import GESTURE, STILL, UNIFORM, Trigger

HALF = (1344, 972)


def jpeg(seed: int = 0, *, blur: float = 0.0, size=HALF, block: int = 8) -> bytes:
    width, height = size
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (height // block, width // block, 3), dtype=np.uint8)
    image = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    if blur:
        image = cv2.GaussianBlur(image, (0, 0), blur)
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    return data.tobytes()


def candidate(stamp: float, data: bytes, *, source: str = UNIFORM, weight: float = 0.0) -> Candidate:
    return Candidate(Trigger(source, source, weight, 1.0, stamp), stamp, data)


def choose(candidates, *, speeds=None, boxes=None, deadline=1.0, **kwargs):
    speeds = speeds or {}
    boxes = boxes or {}
    return select(
        candidates,
        speed_at=lambda stamp: speeds.get(stamp),
        bbox_at=lambda stamp: boxes.get(stamp),
        stillness_weight=0.5,
        sharpness_weight=0.5,
        deadline_sec=deadline,
        **kwargs,
    )


class Ticks:
    """A clock that moves ``step`` seconds every time it is read."""

    def __init__(self, step: float):
        self.step = step
        self.now = 0.0

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def test_the_frame_header_gives_the_size_without_decoding():
    assert jpeg_size(jpeg(size=(320, 240))) == (320, 240)
    assert jpeg_size(b"not a jpeg") is None
    assert jpeg_size(b"\xff\xd8\xff") is None


def test_a_cut_jpeg_does_not_look_like_one():
    data = jpeg(size=(64, 48))
    assert looks_like_jpeg(data)
    assert not looks_like_jpeg(data[: len(data) // 2])
    assert not looks_like_jpeg(b"\x89PNG" + data[4:])


def test_ranks_run_from_worst_to_best_and_skip_the_unknown():
    assert ranks([3.0, 1.0, None, 2.0], higher_is_better=True) == [1.0, 0.0, None, 0.5]
    assert ranks([3.0, 1.0, 2.0], higher_is_better=False) == [0.0, 1.0, 0.5]
    assert ranks([None, 7.0], higher_is_better=True) == [None, 1.0]
    assert ranks([None], higher_is_better=True) == [None]
    assert ranks([2.0, 2.0, 1.0, 3.0], higher_is_better=True) == [0.5, 0.5, 0.0, 1.0]
    assert ranks([5.0, 5.0], higher_is_better=False) == [0.5, 0.5]


def test_blur_lowers_the_sharpness():
    sharp, blurred = jpeg(1), jpeg(1, blur=4.0)
    assert laplacian_sharpness(sharp, None) > 5 * laplacian_sharpness(blurred, None)
    assert laplacian_sharpness(b"\xff\xd8garbage\xff\xd9", None) is None


def test_the_sharpness_is_measured_inside_the_person_box():
    # Left half sharp, right half blurred, in both; the box picks the half.
    image = cv2.imdecode(np.frombuffer(jpeg(2), np.uint8), cv2.IMREAD_COLOR)
    image[:, HALF[0] // 2 :] = cv2.GaussianBlur(image[:, HALF[0] // 2 :], (0, 0), 4.0)
    data = cv2.imencode(".jpg", image)[1].tobytes()
    left = laplacian_sharpness(data, (0, 0, HALF[0] // 2 - 8, HALF[1]))
    right = laplacian_sharpness(data, (HALF[0] // 2 + 8, 0, HALF[0], HALF[1]))
    assert left > 5 * right
    # A box too small to measure alone: the whole frame.
    assert laplacian_sharpness(data, (10, 10, 20, 20)) == laplacian_sharpness(data, None)


def test_the_sharper_of_two_equal_triggers_wins():
    choice = choose([candidate(1.0, jpeg(3)), candidate(2.0, jpeg(3, blur=4.0))])
    assert choice.candidate.stamp == 1.0
    assert choice.scoring == "full"
    assert choice.sharpness_rank == 1.0


def test_the_stiller_of_two_equal_triggers_wins():
    data = jpeg(4)
    choice = choose([candidate(1.0, data), candidate(2.0, data)], speeds={1.0: 0.1, 2.0: 0.9})
    assert choice.candidate.stamp == 1.0
    assert choice.speed == 0.1


def test_on_a_tie_the_later_frame_wins():
    data = jpeg(5, size=(64, 48))
    assert choose([candidate(1.0, data), candidate(2.0, data)]).candidate.stamp == 2.0


def test_a_gesture_beats_a_held_pose_however_blurred_and_moving():
    gesture = candidate(1.0, jpeg(6, blur=6.0), source=GESTURE, weight=3.0)
    pose = candidate(2.0, jpeg(6), source=STILL, weight=1.5)
    choice = choose([gesture, pose], speeds={1.0: 2.0, 2.0: 0.01})
    assert choice.candidate is gesture


def test_past_the_deadline_the_sharpness_is_left_out_for_all():
    sharp, blurred = jpeg(7), jpeg(7, blur=4.0)
    candidates = [candidate(1.0, sharp), candidate(2.0, blurred)]
    # Every clock read takes 0.6 s: the second frame is past a 1 s deadline.
    choice = choose(candidates, deadline=1.0, clock=Ticks(0.6))
    assert choice.scoring == "partial"
    assert choice.sharpness_rank is None
    assert choice.candidate.stamp == 2.0  # sharpness left out: the tie goes to the later


def test_no_decoder_is_the_triggers_and_the_stillness_only():
    def missing(_jpeg, _bbox):
        raise ImportError("no cv2")

    choice = choose([candidate(1.0, b"a"), candidate(2.0, b"b")], speeds={1.0: 0.1, 2.0: 0.5}, sharpness=missing)
    assert choice.scoring == "none"
    assert choice.candidate.stamp == 1.0


def test_a_frame_that_does_not_decode_is_not_chosen_while_another_does():
    broken = candidate(2.0, b"\xff\xd8broken\xff\xd9", source=GESTURE, weight=3.0)
    fine = candidate(1.0, jpeg(8, size=(320, 240)))
    choice = choose([broken, fine])
    assert choice.candidate is fine
    assert choice.undecodable == 1


def test_when_none_decodes_one_is_still_chosen():
    choice = choose([candidate(1.0, b"\xff\xd8x\xff\xd9"), candidate(2.0, b"\xff\xd8y\xff\xd9")])
    assert choice.candidate.stamp == 2.0
    assert choice.undecodable == 2


def test_one_candidate_and_sharpness_off_skip_the_decoding():
    def never(_jpeg, _bbox):
        raise AssertionError("decoded")

    assert choose([candidate(1.0, b"a")], sharpness=never).scoring == "single"
    off = select(
        [candidate(1.0, b"a"), candidate(2.0, b"b")],
        speed_at=lambda _stamp: None,
        bbox_at=lambda _stamp: None,
        stillness_weight=0.5,
        sharpness_weight=0.0,
        deadline_sec=1.0,
        sharpness=never,
    )
    assert off.scoring == "off"


def test_no_candidate_is_an_error():
    with pytest.raises(ValueError):
        choose([])


def test_a_full_round_of_half_frames_scores_well_inside_the_deadline():
    """24 candidates at half resolution: the round's maximum. Measured here, on the board again."""
    candidates = [candidate(float(index), jpeg(index, blur=index % 5)) for index in range(24)]
    boxes = {float(index): (300.0, 100.0, 1000.0, 950.0) for index in range(24)}
    choice = choose(candidates, boxes=boxes, deadline=1.0)
    print(f"24 half frames scored in {choice.elapsed_sec * 1000:.0f} ms")
    assert choice.scoring == "full"
    assert choice.elapsed_sec < 1.0
    assert choice.candidate.stamp % 5 == 0  # one of the unblurred ones
