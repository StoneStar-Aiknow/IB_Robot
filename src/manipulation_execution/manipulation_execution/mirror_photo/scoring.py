"""Picking the round's photo once MIRRORING has ended; no ROS.

During the round nothing is decoded. Afterwards every candidate is decoded at
a quarter size in grey (``cv2.IMREAD_REDUCED_GRAYSCALE_4``) and its sharpness
measured (variance of the Laplacian, inside the person box when there is
one). Each candidate scores::

    trigger weight + stillness_weight * stillness rank + sharpness_weight * sharpness rank

The ranks are within the round only (0 the worst, 1 the best), so no absolute
threshold has to hold across lighting or people. Past ``deadline_sec`` the
decoding stops and the sharpness is left out for every candidate, so a slow
board makes the photo a little worse, never late. A candidate that does not
decode is never chosen while another one does.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .triggers import Trigger

# Smaller than this (quarter-size pixels) the person box is not worth measuring alone.
MIN_CROP_PX = 16
REDUCTION = 4


@dataclass(frozen=True)
class Candidate:
    trigger: Trigger
    # Camera time of the frame.
    stamp: float
    jpeg: bytes
    # Person bounding box from the PEAR yolox frame at this stamp: (x1, y1, x2, y2) in source pixels.
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class Choice:
    candidate: Candidate
    total: float
    stillness_rank: float | None
    sharpness_rank: float | None
    speed: float | None
    sharpness: float | None
    # full: every candidate measured; partial: the deadline passed, sharpness left
    # out; none: no decoder; off: sharpness_weight 0; single: one candidate.
    scoring: str
    undecodable: int
    elapsed_sec: float


def jpeg_size(data: bytes) -> tuple[int, int] | None:
    """``(width, height)`` from the JPEG frame header, without decoding; None if there is none."""
    if data[:2] != b"\xff\xd8":
        return None
    index = 2
    while index + 4 <= len(data):
        if data[index] != 0xFF:
            return None
        marker = data[index + 1]
        if marker == 0xFF:
            index += 1
            continue
        if marker in (0x01, *range(0xD0, 0xD8)):
            index += 2
            continue
        length = int.from_bytes(data[index + 2 : index + 4], "big")
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if index + 9 > len(data):
                return None
            height = int.from_bytes(data[index + 5 : index + 7], "big")
            width = int.from_bytes(data[index + 7 : index + 9], "big")
            return (width, height) if width and height else None
        if marker == 0xDA or length < 2:
            return None
        index += 2 + length
    return None


def looks_like_jpeg(data: bytes) -> bool:
    """Start and end markers both there: the cheap check a candidate passes before it is kept."""
    return len(data) > 4 and data[:2] == b"\xff\xd8" and data.rstrip(b"\x00")[-2:] == b"\xff\xd9"


def laplacian_sharpness(jpeg: bytes, bbox: Sequence[float] | None) -> float | None:
    """Variance of the Laplacian of the quarter-size grey image, inside ``bbox`` if usable; None if it does not decode."""
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if image is None or image.size == 0:
        return None
    if bbox is not None:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = (int(round(value / REDUCTION)) for value in bbox)
        x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
        y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
        if x2 - x1 >= MIN_CROP_PX and y2 - y1 >= MIN_CROP_PX:
            image = image[y1:y2, x1:x2]
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def ranks(values: Sequence[float | None], *, higher_is_better: bool) -> list[float | None]:
    """Each known value's rank among the known ones, 0 (worst) to 1 (best); equal values share a rank, one known value ranks 1."""
    known = sorted(
        (index for index, value in enumerate(values) if value is not None),
        key=lambda index: values[index] if higher_is_better else -values[index],
    )
    result: list[float | None] = [None] * len(values)
    if len(known) == 1:
        result[known[0]] = 1.0
        return result
    start = 0
    while start < len(known):
        end = start
        while end + 1 < len(known) and values[known[end + 1]] == values[known[start]]:
            end += 1
        for index in known[start : end + 1]:
            result[index] = (start + end) / 2 / (len(known) - 1)
        start = end + 1
    return result


def select(
    candidates: Sequence[Candidate],
    *,
    speed_at: Callable[[float], float | None],
    stillness_weight: float,
    sharpness_weight: float,
    deadline_sec: float,
    sharpness: Callable[[bytes, Sequence[float] | None], float | None] = laplacian_sharpness,
    clock: Callable[[], float] = time.monotonic,
) -> Choice:
    """The best of ``candidates`` (not empty); see the module docstring."""
    if not candidates:
        raise ValueError("no candidate to choose from")
    started = clock()
    speeds = [speed_at(candidate.stamp) for candidate in candidates]
    measured: list[float | None] = [None] * len(candidates)
    decoded = [False] * len(candidates)
    scoring = "single" if len(candidates) == 1 else "off" if sharpness_weight <= 0.0 else "full"
    if scoring == "full":
        for index, candidate in enumerate(candidates):
            if clock() - started > deadline_sec:
                scoring = "partial"
                break
            try:
                measured[index] = sharpness(candidate.jpeg, candidate.bbox)
            except ImportError:
                scoring = "none"
                break
            except Exception:  # noqa: BLE001 - a frame that cannot be measured is one that did not decode
                measured[index] = None
            decoded[index] = True
    undecodable = sum(1 for index in range(len(candidates)) if decoded[index] and measured[index] is None)
    usable = [index for index in range(len(candidates)) if not (decoded[index] and measured[index] is None)]
    if not usable:
        usable = list(range(len(candidates)))
    use_sharpness = scoring == "full"
    still_ranks = ranks([speeds[index] for index in usable], higher_is_better=False)
    sharp_ranks = ranks([measured[index] for index in usable], higher_is_better=True) if use_sharpness else None
    best = None
    for position, index in enumerate(usable):
        candidate = candidates[index]
        total = candidate.trigger.weight + stillness_weight * (still_ranks[position] or 0.0)
        if sharp_ranks is not None:
            total += sharpness_weight * (sharp_ranks[position] or 0.0)
        # Ties go to the later frame: the person has settled more.
        key = (total, candidate.stamp)
        if best is None or key > best[0]:
            best = (key, position, index)
    _, position, index = best
    return Choice(
        candidate=candidates[index],
        total=best[0][0],
        stillness_rank=still_ranks[position],
        sharpness_rank=sharp_ranks[position] if sharp_ranks is not None else None,
        speed=speeds[index],
        sharpness=measured[index],
        scoring=scoring,
        undecodable=undecodable,
        elapsed_sec=clock() - started,
    )
