"""Robot-agnostic retargeter protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .frames import CaptureTrack

HUMAN_SIDES = ("left", "right")


@dataclass(frozen=True)
class RetargetRequest:
    track: CaptureTrack
    side: str  # "left" or "right": the human's anatomical arm, never mirrored


@dataclass(frozen=True)
class RetargetOutput:
    channels: tuple[str, ...]
    values: np.ndarray  # (N, C) radians, raw: unsmoothed and on the track's time axis
    timestamps: np.ndarray  # (N,) seconds, strictly increasing


class Retargeter(Protocol):
    """Map one human arm to robot joint angles, frame by frame.

    Everything robot-specific lives behind this protocol: the channels it
    drives, the range each one may use, and how far the idle sway may swing
    it. Smoothing, resampling, scoring and plan assembly only see arrays.
    """

    name: str
    channels: tuple[str, ...]
    working_range: tuple[tuple[float, float], ...]  # radians, per channel
    idle_amplitude: tuple[float, ...]  # radians, per channel

    def retarget(self, request: RetargetRequest) -> RetargetOutput: ...
