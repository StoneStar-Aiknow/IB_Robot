"""Turn the recorded PEAR frames into a clean, time-ordered SMPL-X track."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .rotations import decode_rotation_6d

POSE_FIELD = "smplx_pose_raw"
# 52 packed 6D rotations: 0 root, 1-21 body, 22-36 left hand, 37-51 right hand.
ROTATION_COUNT = 52
POSE_DIM = ROTATION_COUNT * 6
_NANOSEC_PER_SEC = 1_000_000_000


@dataclass(frozen=True)
class CaptureTrack:
    """Robot-agnostic human track on the capture's own, non-uniform time axis."""

    t: np.ndarray  # (N,) seconds, relative to the first kept frame, strictly increasing
    rotations: np.ndarray  # (N, 52, 3, 3) local joint rotations
    dropped: int = 0  # frames rejected while parsing

    def __len__(self) -> int:
        return int(self.t.shape[0])

    @property
    def span_sec(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self) else 0.0


def parse_frames(frames: Sequence[Mapping[str, object]]) -> CaptureTrack:
    """Parse, order and de-duplicate the frames captured by the imitation node.

    Only relative time is used: the stamps come from the image header clock,
    which is not the clock the capture window was measured with, so their
    absolute value means nothing here. PEAR answers asynchronously, so arrival
    order is not capture order and frames are sorted by stamp; of several
    frames sharing a stamp only the first to arrive is kept. Malformed or
    non-finite frames are dropped and counted rather than raised on.
    """
    stamped: list[tuple[int, np.ndarray]] = []
    dropped = 0
    for frame in frames:
        try:
            sec = int(frame["stamp_sec"])
            nanosec = int(frame["stamp_nanosec"])
            pose = np.asarray(frame[POSE_FIELD], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            dropped += 1
            continue
        if not 0 <= nanosec < _NANOSEC_PER_SEC or pose.shape != (POSE_DIM,) or not np.all(np.isfinite(pose)):
            dropped += 1
            continue
        stamped.append((sec * _NANOSEC_PER_SEC + nanosec, pose))

    stamped.sort(key=lambda item: item[0])  # stable: the first arrival wins a tie
    stamps: list[int] = []
    poses: list[np.ndarray] = []
    for stamp, pose in stamped:
        if stamps and stamp <= stamps[-1]:
            dropped += 1
            continue
        stamps.append(stamp)
        poses.append(pose)
    if not stamps:
        return CaptureTrack(np.zeros(0), np.zeros((0, ROTATION_COUNT, 3, 3)), dropped)

    rotations = decode_rotation_6d(np.stack(poses))
    # A zero or degenerate 6D pair decodes to a matrix with missing columns.
    valid = np.all(np.abs(np.linalg.det(rotations) - 1.0) < 1e-3, axis=1)
    dropped += int(np.count_nonzero(~valid))
    relative_ns = np.asarray(stamps, dtype=np.int64)[valid]
    if relative_ns.size:
        relative_ns = relative_ns - relative_ns[0]
    return CaptureTrack(relative_ns.astype(np.float64) / _NANOSEC_PER_SEC, rotations[valid], dropped)
