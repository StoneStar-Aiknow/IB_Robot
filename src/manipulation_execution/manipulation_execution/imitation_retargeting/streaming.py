"""Streaming retargeting: PEAR frames in, a 50 Hz arm pose stream out.

The offline entry point (``retarget_capture``) turns a finished capture into a
plan. A streaming retargeter instead follows a live person, and the live node
(``motion_imitation_retarget_node``) drives it through ``RetargetStream``:

* ``retarget`` maps one PEAR frame to 14 raw joint angles. It is the heavy step
  (up to ~0.2 s when the wrist branch guard widens its window), so the node
  calls it on a worker thread, one frame at a time, never concurrently.
* ``push_pose``, ``reset_input``, ``start_round`` and ``poll`` are cheap and
  share state with each other; the node serialises them under one lock.
* ``reset_input`` touches the same per-person state as ``retarget`` and is
  called on the worker thread, just before the first frame after a gap.

A round is open-ended: ``start_round`` holds the neutral pose briefly, then
cross-fades from it to the person, then follows until the next
``start_round``. When the person comes back after a gap, the node starts a new
round from the last command instead, so the arm does not chase them at full speed. Ending a round is not the stream's business: the session stops
taking the preview and walks the arm back itself.

Robots register a factory in ``_KINDS``; the factory is imported on first use
so a robot's dependencies are only paid by the robot that needs them.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Protocol

import numpy as np


class RetargetStream(Protocol):
    joint_names: tuple[str, ...]  # the order of every pose this stream returns

    def retarget(self, t_camera: float, frame_id: int, smplx_pose_raw: np.ndarray) -> np.ndarray:
        """One PEAR frame -> raw joint angles in radians. Worker thread only; may raise."""

    def push_pose(self, t_camera: float, pose: np.ndarray) -> None:
        """Queue a pose ``retarget`` returned, at its camera time."""

    def reset_input(self) -> None:
        """The input broke off; forget the per-person continuity (worker thread)."""

    def start_round(self, from_pose: Sequence[float] | None = None) -> None:
        """Start a round from the neutral pose, or from ``from_pose``."""

    def poll(self, now: float) -> list[np.ndarray]:
        """The poses due by ``now``, oldest first, one per 50 Hz tick; ``[]`` while filling.

        ``now`` is on the clock the frames' camera times were stamped on.
        """

    def status(self) -> dict:
        """JSON-ready figures about the current round."""


# kind -> "module:factory", relative to this package.
_KINDS = {"x2_followme": ".x2.followme:create"}


def create_stream(kind: str, *, base_pose: Sequence[float], **params) -> RetargetStream:
    """A stream of the registered ``kind``; ``base_pose`` is the neutral pose in its joint order."""
    if kind not in _KINDS:
        raise ValueError(f"unknown streaming retargeter {kind!r}; known: {sorted(_KINDS)}")
    module, factory = _KINDS[kind].split(":")
    return getattr(importlib.import_module(module, __package__), factory)(base_pose=base_pose, **params)
