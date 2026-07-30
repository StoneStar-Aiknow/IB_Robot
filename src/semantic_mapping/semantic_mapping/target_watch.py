"""Bounded watched-target orchestration outcomes."""

from dataclasses import dataclass

import numpy as np

from .online_lifecycle import OnlineLifecycleCoordinator


@dataclass(frozen=True)
class WatchOutcome:
    outcome: str
    message: str


class TargetWatch:
    def __init__(self, lifecycle: OnlineLifecycleCoordinator, object_id: str, max_attempts: int):
        if max_attempts <= 0:
            raise ValueError("maximum watch attempts must be positive")
        self.lifecycle = lifecycle
        self.object_id = object_id
        self.max_attempts = max_attempts
        lifecycle.begin_watch(object_id)

    def observe(self, position: np.ndarray, embedding: np.ndarray) -> WatchOutcome:
        if self.lifecycle.observe_remote_identity(self.object_id, position, embedding):
            return WatchOutcome("replan", "watched object moved to a stable new position")
        return WatchOutcome("continue", "observation did not confirm stable movement")

    def search_failed(self, details: dict | None = None) -> WatchOutcome:
        if self.lifecycle.record_search_failure(
            self.object_id,
            max_attempts=self.max_attempts,
            details=details,
        ):
            return WatchOutcome("lost", "bounded target search exhausted")
        return WatchOutcome("continue", "additional approved search views remain")
