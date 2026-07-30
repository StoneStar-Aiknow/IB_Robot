"""P50/P95 semantic mapping timing and throughput reports."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class FrameTiming:
    inference_ms: float
    serialization_ms: float
    queue_wait_ms: float
    service_round_trip_ms: float
    fusion_commit_ms: float
    end_to_end_ms: float

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")


class TimingReporter:
    def __init__(self, *, backend: str, batch_size: int):
        if not backend:
            raise ValueError("backend is required")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.backend = backend
        self.batch_size = batch_size
        self._samples: list[FrameTiming] = []
        self._dropped_frames = 0

    def record(self, timing: FrameTiming) -> None:
        self._samples.append(timing)

    def record_drop(self, count: int = 1) -> None:
        if count <= 0:
            raise ValueError("drop count must be positive")
        self._dropped_frames += count

    def report(self, *, elapsed_sec: float) -> dict:
        if elapsed_sec <= 0.0 or not np.isfinite(elapsed_sec):
            raise ValueError("elapsed_sec must be a finite positive number")
        stages = {}
        for field_name in FrameTiming.__dataclass_fields__:
            values = np.asarray([getattr(sample, field_name) for sample in self._samples], dtype=np.float64)
            stages[field_name] = {
                "p50_ms": float(np.percentile(values, 50)) if len(values) else 0.0,
                "p95_ms": float(np.percentile(values, 95)) if len(values) else 0.0,
            }
        return {
            "backend": self.backend,
            "batch_size": self.batch_size,
            "processed_frames": len(self._samples),
            "dropped_frames": self._dropped_frames,
            "elapsed_sec": float(elapsed_sec),
            "throughput_fps": len(self._samples) / float(elapsed_sec),
            "stages": stages,
        }
