"""Native RoboTwin artifact reporter for episode videos."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_runtime.models import EpisodeResult, NativeArtifact, RunManifest, RunSummary, StepEvent
from benchmark_runtime.native_report import NativeReportExporter


class RobotwinNativeReporter(NativeReportExporter):
    """Persist the configured RoboTwin camera frames as MP4 artifacts."""

    def __init__(self) -> None:
        self._native_root: Path | None = None
        self._task: dict[str, Any] | None = None
        self._artifacts: list[NativeArtifact] = []

    def on_run_started(self, manifest: RunManifest) -> None:
        self._native_root = Path(manifest.output_ref) / "native" / "robotwin"
        self._native_root.mkdir(parents=True, exist_ok=True)

    def on_step(self, event: StepEvent) -> None:
        del event

    def on_task_started(self, *args: Any, **kwargs: Any) -> None:
        del args
        if self._native_root is None:
            raise RuntimeError("on_run_started must be called before opening a task")
        # Defense in depth: the evaluation parser already rejects
        # single_video=true (Phase 1 has no task-level streaming writer, and
        # per-episode finalize would overwrite video.mp4 each episode).
        if bool(kwargs.get("single_video", False)):
            raise RuntimeError(
                "single_video=true is not supported by the RoboTwin native reporter in Phase 1; "
                "the evaluation parser must reject it before tasks start"
            )
        task_id = int(kwargs.get("task_id", 0))
        task_dir = self._native_root / "tasks" / f"task_{task_id:03d}"
        self._task = {
            "task_id": task_id,
            "suite": str(kwargs.get("suite", "robotwin")),
            "video_enabled": bool(kwargs.get("video_enabled", False)),
            "fps": int(kwargs.get("fps", 10)),
            "camera_name": str(kwargs.get("camera_name", "head_camera")),
            "video_dir": task_dir / "videos",
            "episodes": {},
        }
        self._task["video_dir"].mkdir(parents=True, exist_ok=True)

    def on_episode_reset(self, observations: Mapping[str, Any], *, episode_index: int, **kwargs: Any) -> None:
        del kwargs
        task = self._require_task()
        if not task["video_enabled"]:
            # Never accumulate frames when video is disabled: full-frame RGB
            # buffers would otherwise grow linearly with total episode count.
            return
        task["episodes"][int(episode_index)] = {
            "frames": [self._extract_frame(observations, task["camera_name"])],
            "artifact": None,
        }

    def on_native_step(
        self,
        observations: Mapping[str, Any],
        *,
        episode_index: int,
        result: EpisodeResult | None = None,
        **kwargs: Any,
    ) -> None:
        del result, kwargs
        task = self._require_task()
        if not task["video_enabled"]:
            return
        episode = task["episodes"].get(int(episode_index))
        if episode is None:
            raise RuntimeError(f"episode {episode_index} must be reset before stepping")
        if episode["artifact"] is not None:
            return
        episode["frames"].append(self._extract_frame(observations, task["camera_name"]))

    def on_episode_finished(self, result: EpisodeResult) -> None:
        self.finalize_episode(result)

    def artifact_status(
        self, *, scope: str, task_id: int | None = None, episode_index: int | None = None
    ) -> dict[str, Any]:
        del scope
        if self._task is None or (task_id is not None and task_id != self._task["task_id"]):
            return {"artifacts": [], "failures": []}
        artifacts = [
            {
                "ref": artifact.path,
                "kind": artifact.kind,
                "scope": artifact.scope,
                "required": True,
                "status": "available",
                "metadata": dict(artifact.metadata),
            }
            for artifact in self._artifacts
            if episode_index is None or artifact.episode_id == episode_index
        ]
        return {"artifacts": artifacts, "failures": []}

    def finalize_episode(self, result: EpisodeResult) -> list[NativeArtifact]:
        task = self._require_task()
        episode = task["episodes"].get(result.episode_index)
        if episode is None or episode["artifact"] is not None or not task["video_enabled"]:
            return []

        import imageio.v2 as imageio

        path = task["video_dir"] / f"episode_{result.episode_index:03d}.mp4"
        imageio.mimsave(str(path), episode["frames"], fps=task["fps"], codec="libx264")
        frame_count = len(episode["frames"])
        artifact = NativeArtifact(
            name=path.name,
            kind="video",
            path=self._artifact_path(path),
            mime_type="video/mp4",
            scope="episode",
            suite=task["suite"],
            task_id=task["task_id"],
            episode_id=result.episode_index,
            metadata={
                "fps": task["fps"],
                "camera_name": task["camera_name"],
                "frame_count": frame_count,
            },
        )
        self._artifacts.append(artifact)
        # Release the accumulated frames immediately: artifact metadata is
        # already captured in self._artifacts, so keeping the episode entry
        # would pin ~0.9 MB per frame for the whole run.
        del task["episodes"][result.episode_index]
        return [artifact]

    def finalize_task(self) -> list[NativeArtifact]:
        return list(self._artifacts)

    def finalize(self, summary: RunSummary) -> list[NativeArtifact]:
        del summary
        return list(self._artifacts)

    @staticmethod
    def _extract_frame(observations: Mapping[str, Any], camera_name: str) -> np.ndarray:
        camera_block = observations.get("observation")
        if not isinstance(camera_block, Mapping):
            raise ValueError("native observation is missing the observation mapping")
        camera = camera_block.get(camera_name)
        if not isinstance(camera, Mapping) or not isinstance(camera.get("rgb"), np.ndarray):
            raise ValueError(f"native observation is missing {camera_name}.rgb")
        frame = np.asarray(camera["rgb"], dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"native video frame must be HWC RGB, got {frame.shape}")
        return np.ascontiguousarray(frame)

    def _require_task(self) -> dict[str, Any]:
        if self._task is None:
            raise RuntimeError("on_task_started must be called before recording episodes")
        return self._task

    def _artifact_path(self, path: Path) -> str:
        if self._native_root is None:
            return str(path)
        run_root = self._native_root.parent.parent
        try:
            return str(path.relative_to(run_root))
        except ValueError:
            return str(path)


def create_native_reporter() -> NativeReportExporter:
    return RobotwinNativeReporter()
