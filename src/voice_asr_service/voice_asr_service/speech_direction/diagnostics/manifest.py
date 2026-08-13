"""版本化会话 Manifest 及同目录原子更新。"""

from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
_AUDIO_FORMAT = {"encoding": "PCM_S16LE", "sample_width_bytes": 2}
_TERMINAL_STATES = {"completed", "diagnostics_disabled"}


def _utc_now() -> str:
    """返回可排序、带时区的 UTC 时间。"""
    return datetime.now(timezone.utc).isoformat()


class SessionManifest:
    """维护单个高通量维测会话的版本化索引。"""

    def __init__(
        self,
        session_dir: str | os.PathLike[str],
        *,
        session_id: str,
        sample_rate: int,
        rollover_seconds: int,
        save_raw6ch: bool,
        save_enh4ch: bool,
        save_frame_metrics: bool,
        save_gray_events: bool,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self._session_dir = Path(session_dir)
        self._manifest_path = self._session_dir / "manifest.json"
        self._temporary_path = self._session_dir / "manifest.json.tmp"
        self._ownership_path = self._session_dir / ".manifest.owner"
        self._sample_rate = sample_rate

        self._session_dir.mkdir(parents=True, exist_ok=True)
        # 兼容没有 ownership 标记的旧目录，同时防止覆盖已有 Manifest。
        if self._manifest_path.exists():
            raise FileExistsError(f"会话 Manifest 已存在: {self._manifest_path}")
        try:
            owner_fd = os.open(
                self._ownership_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as error:
            raise FileExistsError(f"会话 Manifest 所有权已被认领: {self._manifest_path}") from error
        # ownership 标记永久保留；即使后续初始化失败，也禁止复用该会话目录。
        try:
            os.write(owner_fd, session_id.encode("utf-8"))
            os.fsync(owner_fd)
        finally:
            os.close(owner_fd)

        for relative_dir in ("audio/full", "metrics", "events", "reports"):
            (self._session_dir / relative_dir).mkdir(parents=True, exist_ok=True)

        document: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id,
            "created_at": _utc_now(),
            "closed_at": None,
            "status": "recording",
            "sample_rate": sample_rate,
            "rollover_seconds": rollover_seconds,
            "audio_format": dict(_AUDIO_FORMAT),
            "thresholds": None if thresholds is None else {key: float(value) for key, value in thresholds.items()},
            "streams": {
                "raw6ch": {
                    "enabled": save_raw6ch,
                    "channels": 6,
                    "volumes": [],
                },
                "enh4ch": {
                    "enabled": save_enh4ch,
                    "channels": 4,
                    "volumes": [],
                },
                "frame_metrics": {
                    "enabled": save_frame_metrics,
                    "volumes": [],
                },
            },
            "capture": {
                "frames_captured": 0,
                "ring_capacity_frames": 0,
                "ring_overwrite_events": 0,
                "ring_overwritten_frames": 0,
                "pipeline_gap_count": 0,
                "pipeline_gap_frames": 0,
            },
            "gray_events": {
                "enabled": save_gray_events,
                "path": "events/gray_events.jsonl",
                "count": 0,
            },
            "recorder": {
                "state": "recording",
                "disabled_reason": None,
                "disabled_session_sample": None,
                "disabled_session_seconds": None,
                "dropped_count": 0,
                "raw_ingest": {
                    "packets_accepted": 0,
                    "frames_accepted": 0,
                    "packets_dropped": 0,
                    "frames_dropped": 0,
                },
            },
        }
        self._persist(document)
        self._document = document

    def add_closed_volume(
        self,
        stream: str,
        *,
        path: str,
        index: int,
        start_sample: int,
        end_sample: int,
        frame_count: int,
        wav_format: dict | None = None,
    ) -> None:
        """登记一个已关闭卷，采样范围使用半开区间 ``[start, end)``。"""
        candidate = copy.deepcopy(self._document)
        volume = {
            "path": path,
            "index": index,
            "start_sample": start_sample,
            "end_sample": end_sample,
            # 秒数只能从整数采样位置和会话采样率派生。
            "start_seconds": start_sample / self._sample_rate,
            "end_seconds": end_sample / self._sample_rate,
            "frame_count": frame_count,
        }
        if wav_format is not None:
            volume["wav_format"] = copy.deepcopy(wav_format)
        candidate["streams"][stream]["volumes"].append(volume)
        self._commit(candidate)

    def set_capture_stats(self, stats: dict[str, int]) -> None:
        """记录采集、Ring覆盖和pipeline gap终态统计。"""
        candidate = copy.deepcopy(self._document)
        for key in candidate["capture"]:
            if key in stats:
                candidate["capture"][key] = int(stats[key])
        self._commit(candidate)

    def set_raw_ingest_stats(self, stats: dict[str, int]) -> None:
        """记录采集raw旁路接受和丢弃的包/帧数。"""
        candidate = copy.deepcopy(self._document)
        target = candidate["recorder"]["raw_ingest"]
        for key in target:
            if key in stats:
                target[key] = int(stats[key])
        self._commit(candidate)

    def set_gray_event_count(self, count: int) -> None:
        """立即持久化灰区事件数量；终态点调用，不依赖后续持久化点。

        历史上此处只改内存、交给后续生命周期一并落盘，但中断退出时后续
        complete/mark_disabled 可能被截断，导致 count 永久丢失。故改为即时 _commit。
        """
        candidate = copy.deepcopy(self._document)
        candidate["gray_events"]["count"] = count
        self._commit(candidate)

    def mark_disabled(
        self,
        *,
        reason: str,
        disabled_session_sample: int | None,
        dropped_count: int,
    ) -> None:
        """记录写入故障并把本会话置为不可逆的停用终态。"""
        if self._document["status"] in _TERMINAL_STATES:
            return
        candidate = copy.deepcopy(self._document)
        candidate["status"] = "diagnostics_disabled"
        candidate["closed_at"] = _utc_now()
        candidate["recorder"].update(
            {
                "state": "diagnostics_disabled",
                "disabled_reason": reason,
                "disabled_session_sample": disabled_session_sample,
                "disabled_session_seconds": (
                    None if disabled_session_sample is None else disabled_session_sample / self._sample_rate
                ),
                "dropped_count": dropped_count,
            }
        )
        self._commit(candidate)

    def complete(self, *, dropped_count: int) -> None:
        """幂等地标记会话正常完成，不覆盖任何既有终态。"""
        if self._document["status"] in _TERMINAL_STATES:
            return
        candidate = copy.deepcopy(self._document)
        candidate["status"] = "completed"
        candidate["closed_at"] = _utc_now()
        candidate["recorder"]["state"] = "completed"
        candidate["recorder"]["dropped_count"] = dropped_count
        self._commit(candidate)

    def snapshot(self) -> dict[str, object]:
        """返回不会影响内部状态的 Manifest 快照。"""
        return copy.deepcopy(self._document)

    def _commit(self, candidate: dict[str, Any]) -> None:
        """持久化成功后才发布候选状态，失败时保持内存与磁盘旧状态。"""
        self._persist(candidate)
        self._document = candidate

    def _persist(self, document: dict[str, Any]) -> None:
        """flush 并 fsync 临时文件后，以 os.replace 原子替换正式文件。"""
        with self._temporary_path.open("w", encoding="utf-8") as temporary_file:
            json.dump(
                document,
                temporary_file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(self._temporary_path, self._manifest_path)


__all__ = ["SCHEMA_VERSION", "SessionManifest"]
