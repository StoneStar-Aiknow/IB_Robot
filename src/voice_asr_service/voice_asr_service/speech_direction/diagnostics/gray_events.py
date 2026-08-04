"""实时灰区帧合并与事件摘要落盘。"""

from __future__ import annotations

import json
import math
import os
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path


def _require_non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} 必须是非 bool 的整数")
    if value < 0:
        raise ValueError(f"{name} 不能为负数")
    return value


def _require_positive_int(value: object, name: str) -> int:
    value = _require_non_negative_int(value, name)
    if value == 0:
        raise ValueError(f"{name} 必须是正数")
    return value


def _require_finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} 必须是有限数值")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} 必须是有限数值")
    return result


@dataclass
class _ActiveEvent:
    start_sample: int
    end_sample: int
    frame_count: int = 0
    max_vad_probability: float = 0.0
    max_rms: float = 0.0
    doa_degrees: list[int] = field(default_factory=list)


class GrayEventTracker:
    """按整数采样时轴合并灰区帧，并仅输出事件 JSONL 摘要。"""

    def __init__(
        self,
        session_dir: str | os.PathLike[str],
        *,
        sample_rate: int,
        merge_gap_samples: int,
    ) -> None:
        self._sample_rate = _require_positive_int(sample_rate, "sample_rate")
        self._merge_gap_samples = _require_non_negative_int(merge_gap_samples, "merge_gap_samples")
        self._path = Path(session_dir) / "events" / "gray_events.jsonl"
        self._active: _ActiveEvent | None = None
        self._last_input_end: int | None = None
        self._event_index = 0
        self._closed = False
        self._failed = False

    def observe(
        self,
        *,
        session_sample: int,
        sample_count: int,
        is_gray: bool,
        vad_probability: float,
        rms: float,
        frame_doa_degree: int | None,
    ) -> None:
        """观察一个不重叠帧；非法输入在任何状态变化和写盘前被拒绝。"""
        if self._failed:
            raise RuntimeError("GrayEventTracker 处于不可恢复的失败状态")
        session_sample = _require_non_negative_int(session_sample, "session_sample")
        sample_count = _require_positive_int(sample_count, "sample_count")
        if not isinstance(is_gray, bool):
            raise TypeError("is_gray 必须是 bool")
        vad_probability = _require_finite_number(vad_probability, "vad_probability")
        if not 0.0 <= vad_probability <= 1.0:
            raise ValueError("vad_probability 必须位于 [0, 1]")
        rms = _require_finite_number(rms, "rms")
        if rms < 0.0:
            raise ValueError("rms 不能为负数")
        if frame_doa_degree is not None:
            if isinstance(frame_doa_degree, bool) or not isinstance(frame_doa_degree, int):
                raise TypeError("frame_doa_degree 必须是整数或 None")
            if not 0 <= frame_doa_degree < 360:
                raise ValueError("frame_doa_degree 必须位于 [0, 360)")
        if self._closed:
            raise RuntimeError("GrayEventTracker 已关闭")

        frame_end = session_sample + sample_count
        # 顺序校验必须早于事件闭合，避免坏帧让活动事件被提前写出。
        if self._last_input_end is not None and session_sample < self._last_input_end:
            raise ValueError("输入采样区间必须按顺序且不能重叠")

        if is_gray:
            if self._active is not None and session_sample - self._active.end_sample > self._merge_gap_samples:
                self._close_active("gap_exceeded")
            if self._active is None:
                self._active = _ActiveEvent(
                    start_sample=session_sample,
                    end_sample=frame_end,
                )
            active = self._active
            # 非灰 gap 只用于合并判定，不计入事件帧数和统计摘要。
            active.end_sample = frame_end
            active.frame_count += 1
            active.max_vad_probability = max(active.max_vad_probability, vad_probability)
            active.max_rms = max(active.max_rms, rms)
            if frame_doa_degree is not None:
                active.doa_degrees.append(frame_doa_degree)
        elif self._active is not None and frame_end - self._active.end_sample > self._merge_gap_samples:
            self._close_active("gap_exceeded")

        self._last_input_end = frame_end

    def close(self, *, closed_by: str = "session_stop") -> None:
        """以指定原因闭合活动事件；重复关闭不产生额外记录。"""
        if self._failed:
            raise RuntimeError("GrayEventTracker 处于不可恢复的失败状态")
        if not isinstance(closed_by, str):
            raise TypeError("closed_by 必须是字符串")
        if not closed_by:
            raise ValueError("closed_by 不能为空")
        if self._closed:
            return
        self._close_active(closed_by)
        self._closed = True

    def _close_active(self, closed_by: str) -> None:
        active = self._active
        if active is None:
            return
        event_index = self._event_index + 1
        doa_summary: dict[str, int | float | None]
        if active.doa_degrees:
            doa_summary = {
                "min": min(active.doa_degrees),
                "max": max(active.doa_degrees),
                "mean": sum(active.doa_degrees) / len(active.doa_degrees),
            }
        else:
            doa_summary = {"min": None, "max": None, "mean": None}
        # 秒数永远即时由整数 sample/rate 推导，不保存独立浮点时钟状态。
        record = {
            "event_index": event_index,
            "start_sample": active.start_sample,
            "end_sample": active.end_sample,
            "start_seconds": active.start_sample / self._sample_rate,
            "end_seconds": active.end_sample / self._sample_rate,
            "frame_count": active.frame_count,
            "max_vad_probability": active.max_vad_probability,
            "max_rms": active.max_rms,
            "doa_degree": doa_summary,
            "closed_by": closed_by,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        self._append_line(line)
        # 只有完整写出并同步 JSONL 行后才发布索引并清空活动事件。
        self._event_index = event_index
        self._active = None

    def _append_line(self, line: bytes) -> None:
        """事务式追加一行；失败时回滚到追加前字节偏移。"""
        event_file = self._path.open("a+b")
        event_file.seek(0, os.SEEK_END)
        original_offset = event_file.tell()
        operation_error: BaseException | None = None
        try:
            # 二进制 file.write 允许无异常短写，必须循环直至整行全部提交。
            view = memoryview(line)
            offset = 0
            while offset < len(view):
                written = event_file.write(view[offset:])
                if written is None or written <= 0:
                    raise OSError("JSONL write 短写后未取得进展")
                offset += written
            event_file.flush()
            os.fsync(event_file.fileno())
        except BaseException as error:
            operation_error = error

        if operation_error is not None:
            try:
                # 回滚同样落盘，确保异常返回后磁盘上没有残行可被后续读取。
                event_file.seek(original_offset)
                event_file.truncate(original_offset)
                event_file.flush()
                os.fsync(event_file.fileno())
                event_file.close()
            except BaseException as rollback_error:
                self._failed = True
                with suppress(BaseException):
                    event_file.close()
                raise RuntimeError("灰区事件 JSONL 回滚失败，tracker 已进入不可恢复状态") from rollback_error
            raise operation_error

        try:
            event_file.close()
        except BaseException as close_error:
            # close 失败时也不能承诺数据已提交，重新打开文件回滚追加区间。
            try:
                with self._path.open("r+b") as rollback_file:
                    rollback_file.seek(original_offset)
                    rollback_file.truncate(original_offset)
                    rollback_file.flush()
                    os.fsync(rollback_file.fileno())
            except BaseException as rollback_error:
                self._failed = True
                raise RuntimeError("灰区事件 JSONL 回滚失败，tracker 已进入不可恢复状态") from rollback_error
            raise close_error


__all__ = ["GrayEventTracker"]
