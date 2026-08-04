"""连续音频与帧指标的滚动分卷写入器。"""

from __future__ import annotations

import json
import os
import wave
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ClosedVolume:
    """已关闭且可安全登记到 Manifest 的卷。"""

    stream: str
    path: str
    index: int
    start_sample: int
    end_sample: int
    frame_count: int
    wav_format: dict[str, int | str] | None = None


ClosedCallback = Callable[[ClosedVolume], None]


def _require_positive_int(value: object, name: str) -> int:
    """拒绝 bool 和浮点等隐式整数，只接受严格正整数。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} 必须是非 bool 的整数")
    if value <= 0:
        raise ValueError(f"{name} 必须是正数")
    return value


class RollingAudioWriter:
    """按会话采样位置将 float32 多通道音频写为滚动 WAV 卷。"""

    _STREAM_CHANNELS = {"raw6ch": 6, "enh4ch": 4}

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        *,
        stream: str,
        sample_rate: int,
        channels: int,
        rollover_samples: int,
        on_closed: ClosedCallback,
    ) -> None:
        expected_channels = self._STREAM_CHANNELS.get(stream)
        if expected_channels is None:
            raise ValueError(f"不支持的音频流: {stream}")
        if channels != expected_channels:
            raise ValueError(f"{stream} 通道数必须为 {expected_channels}，实际为 {channels}")
        sample_rate = _require_positive_int(sample_rate, "sample_rate")
        rollover_samples = _require_positive_int(rollover_samples, "rollover_samples")

        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._stream = stream
        self._sample_rate = sample_rate
        self._channels = channels
        self._rollover_samples = rollover_samples
        self._on_closed = on_closed
        self._index = 0
        self._wav_file: wave.Wave_write | None = None
        self._path: Path | None = None
        self._start_sample: int | None = None
        self._next_sample: int | None = None
        self._rollover_end: int | None = None
        self._frame_count = 0

    def write(self, *, start_sample: int, samples: np.ndarray) -> None:
        """写入半开采样区间；跨分卷边界时拆分输入块。"""
        # 必须在 gap 判断和任何文件操作前拒绝隐式整数，保护已打开卷不变。
        if isinstance(start_sample, bool) or not isinstance(start_sample, int):
            raise TypeError("start_sample 必须是非 bool 的整数")
        if start_sample < 0:
            raise ValueError("start_sample 不能为负数")
        if not isinstance(samples, np.ndarray) or samples.dtype != np.float32:
            raise TypeError("samples 必须是 float32 numpy.ndarray")
        if samples.ndim != 2 or samples.shape[1] != self._channels:
            raise ValueError(f"samples 必须是 (frames, {self._channels})")
        if samples.shape[0] == 0:
            return

        # 采样位置不连续时立即关闭旧卷，绝不用静音补齐缺口或覆盖倒退区间。
        if self._next_sample is not None and start_sample != self._next_sample:
            self._close_volume()

        offset = 0
        position = start_sample
        while offset < samples.shape[0]:
            if self._wav_file is None:
                self._open_volume(position)
            assert self._rollover_end is not None
            available = self._rollover_end - position
            frame_count = min(available, samples.shape[0] - offset)
            chunk = samples[offset : offset + frame_count]
            # WAV 的 16 位 PCM 使用小端有符号整数；先裁剪可避免 1.0 溢出回绕。
            pcm = np.clip(chunk * 32768.0, -32768, 32767).astype("<i2")
            assert self._wav_file is not None
            self._wav_file.writeframesraw(pcm.tobytes(order="C"))
            self._frame_count += frame_count
            position += frame_count
            offset += frame_count
            self._next_sample = position
            if position == self._rollover_end:
                self._close_volume()

    def close(self) -> None:
        """关闭非空尾卷；空 writer 不产生回调。"""
        self._close_volume()

    def _open_volume(self, start_sample: int) -> None:
        self._index += 1
        self._path = self._output_dir / f"{self._stream}_{self._index:06d}.wav"
        self._start_sample = start_sample
        self._next_sample = start_sample
        # 使用整数采样位置计算固定会话边界，不引入浮点秒累计误差。
        self._rollover_end = (start_sample // self._rollover_samples + 1) * self._rollover_samples
        self._frame_count = 0
        wav_file = wave.open(str(self._path), "wb")  # noqa: SIM115 - 分卷跨 write 持有句柄
        wav_file.setnchannels(self._channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(self._sample_rate)
        self._wav_file = wav_file

    def _close_volume(self) -> None:
        if self._wav_file is None:
            return
        wav_file = self._wav_file
        path = self._path
        start_sample = self._start_sample
        end_sample = self._next_sample
        frame_count = self._frame_count
        self._wav_file = None
        wav_file.close()
        assert path is not None and start_sample is not None and end_sample is not None
        # 仅在文件完整关闭后发布，回调方不会观察到仍在写入的卷。
        self._on_closed(
            ClosedVolume(
                stream=self._stream,
                path=path.name,
                index=self._index,
                start_sample=start_sample,
                end_sample=end_sample,
                frame_count=frame_count,
                wav_format={
                    "encoding": "PCM_S16LE",
                    "channels": self._channels,
                    "sample_rate": self._sample_rate,
                    "sample_width_bytes": 2,
                },
            )
        )
        self._path = None
        self._start_sample = None
        self._next_sample = None
        self._rollover_end = None
        self._frame_count = 0


class RollingMetricsWriter:
    """按整数会话采样边界滚动写入帧指标 JSONL。"""

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        *,
        sample_rate: int,
        rollover_samples: int,
        on_closed: ClosedCallback,
    ) -> None:
        sample_rate = _require_positive_int(sample_rate, "sample_rate")
        rollover_samples = _require_positive_int(rollover_samples, "rollover_samples")
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._sample_rate = sample_rate
        self._rollover_samples = rollover_samples
        self._on_closed = on_closed
        self._index = 0
        self._file = None
        self._path: Path | None = None
        self._bucket: int | None = None
        self._start_sample: int | None = None
        self._end_sample: int | None = None
        self._last_session_sample: int | None = None
        self._frame_count = 0

    def write(self, record: Mapping[str, object]) -> None:
        """写入一条指标，并从 session_sample 唯一派生秒数。"""
        if "session_sample" not in record:
            raise ValueError("指标记录缺少 session_sample")
        session_sample = record["session_sample"]
        if isinstance(session_sample, bool) or not isinstance(session_sample, int):
            raise TypeError("session_sample 必须是整数")
        if session_sample < 0:
            raise ValueError("session_sample 不能为负数")
        # 先校验再做滚动或写盘，拒绝路径不得改变当前卷文件和 metadata。
        if self._last_session_sample is not None and session_sample <= self._last_session_sample:
            raise ValueError("session_sample 必须严格递增")

        bucket = session_sample // self._rollover_samples
        if self._file is not None and bucket != self._bucket:
            self._close_volume()
        if self._file is None:
            self._open_volume(session_sample, bucket)

        # 覆盖调用方可能提供的浮点时间，保证时轴只来自整数 sample SSOT。
        serialized = dict(record)
        serialized["session_seconds"] = session_sample / self._sample_rate
        assert self._file is not None
        self._file.write(json.dumps(serialized, ensure_ascii=False, sort_keys=True) + "\n")
        self._frame_count += 1
        self._end_sample = session_sample + 1
        self._last_session_sample = session_sample

    def close(self) -> None:
        """关闭非空尾卷；空 writer 不产生回调。"""
        self._close_volume()

    def _open_volume(self, start_sample: int, bucket: int) -> None:
        self._index += 1
        self._path = self._output_dir / f"frames_{self._index:06d}.jsonl"
        self._file = self._path.open("w", encoding="utf-8")
        self._bucket = bucket
        self._start_sample = start_sample
        self._end_sample = start_sample
        self._frame_count = 0

    def _close_volume(self) -> None:
        if self._file is None:
            return
        metrics_file = self._file
        path = self._path
        start_sample = self._start_sample
        end_sample = self._end_sample
        frame_count = self._frame_count
        self._file = None
        metrics_file.close()
        assert path is not None and start_sample is not None and end_sample is not None
        # JSONL 同样只在 flush/close 完成后通知 Manifest 层。
        self._on_closed(
            ClosedVolume(
                stream="frame_metrics",
                path=path.name,
                index=self._index,
                start_sample=start_sample,
                end_sample=end_sample,
                frame_count=frame_count,
            )
        )
        self._path = None
        self._bucket = None
        self._start_sample = None
        self._end_sample = None
        self._frame_count = 0


__all__ = ["ClosedVolume", "RollingAudioWriter", "RollingMetricsWriter"]
