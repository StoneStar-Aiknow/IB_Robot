"""按会话采样范围离线导出灰区 PCM WAV。"""

from __future__ import annotations

import os
import uuid
import warnings
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .report import SessionReportData


class AudioCoverageError(ValueError):
    """请求区间未被音频分卷连续覆盖。"""


def extract_interval_from_volumes(
    volumes: Sequence[Mapping[str, object]],
    *,
    session_dir: Path,
    start_sample: int,
    end_sample: int,
    channels: int,
) -> np.ndarray:
    """从一个或多个相邻 WAV 分卷精确截取半开采样区间。"""
    if (
        isinstance(start_sample, bool)
        or isinstance(end_sample, bool)
        or not isinstance(start_sample, int)
        or not isinstance(end_sample, int)
        or start_sample < 0
        or end_sample <= start_sample
    ):
        raise ValueError("start_sample 必须是非负整数且小于 end_sample")
    if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
        raise ValueError("channels 必须是正整数")

    cursor = start_sample
    pieces: list[np.ndarray] = []
    for volume in volumes:
        volume_start = int(volume["start_sample"])
        volume_end = int(volume["end_sample"])
        if volume_end <= cursor or volume_start >= end_sample:
            continue
        if volume_start > cursor:
            raise AudioCoverageError(f"audio coverage gap [{cursor}, {min(volume_start, end_sample)})")

        piece_start = max(cursor, volume_start)
        piece_end = min(end_sample, volume_end)
        path = session_dir / str(volume["path"])
        with wave.open(str(path), "rb") as wav_file:
            if wav_file.getnchannels() != channels or wav_file.getsampwidth() != 2:
                raise ValueError(f"{path}: WAV 格式必须为 {channels}ch PCM_S16LE")
            wav_file.setpos(piece_start - volume_start)
            payload = wav_file.readframes(piece_end - piece_start)
        expected_values = (piece_end - piece_start) * channels
        samples = np.frombuffer(payload, dtype="<i2")
        if samples.size != expected_values:
            raise AudioCoverageError(f"{path}: WAV 数据不足以覆盖 [{piece_start}, {piece_end})")
        pieces.append(samples.reshape(-1, channels).copy())
        cursor = piece_end
        if cursor == end_sample:
            break

    if cursor < end_sample:
        raise AudioCoverageError(f"audio coverage gap [{cursor}, {end_sample})")
    return np.concatenate(pieces, axis=0)


def _write_wav(path: Path, samples: np.ndarray, *, channels: int, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.astype("<i2", copy=False).tobytes())


def extract_gray_audio(data: SessionReportData, *, output_dir: Path, overwrite: bool) -> tuple[Path, ...]:
    """导出各灰区事件的原始 6 通道与增强 4 通道 WAV。"""
    streams = data.manifest.get("streams")
    if not isinstance(streams, Mapping):
        raise ValueError("manifest.streams 必须是 mapping")

    jobs: list[tuple[Path, str, int, int, int, Sequence[Mapping[str, object]], str | None]] = []
    for event in data.gray_events:
        event_index = int(event["event_index"])
        requested_start = int(event["start_sample"])
        requested_end = int(event["end_sample"])
        if requested_start < 0 or requested_end <= requested_start:
            raise ValueError("灰区 start_sample 必须小于 end_sample")
        for stream_name, channels in (("raw6ch", 6), ("enh4ch", 4)):
            stream = streams.get(stream_name)
            if not isinstance(stream, Mapping):
                continue
            volumes = stream.get("volumes")
            if not isinstance(volumes, Sequence):
                raise ValueError(f"manifest.streams.{stream_name}.volumes 必须是 sequence")
            if not volumes:
                warnings.warn(
                    f"{stream_name} volumes 为空；requested=[{requested_start}, "
                    f"{requested_end}) effective=none，跳过导出",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            # 两种音频流使用同一 coverage 规则：只裁剪首尾，内部缺口交给截取器报错。
            actual_start = max(requested_start, int(volumes[0]["start_sample"]))
            actual_end = min(requested_end, int(volumes[-1]["end_sample"]))
            clipping_warning: str | None = None
            if actual_start >= actual_end:
                warnings.warn(
                    f"{stream_name} requested=[{requested_start}, {requested_end}) effective=none，跳过导出",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if (actual_start, actual_end) != (requested_start, requested_end):
                clipping_warning = (
                    f"{stream_name} requested=[{requested_start}, {requested_end}) "
                    f"effective=[{actual_start}, {actual_end})"
                )
            final_path = output_dir / f"gray_{event_index:06d}_{stream_name}.wav"
            jobs.append(
                (
                    final_path,
                    stream_name,
                    channels,
                    actual_start,
                    actual_end,
                    volumes,
                    clipping_warning,
                )
            )

    # 覆盖检查必须早于目录创建和任一写盘，避免部分输出。
    if not overwrite:
        for final_path, *_rest in jobs:
            if final_path.exists():
                raise FileExistsError(f"输出已存在: {final_path}")
    if not jobs:
        return ()

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for final_path, _stream_name, channels, start, end, volumes, clipping_warning in jobs:
        if clipping_warning is not None:
            warnings.warn(clipping_warning, RuntimeWarning, stacklevel=2)
        temporary = output_dir / f".{final_path.name}.tmp.{uuid.uuid4().hex}"
        try:
            samples = extract_interval_from_volumes(
                volumes,
                session_dir=data.session_dir,
                start_sample=start,
                end_sample=end,
                channels=channels,
            )
            _write_wav(temporary, samples, channels=channels, sample_rate=data.sample_rate)
            os.replace(temporary, final_path)
            outputs.append(final_path)
        finally:
            temporary.unlink(missing_ok=True)
    return tuple(outputs)


__all__ = ["AudioCoverageError", "extract_gray_audio", "extract_interval_from_volumes"]
