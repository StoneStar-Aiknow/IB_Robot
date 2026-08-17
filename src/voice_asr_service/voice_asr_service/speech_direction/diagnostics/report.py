"""离线维测会话校验及严格双依赖 HTML/PNG 报告生成。

报告不提供 fallback。每次先渲染唯一临时文件，再依次替换最终 HTML/PNG。
同一会话目录不支持并发生成报告；第二次替换失败时可能留下半套新输出。
"""

from __future__ import annotations

import importlib
import json
import math
import os
import uuid
import wave
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TIMING_BREAKDOWN_FIELDS = (
    "fullsubnet_elapsed_ms",
    "silero_vad_elapsed_ms",
    "srp_elapsed_ms",
    "other_elapsed_ms",
)


class ReportTransactionError(RuntimeError):
    """双文件依次提交时无法完整生成报告。"""


class ReportValidationError(ValueError):
    """会话工件不符合离线报告输入契约。"""


@dataclass(frozen=True)
class SessionReportData:
    """经过完整校验、可供后续离线报告阶段消费的会话数据。"""

    manifest: Mapping[str, object]
    session_dir: Path
    sample_rate: int
    x_min_seconds: float
    x_max_seconds: float
    frame_metrics: tuple[Mapping[str, object], ...]
    gray_events: tuple[Mapping[str, object], ...]
    stream_coverages: Mapping[str, tuple[tuple[int, int], ...]]
    rollover_boundaries: tuple[int, ...]
    warnings: tuple[str, ...]


def _fail(location: str, message: str) -> ReportValidationError:
    return ReportValidationError(f"{location}: {message}")


def _require_object(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _fail(location, "必须是 object")
    return value


def _require_keys(value: dict[str, Any], location: str, required: set[str], optional: set[str] = frozenset()) -> None:
    missing = required - value.keys()
    extra = value.keys() - required - optional
    if missing:
        raise _fail(location, f"缺少字段 {sorted(missing)}")
    if extra:
        raise _fail(location, f"包含未知字段 {sorted(extra)}")


def _require_int(value: object, location: str, *, minimum: int = 0) -> int:
    # JSON bool 是 int 的子类，必须显式排除，避免损坏的采样位置混入时轴。
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(location, "必须是非 bool 的整数")
    if value < minimum:
        raise _fail(location, f"必须大于等于 {minimum}")
    return value


def _require_bool(value: object, location: str) -> bool:
    if not isinstance(value, bool):
        raise _fail(location, "必须是 bool")
    return value


def _require_string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(location, "必须是非空字符串")
    return value


def _require_number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _fail(location, "必须是数值")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise _fail(location, f"无法转换为有限数值: {error}") from error
    if not math.isfinite(result):
        raise _fail(location, "必须是有限数值")
    return result


def _require_ranged_number(
    value: object, location: str, *, minimum: float = 0.0, maximum: float | None = None
) -> float:
    result = _require_number(value, location)
    if result < minimum or (maximum is not None and result > maximum):
        raise _fail(location, f"必须位于 [{minimum}, {maximum if maximum is not None else '∞'})")
    return result


def _require_doa(value: object, location: str) -> int | None:
    if value is None:
        return None
    result = _require_int(value, location)
    if result >= 360:
        raise _fail(location, "必须位于 [0, 360)")
    return result


def _require_derived_seconds(value: object, sample: int, sample_rate: int, location: str) -> None:
    seconds = _require_number(value, location)
    if seconds != sample / sample_rate:
        raise _fail(location, f"必须等于 sample/rate ({sample}/{sample_rate})")


def _resolve_artifact(session_dir: Path, raw_path: object, location: str) -> Path:
    relative = _require_string(raw_path, location)
    lexical = Path(relative)
    if lexical.is_absolute() or ".." in lexical.parts:
        raise _fail(location, f"词法路径禁止绝对路径或 '..': {relative}")
    try:
        candidate = (session_dir / lexical).resolve(strict=True)
        candidate.relative_to(session_dir)
        if not candidate.is_file():
            raise _fail(location, f"不是文件: {relative}")
    except ReportValidationError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise _fail(location, f"路径不存在、越界或解析失败: {relative}: {error}") from error
    return candidate


def _artifact_identity(path: Path, location: str) -> tuple[int, int]:
    try:
        stat = path.stat()
    except OSError as error:
        raise _fail(location, f"读取文件 identity 失败: {error}") from error
    return stat.st_dev, stat.st_ino


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    try:
        # 逐行消费，避免大型维测 JSONL 被一次性读入临时字符串。
        with path.open("r", encoding="utf-8") as jsonl_file:
            for line_number, line in enumerate(jsonl_file, 1):
                if not line.strip():
                    raise _fail(f"{path}:line {line_number}", "空行不是 JSON object")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise _fail(f"{path}:line {line_number}", f"JSON 解析失败: {error.msg}") from error
                if not isinstance(record, dict):
                    raise _fail(f"{path}:line {line_number}", "每行必须是 object")
                records.append(record)
    except ReportValidationError:
        raise
    except (OSError, UnicodeError) as error:
        raise _fail(str(path), f"读取失败: {error}") from error
    return tuple(records)


def _validate_volume(
    volume_value: object,
    *,
    location: str,
    expected_index: int,
    sample_rate: int,
    session_dir: Path,
) -> tuple[dict[str, Any], Path, int, int, int]:
    volume = _require_object(volume_value, location)
    _require_keys(
        volume,
        location,
        {
            "path",
            "index",
            "start_sample",
            "end_sample",
            "start_seconds",
            "end_seconds",
            "frame_count",
        },
        {"wav_format"},
    )
    index = _require_int(volume["index"], f"{location}.index", minimum=1)
    if index != expected_index:
        raise _fail(
            f"{location}.index",
            f"volume order 顺序要求 index 连续且等于 {expected_index}",
        )
    start = _require_int(volume["start_sample"], f"{location}.start_sample")
    end = _require_int(volume["end_sample"], f"{location}.end_sample")
    if end <= start:
        raise _fail(location, "end_sample 必须大于 start_sample")
    frame_count = _require_int(volume["frame_count"], f"{location}.frame_count")
    _require_derived_seconds(volume["start_seconds"], start, sample_rate, f"{location}.start_seconds")
    _require_derived_seconds(volume["end_seconds"], end, sample_rate, f"{location}.end_seconds")
    path = _resolve_artifact(session_dir, volume["path"], f"{location}.path")
    return volume, path, start, end, frame_count


def _validate_wav(
    path: Path,
    volume: dict[str, Any],
    *,
    stream: str,
    channels: int,
    sample_rate: int,
    frame_count: int,
    location: str,
) -> None:
    wav_format = _require_object(volume.get("wav_format"), f"{location}.wav_format")
    _require_keys(
        wav_format,
        f"{location}.wav_format",
        {"encoding", "channels", "sample_rate", "sample_width_bytes"},
    )
    if wav_format["encoding"] != "PCM_S16LE":
        raise _fail(f"{location}.wav_format.encoding", "必须是 PCM_S16LE")
    manifest_channels = _require_int(wav_format["channels"], f"{location}.wav_format.channels", minimum=1)
    manifest_rate = _require_int(wav_format["sample_rate"], f"{location}.wav_format.sample_rate", minimum=1)
    manifest_width = _require_int(
        wav_format["sample_width_bytes"], f"{location}.wav_format.sample_width_bytes", minimum=1
    )
    if (manifest_channels, manifest_rate, manifest_width) != (channels, sample_rate, 2):
        raise _fail(f"{location}.wav_format", f"{stream} WAV 声明格式不匹配")
    try:
        with wave.open(str(path), "rb") as wav_file:
            actual = (
                wav_file.getnchannels(),
                wav_file.getframerate(),
                wav_file.getsampwidth(),
                wav_file.getnframes(),
            )
            payload = wav_file.readframes(wav_file.getnframes())
    except (OSError, EOFError, wave.Error) as error:
        raise _fail(str(path), f"WAV 读取失败: {error}") from error
    labels = ("channels 通道", "sample rate 采样率", "sample width 采样宽度", "frame count 帧数")
    expected = (channels, sample_rate, 2, frame_count)
    for label, actual_value, expected_value in zip(labels, actual, expected, strict=True):
        if actual_value != expected_value:
            raise _fail(str(path), f"{label}={actual_value}，期望 {expected_value}")
    expected_bytes = frame_count * channels * 2
    if len(payload) != expected_bytes:
        raise _fail(
            str(path),
            f"WAV payload/帧不完整: {len(payload)} bytes，期望 {expected_bytes}",
        )


def _validate_metric_records(
    path: Path,
    records: tuple[dict[str, Any], ...],
    *,
    sample_rate: int,
    start: int,
    end: int,
    frame_count: int,
) -> None:
    if len(records) != frame_count:
        raise _fail(str(path), f"frame_count={frame_count}，实际行数={len(records)}")
    previous: int | None = None
    required = {
        "session_sample",
        "sample_count",
        "segment_seq",
        "vad_probability",
        "is_speech",
        "is_gray",
        "rms",
        "frame_doa_degree",
        "segment_doa_degree",
        "inference_elapsed_ms",
        "session_seconds",
    }
    for line_number, record in enumerate(records, 1):
        location = f"{path}:line {line_number}"
        _require_keys(
            record,
            location,
            required,
            optional=set(_TIMING_BREAKDOWN_FIELDS)
            | {
                "model_batch_samples",
                "processing_tick_samples",
                "srp_hop_samples",
                "stage_executed",
                "vad_frame_samples",
            },
        )
        sample = _require_int(record["session_sample"], f"{location}.session_sample")
        _require_int(record["sample_count"], f"{location}.sample_count", minimum=1)
        _require_int(record["segment_seq"], f"{location}.segment_seq")
        _require_ranged_number(record["vad_probability"], f"{location}.vad_probability", maximum=1.0)
        _require_bool(record["is_speech"], f"{location}.is_speech")
        _require_bool(record["is_gray"], f"{location}.is_gray")
        _require_ranged_number(record["rms"], f"{location}.rms")
        _require_doa(record["frame_doa_degree"], f"{location}.frame_doa_degree")
        _require_doa(record["segment_doa_degree"], f"{location}.segment_doa_degree")
        total_elapsed = _require_ranged_number(record["inference_elapsed_ms"], f"{location}.inference_elapsed_ms")
        # 历史会话可完全不含分项；新会话必须四项齐全，禁止部分字段造成误判。
        present_breakdown = [field for field in _TIMING_BREAKDOWN_FIELDS if field in record]
        if present_breakdown and len(present_breakdown) != len(_TIMING_BREAKDOWN_FIELDS):
            missing_breakdown = sorted(set(_TIMING_BREAKDOWN_FIELDS) - record.keys())
            raise _fail(location, f"耗时分项字段必须全有或全无，缺少 {missing_breakdown}")
        if present_breakdown:
            breakdown = [
                _require_ranged_number(record[field], f"{location}.{field}") for field in _TIMING_BREAKDOWN_FIELDS
            ]
            if not math.isclose(sum(breakdown), total_elapsed, rel_tol=1e-6, abs_tol=1e-6):
                raise _fail(location, "耗时分项之和必须等于 inference_elapsed_ms")
        _require_derived_seconds(record["session_seconds"], sample, sample_rate, f"{location}.session_seconds")
        if previous is not None and sample <= previous:
            raise _fail(f"{location}.session_sample", "必须严格递增")
        if not start <= sample < end:
            raise _fail(f"{location}.session_sample", f"不在卷区间 [{start}, {end})")
        previous = sample
    if records and records[0]["session_sample"] != start:
        raise _fail(f"{path}:line 1", "首行 session_sample 与卷 start_sample 不一致")
    if records and records[-1]["session_sample"] + 1 != end:
        raise _fail(f"{path}:line {len(records)}", "末行 session_sample 与卷 end_sample 不一致")


def _validate_gray_records(path: Path, records: tuple[dict[str, Any], ...], *, sample_rate: int) -> None:
    required = {
        "event_index",
        "start_sample",
        "end_sample",
        "start_seconds",
        "end_seconds",
        "frame_count",
        "max_vad_probability",
        "max_rms",
        "doa_degree",
        "closed_by",
    }
    previous_end: int | None = None
    for line_number, record in enumerate(records, 1):
        location = f"{path}:line {line_number}"
        _require_keys(record, location, required)
        event_index = _require_int(record["event_index"], f"{location}.event_index", minimum=1)
        if event_index != line_number:
            raise _fail(f"{location}.event_index", f"必须从 1 连续，期望 {line_number}")
        for prefix in ("start", "end"):
            sample_key = f"{prefix}_sample"
            seconds_key = f"{prefix}_seconds"
            sample = _require_int(record[sample_key], f"{location}.{sample_key}")
            _require_derived_seconds(record[seconds_key], sample, sample_rate, f"{location}.{seconds_key}")
        if record["end_sample"] <= record["start_sample"]:
            raise _fail(location, "end_sample 必须大于 start_sample")
        if previous_end is not None and record["start_sample"] < previous_end:
            raise _fail(location, "gray event 时间 order overlap 重叠")
        previous_end = record["end_sample"]
        _require_int(record["frame_count"], f"{location}.frame_count", minimum=1)
        _require_ranged_number(record["max_vad_probability"], f"{location}.max_vad_probability", maximum=1.0)
        _require_ranged_number(record["max_rms"], f"{location}.max_rms")
        _require_string(record["closed_by"], f"{location}.closed_by")
        doa = _require_object(record["doa_degree"], f"{location}.doa_degree")
        _require_keys(doa, f"{location}.doa_degree", {"min", "max", "mean"})
        values = (doa["min"], doa["max"], doa["mean"])
        if all(value is None for value in values):
            continue
        if any(value is None for value in values):
            raise _fail(f"{location}.doa_degree", "min/max/mean 必须同时为 null 或数值")
        minimum = _require_doa(doa["min"], f"{location}.doa_degree.min")
        maximum = _require_doa(doa["max"], f"{location}.doa_degree.max")
        mean = _require_ranged_number(doa["mean"], f"{location}.doa_degree.mean", maximum=359.999999999)
        assert minimum is not None and maximum is not None
        if not minimum <= mean <= maximum:
            raise _fail(f"{location}.doa_degree", "必须满足 min <= mean <= max")


def load_session_report_data(session_dir: str | Path) -> SessionReportData:
    """加载并严格校验 Manifest 指向的全部离线维测工件。"""
    try:
        root = Path(session_dir).resolve(strict=True)
        if not root.is_dir():
            raise _fail(str(root), "会话目录不存在")
        manifest_path = root / "manifest.json"
        # 本 loader 面向 recorder 停止后的本机离线快照；不支持生成期间并发替换工件。
        if manifest_path.is_symlink():
            raise _fail(str(manifest_path), "manifest.json 禁止 symlink")
        resolved_manifest = manifest_path.resolve(strict=True)
        resolved_manifest.relative_to(root)
        if resolved_manifest != manifest_path:
            raise _fail(str(manifest_path), "manifest.json resolve 越界")
    except ReportValidationError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise _fail(str(session_dir), f"会话或 manifest 路径解析失败: {error}") from error
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise _fail(str(manifest_path), f"读取或解析失败: {error}") from error
    manifest = _require_object(manifest, str(manifest_path))
    _require_keys(
        manifest,
        str(manifest_path),
        {
            "schema_version",
            "session_id",
            "created_at",
            "closed_at",
            "status",
            "sample_rate",
            "rollover_seconds",
            "audio_format",
            "streams",
            "gray_events",
            "recorder",
        },
        optional={"capture", "timing_contract", "thresholds"},
    )
    schema_version = _require_int(manifest["schema_version"], f"{manifest_path}.schema_version", minimum=1)
    if schema_version != 1:
        raise _fail(f"{manifest_path}.schema_version", f"不支持版本 {schema_version}")
    _require_string(manifest["session_id"], f"{manifest_path}.session_id")
    _require_string(manifest["created_at"], f"{manifest_path}.created_at")
    status = _require_string(manifest["status"], f"{manifest_path}.status")
    if status not in {"recording", "completed", "diagnostics_disabled"}:
        raise _fail(f"{manifest_path}.status", f"未知状态 {status}")
    if status == "recording":
        if manifest["closed_at"] is not None:
            raise _fail(f"{manifest_path}.closed_at", "recording 状态必须为 null")
    elif manifest["closed_at"] is None:
        raise _fail(f"{manifest_path}.closed_at", "终态必须有关闭时间")
    else:
        _require_string(manifest["closed_at"], f"{manifest_path}.closed_at")
    sample_rate = _require_int(manifest["sample_rate"], f"{manifest_path}.sample_rate", minimum=1)
    rollover_seconds = _require_int(manifest["rollover_seconds"], f"{manifest_path}.rollover_seconds", minimum=1)
    audio_format = _require_object(manifest["audio_format"], f"{manifest_path}.audio_format")
    _require_keys(audio_format, f"{manifest_path}.audio_format", {"encoding", "sample_width_bytes"})
    if audio_format["encoding"] != "PCM_S16LE":
        raise _fail(f"{manifest_path}.audio_format.encoding", "必须是 PCM_S16LE")
    if (
        _require_int(audio_format["sample_width_bytes"], f"{manifest_path}.audio_format.sample_width_bytes", minimum=1)
        != 2
    ):
        raise _fail(f"{manifest_path}.audio_format.sample_width_bytes", "必须是 2")

    streams = _require_object(manifest["streams"], f"{manifest_path}.streams")
    _require_keys(streams, f"{manifest_path}.streams", {"raw6ch", "enh4ch", "frame_metrics"})
    all_metrics: list[dict[str, Any]] = []
    coverages: dict[str, tuple[tuple[int, int], ...]] = {}
    max_end = 0
    warnings: list[str] = []
    artifact_identities: dict[tuple[int, int], str] = {}
    for stream, expected_channels in (("raw6ch", 6), ("enh4ch", 4), ("frame_metrics", None)):
        stream_location = f"{manifest_path}.streams.{stream}"
        stream_data = _require_object(streams[stream], stream_location)
        required = {"enabled", "volumes"} | ({"channels"} if expected_channels is not None else set())
        _require_keys(stream_data, stream_location, required)
        enabled = _require_bool(stream_data["enabled"], f"{stream_location}.enabled")
        if expected_channels is not None:
            channels = _require_int(stream_data["channels"], f"{stream_location}.channels", minimum=1)
            if channels != expected_channels:
                raise _fail(f"{stream_location}.channels", f"必须是 {expected_channels}")
        volumes = stream_data["volumes"]
        if not isinstance(volumes, list):
            raise _fail(f"{stream_location}.volumes", "必须是 array")
        if not enabled and volumes:
            raise _fail(f"{stream_location}.enabled", "false 时 volumes 必须为空")
        if stream == "frame_metrics" and not volumes:
            # recording 初期或首条指标写入前故障时，空指标流仍是合法离线快照。
            warnings.append("无帧指标，报告数据不足；HTML/PNG 报告生成阶段将拒绝生成")
        intervals: list[tuple[int, int]] = []
        previous_end: int | None = None
        for offset, value in enumerate(volumes):
            location = f"{stream_location}.volumes[{offset}]"
            volume, path, start, end, frame_count = _validate_volume(
                value,
                location=location,
                expected_index=offset + 1,
                sample_rate=sample_rate,
                session_dir=root,
            )
            identity = _artifact_identity(path, f"{location}.path")
            if identity in artifact_identities:
                raise _fail(
                    f"{location}.path",
                    f"duplicate/重复工件引用，已由 {artifact_identities[identity]} 引用",
                )
            artifact_identities[identity] = f"{location}.path"
            if previous_end is not None:
                if start < previous_end:
                    raise _fail(location, f"volume order overlap 重叠: {start} < {previous_end}")
                if start > previous_end and expected_channels is not None:
                    warnings.append(f"{stream} coverage gap [{previous_end}, {start})")
            previous_end = end
            max_end = max(max_end, end)
            if expected_channels is not None:
                if frame_count != end - start:
                    raise _fail(f"{location}.frame_count", "音频帧数必须等于 end_sample-start_sample")
                _validate_wav(
                    path,
                    volume,
                    stream=stream,
                    channels=expected_channels,
                    sample_rate=sample_rate,
                    frame_count=frame_count,
                    location=location,
                )
                intervals.append((start, end))
            else:
                records = _read_jsonl(path)
                _validate_metric_records(
                    path,
                    records,
                    sample_rate=sample_rate,
                    start=start,
                    end=end,
                    frame_count=frame_count,
                )
                all_metrics.extend(records)
        if expected_channels is not None:
            coverages[stream] = tuple(intervals)

    gray_location = f"{manifest_path}.gray_events"
    gray_data = _require_object(manifest["gray_events"], gray_location)
    _require_keys(gray_data, gray_location, {"enabled", "path", "count"})
    gray_enabled = _require_bool(gray_data["enabled"], f"{gray_location}.enabled")
    gray_count = _require_int(gray_data["count"], f"{gray_location}.count")
    gray_path_text = _require_string(gray_data["path"], f"{gray_location}.path")
    gray_lexical = Path(gray_path_text)
    if gray_lexical.is_absolute() or ".." in gray_lexical.parts:
        raise _fail(f"{gray_location}.path", "词法路径禁止绝对路径或 '..'")
    try:
        gray_candidate = (root / gray_lexical).resolve()
        gray_candidate.relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise _fail(f"{gray_location}.path", f"路径越界或解析失败: {error}") from error
    if not gray_enabled:
        if gray_count != 0 or gray_candidate.exists():
            raise _fail(f"{gray_location}.enabled", "false 时 count 必须为 0 且文件不得存在")
        gray_records: tuple[dict[str, Any], ...] = ()
    else:
        if gray_count == 0 and not gray_candidate.exists():
            gray_records = ()
        else:
            gray_path = _resolve_artifact(root, gray_data["path"], f"{gray_location}.path")
            gray_identity = _artifact_identity(gray_path, f"{gray_location}.path")
            if gray_identity in artifact_identities:
                raise _fail(f"{gray_location}.path", "duplicate/重复工件引用")
            artifact_identities[gray_identity] = f"{gray_location}.path"
            gray_records = _read_jsonl(gray_path)
            if len(gray_records) != gray_count:
                raise _fail(str(gray_path), f"count={gray_count}，实际行数={len(gray_records)}")
            _validate_gray_records(gray_path, gray_records, sample_rate=sample_rate)
    for record in gray_records:
        max_end = max(max_end, int(record["end_sample"]))

    recorder_location = f"{manifest_path}.recorder"
    recorder = _require_object(manifest["recorder"], recorder_location)
    _require_keys(
        recorder,
        recorder_location,
        {"state", "disabled_reason", "disabled_session_sample", "disabled_session_seconds", "dropped_count"},
        optional={"raw_ingest"},
    )
    recorder_state = _require_string(recorder["state"], f"{recorder_location}.state")
    if recorder_state != status:
        raise _fail(f"{recorder_location}.state", "必须与 status 一致")
    dropped_count = _require_int(recorder["dropped_count"], f"{recorder_location}.dropped_count")
    disabled_sample = recorder["disabled_session_sample"]
    disabled_seconds = recorder["disabled_session_seconds"]
    if disabled_sample is not None:
        sample = _require_int(disabled_sample, f"{recorder_location}.disabled_session_sample")
        _require_derived_seconds(disabled_seconds, sample, sample_rate, f"{recorder_location}.disabled_session_seconds")
    elif disabled_seconds is not None:
        raise _fail(f"{recorder_location}.disabled_session_seconds", "sample 为空时必须为 null")
    disabled_reason = recorder["disabled_reason"]
    if disabled_reason is not None and not isinstance(disabled_reason, str):
        raise _fail(f"{recorder_location}.disabled_reason", "必须是字符串或 null")
    if status == "diagnostics_disabled":
        _require_string(disabled_reason, f"{recorder_location}.disabled_reason")
        if (disabled_sample is None) != (disabled_seconds is None):
            raise _fail(recorder_location, "disabled sample/seconds 必须同时存在或同时为 null")
    elif disabled_reason is not None or disabled_sample is not None or disabled_seconds is not None:
        raise _fail(recorder_location, "非停用状态的 disabled 字段必须全部为 null")
    if status == "recording":
        warnings.append("session status is recording; artifacts may be incomplete")
    if status == "diagnostics_disabled":
        warnings.append(f"session diagnostics_disabled: {recorder['disabled_reason'] or 'unknown reason'}")
    if dropped_count:
        warnings.append(f"recorder dropped_count={dropped_count}")

    rollover_samples = sample_rate * rollover_seconds
    boundaries = tuple(range(rollover_samples, max_end, rollover_samples))
    return SessionReportData(
        manifest=manifest,
        session_dir=root,
        sample_rate=sample_rate,
        x_min_seconds=0.0,
        x_max_seconds=max_end / sample_rate,
        frame_metrics=tuple(all_metrics),
        gray_events=gray_records,
        stream_coverages=coverages,
        rollover_boundaries=boundaries,
        warnings=tuple(warnings),
    )


def require_report_dependencies() -> None:
    """同时检查 HTML 与 PNG 渲染依赖，不允许生成降级或半套报告。"""
    missing: list[str] = []
    for dependency in ("plotly", "matplotlib"):
        try:
            importlib.import_module(dependency)
        except ImportError:
            missing.append(dependency)
    if missing:
        raise ImportError(f"缺少离线报告依赖: {', '.join(missing)}")


def _metrics_coverages(data: SessionReportData) -> tuple[tuple[int, int], ...]:
    streams = data.manifest.get("streams", {})
    metrics = streams.get("frame_metrics", {}) if isinstance(streams, Mapping) else {}
    volumes = metrics.get("volumes", ()) if isinstance(metrics, Mapping) else ()
    return tuple((int(volume["start_sample"]), int(volume["end_sample"])) for volume in volumes)


def _report_title(data: SessionReportData) -> str:
    status = data.manifest.get("status", "unknown")
    warning_text = " | ".join(data.warnings) if data.warnings else "none"
    return f"Speech Direction Diagnostics | status={status} | warnings={warning_text}"


def _coverage_rows(data: SessionReportData):
    return (
        ("raw6ch", data.stream_coverages.get("raw6ch", ()), 3.0, "blue"),
        ("enh4ch", data.stream_coverages.get("enh4ch", ()), 2.0, "green"),
        ("metrics", _metrics_coverages(data), 1.0, "orange"),
    )


def _timing_breakdown_available(data: SessionReportData) -> bool:
    """仅当每条指标都具有完整分项时绘制分项曲线。"""
    return bool(data.frame_metrics) and all(
        all(field in item for field in _TIMING_BREAKDOWN_FIELDS) for item in data.frame_metrics
    )


def _report_thresholds(data: SessionReportData) -> Mapping[str, float]:
    """优先读取会话快照，否则读取当前 SpeechDirectionConfig，禁止 renderer 写死门限。"""
    thresholds = data.manifest.get("thresholds")
    if isinstance(thresholds, Mapping):
        return {
            str(key): float(value)
            for key, value in thresholds.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
    # 旧会话尚未记录 thresholds；这里复用正式配置对象，避免复制默认常量。
    try:
        from ..config import SpeechDirectionConfig

        gray_region = SpeechDirectionConfig().gray_region
        return {
            "rms_threshold": float(gray_region.rms_threshold),
            "vad_threshold": float(gray_region.vad_threshold),
            "seg_max_rms_threshold": float(gray_region.seg_max_rms_threshold),
        }
    except (ImportError, AttributeError, TypeError, ValueError):
        return {}


def _build_matplotlib_figure(data: SessionReportData):
    """构造五行严格共享 X 轴的 Matplotlib figure，供 PNG 与结构测试共用。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
    times = [float(item["session_seconds"]) for item in data.frame_metrics]
    # RMS 与 VAD 的数值量纲不同，分别绘制以避免共轴缩放掩盖波动。
    axes[0].plot(times, [float(item["rms"]) for item in data.frame_metrics], label="RMS")
    axes[0].set_ylabel("RMS")
    thresholds = _report_thresholds(data)
    if "rms_threshold" in thresholds:
        axes[0].axhline(thresholds["rms_threshold"], color="tab:red", linestyle="--", label="RMS threshold")
    axes[1].plot(
        times,
        [float(item["vad_probability"]) for item in data.frame_metrics],
        label="VAD probability",
    )
    axes[1].set_ylabel("VAD probability")
    axes[1].set_ylim(-0.05, 1.05)
    if "vad_threshold" in thresholds:
        axes[1].axhline(thresholds["vad_threshold"], color="tab:red", linestyle="--", label="VAD threshold")

    frame_points = [
        (float(item["session_seconds"]), item["frame_doa_degree"])
        for item in data.frame_metrics
        if item["frame_doa_degree"] is not None
    ]
    segment_points = [
        (float(item["session_seconds"]), item["segment_doa_degree"])
        for item in data.frame_metrics
        if item["segment_doa_degree"] is not None
    ]
    if frame_points:
        axes[2].scatter(*zip(*frame_points, strict=True), s=14, label="frame DOA")
    if segment_points:
        axes[2].scatter(*zip(*segment_points, strict=True), s=70, marker="*", label="segment DOA")
    axes[2].set_ylabel("DOA (deg)")
    axes[2].set_ylim(-10, 370)

    axes[3].plot(
        times,
        [float(item["inference_elapsed_ms"]) for item in data.frame_metrics],
        label="total",
        linewidth=2.0,
    )
    if _timing_breakdown_available(data):
        for field, label in (
            ("fullsubnet_elapsed_ms", "FullSubNet"),
            ("silero_vad_elapsed_ms", "Silero VAD"),
            ("srp_elapsed_ms", "SRP"),
            ("other_elapsed_ms", "other"),
        ):
            axes[3].plot(times, [float(item[field]) for item in data.frame_metrics], label=label)
    axes[3].set_ylabel("latency (ms)")

    # coverage 逐卷单独绘制，真实缺口不会被折线跨接或补齐。
    for stream, intervals, y_value, color in _coverage_rows(data):
        for index, (start, end) in enumerate(intervals, 1):
            axes[4].hlines(
                y_value,
                start / data.sample_rate,
                end / data.sample_rate,
                linewidth=8,
                color=color,
                label=stream if index == 1 else None,
            )
    axes[4].set_yticks((1.0, 2.0, 3.0), labels=("metrics", "enh4ch", "raw6ch"))
    axes[4].set_ylabel("coverage")
    axes[4].set_xlabel("session time (s)")

    for axis in axes:
        # 灰区与分卷边界必须逐行落图，不能依赖视觉上跨 subplot 的全局装饰。
        for event in data.gray_events:
            axis.axvspan(
                int(event["start_sample"]) / data.sample_rate,
                int(event["end_sample"]) / data.sample_rate,
                color="lightgray",
                alpha=0.35,
            )
        for boundary in data.rollover_boundaries:
            axis.axvline(boundary / data.sample_rate, color="gray", linestyle="--", alpha=0.7)
        axis.set_xlim(data.x_min_seconds, data.x_max_seconds)
        axis.grid(True, alpha=0.25)
        if axis is not axes[4]:
            axis.legend(loc="upper right")
    fig.suptitle(_report_title(data), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _build_plotly_figure(data: SessionReportData):
    """构造五行严格匹配 X 轴的 Plotly figure，供 HTML 与结构测试共用。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("RMS", "VAD probability", "Frame / Segment DOA", "Inference latency", "Coverage"),
        vertical_spacing=0.05,
    )
    times = [float(item["session_seconds"]) for item in data.frame_metrics]
    thresholds = _report_thresholds(data)
    fig.add_trace(
        go.Scatter(x=times, y=[item["rms"] for item in data.frame_metrics], mode="lines", name="RMS"), row=1, col=1
    )
    if "rms_threshold" in thresholds:
        fig.add_hline(
            y=thresholds["rms_threshold"],
            line={"color": "red", "dash": "dash"},
            annotation_text="RMS threshold",
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=times, y=[item["vad_probability"] for item in data.frame_metrics], mode="lines", name="VAD probability"
        ),
        row=2,
        col=1,
    )
    if "vad_threshold" in thresholds:
        fig.add_hline(
            y=thresholds["vad_threshold"],
            line={"color": "red", "dash": "dash"},
            annotation_text="VAD threshold",
            row=2,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=times, y=[item["frame_doa_degree"] for item in data.frame_metrics], mode="markers", name="frame DOA"
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=[item["segment_doa_degree"] for item in data.frame_metrics],
            mode="markers",
            marker={"symbol": "star", "size": 10},
            name="segment DOA",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=[item["inference_elapsed_ms"] for item in data.frame_metrics],
            mode="lines",
            line={"width": 3},
            name="total",
        ),
        row=4,
        col=1,
    )
    if _timing_breakdown_available(data):
        for field, label in (
            ("fullsubnet_elapsed_ms", "FullSubNet"),
            ("silero_vad_elapsed_ms", "Silero VAD"),
            ("srp_elapsed_ms", "SRP"),
            ("other_elapsed_ms", "other"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=[item[field] for item in data.frame_metrics],
                    mode="lines",
                    name=label,
                ),
                row=4,
                col=1,
            )
    for stream, intervals, y_value, color in _coverage_rows(data):
        for index, (start, end) in enumerate(intervals, 1):
            fig.add_trace(
                go.Scatter(
                    x=(start / data.sample_rate, end / data.sample_rate),
                    y=(y_value, y_value),
                    mode="lines",
                    line={"width": 10, "color": color},
                    name=f"{stream} #{index}",
                    showlegend=index == 1,
                ),
                row=5,
                col=1,
            )

    for row in range(1, 6):
        for event in data.gray_events:
            fig.add_vrect(
                x0=int(event["start_sample"]) / data.sample_rate,
                x1=int(event["end_sample"]) / data.sample_rate,
                fillcolor="lightgray",
                opacity=0.35,
                line_width=0,
                row=row,
                col=1,
            )
        for boundary in data.rollover_boundaries:
            fig.add_vline(
                x=boundary / data.sample_rate,
                line={"color": "gray", "dash": "dash"},
                row=row,
                col=1,
            )
    # shared_xaxes 负责交互联动；显式 matches/range 防止 Plotly 自动范围造成细微偏差。
    fig.layout.xaxis.update(matches=None)
    for axis_name in ("xaxis", "xaxis2", "xaxis3", "xaxis4", "xaxis5"):
        fig.layout[axis_name].update(range=(data.x_min_seconds, data.x_max_seconds))
    for axis_name in ("xaxis2", "xaxis3", "xaxis4", "xaxis5"):
        fig.layout[axis_name].update(matches="x")
    fig.update_yaxes(range=(-0.05, 1.05), row=2, col=1)
    fig.update_yaxes(range=(-10, 370), row=3, col=1)
    fig.update_yaxes(tickvals=(1, 2, 3), ticktext=("metrics", "enh4ch", "raw6ch"), row=5, col=1)
    fig.update_xaxes(title_text="session time (s)", row=5, col=1)
    fig.update_layout(title=_report_title(data), height=1250)
    return fig


def _write_plotly_html(data: SessionReportData, path: Path) -> None:
    _build_plotly_figure(data).write_html(path, include_plotlyjs=True, full_html=True)


def _write_matplotlib_png(data: SessionReportData, path: Path) -> None:
    import matplotlib.pyplot as plt

    figure = _build_matplotlib_figure(data)
    try:
        figure.savefig(path, dpi=120, format="png")
    finally:
        plt.close(figure)


class DiagnosticsReport:
    """消费已校验数据生成 HTML 与 PNG；同一会话目录不得并发执行。"""

    def __init__(self, session_dir: str | Path) -> None:
        self.session_dir = Path(session_dir)

    def generate(self, *, overwrite: bool = False, data: SessionReportData | None = None) -> dict[str, Path]:
        if data is None:
            require_report_dependencies()
            data = load_session_report_data(self.session_dir)
        else:
            # 复用调用方已验证的同一快照，但禁止把其他会话的数据写入当前报告目录。
            if self.session_dir.resolve() != data.session_dir.resolve():
                raise ReportValidationError("已验证数据的 session_dir 与报告实例 session_dir 不一致")
        if not data.frame_metrics:
            raise ReportValidationError("指标数据不足：frame_metrics 为空，无法渲染 HTML/PNG")

        reports_dir = data.session_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        final_paths = {
            "html": reports_dir / "doa_curves.html",
            "png": reports_dir / "doa_curves.png",
        }
        if not overwrite:
            existing = [path for path in final_paths.values() if path.exists()]
            if existing:
                raise FileExistsError(f"报告已存在: {existing[0]}")

        # 每次调用使用独立 UUID 临时文件；约定调用方不得并发生成同一会话报告。
        temporary_id = uuid.uuid4().hex
        temporary_paths = {
            "html": reports_dir / f".doa_curves.tmp.{temporary_id}.html",
            "png": reports_dir / f".doa_curves.tmp.{temporary_id}.png",
        }
        try:
            _write_plotly_html(data, temporary_paths["html"])
            _write_matplotlib_png(data, temporary_paths["png"])
            os.replace(temporary_paths["html"], final_paths["html"])
            try:
                os.replace(temporary_paths["png"], final_paths["png"])
            except Exception as error:
                raise ReportTransactionError(
                    "HTML 已替换但 PNG 替换失败，可能产生半套输出；请检查输出并带 --overwrite 重跑"
                ) from error
            return final_paths
        finally:
            # 仅清理本次 UUID 对应的临时文件，不触碰旧 final 或其他调用的文件。
            for temporary_path in temporary_paths.values():
                temporary_path.unlink(missing_ok=True)


__all__ = [
    "DiagnosticsReport",
    "ReportTransactionError",
    "ReportValidationError",
    "SessionReportData",
    "load_session_report_data",
    "require_report_dependencies",
]
