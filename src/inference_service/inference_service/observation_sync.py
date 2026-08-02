"""Timestamp mapping and synchronized selection for streamed observations."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass

from robot_config.contract_utils import StreamBuffer

RTP_CLOCK_RATE = 90_000
_RTP_MODULUS = 1 << 32
_RTP_HALF_MODULUS = 1 << 31


@dataclass(frozen=True, slots=True)
class SynchronizationIssue:
    reason: str
    observation_key: str
    stream_id: str
    details: Mapping[str, object]


class ObservationSynchronizationError(RuntimeError):
    code = "observation_not_ready"
    recoverable = True
    stage = "observation_sync"

    def __init__(self, issues: list[SynchronizationIssue]) -> None:
        self.issues = tuple(issues)
        self.details = {
            "streams": [
                {
                    "reason": issue.reason,
                    "observation_key": issue.observation_key,
                    "stream_id": issue.stream_id,
                    **dict(issue.details),
                }
                for issue in issues
            ]
        }
        summary = ", ".join(f"{issue.stream_id} ({issue.observation_key}): {issue.reason}" for issue in issues)
        super().__init__(f"streamed observations are not ready: {summary}")


@dataclass(frozen=True, slots=True)
class StreamSelection:
    observation_key: str
    stream_id: str
    buffer: StreamBuffer
    timestamp_mapping_ready: bool = True
    keyframe_ready: bool = True
    pad_before_first: bool = False


@dataclass(frozen=True, slots=True)
class SelectedStreamValue:
    observation_key: str
    stream_id: str
    capture_timestamp_ns: int
    receive_timestamp_ns: int
    value: object


class RtpTimestampMapper:
    """Map 32-bit 90 kHz RTP timestamps into a session capture-time clock."""

    def __init__(self, max_mapping_age_ns: int, *, observation_key: str, stream_id: str) -> None:
        if max_mapping_age_ns <= 0:
            raise ValueError("max_mapping_age_ns must be positive")
        if not observation_key or not stream_id:
            raise ValueError("timestamp mapper requires observation_key and stream_id")
        self.max_mapping_age_ns = int(max_mapping_age_ns)
        self.observation_key = observation_key
        self.stream_id = stream_id
        self._lock = threading.RLock()
        self.reset()

    def reset(self, session_generation: int = 0) -> None:
        if session_generation < 0:
            raise ValueError("session_generation cannot be negative")
        with self._lock:
            self.session_generation = int(session_generation)
            self._rtp_timestamp: int | None = None
            self._capture_timestamp_ns: int | None = None
            self._mapping_receive_time_ns: int | None = None

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._rtp_timestamp is not None

    def update(
        self,
        rtp_timestamp: int,
        capture_timestamp_ns: int,
        receive_time_ns: int,
        *,
        session_generation: int,
    ) -> None:
        if session_generation < 1:
            raise ValueError("timestamp mappings require a positive session generation")
        if not 0 <= rtp_timestamp < _RTP_MODULUS:
            raise ValueError("rtp_timestamp must fit in uint32")
        if capture_timestamp_ns < 0 or receive_time_ns < 0:
            raise ValueError("capture and receive timestamps cannot be negative")
        with self._lock:
            if self.session_generation != session_generation:
                self.reset(session_generation)
            self._rtp_timestamp = int(rtp_timestamp)
            self._capture_timestamp_ns = int(capture_timestamp_ns)
            self._mapping_receive_time_ns = int(receive_time_ns)

    def map(self, rtp_timestamp: int, *, now_ns: int, session_generation: int) -> int:
        if not 0 <= rtp_timestamp < _RTP_MODULUS:
            raise ValueError("rtp_timestamp must fit in uint32")
        with self._lock:
            if session_generation != self.session_generation or self._rtp_timestamp is None:
                raise ObservationSynchronizationError([self._mapping_issue("unmapped", session_generation)])
            assert self._capture_timestamp_ns is not None
            assert self._mapping_receive_time_ns is not None
            mapping_age_ns = max(0, int(now_ns) - self._mapping_receive_time_ns)
            if mapping_age_ns > self.max_mapping_age_ns:
                raise ObservationSynchronizationError(
                    [
                        self._mapping_issue(
                            "stale",
                            session_generation,
                            constraint="timestamp_mapping",
                            mapping_age_ns=mapping_age_ns,
                            max_mapping_age_ns=self.max_mapping_age_ns,
                        )
                    ]
                )
            delta = (int(rtp_timestamp) - self._rtp_timestamp + _RTP_HALF_MODULUS) % _RTP_MODULUS
            delta -= _RTP_HALF_MODULUS
            return self._capture_timestamp_ns + round(delta * 1_000_000_000 / RTP_CLOCK_RATE)

    def _mapping_issue(self, reason: str, session_generation: int, **details: object) -> SynchronizationIssue:
        return SynchronizationIssue(
            reason=reason,
            observation_key=self.observation_key,
            stream_id=self.stream_id,
            details={"session_generation": session_generation, **details},
        )


def select_synchronized_streams(
    streams: Mapping[str, StreamSelection],
    target_timestamp_ns: int,
    *,
    now_ns: int,
    max_inter_camera_skew_ns: int,
) -> dict[str, SelectedStreamValue]:
    """Select all required streams against one target or fail as one operation."""
    if max_inter_camera_skew_ns < 0:
        raise ValueError("max_inter_camera_skew_ns cannot be negative")
    selected: dict[str, SelectedStreamValue] = {}
    issues: list[SynchronizationIssue] = []
    for observation_key, stream in streams.items():
        if observation_key != stream.observation_key:
            raise ValueError(f"stream mapping key {observation_key!r} does not match its observation key")
        if not stream.timestamp_mapping_ready:
            issues.append(_stream_issue(stream, "unmapped"))
            continue
        if not stream.keyframe_ready:
            issues.append(_stream_issue(stream, "pre_keyframe"))
            continue
        item, issue = stream.buffer.select_entry(target_timestamp_ns, now_ns=now_ns)
        if issue is not None and issue.get("reason") == "newer_than_request" and stream.pad_before_first:
            item = stream.buffer.first_entry()
            issue = None if item is not None else issue
        if issue is not None:
            reason = str(issue["reason"])
            if reason == "newer_than_request":
                reason = "missing"
            issues.append(
                _stream_issue(stream, reason, **{key: value for key, value in issue.items() if key != "reason"})
            )
            continue
        assert item is not None
        capture_timestamp_ns, receive_timestamp_ns, value = item
        selected[observation_key] = SelectedStreamValue(
            observation_key=observation_key,
            stream_id=stream.stream_id,
            capture_timestamp_ns=capture_timestamp_ns,
            receive_timestamp_ns=receive_timestamp_ns,
            value=value,
        )
    if issues:
        raise ObservationSynchronizationError(issues)

    timestamps = [item.capture_timestamp_ns for item in selected.values()]
    if timestamps and max(timestamps) - min(timestamps) > max_inter_camera_skew_ns:
        issues = [
            _stream_issue(
                streams[key],
                "skewed",
                selected_timestamp_ns=item.capture_timestamp_ns,
                minimum_timestamp_ns=min(timestamps),
                maximum_timestamp_ns=max(timestamps),
                max_inter_camera_skew_ns=max_inter_camera_skew_ns,
            )
            for key, item in selected.items()
        ]
        raise ObservationSynchronizationError(issues)
    return selected


def _stream_issue(stream: StreamSelection, reason: str, **details: object) -> SynchronizationIssue:
    return SynchronizationIssue(
        reason=reason,
        observation_key=stream.observation_key,
        stream_id=stream.stream_id,
        details=details,
    )
