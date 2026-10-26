"""Offline clock mapping with explicit provenance and conditional error budgets.

The reference is the experiment's PTP grandmaster PHC, not UTC. Original clocks
are never changed. No interpolation across observed faults or discontinuities.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

CLOCKS = ("monotonic_ns", "monotonic_raw_ns", "realtime_ns")
MAX_CAPTURE_BYTES = 64 * 1024 * 1024


def _integer(value: object, name: str, *, signed: bool = False) -> int:
    if type(value) is not int or abs(value) >= 2**63 or (not signed and value < 0):
        raise ValueError(f"{name} must be a {'signed ' if signed else 'non-negative '}int64")
    return value


@dataclass(frozen=True)
class CalibrationPolicy:
    max_uncertainty_ns: int = 50_000
    max_gap_ns: int = 100_000_000
    max_status_age_ns: int = 2_500_000_000
    max_sync_age_ns: int = 2_500_000_000
    max_clock_rate_ppm: int = 1000
    sync_error_growth_ppm: int = 10
    # Instrument/PHY timestamp bias is not measured by the PTP servo.
    timestamp_bias_budget_ns: int = 1000
    settle_updates: int = 3

    def __post_init__(self):
        for name, value in vars(self).items():
            _integer(value, name)
            if value == 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class CaptureIdentity:
    session_id: str
    machine_id: str
    boot_id: str
    interface: str
    phc: str
    grandmaster_identity: str
    domain: int

    def __post_init__(self):
        for name, value in vars(self).items():
            if name == "domain":
                if type(value) is not int or not 0 <= value <= 127:
                    raise ValueError("domain must be in 0..127")
            elif not isinstance(value, str) or not value or len(value) > 256 or value.strip() != value:
                raise ValueError(f"invalid {name}")

    @property
    def reference_id(self) -> tuple[str, int, str]:
        return self.session_id, self.domain, self.grandmaster_identity


@dataclass(frozen=True)
class PtpStatus:
    observed_monotonic_ns: int
    domain: int
    clock_identity: str
    grandmaster_identity: str
    port_state: str
    timestamping: str
    interface: str
    gm_present: bool
    master_offset_ns: int
    mean_path_delay_ns: int
    ingress_time_ns: int
    gm_time_base_indicator: int

    def __post_init__(self):
        for name in ("observed_monotonic_ns", "domain", "ingress_time_ns", "gm_time_base_indicator"):
            _integer(getattr(self, name), name)
        for name in ("master_offset_ns", "mean_path_delay_ns"):
            _integer(getattr(self, name), name, signed=True)
        if type(self.gm_present) is not bool:
            raise ValueError("gm_present must be boolean")
        for name in ("clock_identity", "grandmaster_identity", "port_state", "timestamping", "interface"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"invalid {name}")


@dataclass(frozen=True)
class ClockSample:
    monotonic_ns: int
    monotonic_raw_ns: int
    realtime_ns: int
    boottime_ns: int
    phc_ns: int
    bracket_ns: int
    status: PtpStatus | None

    def __post_init__(self):
        for name in (*CLOCKS, "boottime_ns", "phc_ns", "bracket_ns"):
            _integer(getattr(self, name), name)
        if self.status is not None and not isinstance(self.status, PtpStatus):
            raise ValueError("invalid PTP status")


@dataclass(frozen=True)
class MappedTime:
    timestamp_ns: int | None
    uncertainty_ns: int | None
    reason: str = ""
    segment_id: int | None = None

    @property
    def valid(self) -> bool:
        return self.timestamp_ns is not None


def sample_quality(sample: ClockSample, identity: CaptureIdentity, policy: CalibrationPolicy) -> tuple[int, str]:
    status = sample.status
    if status is None:
        return 0, "missing_ptp_status"
    if status.domain != identity.domain or status.grandmaster_identity != identity.grandmaster_identity:
        return 0, "reference_changed"
    if status.timestamping != "HARDWARE" or status.interface != identity.interface:
        return 0, "wrong_timestamp_source"
    age = sample.monotonic_ns - status.observed_monotonic_ns
    if not 0 <= age <= policy.max_status_age_ns:
        return 0, "stale_ptp_status"
    sync_error = 0
    if status.port_state == "MASTER" and status.clock_identity == identity.grandmaster_identity:
        pass
    elif status.port_state == "SLAVE" and status.gm_present:
        sync_age = sample.phc_ns - status.ingress_time_ns
        if status.ingress_time_ns == 0 or not 0 <= sync_age <= policy.max_sync_age_ns:
            return 0, "stale_sync"
        if status.mean_path_delay_ns < 0:
            return 0, "negative_path_delay"
        # Full mean path delay bounds asymmetry only under non-negative path
        # delays. Offset/growth and timestamp-bias budgets remain assumptions.
        sync_error = (
            abs(status.master_offset_ns)
            + status.mean_path_delay_ns
            + math.ceil(sync_age * policy.sync_error_growth_ppm / 1_000_000)
        )
    else:
        return 0, "ptp_not_locked"
    error = sample.bracket_ns + sync_error + policy.timestamp_bias_budget_ns
    return error, "uncertainty_exceeded" if error > policy.max_uncertainty_ns else ""


class _ClockIntervals:
    """Index original closed intervals, including backward steps and plateaus.

    Three matches suffice to prove ambiguity. Never sort or discard the original
    sample sequence: interval IDs still feed the existing quality/epoch checks.
    """

    def __init__(self, intervals):
        self.center = sorted((a + b) // 2 for a, b, _ in intervals)[len(intervals) // 2]
        left, right, crossing = [], [], []
        for interval in intervals:
            a, b, _ = interval
            (left if b < self.center else right if a > self.center else crossing).append(interval)
        self.starts = sorted(crossing)
        self.ends = sorted(crossing, key=lambda item: item[1], reverse=True)
        self.left = _ClockIntervals(left) if left else None
        self.right = _ClockIntervals(right) if right else None

    def find(self, timestamp, matches):
        if len(matches) >= 3:
            return
        if timestamp <= self.center:
            for a, _, index in self.starts:
                if a > timestamp or len(matches) >= 3:
                    break
                matches.append(index)
            child = self.left if timestamp < self.center else None
        else:
            for _, b, index in self.ends:
                if b < timestamp or len(matches) >= 3:
                    break
                matches.append(index)
            child = self.right
        if child is not None and len(matches) < 3:
            child.find(timestamp, matches)


class ClockCalibration:
    def __init__(self, identity: CaptureIdentity, samples: list[ClockSample], policy: CalibrationPolicy):
        self.identity, self.samples, self.policy = identity, tuple(samples), policy
        self._ticks = [sample.monotonic_ns for sample in samples]
        self._ordered_ticks = {}
        self._interval_indexes = {}
        for clock in CLOCKS:
            ticks = [getattr(sample, clock) for sample in samples]
            if all(b > a for a, b in zip(ticks, ticks[1:], strict=False)):
                self._ordered_ticks[clock] = ticks
            elif len(ticks) > 1:
                self._interval_indexes[clock] = _ClockIntervals(
                    [(min(a, b), max(a, b), i) for i, (a, b) in enumerate(zip(ticks, ticks[1:], strict=False))]
                )
        self._segment_cache = {}
        if any(b <= a for a, b in zip(self._ticks, self._ticks[1:], strict=False)):
            raise ValueError("capture samples must be strictly monotonic; never sort across a reboot")
        self.quality = []
        good_updates, previous_update, previous_epoch = 0, None, None
        reference_discontinuity = False
        for sample in samples:
            error, reason = sample_quality(sample, identity, policy)
            if self.quality and not reason and not self.quality[-1][1]:
                previous = samples[len(self.quality) - 1]
                dt = sample.monotonic_ns - previous.monotonic_ns
                tolerance = (
                    previous.bracket_ns + sample.bracket_ns + math.ceil(dt * policy.max_clock_rate_ppm / 1_000_000)
                )
                if abs((sample.phc_ns - previous.phc_ns) - dt) > tolerance:
                    reference_discontinuity = True
            if reference_discontinuity:
                # A reference jump cannot be repaired by interpolating later
                # samples under the same experiment epoch. Start a new capture.
                reason = "reference_discontinuity"
            epoch = (sample.status.clock_identity, sample.status.gm_time_base_indicator) if sample.status else None
            if reason or epoch != previous_epoch:
                good_updates = 0
            update = sample.status.observed_monotonic_ns if sample.status else None
            if not reason and update != previous_update:
                good_updates += 1
            previous_update, previous_epoch = update, epoch
            if not reason and good_updates < policy.settle_updates:
                reason = "settling"
            self.quality.append((error, reason))

    def _interval_reason(self, index: int, clock: str) -> str:
        left, right = self.samples[index : index + 2]
        if self.quality[index][1] or self.quality[index + 1][1]:
            return self.quality[index][1] or self.quality[index + 1][1]
        dt = right.monotonic_ns - left.monotonic_ns
        if dt > self.policy.max_gap_ns:
            return "sample_gap"
        if (left.status.clock_identity, left.status.gm_time_base_indicator) != (
            right.status.clock_identity,
            right.status.gm_time_base_indicator,
        ):
            return "clock_epoch_changed"
        tolerance = left.bracket_ns + right.bracket_ns + math.ceil(dt * self.policy.max_clock_rate_ppm / 1_000_000)
        if abs((right.boottime_ns - left.boottime_ns) - dt) > tolerance:
            return "suspend_or_clock_step"
        for name in {clock, "phc_ns"}:
            delta = getattr(right, name) - getattr(left, name)
            if delta <= 0 or abs(delta - dt) > tolerance:
                return "clock_step_or_rate_exceeded"
        return ""

    def map_time(self, timestamp_ns: int, *, clock: str, machine_id: str, boot_id: str) -> MappedTime:
        _integer(timestamp_ns, "timestamp_ns")
        if clock not in CLOCKS:
            raise ValueError(f"unsupported clock: {clock}")
        if (machine_id, boot_id) != (self.identity.machine_id, self.identity.boot_id):
            return MappedTime(None, None, "host_or_boot_mismatch")
        if clock in self._ordered_ticks:
            index = bisect_right(self._ordered_ticks[clock], timestamp_ns) - 1
            indices = [min(index, len(self.samples) - 2)] if index >= 0 else []
        else:
            # Realtime can repeat after a backward step; ambiguous epochs must
            # not be silently sorted into a plausible-looking time axis.
            indices = []
            self._interval_indexes[clock].find(timestamp_ns, indices)
            indices.sort()
            if len(indices) > 1:
                # Adjacent intervals sharing one exact sample are not ambiguous.
                if (
                    len(indices) == 2
                    and indices[1] == indices[0] + 1
                    and timestamp_ns == getattr(self.samples[indices[1]], clock)
                ):
                    indices = indices[1:]
                else:
                    return MappedTime(None, None, "ambiguous_clock_epoch")
        if not indices or indices[0] < 0:
            return MappedTime(None, None, "outside_capture")
        index = indices[0]
        left, right = self.samples[index : index + 2]
        a, b = getattr(left, clock), getattr(right, clock)
        if not min(a, b) <= timestamp_ns <= max(a, b):
            return MappedTime(None, None, "outside_capture")
        reason = self._interval_reason(index, clock)
        if reason:
            return MappedTime(None, None, reason)
        delta = timestamp_ns - a
        mapped = left.phc_ns + (delta * (right.phc_ns - left.phc_ns) + (b - a) // 2) // (b - a)
        distance = min(delta, b - timestamp_ns)
        uncertainty = max(self.quality[index][0], self.quality[index + 1][0])
        # Bounded-rate interpolation allowance; includes the measured slope.
        uncertainty += math.ceil(2 * distance * self.policy.max_clock_rate_ppm / 1_000_000) + 1
        if uncertainty > self.policy.max_uncertainty_ns:
            return MappedTime(None, uncertainty, "uncertainty_exceeded")
        if clock not in self._segment_cache:
            segments, segment, valid_previous = [], 0, False
            for i in range(len(self.samples) - 1):
                valid = not self._interval_reason(i, clock)
                if valid and not valid_previous:
                    segment += 1
                segments.append(segment if valid else None)
                valid_previous = valid
            self._segment_cache[clock] = segments
        return MappedTime(mapped, uncertainty, segment_id=self._segment_cache[clock][index])

    def summary(self) -> dict:
        reasons: dict[str, int] = {}
        errors = []
        drifts = []
        for i in range(len(self.samples) - 1):
            a, b = self.samples[i : i + 2]
            result = self.map_time(
                (a.monotonic_ns + b.monotonic_ns) // 2,
                clock="monotonic_ns",
                machine_id=self.identity.machine_id,
                boot_id=self.identity.boot_id,
            )
            reasons[result.reason or "valid"] = reasons.get(result.reason or "valid", 0) + 1
            if result.valid:
                errors.append(result.uncertainty_ns)
                drifts.append(((b.phc_ns - a.phc_ns) / (b.monotonic_ns - a.monotonic_ns) - 1) * 1_000_000)
        return {
            "samples": len(self.samples),
            "intervals": reasons,
            "max_estimated_uncertainty_ns": max(errors, default=None),
            "reference_rate_ppm_range": [min(drifts), max(drifts)] if drifts else None,
        }


def load_calibration(path: str | Path) -> ClockCalibration:
    path = Path(path)
    if path.stat().st_size > MAX_CAPTURE_BYTES:
        raise ValueError("calibration capture exceeds 64 MiB")
    line_number = 1
    try:
        with path.open(encoding="utf-8") as stream:
            header = json.loads(next(stream))
            if (
                not isinstance(header, dict)
                or header.get("schema_version") != 1
                or header.get("kind") != "clock_calibration"
            ):
                raise ValueError("unsupported calibration schema")
            identity = CaptureIdentity(**header["identity"])
            policy = CalibrationPolicy(**header["policy"])
            samples = []
            for line in stream:
                line_number += 1
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("expected a record object")
                if value["kind"] == "ptp_status":
                    continue
                if value["kind"] != "sample":
                    raise ValueError("unknown calibration record")
                data = value["sample"]
                status = data["status"]
                samples.append(ClockSample(**{**data, "status": PtpStatus(**status) if status is not None else None}))
        return ClockCalibration(identity, samples, policy)
    except (StopIteration, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid calibration capture {path}:{line_number}: {exc or 'empty file'}") from exc


def reference_difference(
    start: MappedTime,
    end: MappedTime,
    start_calibration: ClockCalibration,
    end_calibration: ClockCalibration,
    *,
    max_uncertainty_ns: int = 100_000,
) -> MappedTime:
    """Signed delta on one pinned reference, carrying both hosts' error budgets."""
    if start_calibration.identity.reference_id != end_calibration.identity.reference_id:
        return MappedTime(None, None, "reference_mismatch")
    if not start.valid or not end.valid:
        return MappedTime(None, None, start.reason or end.reason)
    uncertainty = start.uncertainty_ns + end.uncertainty_ns
    if uncertainty > max_uncertainty_ns:
        return MappedTime(None, uncertainty, "uncertainty_exceeded")
    return MappedTime(end.timestamp_ns - start.timestamp_ns, uncertainty)
