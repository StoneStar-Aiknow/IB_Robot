"""Calibrated trace merge and portable, UI-neutral event archive.

Unmapped records remain inspectable but never enter the shared timeline.
Provenance comes from the emitting process, never from the merge machine.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from .clock_calibration import CalibrationPolicy, CaptureIdentity, ClockCalibration, load_calibration
from .model import EventOrigin, TraceDataset, TraceEvent, TracepointDefinition
from .parsing import parse_trace

MERGED_KIND = "ibrobot-merged-trace"
MAX_MERGE_EVENTS = 1_000_000
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_RECORD_BYTES = 1024 * 1024
# Explicitly documented realtime boundary, not arbitrary user *_ns attributes.
_BOUNDARY_FIELDS = {"first_action_execute": ("publish_end_ns",)}


def _source_identity(event: TraceEvent) -> tuple:
    origin = event.origin
    return (origin.machine_id, origin.boot_id, origin.process_instance, origin.event_sequence)


def _process_namespace(event: TraceEvent) -> str:
    identity = _source_identity(event)[:3]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]


def _calibrate(event: TraceEvent, cal: ClockCalibration, source_id: str, calibration_hash: str) -> TraceEvent:
    origin = event.origin
    mapping = {
        "status": "uncalibrated",
        "source_id": source_id,
        "calibration_sha256": calibration_hash,
        "original_timestamp_ns": event.timestamp_ns,
        "original_clock": event.clock,
        "original_fields": dict(event.fields),
        "reference_id": list(cal.identity.reference_id),
        "uncertainty_ns": None,
        "reason": "",
    }
    reason = ""
    if event.clock != "realtime" or event.time_mapping:
        reason = "unsupported_or_already_mapped_clock"
    elif (
        not all((origin.machine_id, origin.boot_id, origin.process_instance, origin.capture_session))
        or origin.event_sequence is None
    ):
        reason = "missing_event_provenance"
        mapping["missing_fields"] = [
            key for key in ("machine_id", "boot_id", "process_instance", "capture_session") if not getattr(origin, key)
        ]
        if origin.event_sequence is None:
            mapping["missing_fields"].append("event_sequence")
    elif origin.capture_session != cal.identity.session_id:
        reason = "capture_session_mismatch"
    result = None
    if not reason:
        # timestamp_ns is the event's authoritative realtime boundary. The
        # emission monotonic_ns may be later for deferred/explicit events.
        result = cal.map_time(
            event.timestamp_ns, clock="realtime_ns", machine_id=origin.machine_id, boot_id=origin.boot_id
        )
        reason = result.reason
    fields = dict(event.fields)
    if not reason:
        mapping.update(uncertainty_ns=result.uncertainty_ns, segment_id=result.segment_id)
        for key in _BOUNDARY_FIELDS.get(event.name, ()):
            if key not in fields:
                continue
            value = fields[key]
            if type(value) is not int or not 0 <= value < 2**63:
                reason = f"invalid_boundary:{key}"
                break
            boundary = cal.map_time(value, clock="realtime_ns", machine_id=origin.machine_id, boot_id=origin.boot_id)
            if not boundary.valid or boundary.segment_id != result.segment_id:
                reason = f"uncalibrated_boundary:{key}:{boundary.reason or 'clock_segment_mismatch'}"
                break
            fields[key] = boundary.timestamp_ns
            mapping.setdefault("boundary_uncertainty_ns", {})[key] = boundary.uncertainty_ns
            if key == "publish_end_ns":
                mapping["end_ns_uncertainty_ns"] = boundary.uncertainty_ns
        if not reason:
            namespace = _process_namespace(event)
            for key in ("span_id", "parent_span_id"):
                if fields.get(key):
                    fields[key] = f"{namespace}:{fields[key]}"
            mapping["status"] = "calibrated"
            return replace(
                event, timestamp_ns=result.timestamp_ns, clock="ptp_reference", fields=fields, time_mapping=mapping
            )
    mapping["reason"] = reason
    return replace(event, time_mapping=mapping)


def merge_datasets(
    sources: list[tuple[TraceDataset, ClockCalibration, str, str]], *, max_events: int = MAX_MERGE_EVENTS
) -> TraceDataset:
    if not 2 <= len(sources) <= 32:
        raise ValueError("merge requires 2..32 trace/calibration pairs")
    if type(max_events) is not int or not 1 <= max_events <= MAX_MERGE_EVENTS:
        raise ValueError("invalid merge event limit")
    references = {cal.identity.reference_id for _, cal, _, _ in sources}
    if len(references) != 1:
        raise ValueError("all calibrations must use the same session/domain/grandmaster")
    identities = [source_id for _, _, source_id, _ in sources]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate merge input")
    if sum(len(dataset.events) + len(dataset.uncalibrated_events) for dataset, _, _, _ in sources) > max_events:
        raise ValueError("merge event limit exceeded")
    result = TraceDataset(
        metadata={
            "kind": MERGED_KIND,
            "reference_id": list(next(iter(references))),
            "sources": [],
            "quality": "conditional engineering estimate; see calibration policy",
        }
    )
    seen = {}
    duplicate_count = 0
    for dataset, cal, source_id, calibration_hash in sources:
        if dataset.metadata.get("kind") == MERGED_KIND or dataset.uncalibrated_events:
            raise ValueError("merge original captures, not already merged datasets")
        result.definitions.extend(dataset.definitions)
        result.warnings.extend(dataset.warnings[:100])
        result.metadata["sources"].append(
            {
                "source_id": source_id,
                "source_metadata": dataset.metadata,
                "calibration_identity": asdict(cal.identity),
                "calibration_policy": asdict(cal.policy),
                "calibration_sha256": calibration_hash,
                "calibration_summary": cal.summary(),
            }
        )
        for event in dataset.events:
            key = _source_identity(event)
            if all(value is not None and value != "" for value in key):
                # Compare the original payload, ignoring importer path/order.
                payload = (
                    event.timestamp_ns,
                    event.monotonic_ns,
                    event.name,
                    event.fields,
                    event.clock,
                    {key: value for key, value in asdict(event.origin).items() if key not in {"path", "source"}},
                )
                if key in seen:
                    previous, previous_hash = seen[key]
                    if previous != payload or previous_hash != calibration_hash:
                        raise ValueError("conflicting payload/calibration for the same emitted event identity")
                    duplicate_count += 1
                    continue
                seen[key] = (payload, calibration_hash)
            mapped = _calibrate(event, cal, source_id, calibration_hash)
            (result.events if mapped.time_mapping["status"] == "calibrated" else result.uncalibrated_events).append(
                mapped
            )
    result.sort()
    result.metadata.update(
        calibrated_event_count=len(result.events),
        uncalibrated_event_count=len(result.uncalibrated_events),
        duplicate_event_count=duplicate_count,
        rejection_reasons=dict(Counter(event.time_mapping["reason"] for event in result.uncalibrated_events)),
    )
    if result.uncalibrated_events:
        result.warnings.append(
            f"{len(result.uncalibrated_events)} uncalibrated events retained outside the shared timeline"
        )
        missing = {}
        for event in result.uncalibrated_events:
            for key in event.time_mapping.get("missing_fields", []):
                counts = missing.setdefault(event.time_mapping["source_id"], {})
                counts[key] = counts.get(key, 0) + 1
        result.metadata["missing_provenance_by_source"] = missing
        for source_id, counts in missing.items():
            result.warnings.append(
                f"Source {source_id}: missing provenance {counts}; check emitter identity files and IB_TRACE_SESSION_ID before capture"
            )
    return result


def merge_traces(inputs: list[tuple[Path, Path]], *, max_events: int = MAX_MERGE_EVENTS) -> TraceDataset:
    if not 2 <= len(inputs) <= 32 or type(max_events) is not int or not 1 <= max_events <= MAX_MERGE_EVENTS:
        raise ValueError("merge requires 2..32 inputs and 1..1000000 max-events")
    sources = []
    total = 0
    for source, calibration in inputs:
        cal = load_calibration(calibration)
        dataset = parse_trace(source, max_events=max_events - total + 1)
        total += len(dataset.events)
        if total > max_events:
            raise ValueError("merge event limit exceeded")
        sources.append((dataset, cal, str(source.resolve()), hashlib.sha256(calibration.read_bytes()).hexdigest()))
    return merge_datasets(sources, max_events=max_events)


def write_merged_trace(path: Path, dataset: TraceDataset) -> None:
    if dataset.metadata.get("kind") != MERGED_KIND:
        raise ValueError("expected a calibrated merged dataset")
    with path.open("x", encoding="utf-8") as stream:

        def emit(record):
            encoded = json.dumps(record, sort_keys=True, allow_nan=False)
            if len(encoded.encode()) > MAX_RECORD_BYTES:
                raise ValueError("merged record size limit exceeded")
            stream.write(encoded + "\n")

        emit(
            {
                "kind": MERGED_KIND,
                "schema_version": 1,
                "metadata": dataset.metadata,
                "warnings": dataset.warnings,
                "definitions": [definition.to_dict() for definition in dataset.definitions],
            }
        )
        for kind, events in (("event", dataset.events), ("uncalibrated_event", dataset.uncalibrated_events)):
            for event in events:
                emit({"kind": kind, "event": event.to_dict()})
        emit(
            {
                "kind": "end",
                "event_count": len(dataset.events),
                "uncalibrated_event_count": len(dataset.uncalibrated_events),
            }
        )


def read_merged_trace(path: Path, *, max_events: int | None = None) -> TraceDataset:
    try:
        return _read_merged_trace(path, max_events=max_events)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid merged archive structure: {exc}") from exc


def _read_merged_trace(path: Path, *, max_events: int | None = None) -> TraceDataset:
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("merged archive exceeds size limit")
    limit = min(max_events, MAX_MERGE_EVENTS) if max_events is not None else MAX_MERGE_EVENTS
    if type(limit) is not int or limit < 1:
        raise ValueError("invalid merged event limit")
    with path.open(encoding="utf-8") as stream:

        def read():
            line = stream.readline(MAX_RECORD_BYTES + 1)
            if not line or len(line.encode()) > MAX_RECORD_BYTES or not line.endswith("\n"):
                raise ValueError("incomplete or oversized merged archive record")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("invalid merged archive record")
            return value

        header = read()
        if header.get("kind") != MERGED_KIND or header.get("schema_version") != 1:
            raise ValueError("unsupported merged trace schema")
        dataset = TraceDataset(
            metadata=header["metadata"],
            warnings=header["warnings"],
            definitions=[TracepointDefinition(**value) for value in header["definitions"]],
        )
        if (
            not isinstance(dataset.metadata, dict)
            or dataset.metadata.get("kind") != MERGED_KIND
            or not isinstance(dataset.warnings, list)
        ):
            raise ValueError("invalid merged metadata")
        sources = dataset.metadata.get("sources")
        legacy_sources = "sources" not in dataset.metadata
        if legacy_sources:
            sources = []  # v1 archives without source policies keep the original 50us ceiling.
        if not isinstance(sources, list):
            raise ValueError("missing merged source policies")
        policies = {}
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError("invalid merged source policy")
            source_id, digest = source["source_id"], source["calibration_sha256"]
            if not isinstance(source_id, str) or not source_id or not isinstance(digest, str) or not digest:
                raise ValueError("invalid merged source identity")
            if source_id in policies:
                raise ValueError("duplicate merged source identity")
            policy = CalibrationPolicy(**source["calibration_policy"])
            identity = CaptureIdentity(**source["calibration_identity"])
            if list(identity.reference_id) != dataset.metadata.get("reference_id"):
                raise ValueError("merged source reference mismatch")
            policies[source_id] = (digest, policy, identity)
        while True:
            record = read()
            if record["kind"] == "end":
                if (record["event_count"], record["uncalibrated_event_count"]) != (
                    len(dataset.events),
                    len(dataset.uncalibrated_events),
                ) or stream.read(1):
                    raise ValueError("merged trace footer/count mismatch")
                break
            if len(dataset.events) + len(dataset.uncalibrated_events) >= limit:
                raise ValueError("merged event limit exceeded")
            if record["kind"] not in {"event", "uncalibrated_event"}:
                raise ValueError("invalid merged record kind")
            raw = record["event"]
            event = TraceEvent(**{**raw, "origin": EventOrigin(**raw["origin"])})
            if not isinstance(event.time_mapping, dict):
                raise ValueError("invalid mapping metadata")
            if (
                type(event.timestamp_ns) is not int
                or not 0 < event.timestamp_ns < 2**63
                or not isinstance(event.fields, dict)
            ):
                raise ValueError("invalid merged event timestamp or fields")
            if record["kind"] == "event":
                mapping = event.time_mapping
                source_id = mapping.get("source_id")
                if not legacy_sources and (not isinstance(source_id, str) or source_id not in policies):
                    raise ValueError("invalid calibrated event source")
                digest, policy, identity = (
                    policies[source_id] if not legacy_sources else (None, CalibrationPolicy(), None)
                )
                if (
                    event.clock != "ptp_reference"
                    or mapping.get("status") != "calibrated"
                    or mapping.get("reference_id") != dataset.metadata.get("reference_id")
                    or type(mapping.get("uncertainty_ns")) is not int
                    or not 0 <= mapping["uncertainty_ns"] <= policy.max_uncertainty_ns
                    or (not legacy_sources and mapping.get("calibration_sha256") != digest)
                    or (
                        identity is not None
                        and (event.origin.machine_id, event.origin.boot_id, event.origin.capture_session)
                        != (identity.machine_id, identity.boot_id, identity.session_id)
                    )
                ):
                    raise ValueError("invalid calibrated event")
                dataset.events.append(event)
            else:
                if event.time_mapping.get("status") != "uncalibrated":
                    raise ValueError("invalid uncalibrated event")
                dataset.uncalibrated_events.append(event)
    dataset.sort()
    return dataset
