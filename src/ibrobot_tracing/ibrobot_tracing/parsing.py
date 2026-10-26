"""Parse structured IBTRACE1 events from logs or CTF traces."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from .model import EventOrigin, TraceDataset, TraceEvent, TracepointDefinition

_STRUCTURED_PREFIX = "IBTRACE1 "
_BABELTRACE_MESSAGE = re.compile(r'\bmsg = ("(?:[^"\\]|\\.)*")')
_MAX_TRACEPOINT_DEFINITIONS = 100_000
_MAX_PARSE_WARNINGS = 1_000


def _event_message(text: str) -> str:
    """Extract and unescape the Python logging message from Babeltrace output."""
    # Babeltrace 1 renders nested quotes in Python log messages without
    # escaping them. Recover a structured message from its following field and
    # validate the JSON before falling back to the normal quoted-string parser.
    message_marker = 'msg = "' + _STRUCTURED_PREFIX
    marker = text.find(message_marker)
    if marker >= 0:
        message_start = marker + len('msg = "')
        message_end = text.rfind('", logger_name = "')
        if message_end > message_start:
            message = text[message_start:message_end]
            try:
                json.loads(message[len(_STRUCTURED_PREFIX) :])
            except json.JSONDecodeError:
                pass
            else:
                return message
    match = _BABELTRACE_MESSAGE.search(text)
    if match is None:
        return text
    try:
        return str(json.loads(match.group(1)))
    except json.JSONDecodeError:
        return text


def _provider(text: str) -> str:
    match = re.search(r"\b(ib_trace\.[A-Za-z0-9_.-]+)\b", text)
    return match.group(1) if match else ""


def _structured_event(line: str, origin: EventOrigin, sequence: int) -> TraceEvent | None:
    try:
        message = _event_message(line)
        marker = message.find(_STRUCTURED_PREFIX)
        if marker < 0:
            return None
        payload = json.loads(message[marker + len(_STRUCTURED_PREFIX) :])
        if not isinstance(payload, dict):
            return None
        timestamp_ns = payload.get("timestamp_ns")
        schema_version = payload.get("schema_version", 1)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        type(timestamp_ns) is not int
        or not 0 < timestamp_ns < 2**63
        or type(schema_version) is not int
        or schema_version != 1
    ):
        return None
    name = payload.get("event")
    fields = payload.get("fields", {})
    clock = payload.get("clock", "realtime")
    if not isinstance(name, str) or not name or not isinstance(fields, dict) or clock != "realtime":
        return None
    monotonic_ns = payload.get("monotonic_ns")
    if monotonic_ns is not None and (type(monotonic_ns) is not int or not 0 <= monotonic_ns < 2**63):
        return None
    provenance = payload.get("provenance", {})
    if not isinstance(provenance, dict):
        return None
    text_keys = ("host", "machine_id", "boot_id", "process_instance", "capture_session")
    int_keys = ("process_id", "thread_id", "event_sequence")
    if any(not isinstance(provenance.get(key, ""), str) or len(provenance.get(key, "")) > 1024 for key in text_keys):
        return None
    if any(
        key in provenance and (type(provenance[key]) is not int or not 0 <= provenance[key] < 2**63) for key in int_keys
    ):
        return None
    origin = replace(origin, **{key: provenance[key] for key in (*text_keys, *int_keys) if key in provenance})
    return TraceEvent(
        timestamp_ns=timestamp_ns,
        name=name,
        fields=dict(fields),
        origin=origin,
        clock=clock,
        schema_version=schema_version,
        sequence=sequence,
        monotonic_ns=monotonic_ns,
    )


def _tracepoint_definition(event: TraceEvent) -> TracepointDefinition | None:
    if event.name != "_tracepoint_definition":
        return None
    kind = str(event.field("kind", ""))
    name = str(event.field("name", ""))
    if kind not in {"event", "span"} or not name:
        return None
    return TracepointDefinition(
        kind=kind,
        component_id=str(event.field("component_id", "")),
        name=name,
        origin=str(event.field("origin", "")),
        description=str(event.field("description", "")),
    )


def parse_lines(
    lines: Iterable[str],
    *,
    source: str = "<memory>",
    max_events: int | None = None,
) -> TraceDataset:
    dataset = TraceDataset(metadata={"source": source})
    trace_records = 0
    definition_records = 0
    parse_warning_count = 0
    definition_limit = min(max_events or _MAX_TRACEPOINT_DEFINITIONS, _MAX_TRACEPOINT_DEFINITIONS)

    def warn(message: str) -> None:
        nonlocal parse_warning_count
        if parse_warning_count < _MAX_PARSE_WARNINGS:
            dataset.warnings.append(message)
        elif parse_warning_count == _MAX_PARSE_WARNINGS:
            dataset.warnings.append("Additional trace parse warnings suppressed")
        parse_warning_count += 1

    for sequence, line in enumerate(lines):
        if "ib_trace." not in line and _STRUCTURED_PREFIX not in line:
            continue
        origin = EventOrigin(source="text", path=source, provider=_provider(line))
        if _STRUCTURED_PREFIX in line:
            event = _structured_event(line, origin, sequence)
        else:
            warn(f"Unsupported legacy trace record {source}:{sequence + 1}; only IBTRACE1 is accepted")
            continue
        if event is None:
            warn(f"Could not parse trace record {source}:{sequence + 1}")
        elif event.name == "_tracepoint_definition":
            definition_records += 1
            if definition_records > definition_limit:
                if definition_records == definition_limit + 1:
                    dataset.warnings.append(f"Tracepoint definition limit ({definition_limit}) reached")
                continue
            definition = _tracepoint_definition(event)
            if definition is None:
                warn(f"Invalid tracepoint definition {source}:{sequence + 1}")
            else:
                dataset.definitions.append(definition)
        else:
            trace_records += 1
            if max_events is not None and trace_records > max_events:
                dataset.warnings.append(f"Trace record limit ({max_events}) reached")
                break
            dataset.events.append(event)
    dataset.sort()
    return dataset


def parse_log(path: Path, *, max_events: int | None = None) -> TraceDataset:
    with path.open(encoding="utf-8", errors="replace") as stream:
        return parse_lines(stream, source=str(path), max_events=max_events)


def parse_babeltrace_text(text: str, *, source: str = "<babeltrace>") -> TraceDataset:
    return parse_lines(text.splitlines(), source=source)


def parse_ctf(path: Path, *, max_events: int | None = None) -> TraceDataset:
    for executable in ("babeltrace2", "babeltrace"):
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as output:
            try:
                result = subprocess.run(
                    [executable, str(path)],
                    stdout=output,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=120,
                    check=False,
                )
            except FileNotFoundError:
                continue
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"{executable} timed out while reading {path}") from exc
            if result.returncode == 0:
                output.seek(0)
                dataset = parse_lines(output, source=str(path), max_events=max_events)
                dataset.events = [
                    replace(event, origin=replace(event.origin, source="ctf")) for event in dataset.events
                ]
                dataset.metadata["reader"] = executable
                return dataset
    raise RuntimeError("babeltrace2 not found or failed to read the CTF trace")


def parse_trace(path: Path, *, source_kind: str = "auto", max_events: int | None = None) -> TraceDataset:
    if not path.exists():
        raise FileNotFoundError(path)
    kind = source_kind
    if kind == "auto":
        kind = "ctf" if path.is_dir() else "log"
        if path.is_file() and path.suffix == ".ibtrace":
            kind = "merged"
    if kind == "merged":
        from .merging import read_merged_trace

        return read_merged_trace(path, max_events=max_events)
    if kind == "ctf":
        return parse_ctf(path, max_events=max_events)
    if kind == "log":
        return parse_log(path, max_events=max_events)
    raise ValueError(f"Unsupported trace source kind: {source_kind}")
