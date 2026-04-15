#!/usr/bin/env python3
"""
Analyze IB-Robot tracing data.

Parses CTF traces (ros2_tracing / LTTng) or log-based traces to produce:
- request-level stage latency breakdowns
- observation ingress / sampling freshness summaries
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TraceEvent:
    timestamp_ns: int
    event_name: str
    fields: dict[str, Any] = field(default_factory=dict)

    @property
    def request_id(self) -> str:
        return str(self.fields.get("request_id", ""))


@dataclass
class RequestTrace:
    request_id: str
    events: list[TraceEvent] = field(default_factory=list)

    def get(self, name: str) -> TraceEvent | None:
        for event in self.events:
            if event.event_name == name:
                return event
        return None

    def span_ms(self, start: str, end: str) -> float | None:
        start_event = self.get(start)
        end_event = self.get(end)
        if start_event and end_event:
            return (end_event.timestamp_ns - start_event.timestamp_ns) / 1_000_000
        return None

    def field_ms(self, name: str, field_name: str) -> float | None:
        event = self.get(name)
        if event is None:
            return None
        return _to_float(event.fields.get(field_name))


def _field_pairs(text: str) -> dict[str, str]:
    pairs = {}
    for key, value in re.findall(
        r'([A-Za-z_][A-Za-z0-9_]*)=("[^"]*"|\[[^\]]*\]|[^\s,}]+)',
        text,
    ):
        pairs[key] = value.strip('",')
    return pairs


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip('",'))
    except ValueError:
        return None


def _percentile(sorted_values: list[float], ratio: float) -> float:
    index = min(len(sorted_values) - 1, int((len(sorted_values) - 1) * ratio))
    return sorted_values[index]


def _extract_event_payload(text: str) -> tuple[str, str] | None:
    matches = list(re.finditer(r"\[(\w+)\]\s*", text))
    if not matches:
        return None
    match = matches[-1]
    return match.group(1), text[match.end() :]


def _parse_timestamp_ns(text: str) -> int | None:
    wall_clock_match = re.match(r"\[(\d{2}):(\d{2}):(\d{2})\.(\d{9})\]", text)
    if wall_clock_match:
        hour, minute, second, nanos = wall_clock_match.groups()
        return ((int(hour) * 60 + int(minute)) * 60 + int(second)) * 1_000_000_000 + int(nanos)

    ros_log_match = re.search(r"\[(\d+)\.(\d{1,9})\]", text)
    if ros_log_match:
        seconds, nanos = ros_log_match.groups()
        return int(seconds) * 1_000_000_000 + int(nanos.ljust(9, "0"))

    raw_match = re.match(r"\[(\d+)\]", text)
    if raw_match:
        return int(raw_match.group(1))

    return None


def parse_log_events(path: Path) -> list[TraceEvent]:
    """Parse events from logging-based trace output."""
    events = []
    with path.open() as stream:
        for line in stream:
            if "ib_trace." not in line:
                continue
            payload = _extract_event_payload(line)
            if payload is None:
                continue
            event_name, field_text = payload
            fields = _field_pairs(field_text)
            ts_ns = int(fields.pop("_ts_ns", "0") or "0")
            if ts_ns == 0:
                parsed_ts = _parse_timestamp_ns(line)
                if parsed_ts is None:
                    continue
                ts_ns = parsed_ts
            events.append(TraceEvent(timestamp_ns=ts_ns, event_name=event_name, fields=fields))
    return events


def parse_ctf_events(trace_dir: Path) -> list[TraceEvent]:
    """Parse events from CTF trace via babeltrace2."""
    for cmd in ["babeltrace2", "babeltrace"]:
        try:
            result = subprocess.run(
                [cmd, str(trace_dir)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                return _parse_bt_output(result.stdout)
        except FileNotFoundError:
            continue
    print(
        "Error: babeltrace2 not found. Install: sudo apt install babeltrace2",
        file=sys.stderr,
    )
    return []


def _parse_bt_output(output: str) -> list[TraceEvent]:
    """Extract ib_trace events from babeltrace output."""
    events = []
    for line in output.splitlines():
        if "ib_trace." not in line:
            continue
        payload = _extract_event_payload(line)
        if payload is None:
            continue
        event_name, field_text = payload
        fields = _field_pairs(field_text)
        timestamp_ns = _parse_timestamp_ns(line)
        if timestamp_ns is None:
            continue
        events.append(
            TraceEvent(
                timestamp_ns=timestamp_ns,
                event_name=event_name,
                fields=fields,
            )
        )
    return events


def group_by_request(events: list[TraceEvent]) -> dict[str, RequestTrace]:
    traces: dict[str, RequestTrace] = {}
    for event in events:
        request_id = event.request_id
        if not request_id:
            continue
        trace = traces.setdefault(request_id, RequestTrace(request_id=request_id))
        trace.events.append(event)
    for trace in traces.values():
        trace.events.sort(key=lambda item: item.timestamp_ns)
    return traces


def analyze_requests(traces: dict[str, RequestTrace]) -> list[dict[str, Any]]:
    results = []
    for trace in traces.values():
        row = {"request_id": trace.request_id}
        for label, start, end in [
            ("obs_frame_ms", "dispatch_request", "obs_frame"),
            ("dispatch_to_infer_ms", "dispatch_request", "inference_begin"),
            ("preprocess_ms", "preprocess_begin", "preprocess_end"),
            ("inference_ms", "inference_begin", "inference_end"),
            ("postprocess_ms", "postprocess_begin", "postprocess_end"),
            ("queue_refill_ms", "dispatch_request", "queue_refill"),
            ("refill_to_execute_ms", "queue_refill", "action_execute"),
            ("network_ms", "edge_publish", "edge_receive"),
            ("total_ms", "dispatch_request", "action_execute"),
        ]:
            value = trace.span_ms(start, end)
            if value is not None:
                row[label] = round(value, 2)
        for label, event_name, field_name in [
            ("action_chunk_publish_ms", "action_chunk_publish", "publish_ms"),
            ("dispatch_decode_ms", "dispatch_decode", "decode_ms"),
            ("execute_publish_ms", "action_execute", "publish_ms"),
        ]:
            value = trace.field_ms(event_name, field_name)
            if value is not None:
                row[label] = round(value, 2)
        results.append(row)
    return results


def summarize_observations(events: list[TraceEvent]) -> dict[str, dict[str, list[float]]]:
    summary: dict[str, dict[str, list[float]]] = {}
    for event in events:
        key = str(event.fields.get("key", ""))
        if not key:
            continue
        bucket = summary.setdefault(key, {"transport_ms": [], "age_ms": []})
        if event.event_name == "obs_receive":
            transport_ms = _to_float(event.fields.get("transport_ms"))
            if transport_ms is not None:
                bucket["transport_ms"].append(transport_ms)
        elif event.event_name == "obs_sample" and str(event.fields.get("ready", "0")) == "1":
            age_ms = _to_float(event.fields.get("age_ms"))
            if age_ms is not None:
                bucket["age_ms"].append(age_ms)
    return summary


def print_stats(
    request_results: list[dict[str, Any]],
    observation_summary: dict[str, dict[str, list[float]]],
    fmt: str = "text",
):
    if fmt == "json":
        print(
            json.dumps(
                {
                    "requests": request_results,
                    "observations": observation_summary,
                },
                indent=2,
            )
        )
        return

    fields = [
        "obs_frame_ms",
        "dispatch_to_infer_ms",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "action_chunk_publish_ms",
        "dispatch_decode_ms",
        "queue_refill_ms",
        "refill_to_execute_ms",
        "execute_publish_ms",
        "network_ms",
        "total_ms",
    ]
    labels = {
        "obs_frame_ms": "Obs sampling",
        "dispatch_to_infer_ms": "Dispatch→Infer",
        "preprocess_ms": "Preprocess",
        "inference_ms": "Inference",
        "postprocess_ms": "Postprocess",
        "action_chunk_publish_ms": "Chunk publish",
        "dispatch_decode_ms": "Dispatch decode",
        "queue_refill_ms": "Dispatch→Refill",
        "refill_to_execute_ms": "Refill→Execute",
        "execute_publish_ms": "Execute publish",
        "network_ms": "Network (edge↔cloud)",
        "total_ms": "Dispatch→Execute",
    }

    print("=" * 70)
    print("IB-Robot Inference Chain Latency Report")
    print(f"Requests: {len(request_results)}")
    print("=" * 70)
    print(f"{'Stage':<25} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8} {'n':>5}")
    print("-" * 70)

    for field_name in fields:
        values = sorted(row[field_name] for row in request_results if field_name in row)
        if not values:
            continue
        count = len(values)
        p50 = _percentile(values, 0.50)
        p95 = _percentile(values, 0.95)
        p99 = _percentile(values, 0.99)
        max_value = values[-1]
        tag = ">>>" if field_name == "total_ms" else "   "
        print(f"{tag}{labels[field_name]:<22} {p50:>7.1f}ms {p95:>7.1f}ms {p99:>7.1f}ms {max_value:>7.1f}ms {count:>4}")

    print("=" * 70)
    if observation_summary:
        print("Observation Ingress / Sampling Summary")
        print("=" * 70)
        print(f"{'Observation':<28} {'recv_p50':>9} {'recv_p95':>9} {'age_p50':>9} {'age_p95':>9}")
        print("-" * 70)
        for key in sorted(observation_summary):
            transport_values = sorted(observation_summary[key]["transport_ms"])
            age_values = sorted(observation_summary[key]["age_ms"])
            recv_p50 = _percentile(transport_values, 0.50) if transport_values else None
            recv_p95 = _percentile(transport_values, 0.95) if transport_values else None
            age_p50 = _percentile(age_values, 0.50) if age_values else None
            age_p95 = _percentile(age_values, 0.95) if age_values else None
            print(
                f"{key:<28} "
                f"{(f'{recv_p50:.1f}ms' if recv_p50 is not None else '-'):>9} "
                f"{(f'{recv_p95:.1f}ms' if recv_p95 is not None else '-'):>9} "
                f"{(f'{age_p50:.1f}ms' if age_p50 is not None else '-'):>9} "
                f"{(f'{age_p95:.1f}ms' if age_p95 is not None else '-'):>9}"
            )
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze IB-Robot trace latency")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace-dir", type=Path, help="CTF trace directory")
    source.add_argument("--log-file", type=Path, help="Log file (IB_TRACE_MODE=log)")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    events = parse_ctf_events(args.trace_dir) if args.trace_dir else parse_log_events(args.log_file)
    if not events:
        print("No ib_trace events found.", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(events)} ib_trace events", file=sys.stderr)
    traces = group_by_request(events)
    print(f"Found {len(traces)} requests", file=sys.stderr)

    request_results = analyze_requests(traces)
    observation_summary = summarize_observations(events)
    print_stats(request_results, observation_summary, args.format)


if __name__ == "__main__":
    main()
