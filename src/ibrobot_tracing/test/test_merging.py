import json
from dataclasses import replace

import pytest

from ibrobot_tracing._span_hierarchy import build_span_hierarchy
from ibrobot_tracing.analysis import AnalysisService
from ibrobot_tracing.cli import main
from ibrobot_tracing.clock_calibration import (
    CalibrationPolicy,
    CaptureIdentity,
    ClockCalibration,
    ClockSample,
    PtpStatus,
)
from ibrobot_tracing.critical_path import CriticalPathQuery, project_critical_path
from ibrobot_tracing.merging import merge_datasets, read_merged_trace, write_merged_trace
from ibrobot_tracing.model import EventOrigin, TraceDataset, TraceEvent
from ibrobot_tracing.parsing import parse_lines, parse_trace
from ibrobot_tracing.projection import TimelineQuery, project_timeline


def calibration(machine_id, session="shared"):
    identity = CaptureIdentity(session, machine_id, f"boot-{machine_id}", "eth0", "/dev/ptp0", "gm", 97)
    samples = []
    for index in range(8):
        mono = 10_000_000_000 + index * 20_000_000
        phc = 1_700_000_000_000_000_000 + index * 20_000_000
        status = PtpStatus(
            mono - 1_000, 97, "source", "gm", "SLAVE", "HARDWARE", "eth0", True, 0, 100, phc - 1_000_000, 0
        )
        samples.append(ClockSample(mono, mono - 1, phc + 10_000_000_000, mono + 1, phc, 100, status))
    return ClockCalibration(identity, samples, CalibrationPolicy())


def event(machine, boot, process, sequence, timestamp, *, name="mark", session="shared", fields=None):
    return TraceEvent(
        1_700_000_000_000_000_000 + timestamp,
        name,
        {"request_id": "request", **(fields or {})},
        EventOrigin(
            source="ctf",
            host=machine,
            machine_id=machine,
            boot_id=boot,
            process_instance=process,
            capture_session=session,
            event_sequence=sequence,
        ),
        schema_version=1,
    )


def test_merge_calibrates_and_round_trips_only_shared_events(tmp_path):
    datasets = []
    for index, machine in enumerate(("host-a", "host-b")):
        cal = calibration(machine)
        datasets.append(
            (
                TraceDataset(events=[event(machine, f"boot-{machine}", f"proc-{machine}", 1, 10_080_000_000)]),
                cal,
                machine,
                f"cal-{index}",
            )
        )
    merged = merge_datasets(datasets)
    assert len(merged.events) == 2
    assert not merged.uncalibrated_events
    assert all(event.clock == "ptp_reference" for event in merged.events)
    assert merged.events[0].time_mapping["status"] == "calibrated"
    output = tmp_path / "merged.ibtrace"
    write_merged_trace(output, merged)
    loaded = read_merged_trace(output)
    assert len(loaded.events) == 2
    assert loaded.metadata["kind"] == "ibrobot-merged-trace"


def test_missing_provenance_is_retained_outside_timeline():
    cal = calibration("host-a")
    dataset = TraceDataset(events=[TraceEvent(10_020_000_000, "mark", {"request_id": "request"})])
    merged = merge_datasets(
        [(dataset, cal, "host-a", "cal"), (TraceDataset(), calibration("host-b"), "host-b", "cal-b")]
    )
    assert not merged.events
    assert len(merged.uncalibrated_events) == 1
    assert merged.uncalibrated_events[0].time_mapping["reason"] == "missing_event_provenance"
    assert "machine_id" in merged.uncalibrated_events[0].time_mapping["missing_fields"]
    assert merged.metadata["missing_provenance_by_source"]["host-a"]["capture_session"] == 1


def test_custom_uncertainty_policy_roundtrip_and_tampered_source(tmp_path):
    cal = calibration("host-a")
    cal = ClockCalibration(
        cal.identity, cal.samples, replace(cal.policy, max_uncertainty_ns=100_000, timestamp_bias_budget_ns=60_000)
    )
    dataset = TraceDataset(events=[event("host-a", "boot-host-a", "p", 1, 10_080_000_000)])
    merged = merge_datasets([(dataset, cal, "a", "hash-a"), (TraceDataset(), calibration("host-b"), "b", "hash-b")])
    assert merged.events[0].time_mapping["uncertainty_ns"] > 50_000
    path = tmp_path / "custom.ibtrace"
    write_merged_trace(path, merged)
    assert read_merged_trace(path).events == merged.events
    original = path.read_text().splitlines()
    for key, value in (("source_id", "unknown"), ("calibration_sha256", "wrong"), ("uncertainty_ns", 100_001)):
        lines = original.copy()
        row = json.loads(lines[1])
        row["event"]["time_mapping"][key] = value
        lines[1] = json.dumps(row)
        path.write_text("\n".join(lines) + "\n")
        with pytest.raises(ValueError, match="invalid calibrated event"):
            read_merged_trace(path)
    lines = original.copy()
    header = json.loads(lines[0])
    del header["metadata"]["sources"]
    lines[0] = json.dumps(header)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="invalid calibrated event"):
        read_merged_trace(path)  # Legacy policy-free archives cannot raise the old ceiling.


def test_incomplete_spans_keep_separate_process_lanes():
    sources = []
    for host in ("host-a", "host-b"):
        begin = event(
            host,
            "boot-" + host,
            "p",
            1,
            10_080_000_000,
            name="span_begin",
            fields={"component_id": "worker", "span_id": "work", "span_name": "work"},
        )
        sources.append((TraceDataset(events=[begin]), calibration(host), host, host))
    result = AnalysisService().analyze_dataset(merge_datasets(sources))
    timeline = project_timeline(result, TimelineQuery(request_id="request"))
    spans = [item for item in timeline.items if item.kind == "span"]
    assert len(spans) == 2 and len({item.lane_id for item in spans}) == 2
    assert all(item.status == "incomplete" for item in spans)
    assert {span.time_mapping["source_origin"]["machine_id"] for span in result.spans} == {"host-a", "host-b"}


def test_reference_and_duplicate_inputs_are_rejected():
    dataset = TraceDataset(events=[event("host-a", "boot-host-a", "proc-a", 1, 10_020_000_000)])
    with pytest.raises(ValueError, match="same session"):
        merge_datasets(
            [(dataset, calibration("host-a"), "a", "a"), (dataset, calibration("host-a", session="other"), "b", "b")]
        )
    with pytest.raises(ValueError, match="duplicate merge input"):
        merge_datasets([(dataset, calibration("host-a"), "a", "a"), (dataset, calibration("host-a"), "a", "a")])


def test_conflicting_event_identity_is_rejected():
    left = TraceDataset(events=[event("host-a", "boot-host-a", "proc-a", 1, 10_020_000_000)])
    right = TraceDataset(events=[event("host-a", "boot-host-a", "proc-a", 1, 10_020_000_000, fields={"value": 1})])
    with pytest.raises(ValueError, match="conflicting payload"):
        merge_datasets([(left, calibration("host-a"), "a", "a"), (right, calibration("host-a"), "b", "b")])


def test_archive_rejects_tampered_calibrated_event(tmp_path):
    cal = calibration("host-a")
    merged = merge_datasets(
        [
            (TraceDataset(events=[event("host-a", "boot-host-a", "proc-a", 1, 10_080_000_000)]), cal, "a", "cal"),
            (TraceDataset(), calibration("host-b"), "b", "cal-b"),
        ]
    )
    output = tmp_path / "merged.ibtrace"
    write_merged_trace(output, merged)
    lines = output.read_text().splitlines()
    value = json.loads(lines[1])
    value["event"]["time_mapping"]["status"] = "uncalibrated"
    lines[1] = json.dumps(value)
    output.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="invalid calibrated event"):
        read_merged_trace(output)


def two_host_chain():
    a = calibration("host-a")
    b = calibration("host-b")
    # Different host realtime offsets: raw comparison would be off by 5 s.
    b = ClockCalibration(
        b.identity, [replace(s, realtime_ns=s.realtime_ns + 5_000_000_000) for s in b.samples], b.policy
    )
    left = TraceDataset(
        events=[
            event(
                "host-a",
                "boot-host-a",
                "pa",
                1,
                10_060_000_000,
                name="dispatch_request",
                fields={"component_id": "action_dispatcher.request", "origin": "built-in"},
            ),
            event(
                "host-a",
                "boot-host-a",
                "pa",
                2,
                10_062_000_000,
                name="flow_send",
                fields={"edge_id": "rpc", "flow_id": "f"},
            ),
            event(
                "host-a",
                "boot-host-a",
                "pa",
                3,
                10_100_000_000,
                name="first_action_execute",
                fields={
                    "component_id": "action_dispatcher.execute",
                    "origin": "built-in",
                    "publish_end_ns": 1_700_000_010_105_000_000,
                    "publish_ms": 5,
                },
            ),
        ]
    )
    right = TraceDataset(
        events=[
            event(
                "host-b",
                "boot-host-b",
                "pb",
                1,
                15_065_000_000,
                name="flow_receive",
                fields={"edge_id": "rpc", "flow_id": "f"},
            ),
            event(
                "host-b",
                "boot-host-b",
                "pb",
                2,
                15_067_000_000,
                name="span_begin",
                fields={
                    "span_id": "same",
                    "span_name": "model_call",
                    "component_id": "policy.inference",
                    "origin": "built-in",
                },
            ),
            event(
                "host-b",
                "boot-host-b",
                "pb",
                3,
                15_090_000_000,
                name="span_end",
                fields={
                    "span_id": "same",
                    "span_name": "model_call",
                    "component_id": "policy.inference",
                    "origin": "built-in",
                    "status": "ok",
                    "duration_ns": 22_999_900,
                },
            ),
        ]
    )
    return [(left, a, "edge", "cal-a"), (right, b, "cloud", "cal-b")]


def test_calibrated_flow_request_span_and_critical_path(tmp_path):
    sources = two_host_chain()
    merged = merge_datasets(sources)
    result = AnalysisService().analyze_dataset(merged)
    assert result.flows[0].duration_ms == 3.0
    assert result.flows[0].uncertainty_ns < 100_000
    assert result.spans[0].duration_ms == 22.9999  # preserve locally measured monotonic duration
    hierarchy = build_span_hierarchy(result.spans)
    assert hierarchy.nodes[0].duration_source == "calibrated"
    assert hierarchy.nodes[0].end_ns == result.spans[0].end_ns
    assert hierarchy.nodes[0].duration_ns == 23_000_000
    row = result.request_rows[0]
    assert row["total_ms"] == 45
    assert row["dispatch_to_infer_ms"] == 7
    assert row["total_ms_uncertainty_ns"] < 100_000
    projected = project_critical_path(result, CriticalPathQuery("request"))
    assert projected.totals.by_kind_ns["flow"] == 3_000_000
    last = next(event for event in merged.events if event.name == "first_action_execute")
    assert last.time_mapping["original_fields"]["publish_end_ns"] - last.fields["publish_end_ns"] == 10_000_000_000
    assert sources[0][0].events[-1].clock == "realtime"
    path = tmp_path / "trace.ibtrace"
    write_merged_trace(path, merged)
    loaded = AnalysisService().analyze_dataset(parse_trace(path))
    assert loaded.request_rows == result.request_rows
    assert main(["summary", str(path)]) == 0
    assert main(["events", str(path), "--uncalibrated"]) == 0
    assert main(["export", str(path)]) == 0


def test_timeline_keeps_same_component_on_distinct_hosts_separate():
    sources = two_host_chain()
    for dataset, _, _, _ in sources:
        for item in dataset.events:
            item.fields["component_id"] = "shared-component"
    result = AnalysisService().analyze_dataset(merge_datasets(sources))
    timeline = project_timeline(result, TimelineQuery(request_id="request"))
    component_lanes = [lane for lane in timeline.lanes if lane.kind == "component"]
    assert len(component_lanes) == 2
    assert len({lane.id for lane in component_lanes}) == 2


def test_uncalibrated_request_is_not_partially_reported_as_success():
    sources = two_host_chain()
    sources[1][0].events[0] = replace(
        sources[1][0].events[0], origin=replace(sources[1][0].events[0].origin, boot_id="wrong")
    )
    merged = merge_datasets(sources)
    result = AnalysisService().analyze_dataset(merged)
    assert len(merged.uncalibrated_events) == 1
    assert result.flows[0].status == "uncalibrated"
    assert result.flows[0].duration_ms is None
    assert "total_ms" not in result.request_rows[0]
    assert result.request_rows[0]["clock_status"] == "uncalibrated_events"
    assert result.to_dict()["uncalibrated_events"][0]["time_mapping"]["reason"] == "host_or_boot_mismatch"


def test_process_span_id_collisions_and_duplicate_capture_are_not_conflated():
    sources = two_host_chain()
    a, _, _, _ = sources[0]
    for index, kind in enumerate(("span_begin", "span_end")):
        a.events.append(
            event(
                "host-a",
                "boot-host-a",
                "pa",
                5 + index,
                10_070_000_000 + index * 10_000_000,
                name=kind,
                fields={"span_id": "same", "span_name": "other", "duration_ns": 10_000_000},
            )
        )
    duplicated = sources + [(a, sources[0][1], "edge-copy", "cal-a")]
    merged = merge_datasets(duplicated)
    assert merged.metadata["duplicate_event_count"] == len(a.events)
    result = AnalysisService().analyze_dataset(merged)
    assert len(result.spans) == 2
    assert len({span.span_id for span in result.spans}) == 2


def test_duplicate_flow_endpoints_are_ambiguous():
    sources = two_host_chain()
    original = sources[0][0].events[1]
    sources[0][0].events.append(replace(original, origin=replace(original.origin, event_sequence=99)))
    result = AnalysisService().analyze_dataset(merge_datasets(sources))
    assert result.flows[0].status == "ambiguous"
    assert result.flows[0].duration_ms is None


def test_parser_retains_provenance_and_monotonic_without_overwriting_explicit_time():
    payload = {
        "event": "mark",
        "timestamp_ns": 100,
        "monotonic_ns": 900,
        "provenance": {
            "host": "h",
            "machine_id": "m",
            "boot_id": "b",
            "process_instance": "p",
            "capture_session": "s",
            "process_id": 42,
            "thread_id": 43,
            "event_sequence": 0,
        },
    }
    parsed = parse_lines(["IBTRACE1 " + json.dumps(payload)]).events[0]
    assert parsed.timestamp_ns == 100 and parsed.monotonic_ns == 900
    assert parsed.origin.process_id == 42 and parsed.origin.capture_session == "s"
    payload["monotonic_ns"] = True
    assert not parse_lines(["IBTRACE1 " + json.dumps(payload)]).events


def test_event_boundary_outside_capture_is_rejected_and_incomplete_archive_fails(tmp_path):
    sources = two_host_chain()
    sources[0][0].events[-1].fields["publish_end_ns"] += 1_000_000_000
    merged = merge_datasets(sources)
    assert len(merged.uncalibrated_events) == 1
    path = tmp_path / "truncated.ibtrace"
    write_merged_trace(path, merged)
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        read_merged_trace(path)


def test_split_capture_preserves_local_span_pairing():
    sources = two_host_chain()
    cloud, cal, _, digest = sources.pop()
    sources.extend(
        [
            (TraceDataset(events=cloud.events[:2]), cal, "cloud-a", digest),
            (TraceDataset(events=cloud.events[2:]), cal, "cloud-b", digest),
        ]
    )
    result = AnalysisService().analyze_dataset(merge_datasets(sources))
    assert result.spans[0].status == "ok"
    assert result.spans[0].duration_ms == 22.9999


def test_cli_merge_roundtrip_with_jsonl_logs(tmp_path, capsys):
    args = ["merge"]
    for index, (dataset, cal, _, _) in enumerate(two_host_chain()):
        log = tmp_path / f"host{index}.log"
        calibration_path = tmp_path / f"cal{index}.jsonl"
        from dataclasses import asdict

        calibration_path.write_text(
            "\n".join(
                json.dumps(value)
                for value in [
                    {
                        "kind": "clock_calibration",
                        "schema_version": 1,
                        "identity": asdict(cal.identity),
                        "policy": asdict(cal.policy),
                    },
                    *[{"kind": "sample", "sample": asdict(sample)} for sample in cal.samples],
                ]
            )
            + "\n"
        )
        log.write_text(
            "\n".join(
                "IBTRACE1 "
                + json.dumps(
                    {
                        "event": item.name,
                        "timestamp_ns": item.timestamp_ns,
                        "schema_version": 1,
                        "fields": item.fields,
                        "provenance": {key: value for key, value in asdict(item.origin).items() if value is not None},
                    }
                )
                for item in dataset.events
            )
            + "\n"
        )
        args += ["--input", str(log), str(calibration_path)]
    output = tmp_path / "merged.ibtrace"
    assert main([*args, "--output", str(output)]) == 0
    assert main(["critical-path", str(output), "--request-id", "request", "--format", "json"]) == 0
    assert '"uncertainty_ns"' in capsys.readouterr().out
    assert main([*args, "--output", str(output)]) == 3  # no overwrite


def test_new_provenance_is_protected_and_sequenced(monkeypatch):
    import logging

    from ibrobot_tracing import instrumentation
    from ibrobot_tracing.instrumentation import TraceEmitter

    class Collector(logging.Handler):
        def emit(self, record):
            self.last = record.getMessage()

    collector = Collector()
    logger = logging.Logger("step5", level=logging.INFO)
    logger.addHandler(collector)
    monkeypatch.setenv("IB_TRACE_SESSION_ID", "shared")
    monkeypatch.setattr(instrumentation, "_provenance_state", None)
    emitter = TraceEmitter(logger)
    emitter.event("probe", machine_id="user cannot replace provenance")
    value = json.loads(collector.last.removeprefix("IBTRACE1 "))
    assert value["provenance"]["machine_id"] != value["fields"]["machine_id"]
    assert value["provenance"]["capture_session"] == "shared"
    previous = value["provenance"]["event_sequence"]
    emitter.event("probe")
    assert json.loads(collector.last.removeprefix("IBTRACE1 "))["provenance"]["event_sequence"] == previous + 1


def test_uncalibrated_same_hostname_different_machine_flow_is_not_timed():
    a = TraceEvent(
        100,
        "flow_send",
        {"edge_id": "rpc", "flow_id": "id"},
        origin=EventOrigin(host="same-name", machine_id="a", boot_id="boot-a"),
    )
    b = replace(a, timestamp_ns=200, name="flow_receive", origin=replace(a.origin, machine_id="b"))
    result = AnalysisService().analyze_dataset(TraceDataset(events=[a, b]))
    assert result.flows[0].status == "cross_host_unsynchronized"
    assert result.flows[0].duration_ms is None


def test_concatenated_raw_host_traces_cannot_silently_produce_request_metrics():
    sources = two_host_chain()
    raw = TraceDataset(events=[event for source in sources for event in source[0].events])
    result = AnalysisService().analyze_dataset(raw)
    assert result.request_rows[0] == {"request_id": "request", "clock_status": "cross_host_unsynchronized"}
