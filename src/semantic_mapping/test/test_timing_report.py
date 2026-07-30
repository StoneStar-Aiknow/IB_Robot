"""Tests for semantic mapping P50/P95 timing reports."""

import pytest

from semantic_mapping.timing_report import FrameTiming, TimingReporter


def _timing(value: float) -> FrameTiming:
    return FrameTiming(
        inference_ms=value,
        serialization_ms=value + 1.0,
        queue_wait_ms=value + 2.0,
        service_round_trip_ms=value + 3.0,
        fusion_commit_ms=value + 4.0,
        end_to_end_ms=value + 10.0,
    )


def test_report_includes_all_stages_percentiles_throughput_and_drops() -> None:
    reporter = TimingReporter(backend="cuda", batch_size=4)
    for value in range(1, 101):
        reporter.record(_timing(float(value)))
    reporter.record_drop(3)

    report = reporter.report(elapsed_sec=20.0)

    assert report["backend"] == "cuda"
    assert report["batch_size"] == 4
    assert report["processed_frames"] == 100
    assert report["dropped_frames"] == 3
    assert report["throughput_fps"] == 5.0
    assert set(report["stages"]) == {
        "inference_ms",
        "serialization_ms",
        "queue_wait_ms",
        "service_round_trip_ms",
        "fusion_commit_ms",
        "end_to_end_ms",
    }
    assert report["stages"]["inference_ms"]["p50_ms"] == pytest.approx(50.5)
    assert report["stages"]["inference_ms"]["p95_ms"] == pytest.approx(95.05)
    assert report["stages"]["end_to_end_ms"]["p50_ms"] == pytest.approx(60.5)


def test_empty_report_is_explicit_and_does_not_invent_latency() -> None:
    report = TimingReporter(backend="cpu", batch_size=1).report(elapsed_sec=1.0)

    assert report["processed_frames"] == 0
    assert report["throughput_fps"] == 0.0
    assert report["stages"]["end_to_end_ms"] == {"p50_ms": 0.0, "p95_ms": 0.0}


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TimingReporter(backend="", batch_size=1),
        lambda: TimingReporter(backend="cuda", batch_size=0),
        lambda: FrameTiming(-1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ],
)
def test_invalid_timing_inputs_fail_closed(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_invalid_elapsed_or_drop_count_fails_closed() -> None:
    reporter = TimingReporter(backend="cuda", batch_size=1)
    with pytest.raises(ValueError, match="drop count"):
        reporter.record_drop(0)
    with pytest.raises(ValueError, match="elapsed_sec"):
        reporter.report(elapsed_sec=0.0)
