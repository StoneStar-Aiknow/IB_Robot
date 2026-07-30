"""Tests for semantic mapping backend promotion gates."""

from perception_service.conformance import ConformanceReport
from semantic_mapping.promotion import evaluate_promotion


def _timing() -> dict:
    return {
        "backend": "cuda",
        "batch_size": 8,
        "processed_frames": 100,
        "dropped_frames": 2,
        "throughput_fps": 5.0,
        "stages": {
            "inference_ms": {"p50_ms": 300.0, "p95_ms": 400.0},
            "queue_wait_ms": {"p50_ms": 20.0, "p95_ms": 80.0},
            "end_to_end_ms": {"p50_ms": 450.0, "p95_ms": 700.0},
        },
    }


def test_online_backend_promotes_only_with_conformance_and_bounded_latency() -> None:
    decision = evaluate_promotion(
        mode="online",
        conformance=ConformanceReport(True),
        timing_report=_timing(),
    )

    assert decision.promoted
    assert decision.failures == ()


def test_online_backend_fails_closed_on_latency_queue_drop_and_samples() -> None:
    timing = _timing()
    timing["processed_frames"] = 10
    timing["dropped_frames"] = 10
    timing["stages"]["queue_wait_ms"]["p95_ms"] = 150.0
    timing["stages"]["end_to_end_ms"]["p95_ms"] = 900.0

    decision = evaluate_promotion(
        mode="online",
        conformance=ConformanceReport(True),
        timing_report=timing,
    )

    assert not decision.promoted
    assert set(decision.failures) == {
        "insufficient timing samples",
        "online end-to-end P95 exceeds the latency budget",
        "online queue-wait P95 exceeds the latency budget",
        "online dropped-frame ratio exceeds the budget",
    }


def test_offline_backend_uses_throughput_gate_and_propagates_conformance_failure() -> None:
    timing = _timing()
    timing["throughput_fps"] = 0.5
    decision = evaluate_promotion(
        mode="offline",
        conformance=ConformanceReport(False, failures=("embedding cosine is below threshold",)),
        timing_report=timing,
    )

    assert not decision.promoted
    assert decision.failures == (
        "embedding cosine is below threshold",
        "offline throughput is below the production budget",
    )


def test_batch_and_model_only_reference_budget_are_enforced() -> None:
    timing = _timing()
    timing["batch_size"] = 9
    timing["stages"]["inference_ms"]["p50_ms"] = 327.0
    decision = evaluate_promotion(
        mode="offline",
        conformance=ConformanceReport(True),
        timing_report=timing,
    )

    assert not decision.promoted
    assert "batch size is outside the production bound" in decision.failures
    assert "inference P50 exceeds the reference budget" in decision.failures
