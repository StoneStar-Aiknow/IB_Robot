"""Fail-closed production promotion gates for semantic mapping backends."""

from dataclasses import dataclass

from perception_service.conformance import ConformanceReport


@dataclass(frozen=True)
class PromotionThresholds:
    minimum_samples: int = 100
    maximum_inference_p50_ms: float = 326.0
    maximum_online_end_to_end_p95_ms: float = 750.0
    maximum_online_queue_wait_p95_ms: float = 100.0
    maximum_online_drop_ratio: float = 0.05
    minimum_offline_throughput_fps: float = 1.0
    maximum_batch_size: int = 8


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    failures: tuple[str, ...]


def evaluate_promotion(
    *,
    mode: str,
    conformance: ConformanceReport,
    timing_report: dict,
    thresholds: PromotionThresholds = PromotionThresholds(),
) -> PromotionDecision:
    if mode not in {"online", "offline"}:
        raise ValueError("promotion mode must be 'online' or 'offline'")
    failures = list(conformance.failures)
    if not conformance.passed and not failures:
        failures.append("backend conformance did not pass")

    samples = int(timing_report.get("processed_frames", 0))
    dropped = int(timing_report.get("dropped_frames", 0))
    batch_size = int(timing_report.get("batch_size", 0))
    stages = timing_report.get("stages", {})
    inference_p50 = float(stages.get("inference_ms", {}).get("p50_ms", float("inf")))
    end_to_end_p95 = float(stages.get("end_to_end_ms", {}).get("p95_ms", float("inf")))
    queue_wait_p95 = float(stages.get("queue_wait_ms", {}).get("p95_ms", float("inf")))
    throughput = float(timing_report.get("throughput_fps", 0.0))

    if samples < thresholds.minimum_samples:
        failures.append("insufficient timing samples")
    if batch_size <= 0 or batch_size > thresholds.maximum_batch_size:
        failures.append("batch size is outside the production bound")
    if inference_p50 > thresholds.maximum_inference_p50_ms:
        failures.append("inference P50 exceeds the reference budget")

    if mode == "online":
        total_frames = samples + dropped
        drop_ratio = dropped / total_frames if total_frames else 1.0
        if end_to_end_p95 > thresholds.maximum_online_end_to_end_p95_ms:
            failures.append("online end-to-end P95 exceeds the latency budget")
        if queue_wait_p95 > thresholds.maximum_online_queue_wait_p95_ms:
            failures.append("online queue-wait P95 exceeds the latency budget")
        if drop_ratio > thresholds.maximum_online_drop_ratio:
            failures.append("online dropped-frame ratio exceeds the budget")
    elif throughput < thresholds.minimum_offline_throughput_fps:
        failures.append("offline throughput is below the production budget")

    return PromotionDecision(promoted=not failures, failures=tuple(dict.fromkeys(failures)))
