#!/usr/bin/env python3
"""Benchmark the synchronized PI0.5 policy callback on Ascend NPU."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from inference_manifest import load_inference_manifest
from inference_service.pipeline import create_pipeline_manager
from inference_service.runtime_composition import build_policy_runtime_dependencies

_PIPELINE_ID = "pi05_910b_latency"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p90_ms": percentile(values, 0.90),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
        "std_ms": statistics.pstdev(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark only the synchronized PI0.5 policy callback. Model loading, preprocessing, "
            "postprocessing, output materialization, framework orchestration, and ROS transport are excluded."
        )
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--deployment", default="torch-npu")
    parser.add_argument("--task", default="pick up the object")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1:
        parser.error("warmup and iterations must both be positive")
    return args


def make_inputs(seed: int, task: str) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    return {
        "observation.images.image": rng.random((1, 3, 256, 256), dtype=np.float32),
        "observation.images.image2": rng.random((1, 3, 256, 256), dtype=np.float32),
        "observation.state": np.zeros((1, 8), dtype=np.float32),
        "task": task,
    }


def validate_action(action: object) -> tuple[list[int], float]:
    candidate = action.detach() if hasattr(action, "detach") else action
    candidate = candidate.cpu() if hasattr(candidate, "cpu") else candidate
    array = np.asarray(candidate)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (50, 7):
        raise RuntimeError(f"unexpected action shape: {array.shape}")
    if not np.isfinite(array).all():
        raise RuntimeError("non-finite action output")
    return list(array.shape), float(np.abs(array).max())


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    validated = load_inference_manifest(args.bundle, args.deployment)
    dependencies = build_policy_runtime_dependencies()
    manager = None
    try:
        manager = create_pipeline_manager(
            _PIPELINE_ID,
            validated,
            request_timeout=1800.0,
            default_task=args.task,
            execution_mode="monolithic",
            registry_set=dependencies.registry_set,
            providers=dependencies.providers,
        )
        pipeline = manager.pipelines[_PIPELINE_ID]
        session = pipeline._session_handle._capability_source
        policy = session.policy
        if policy is None:
            raise RuntimeError("loaded PI0.5 session has no policy")

        # Build canonical NPU inputs once. The timed region intentionally contains
        # only the policy callback and the synchronization required for an accurate
        # asynchronous-device measurement.
        processed_inputs = session.preprocess(make_inputs(args.seed, args.task))
        records: list[dict[str, object]] = []
        measured_ms: list[float] = []
        action_shape: list[int] | None = None
        action_absmax = 0.0

        for phase, count in (("warmup", args.warmup), ("measured", args.iterations)):
            for index in range(count):
                torch.npu.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    action = policy.predict_action_chunk(processed_inputs)
                torch.npu.synchronize()
                model_inference_ms = (time.perf_counter() - started) * 1000.0
                if not math.isfinite(model_inference_ms):
                    raise RuntimeError(f"non-finite model inference latency: {model_inference_ms}")

                # Correctness checks are deliberately outside the timed region.
                action_shape, action_absmax = validate_action(action)
                record = {
                    "phase": phase,
                    "iteration": index + 1,
                    "model_inference_ms": model_inference_ms,
                }
                records.append(record)
                if phase == "measured":
                    measured_ms.append(model_inference_ms)
                print(json.dumps(record, ensure_ascii=False), flush=True)

        report = {
            "status": "PASS",
            "metric": "model_inference_ms",
            "timing_scope": "torch.npu.synchronize -> policy.predict_action_chunk -> torch.npu.synchronize",
            "excluded_from_timing": [
                "model_load",
                "preprocess",
                "postprocess",
                "output_materialize_and_validation",
                "framework_orchestration",
                "ros_transport",
            ],
            "python": sys.executable,
            "bundle": str(args.bundle.resolve()),
            "deployment": args.deployment,
            "device_name": torch.npu.get_device_name(0),
            "warmup": args.warmup,
            "iterations": args.iterations,
            "action_shape": action_shape,
            "action_absmax": action_absmax,
            "summary": summarize(measured_ms),
            "records": records,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2, ensure_ascii=False)
        )
        return 0
    finally:
        if manager is not None:
            manager.close()
        dependencies.providers.close()


if __name__ == "__main__":
    raise SystemExit(main())
