#!/usr/bin/env python3
"""Find where a served request spends its time, against the benchmark's 174 ms.

The benchmark called the generator and discriminator directly; a served request
goes through GraspGenXSampler as well, which rebuilds the gripper inputs,
centres the cloud and converts the poses. This times those parts separately so
the difference is attributed rather than guessed at.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# <workspace>/src/<package>/test/<this file>, so the workspace is three
# parents up. A hardcoded default would only ever be right on one host.
WORKSPACE = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[3])
for package in sorted(p.name for p in (WORKSPACE / "src").iterdir() if p.is_dir()):
    sys.path.insert(0, str(WORKSPACE / "src" / package))

from inference_manifest.loader import load_inference_manifest  # noqa: E402
from inference_service.backends.types import RuntimeContext  # noqa: E402
from inference_service.unified_runtime import ExecutionContext, ModelRequest  # noqa: E402
from perception_service.graspgenx_adapter import GraspGenXAdapter  # noqa: E402
from perception_service.model_session_builders import build_graspgenx_session  # noqa: E402

BUNDLE = WORKSPACE / "models/graspgenx_omnihand_left_x2"
RUNS = int(os.environ.get("RUNS", "7"))

totals: dict[str, float] = defaultdict(float)
history: dict[str, list[float]] = defaultdict(list)


def timed(label, function):
    def wrapper(*args, **kwargs):
        torch.npu.synchronize()
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            torch.npu.synchronize()
            totals[label] += (time.perf_counter() - started) * 1e3

    return wrapper


def main() -> None:
    validated = load_inference_manifest(BUNDLE, "torch_npu")
    adapter = GraspGenXAdapter.from_bundle(BUNDLE, model=None, deployment="torch_npu")
    context = RuntimeContext(validated)
    session = build_graspgenx_session(context, adapter=adapter)
    session.load(context)

    module = session._module
    sampler = module.sampler
    model = sampler.model
    generator = model.grasp_generator

    print(f"record_history   {getattr(generator, 'record_history', 'absent')}")
    print(f"om_loop          {type(getattr(generator, 'om_loop', None)).__name__}")
    print(f"num_grasps       {generator.num_grasps_per_object}")
    print(f"diffusion steps  {generator.num_diffusion_iters_eval}", flush=True)

    type(sampler).load_gripper_input = timed("sampler.load_gripper_input", type(sampler).load_gripper_input)
    type(generator).infer = timed("generator.infer", type(generator).infer)
    type(model.grasp_discriminator).infer = timed("discriminator.infer", type(model.grasp_discriminator).infer)

    rng = np.random.default_rng(0)
    extent = np.array([0.08, 0.06, 0.10], dtype=np.float32)
    cloud = (rng.random((2048, 3), dtype=np.float32) - 0.5) * extent
    face = rng.integers(0, 3, size=2048)
    sign = rng.choice([-1.0, 1.0], size=2048).astype(np.float32)
    cloud[np.arange(2048), face] = sign * extent[face] / 2
    cloud += np.array([0.0, 0.0, 0.55], dtype=np.float32)
    request = ModelRequest(
        {
            "observation.object_points": cloud,
            "observation.grasp_sample_count": np.array([adapter.config.num_grasps], np.int64),
        },
        metadata={"service_type": "ibrobot_msgs/srv/GenerateGrasps"},
    )

    for _ in range(2):
        session.execute(request, ExecutionContext(request_id="warmup"))
    totals.clear()

    for index in range(RUNS):
        started = time.perf_counter()
        session.execute(request, ExecutionContext(request_id=f"profile-{index}"))
        wall = (time.perf_counter() - started) * 1e3
        for label, value in totals.items():
            history[label].append(value)
        history["request (wall)"].append(wall)
        totals.clear()
        print(f"  run {index + 1}/{RUNS}  {wall:7.1f} ms", flush=True)

    wall = statistics.median(history.pop("request (wall)"))
    print(f"\n  {'stage':30} {'median ms':>10} {'share':>7}")
    accounted = 0.0
    for label in sorted(history, key=lambda key: -statistics.median(history[key])):
        median = statistics.median(history[label])
        accounted += median
        print(f"  {label:30} {median:10.1f} {median / wall * 100:6.1f}%")
    print(f"  {'-' * 30} {'-' * 10}")
    print(f"  {'sampler overhead (rest)':30} {wall - accounted:10.1f} {(wall - accounted) / wall * 100:6.1f}%")
    print(f"  {'request (wall)':30} {wall:10.1f}")
    session.close()


if __name__ == "__main__":
    main()
