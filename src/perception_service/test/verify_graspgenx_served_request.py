#!/usr/bin/env python3
"""Serve one GraspGenX request through IB_Robot's own runtime, on either device.

Everything below the request goes through the framework: the manifest is
loaded and validated, the backend registry admits the deployment, the session
builder picks the path, and the bundle's `torch_module_loader` builds the
model. For `torch_npu` that means the Ascend encoders plus the compiled
diffusion loop; for `torch_cuda` it is the unmodified upstream sampler.

One script serves both so a comparison cannot drift: the cloud, the request and
the checks are shared, and only the deployment differs.

    DEPLOYMENT=torch_npu  ... # the 310P, with the offline loop
    DEPLOYMENT=torch_cuda ... # a CUDA host, upstream's own path

The grasps are sanity-checked, not compared numerically: finite poses,
orthonormal rotations, confidences in range. Parity against a frozen CUDA
baseline is a separate measurement.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# <workspace>/src/<package>/test/<this file>, so the workspace is three
# parents up. A hardcoded default would only ever be right on one host.
WORKSPACE = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[3])
for package in sorted(p.name for p in (WORKSPACE / "src").iterdir() if p.is_dir()):
    sys.path.insert(0, str(WORKSPACE / "src" / package))

DEPLOYMENT = os.environ.get("DEPLOYMENT", "torch_npu")

# An npu deployment exists only on an Ascend device, and its bundle must carry
# the compiled diffusion loop. Skip at collection rather than fail on a host
# that has neither; a cuda run is requested explicitly and needs no guard.
if DEPLOYMENT == "torch_npu":
    try:
        import torch_npu  # noqa: F401 -- registers the NPU device
    except ImportError as exc:  # pragma: no cover - the common case off-device
        import pytest

        pytest.skip(f"Ascend Torch runtime unavailable: {exc}", allow_module_level=True)

from inference_manifest.loader import load_inference_manifest  # noqa: E402
from inference_service.backends.registry import (  # noqa: E402
    STATIC_BACKEND_DESCRIPTORS,
    BackendRegistry,
)
from inference_service.backends.types import RuntimeContext  # noqa: E402
from inference_service.unified_runtime import ExecutionContext, ModelRequest  # noqa: E402
from perception_service.graspgenx_adapter import GraspGenXAdapter  # noqa: E402
from perception_service.model_session_builders import build_graspgenx_session  # noqa: E402

BUNDLE = WORKSPACE / "models/graspgenx_omnihand_left_x2"
POINTS = int(os.environ.get("POINTS", "2048"))
RUNS = int(os.environ.get("RUNS", "3"))


def object_cloud(points: int) -> np.ndarray:
    """A surface sample of a small box, in metres, the same on both devices."""
    rng = np.random.default_rng(0)
    extent = np.array([0.08, 0.06, 0.10], dtype=np.float32)
    cloud = (rng.random((points, 3), dtype=np.float32) - 0.5) * extent
    face = rng.integers(0, 3, size=points)
    sign = rng.choice([-1.0, 1.0], size=points).astype(np.float32)
    cloud[np.arange(points), face] = sign * extent[face] / 2
    return cloud + np.array([0.0, 0.0, 0.55], dtype=np.float32)


def main() -> None:
    validated = load_inference_manifest(BUNDLE, DEPLOYMENT)
    BackendRegistry(STATIC_BACKEND_DESCRIPTORS).validate(RuntimeContext(validated))
    adapter = GraspGenXAdapter.from_bundle(BUNDLE, model=None, deployment=DEPLOYMENT)
    print(
        f"deployment   {DEPLOYMENT} device={validated.deployment.device} "
        f"num_grasps={adapter.config.num_grasps} topk={adapter.config.topk}"
    )
    print(f"artifacts    {sorted(validated.resolved_artifacts or {})}")

    context = RuntimeContext(validated)
    session = build_graspgenx_session(context, adapter=adapter)
    print(f"session      {type(session).__name__}")

    started = time.perf_counter()
    session.load(context)
    load_seconds = time.perf_counter() - started
    print(f"loaded       {load_seconds:.1f} s", flush=True)

    request = ModelRequest(
        {
            "observation.object_points": object_cloud(POINTS),
            "observation.grasp_sample_count": np.array([adapter.config.num_grasps], dtype=np.int64),
        },
        metadata={"service_type": "ibrobot_msgs/srv/GenerateGrasps"},
    )

    timings = []
    for index in range(RUNS):
        started = time.perf_counter()
        outputs = session.execute(request, ExecutionContext(request_id=f"serve-{index}"))
        timings.append((time.perf_counter() - started) * 1e3)
        print(f"  run {index + 1}/{RUNS}  {timings[-1]:7.1f} ms", flush=True)

    poses = np.asarray(outputs["grasp.poses"])
    confidence = np.asarray(outputs["grasp.confidence"]).reshape(-1)
    print(f"\nposes        {poses.shape} {poses.dtype}")
    print(f"confidence   {confidence.shape} range [{confidence.min():.4f}, {confidence.max():.4f}]")

    rotations = poses[:, :3, :3]
    orthonormal = float(np.abs(rotations @ np.transpose(rotations, (0, 2, 1)) - np.eye(3)).max())
    checks = {
        "poses are finite": bool(np.isfinite(poses).all()),
        "rotations are orthonormal": orthonormal < 2e-3,
        "confidences are finite": bool(np.isfinite(confidence).all()),
        "the requested grasp count came back": poses.shape[0] == adapter.config.num_grasps,
    }
    for label, passed in checks.items():
        print(f"  {'ok  ' if passed else 'FAIL'} {label}")
    print(f"  (max |RR^T - I| = {orthonormal:.2e})")

    # The first request pays for kernel caches and internal-format conversions,
    # so it is reported but kept out of the steady-state figure.
    steady = timings[1:] or timings
    print(f"\nload         {load_seconds:.1f} s   (once per process, not per request)")
    print(f"first        {timings[0]:.1f} ms")
    print(f"steady       {statistics.median(steady):.1f} ms  (min {min(steady):.1f}, max {max(steady):.1f})")
    session.close()

    # Cheap, always-on version of the independence check: no module may come
    # from outside this workspace. It cannot see data files or a library opened
    # by torch.ops.load_library -- those were confirmed once with strace, and
    # the paths are asserted by the loader and custom_ops resolver instead.
    outside = sorted(
        name
        for name, module in sys.modules.items()
        if (path := getattr(module, "__file__", None))
        and "GraspGenX" in path
        and str(WORKSPACE) not in path
        and "site-packages" not in path
    )
    print(f"\nmodules from outside the workspace: {outside or 'none'}")
    if outside or not all(checks.values()):
        raise SystemExit("FAILED")
    print(f"served {DEPLOYMENT} end to end")


if __name__ == "__main__":
    main()
