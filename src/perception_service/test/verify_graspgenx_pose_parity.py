#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Compare the geometric coverage of independently sampled served grasps.

What the existing checks cover and what they do not: the OM comparison puts two
compiled graphs against each other, the served-request script asserts finite
poses, orthonormal rotations and a confidence range, and the package README
reports rotation orthonormality, rotation cosine and confidence ranking. None of
them looks at the pose **translation**, which is what decides whether the
gripper reaches the object. Attention, the FFN and the CPE run in FP16 on the
Ascend path -- rotation degrades to ~2e-3 there against CUDA's ~5e-7 -- and a
systematic translation error (un-centring, denormalisation, a scale slip) would
leave every one of those figures untouched.

So: run this once per device with the same bundle, then compare the two files.

    # on a CUDA host
    DEPLOYMENT=torch_cuda verify_graspgenx_pose_parity.py cuda.npz
    # on the board
    DEPLOYMENT=torch_npu  verify_graspgenx_pose_parity.py npu.npz
    # anywhere
    verify_graspgenx_pose_parity.py --compare cuda.npz npu.npz

The seed fixes the cloud and each device's local RNG stream. CUDA and NPU RNG
streams differ, so samples are NOT paired by rank or by array index. Coverage
uses one-to-one geometric matching within explicit translation/rotation limits;
it is a sampling diagnostic, not a numerical device parity assertion. Use
verify_graspgenx_loop_parity.py to replay identical diffusion inputs.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

WORKSPACE = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[3])
for _package in (
    "inference_manifest",
    "inference_service",
    "model_utils",
    "perception_service",
    "robot_config",
    "torch_models",
):
    sys.path.insert(0, str(WORKSPACE / "src" / _package))

# Importing graspgenx runs a hook that clones gripper descriptions from Hugging
# Face unless both roots exist; serving from a packaged bundle needs neither.
_ASSETS = tempfile.mkdtemp(prefix="graspgenx-parity-")
for _variable in ("GRASPGENX_GRIPPER_CFG_DIR", "GRASPGENX_CHECKPOINT_DIR"):
    os.environ.setdefault(_variable, _ASSETS)


def object_cloud(points: int, seed: int) -> np.ndarray:
    """A compact blob in front of the camera, the shape a segmented object has."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((points, 3)).astype(np.float32) * 0.03) + np.float32([0.45, 0.02, 0.18])


def serve(bundle: Path, deployment: str, cloud: np.ndarray, seed: int = 20261018) -> dict[str, np.ndarray]:
    import torch

    from inference_manifest.loader import load_inference_manifest
    from inference_service.backends.types import RuntimeContext
    from inference_service.unified_runtime import ExecutionContext, ModelRequest
    from perception_service.graspgenx_adapter import GraspGenXAdapter
    from perception_service.model_session_builders import build_graspgenx_session

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    validated = load_inference_manifest(bundle, deployment)
    adapter = GraspGenXAdapter.from_bundle(bundle, model=validated.manifest.model, deployment=validated.deployment)
    context = RuntimeContext(validated)
    session = build_graspgenx_session(context, adapter=adapter)
    session.load(context)
    try:
        inputs = adapter.prepare(cloud)
        if inputs is None:
            raise SystemExit("the generated cloud was rejected as ungraspable; raise --points")
        request = ModelRequest(inputs, metadata={"service_type": "ibrobot_msgs/srv/GenerateGrasps"})
        outputs = session.execute(request, ExecutionContext(request_id="parity"))
        # postprocess reads result.outputs, which is what the plugin hands it;
        # session.execute returns the mapping itself.
        candidates = adapter.postprocess(SimpleNamespace(outputs=outputs))
    finally:
        session.close()

    return {
        "pose": np.stack([c.pose_matrix for c in candidates]).astype(np.float64),
        "confidence": np.asarray([c.confidence for c in candidates], dtype=np.float64),
    }


def geometric_matches(a, b, translation_mm: float, rotation_deg: float):
    """Maximum cardinality one-to-one matching under both pose tolerances."""
    from scipy.optimize import linear_sum_assignment

    if translation_mm <= 0 or rotation_deg <= 0:
        raise ValueError("geometric matching tolerances must be positive")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("pose arrays must be finite")
    translation = np.linalg.norm(a[:, None, :3, 3] - b[None, :, :3, 3], axis=-1) * 1000.0
    rotation = np.degrees(
        np.arccos(
            np.clip(
                (np.einsum("aij,bij->ab", a[:, :3, :3], b[:, :3, :3]) - 1.0) / 2.0,
                -1.0,
                1.0,
            )
        )
    )
    valid = (translation <= translation_mm) & (rotation <= rotation_deg)
    # An invalid edge costs more than all valid edges combined: minimize the
    # number of invalid pairs first, then distances among the valid pairs.
    cost = np.where(valid, (translation / translation_mm + rotation / rotation_deg) / 2, min(len(a), len(b)) + 1)
    i, j = linear_sum_assignment(cost)
    selected = valid[i, j]
    return i[selected], j[selected], translation[i[selected], j[selected]], rotation[i[selected], j[selected]]


def compare(left_path: Path, right_path: Path, translation_mm: float = 5.0, rotation_deg: float = 10.0) -> int:
    left, right = np.load(left_path), np.load(right_path)
    rows = min(len(left["confidence"]), len(right["confidence"]))
    if rows == 0:
        raise SystemExit("one of the runs returned no grasps")
    print(f"{left_path.name}: {len(left['confidence'])} grasps   {right_path.name}: {len(right['confidence'])}\n")

    print(f"Independent samples; geometric coverage within {translation_mm:g} mm and {rotation_deg:g} deg.")
    order_left = np.argsort(-left["confidence"])
    order_right = np.argsort(-right["confidence"])
    for keep in sorted({min(k, rows) for k in (10, 50, 200, rows)}):
        a, b = left["pose"][order_left[:keep]], right["pose"][order_right[:keep]]
        i, _, translation, rotation = geometric_matches(a, b, translation_mm, rotation_deg)
        print(f"  confidence top-{keep:<4} geometric matches {len(i)}/{keep}")
        if len(i):
            print(f"    matched translation max {translation.max():.4f} mm; rotation max {rotation.max():.4f} deg")
    print("\nCoverage is not a numerical parity result; replay fixed noise with verify_graspgenx_loop_parity.py.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, nargs="?", help="Where to write this device's poses.")
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("LEFT", "RIGHT"), help="Compare two runs.")
    parser.add_argument("--bundle", type=Path, default=WORKSPACE / "models/graspgenx_omnihand_left_x2")
    parser.add_argument("--deployment", default=os.environ.get("DEPLOYMENT", "torch_npu"))
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--translation-mm", type=float, default=5.0, help="Geometric match tolerance (default: 5 mm).")
    parser.add_argument("--rotation-deg", type=float, default=10.0, help="Geometric match tolerance (default: 10 deg).")
    arguments = parser.parse_args()

    if arguments.compare:
        return compare(*arguments.compare, arguments.translation_mm, arguments.rotation_deg)
    if arguments.output is None:
        parser.error("give an output path, or --compare two of them")

    cloud = object_cloud(arguments.points, arguments.seed)
    print(f"bundle     {arguments.bundle}")
    print(f"deployment {arguments.deployment}")
    print(f"cloud      {cloud.shape} seed {arguments.seed}", flush=True)
    produced = serve(arguments.bundle, arguments.deployment, cloud, arguments.seed)
    np.savez(
        arguments.output,
        deployment=arguments.deployment,
        seed=arguments.seed,
        points=arguments.points,
        **produced,
    )
    translations = produced["pose"][:, :3, 3]
    print(f"\ngrasps     {len(produced['confidence'])}")
    print(f"confidence [{produced['confidence'].min():.4f}, {produced['confidence'].max():.4f}]")
    print(f"centroid   {translations.mean(axis=0).round(4).tolist()}")
    print(f"wrote      {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
