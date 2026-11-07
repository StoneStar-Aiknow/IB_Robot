"""Offline fixed-path timings only; no ROS node, transport or hardware command."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import yaml
from aimdk_grasp.fixed_pose_backend import FixedBackend
from aimdk_grasp.path_validation import ValidationGate


def benchmark(backend):
    home = dict(zip(backend.ARM_JOINTS, [0.4, 0, 0, -1.2, 0, 0, 0] * 2, strict=True))
    up = backend.library["poses"]["arms_up"]["target"]
    results = {}
    for name, start, end in (("raise", home, up), ("return", up, home)):
        # Warm immutable geometry caches before comparing the two preview paths.
        plan = backend.plan_for_execution(start, [end], [0.0])
        observed = SimpleNamespace(feedback=backend.observation(start))
        restorer = SimpleNamespace(
            ports=SimpleNamespace(prepare=lambda _: None),
            arm_baseline=lambda _, baseline=observed: baseline,
            geometry=backend.geometry,
            guard_arm_motion=lambda *_, baseline=observed: baseline,
        )
        backend.body_geometry.check_path(backend.ARM_JOINTS, plan.traj.positions)
        backend.geometry.check_arm_path(observed.feedback, backend.ARM_JOINTS, plan.traj.positions)
        begin = time.perf_counter()
        assert backend.plan_safe(start, [end], [0.0]).ok
        before = time.perf_counter() - begin
        begin = time.perf_counter()
        assert backend.plan_for_execution(start, [end], [0.0]).ok
        after = time.perf_counter() - begin
        result = {
            "strict_preview_sec": before,
            "candidate_preview_sec": after,
            "trajectory_sec": float(plan.traj.times[-1]),
            "gates": {},
        }
        for hand, body in (("online", "online"), ("off", "online"), ("online", "off"), ("off", "off")):
            gate = ValidationGate(backend, {"hand": hand, "body": body})
            begin = time.perf_counter()
            gate.check(restorer, None, plan, 0.005)
            result["gates"][f"{hand}/{body}"] = {"total_sec": time.perf_counter() - begin, **gate.report}
        results[name] = result
    return {"conditions": "Offline open hands/head zero, warm model; no ROS polling or hardware", "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("profile", "limits", "library", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    backend = FixedBackend(args.profile, yaml.safe_load(Path(args.limits).read_text()), args.library)
    result = benchmark(backend)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
