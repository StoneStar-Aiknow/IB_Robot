"""Export conservative fixed-path validation records without controlling a robot."""

import argparse
import json
from pathlib import Path

import yaml
from aimdk_grasp.fixed_pose_backend import FixedBackend
from aimdk_grasp.path_certificate import compile_certificate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--limits", required=True, help="YAML arm_limits mapping")
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--entry-rad", type=float, required=True)
    parser.add_argument("--tracking-rad", type=float, required=True)
    parser.add_argument("--response-sec", type=float, required=True)
    parser.add_argument("--poll-sec", type=float, default=0.2)
    args = parser.parse_args()
    backend = FixedBackend(args.profile, yaml.safe_load(Path(args.limits).read_text()), args.library)
    data = compile_certificate(
        backend,
        entry_rad=[args.entry_rad] * 14,
        tracking_rad=[args.tracking_rad] * 14,
        response_sec=args.response_sec,
        poll_sec=args.poll_sec,
    )
    Path(args.output).write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(
        json.dumps(
            {
                "digest": data["digest"],
                "routes": [{"proven": list(r["components"]), "refused": r["refused"]} for r in data["routes"]],
            },
            ensure_ascii=False,
        )
    )
    # A record with no proven component is useful evidence, not a usable fast path.
    return 0 if any(r["components"] for r in data["routes"]) else 2
