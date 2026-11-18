#!/usr/bin/env python3
"""Write the template segment whitelist (``segments.json``) from the pose panel's files.

The pose panel (x2-wt-18, MuJoCo) saves one ``x2_template_pose/v1`` file per
template: ``poses.initial`` is P, ``poses.template`` is T, each with 14 arm
and 20 hand values in the native X2 order. This tool checks the joint names,
copies the values, adds the playback timing and runs every check the session
and the data plane run (``template_motion.registry``: CBF arm limits less 1
degree, hand ranges, min-jerk speed for t_in / t_out / t_out_stop). A pose
that fails is reported and left out; nothing is written unless every pose
asked for passed. ``report_sha256`` is the pose file's sha256.

``--scale 0.5 0.75`` also writes ``<name>@50`` / ``<name>@75``: T moved toward
P by that factor (the hands as well), for a first try on the real robot.

    python3 scripts/x2_motion_template/make_segments.py \\
        --poses /mnt/oee/X2-wt/motion-template-sim/poses \\
        --out src/robot_config/config/motion_template/segments.json

Never sends anything to a robot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "manipulation_execution"))

from manipulation_execution.mirror_session.hand_overlay import HAND_HIGH, HAND_LOW  # noqa: E402
from manipulation_execution.template_motion.config import ARM_LIMITS  # noqa: E402
from manipulation_execution.template_motion.registry import SCHEMA, check_segment  # noqa: E402

ARM_JOINTS = [
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_yaw",
        "wrist_pitch",
        "wrist_roll",
    )
]
HAND_JOINTS = [
    f"{side}_hand_{joint}_joint"
    for side in ("left", "right")
    for joint in (
        "thumb_roll",
        "thumb_abad",
        "thumb_mcp",
        "index_abad",
        "index_pip",
        "middle_pip",
        "ring_abad",
        "ring_pip",
        "pinky_abad",
        "pinky_pip",
    )
]
TEMPLATES = ("overhead_heart", "heart", "holy", "xsign_left_front", "xsign_right_front")


def segment_from_pose(path: Path, timing: dict) -> dict:
    content = path.read_bytes()
    pose = json.loads(content)
    if pose.get("schema") != "x2_template_pose/v1":
        raise ValueError(f"{path.name}: not an x2_template_pose/v1 file")
    if pose.get("arm_joint_names") != ARM_JOINTS or pose.get("hand_joint_names") != HAND_JOINTS:
        raise ValueError(f"{path.name}: joint names differ from the native X2 order")
    P, T = pose["poses"]["initial"], pose["poses"]["template"]
    return {
        "P": [float(v) for v in P["arm"]],
        "T": [float(v) for v in T["arm"]],
        "hand_P": [float(v) for v in P["hand"]],
        "hand_T": [float(v) for v in T["hand"]],
        **timing,
        "report_sha256": hashlib.sha256(content).hexdigest(),
    }


def scaled(entry: dict, factor: float) -> dict:
    out = dict(entry)
    out["T"] = [p + factor * (t - p) for p, t in zip(entry["P"], entry["T"], strict=False)]
    out["hand_T"] = [p + factor * (t - p) for p, t in zip(entry["hand_P"], entry["hand_T"], strict=False)]
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--poses", type=Path, required=True, help="directory of <template>.json pose files")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--templates", nargs="*", default=list(TEMPLATES))
    parser.add_argument("--t-in", type=float, default=1.5)
    parser.add_argument("--t-out", type=float, default=1.5)
    parser.add_argument("--t-out-stop", type=float, default=0.6)
    parser.add_argument("--hold-max", type=float, default=4.0)
    parser.add_argument("--start-tol", type=float, default=0.03)
    parser.add_argument("--hand-start-tol", type=float, default=0.05)
    parser.add_argument("--scale", type=float, nargs="*", default=[], help="also write T scaled toward P")
    args = parser.parse_args(argv)
    timing = {
        "t_in": args.t_in,
        "t_out": args.t_out,
        "t_out_stop": args.t_out_stop,
        "hold_max_s": args.hold_max,
        "start_tol_rad": args.start_tol,
        "hand_start_tol": args.hand_start_tol,
    }
    segments, failed = {}, []
    for name in args.templates:
        try:
            entry = segment_from_pose(args.poses / f"{name}.json", timing)
            variants = {name: entry}
            for factor in args.scale:
                if not 0.0 < factor < 1.0:
                    raise ValueError(f"--scale {factor} must be within (0, 1)")
                variants[f"{name}@{round(factor * 100)}"] = scaled(entry, factor)
            for key, value in variants.items():
                segment = check_segment(key, value, ARM_LIMITS, hand_limits=(HAND_LOW, HAND_HIGH))
                segments[key] = value
                peak = max(abs(t - p) for p, t in zip(segment.P, segment.T, strict=False))
                print(f"ok   {key}: max|T-P| {peak:.3f} rad, sha256 {value['report_sha256'][:12]}")
        except (OSError, KeyError, ValueError) as exc:
            failed.append(name)
            print(f"FAIL {name}: {exc}", file=sys.stderr)
    if failed:
        print(f"nothing written: {', '.join(failed)} failed", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"schema": SCHEMA, "segments": segments}, indent=1, sort_keys=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {len(segments)} segments to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
