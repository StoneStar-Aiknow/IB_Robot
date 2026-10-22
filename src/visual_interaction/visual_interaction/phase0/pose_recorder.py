"""Record the camera to disk, then cut it into labelled rounds by time.

Two steps, deliberately separate. ``record`` saves every frame with its camera
stamp and nothing else — no decisions, no sampling — so one pass in front of
the camera can be sliced many different ways afterwards without asking anybody
to stand there again. ``slice`` turns a time range into a round directory the
probes already read, and writes the ground truth alongside it.

Why time ranges rather than a script that prompts for each pose: a pose held
for five seconds gives roughly 150 frames at 30 Hz, and which two of them a
window should use is exactly what the probes are for. Recording the whole run
and cutting it later keeps that choice out of the recording.

The ground truth here is a label a person assigned, which is the thing the
October set could not provide: those labels were derived from the model's own
free text, so they could only measure whether a closed list carries what the
model already perceives. A human label measures whether the model perceives
correctly at all.

Usage (board):
    # 1. Record while somebody runs through the poses. Ctrl-C when done.
    python3 -m visual_interaction.phase0.pose_recorder record \\
        --out /root/poses_20261022 --seconds 180

    # 2. Cut it up. Times are seconds from the start of the recording.
    python3 -m visual_interaction.phase0.pose_recorder slice \\
        --recording /root/poses_20261022 \\
        --cut 5-12=none --cut 15-22=gesture --cut 25-32=gesture \\
        --cut 60-70=holding --cut 75-85=interacting \\
        --out /root/poses_20261022/rounds

Then point action_probe at --frames-root <out> --truth <out>/truth.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

CAMERA_TOPIC = "/aima/hal/sensor/rgb_head_front_center/rgb_image_half/compressed"
# What the relay publishes, measured on the robot. A mismatched subscription
# receives nothing and reports nothing.
CAMERA_QOS = ("best_effort", "volatile")
# Frames per round directory, matching what the vision OM takes.
ROUND_FRAMES = 4
INDEX_NAME = "index.json"


def _subscribe(node, topic: str, on_frame):
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage

    reliability, durability = CAMERA_QOS
    qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT if reliability == "best_effort" else ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE if durability == "volatile" else DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=30,
    )
    return node.create_subscription(CompressedImage, topic, on_frame, qos)


def record(args) -> int:
    import rclpy

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "frames"
    frames_dir.mkdir(exist_ok=True)

    entries: list[dict] = []
    rclpy.init()
    node = rclpy.create_node("pose_recorder")

    def on_frame(message) -> None:
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        name = f"{len(entries):06d}.jpg"
        (frames_dir / name).write_bytes(bytes(message.data))
        entries.append({"file": name, "stamp": stamp})
        if len(entries) % 150 == 0:
            elapsed = entries[-1]["stamp"] - entries[0]["stamp"]
            print(f"  {len(entries)} frames, {elapsed:.0f}s", flush=True)

    _subscribe(node, args.topic, on_frame)
    print(f"pose_recorder: recording {args.topic} to {frames_dir}")
    print(f"pose_recorder: hold each pose at least {args.hold:g}s; Ctrl-C to stop")
    deadline = time.monotonic() + args.seconds if args.seconds > 0 else None
    try:
        while deadline is None or time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if not entries:
        print("pose_recorder: no frames arrived; check the topic and the QoS", file=sys.stderr)
        return 2
    span = entries[-1]["stamp"] - entries[0]["stamp"]
    index = {
        "topic": args.topic,
        "frames": len(entries),
        "span_sec": round(span, 2),
        "rate_hz": round(len(entries) / span, 1) if span > 0 else None,
        "entries": entries,
    }
    (out / INDEX_NAME).write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"pose_recorder: {len(entries)} frames over {span:.1f}s ({index['rate_hz']} Hz) -> {out / INDEX_NAME}")
    print("pose_recorder: now slice it, e.g. --cut 5-12=gesture")
    return 0


def _parse_cut(value: str) -> tuple[float, float, str]:
    """``START-END=LABEL`` in seconds from the start of the recording."""
    try:
        span, label = value.split("=", 1)
        start, end = span.split("-", 1)
        first, last = float(start), float(end)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"--cut wants START-END=LABEL, got {value!r}") from error
    if last <= first:
        raise argparse.ArgumentTypeError(f"--cut {value!r}: end must be after start")
    return first, last, label.strip()


def slice_(args) -> int:
    recording = Path(args.recording)
    index_path = recording / INDEX_NAME
    if not index_path.exists():
        print(f"pose_recorder: {index_path} not found; record first", file=sys.stderr)
        return 2
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entries = index["entries"]
    origin = entries[0]["stamp"]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    truth: dict[str, str] = {}
    produced = 0
    for number, (first, last, label) in enumerate(args.cut, start=1):
        window = [e for e in entries if first <= e["stamp"] - origin <= last]
        if len(window) < ROUND_FRAMES:
            print(
                f"pose_recorder: {first:g}-{last:g}s has {len(window)} frames, need {ROUND_FRAMES}; skipped",
                file=sys.stderr,
            )
            continue
        # Spread the picks across the range rather than taking the first four:
        # consecutive frames at 30 Hz are nearly identical, and a window is
        # supposed to represent the whole range.
        step = (len(window) - 1) / (ROUND_FRAMES - 1)
        picks = [window[round(i * step)] for i in range(ROUND_FRAMES)]
        name = f"{number:04d}"
        target = out / name
        target.mkdir(exist_ok=True)
        for position, entry in enumerate(picks):
            (target / f"{position}.jpg").write_bytes((recording / "frames" / entry["file"]).read_bytes())
        truth[name] = label
        produced += 1
        print(f"pose_recorder: {name} = {label:12s} {first:g}-{last:g}s ({len(window)} frames, picked {ROUND_FRAMES})")

    if not produced:
        print("pose_recorder: no rounds produced", file=sys.stderr)
        return 1
    truth_path = out / "truth.json"
    payload = {
        "_note": (
            "Labels assigned by a person while recording, not derived from model output. "
            "This measures whether the model perceives the pose correctly, which a "
            "model-derived label cannot."
        ),
        "_recording": str(recording),
        **truth,
    }
    truth_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"pose_recorder: {produced} rounds -> {out}")
    print(f"pose_recorder: truth -> {truth_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    recorder = sub.add_parser("record", help="save every frame with its camera stamp")
    recorder.add_argument("--out", required=True, help="directory to write frames and index.json into")
    recorder.add_argument("--topic", default=CAMERA_TOPIC)
    recorder.add_argument("--seconds", type=float, default=0.0, help="0 runs until interrupted")
    recorder.add_argument("--hold", type=float, default=5.0, help="reminder only: seconds to hold each pose")
    recorder.set_defaults(run=record)

    slicer = sub.add_parser("slice", help="cut a recording into labelled round directories")
    slicer.add_argument("--recording", required=True)
    slicer.add_argument("--out", required=True)
    slicer.add_argument(
        "--cut",
        action="append",
        type=_parse_cut,
        required=True,
        metavar="START-END=LABEL",
        help="seconds from the recording's start, e.g. 5-12=gesture (repeatable)",
    )
    slicer.set_defaults(run=slice_)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
