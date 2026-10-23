"""Publish saved frames onto a camera topic, so the rest of the chain is real.

Phase-zero only. This exists because the full chain — launch wiring, interface
resolution, subscription, frame buffer, window sampling, the session's
two-window confirmation and de-duplication, the content boundary, utterance
publication — cannot be exercised by unit tests, and the robot's camera is not
always streaming. Everything downstream of the camera topic runs exactly as it
would on a live stream; only the frames are replayed.

What it is NOT: a substitute for a live-camera run. Replay cannot reproduce
real stamp jitter, dropped frames, exposure changes or a visitor actually
moving, and it says nothing about end-to-end latency. It answers "does the
chain hold together", not "does it work on a robot".

Stamps are written as the host's own clock at publish time, not the stamps the
frames were recorded with. The session judges evidence age from the camera
stamp, so replaying week-old stamps would make every window look stale and
every round would be suppressed before reaching the model.

Usage (board):
    source .shrc_local
    python3 -m visual_interaction.phase0.frame_replay \\
        --frames-root <dir of NNNN/{0,1,2,3}.jpg> --rate 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

DEFAULT_TOPIC = "/aima/hal/sensor/stereo_head_front_left/rgb_image/compressed"


def collect(root: Path) -> list[Path]:
    """Every jpg under root, in directory then filename order."""
    frames = sorted(path for path in root.rglob("*.jpg") if path.is_file())
    if not frames:
        raise SystemExit(f"no .jpg under {root}")
    return frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--rate", type=float, default=30.0, help="publish rate in Hz")
    parser.add_argument("--loop", action="store_true", help="repeat once the frames run out")
    parser.add_argument("--seconds", type=float, default=0.0, help="stop after this long; 0 = until interrupted")
    args = parser.parse_args(argv)

    if args.rate <= 0:
        print("frame_replay: --rate must be positive", file=sys.stderr)
        return 2
    frames = collect(args.frames_root)
    blobs = [path.read_bytes() for path in frames]

    import rclpy
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage

    rclpy.init()
    node = rclpy.create_node("visual_interaction_frame_replay")
    # Matches what the vendor camera publishes and what the session subscribes
    # with, so QoS incompatibility cannot silently drop every frame.
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
    publisher = node.create_publisher(CompressedImage, args.topic, qos)

    print(f"frame_replay: {len(blobs)} frames -> {args.topic} at {args.rate:g} Hz")
    period = 1.0 / args.rate
    deadline = time.monotonic() + args.seconds if args.seconds > 0 else None
    published = 0
    try:
        while True:
            for blob in blobs:
                if deadline is not None and time.monotonic() >= deadline:
                    raise KeyboardInterrupt
                message = CompressedImage()
                # The host clock, deliberately: see the module docstring.
                message.header.stamp = node.get_clock().now().to_msg()
                message.header.frame_id = "replay"
                message.format = "jpeg"
                message.data = list(blob)
                publisher.publish(message)
                published += 1
                if published % int(max(1.0, args.rate)) == 0:
                    print(f"  published {published}", flush=True)
                time.sleep(period)
            if not args.loop:
                break
    except KeyboardInterrupt:
        pass
    finally:
        print(f"frame_replay: published {published} frames")
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
