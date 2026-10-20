"""Publish intervals on ``/audio/play``. Subscribes only; never publishes.

Usage (board, while something else plays audio)::

    source .shrc_local
    python3 -m visual_interaction.phase0.pcm_intervals \\
        --label idle --seconds 60 --output measurement.json

``--label`` names the load condition §12 asks to vary (idle, vl_decode,
jpeg_crop, tts_postprocess, combined); it is recorded, never inferred. What
this sees is arrival at a subscriber, which bounds publish-side trouble only —
the vendor audio buffer and the speaker are not visible from here.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from visual_interaction.phase0.pcm import Chunk, PcmFormat, analyse
from visual_interaction.phase0.report import build_report, write_report

TOOL = "pcm_intervals"
DEFAULT_TOPIC = "/audio/play"
# Longer than any plausible stall, shorter than a pause between sentences.
DEFAULT_GAP_SEC = 1.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="the load condition being measured")
    parser.add_argument("--seconds", type=float, required=True, help="how long to listen")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--utterance-gap", type=float, default=DEFAULT_GAP_SEC)
    args = parser.parse_args(argv)

    import rclpy
    from audio_common_msgs.msg import AudioData
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

    pcm = PcmFormat(sample_rate=args.sample_rate, channels=args.channels)
    chunks: list[Chunk] = []

    rclpy.init()
    node = rclpy.create_node("visual_interaction_pcm_intervals")
    # Matches the playback publisher: reliable, keep_last, depth 10.
    qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
    node.create_subscription(
        AudioData,
        args.topic,
        # monotonic, so a clock step cannot invent a gap.
        lambda message: chunks.append(Chunk(arrived=time.monotonic(), payload_bytes=len(message.data))),
        qos,
    )
    print(f"{TOOL}: listening on {args.topic} for {args.seconds:g}s (label={args.label})")
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        print(f"{TOOL}: stopped early")
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if not chunks:
        print(f"{TOOL}: no PCM arrived; nothing was playing, or the topic is wrong", file=sys.stderr)
        return 2

    try:
        results = analyse(chunks, pcm, gap=args.utterance_gap)
    except ValueError as error:
        print(f"{TOOL}: {error}", file=sys.stderr)
        return 2

    report = build_report(
        tool=TOOL,
        conditions={"label": args.label, "topic": args.topic, "listened_sec": args.seconds},
        results=results,
        repo=args.repo,
        notes=[
            "Arrival at a ROS subscriber. Bounds publish-side trouble only; the vendor audio "
            "buffer and the speaker are not observable here, so this does not establish acoustic continuity.",
        ],
    )
    write_report(args.output, report)

    pooled = results["pooled"]
    print(f"  chunks={len(chunks)} utterances={results['utterances']} period={pcm.chunk_period * 1000:.1f}ms")
    for name in ("interval_ms", "lateness_ms"):
        stat = pooled[name]
        if stat.get("n"):
            print(f"  {name:12s} p50={stat['p50']:7.2f} p95={stat['p95']:7.2f} max={stat['max']:7.2f}")
    over = pooled["intervals_over_period"]["1x"]
    print(f"  intervals over one period: {over['count']} ({over['share'] * 100:.1f}%)")
    print(f"{TOOL}: wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
