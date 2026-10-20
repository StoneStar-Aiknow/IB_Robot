"""Time ``/voice_tts/synthesize`` for short replies. Synthesises; never plays.

Usage (board, with the voice service up)::

    source .shrc_local
    python3 -m visual_interaction.phase0.tts_timing \\
        --label alone --runs 30 --output measurement.json

§12 asks for roughly twenty-character sentences from one group, measured
alone, concurrently with VL, and serialised. ``--label`` names which of those
this run is; the concurrent load is created separately (run the VL tool at the
same time) and recorded here by its label only.

Only ``synthesize`` is called, so nothing is heard: playing needs the speech
coordinator that does not exist yet, and §13 puts acoustic measurement in a
separately authorised experiment.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

from visual_interaction.phase0.report import build_report, write_report
from visual_interaction.phase0.stats import counts, summarise

TOOL = "tts_timing"
DEFAULT_ENDPOINT = "/voice_tts/synthesize"

# Replies this feature would actually say: grounded in a whitelisted category,
# around twenty characters, all of them accepted by ``session.semantics``.
DEFAULT_TEXTS = (
    "你举起的玩偶真有趣。",
    "你举着的牌子我看到了。",
    "你戴的帽子很有特点。",
    "你别着的徽章很醒目。",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="alone | vl_concurrent | serialised | ...")
    parser.add_argument("--runs", type=int, default=30, help="§12 asks for at least 30 after warm-up")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--warmup", type=int, default=2, help="discarded runs before measuring")
    parser.add_argument("--text", action="append", default=None, help="repeatable; defaults to four grounded replies")
    args = parser.parse_args(argv)

    texts = tuple(args.text) if args.text else DEFAULT_TEXTS
    if args.runs <= 0:
        print(f"{TOOL}: --runs must be positive", file=sys.stderr)
        return 2

    import rclpy

    from ibrobot_msgs.srv import SynthesizeSpeech

    rclpy.init()
    node = rclpy.create_node("visual_interaction_tts_timing")
    client = node.create_client(SynthesizeSpeech, args.endpoint)
    try:
        if not client.wait_for_service(timeout_sec=30.0):
            print(f"{TOOL}: {args.endpoint} never appeared", file=sys.stderr)
            return 2

        wall_ms: list[float] = []
        reported_ms: list[float] = []
        audio_sec: list[float] = []
        realtime_factor: list[float] = []
        failures: list[str] = []

        total = args.warmup + args.runs
        for index in range(total):
            text = texts[index % len(texts)]
            request = SynthesizeSpeech.Request()
            request.text = text
            started = time.monotonic()
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=args.timeout)
            elapsed = (time.monotonic() - started) * 1000.0
            response = future.result()

            if index < args.warmup:
                continue
            if response is None:
                failures.append("CALL_TIMEOUT")
                continue
            if not response.success:
                failures.append(response.error_code or "UNKNOWN")
                continue
            wall_ms.append(elapsed)
            reported_ms.append(float(response.inference_time_ms))
            duration = float(response.total_duration_sec)
            audio_sec.append(duration)
            if duration > 0.0:
                # Below 1.0 the synthesis is faster than the audio it makes.
                realtime_factor.append(elapsed / 1000.0 / duration)

        results = {
            "requested_runs": args.runs,
            "succeeded": len(wall_ms),
            "failures": counts(failures),
            "wall_ms": summarise(wall_ms),
            "service_reported_ms": summarise(reported_ms),
            "audio_duration_sec": summarise(audio_sec),
            "realtime_factor": summarise(realtime_factor),
            "texts": [{"text": text, "chars": len(text)} for text in texts],
        }
        report = build_report(
            tool=TOOL,
            conditions={
                "label": args.label,
                "endpoint": args.endpoint,
                "warmup": args.warmup,
                "mean_chars": statistics.mean(len(text) for text in texts),
            },
            results=results,
            repo=args.repo,
            notes=[
                "synthesize only; no audio was played, so this is not an acoustic measurement.",
                "Concurrent load, when any, is named by --label and generated by a separate process.",
            ],
        )
        write_report(args.output, report)

        stat = results["wall_ms"]
        if stat.get("n"):
            print(
                f"  wall p50={stat['p50']:.1f}ms p95={stat['p95']:.1f}ms max={stat['max']:.1f}ms "
                f"({stat['n']} runs, label={args.label})"
            )
        if failures:
            print(f"  failures: {results['failures']}")
        print(f"{TOOL}: wrote {args.output}")
        return 0 if wall_ms else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
