#!/usr/bin/env python3
"""Call /voice_tts/synthesize and print segment results.

Usage: python3 call_tts.py [--text "你好"] [--endpoint /voice_tts/synthesize] [--timeout 600]
"""

import argparse

import rclpy
from rclpy.node import Node

from ibrobot_msgs.srv import SynthesizeSpeech


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text", default="你好，异构算力统一推理框架。")
    p.add_argument("--endpoint", default="/voice_tts/synthesize")
    p.add_argument("--timeout", type=float, default=600)
    args = p.parse_args()

    rclpy.init()
    node = Node("tts_verify")
    cli = node.create_client(SynthesizeSpeech, args.endpoint)
    if not cli.wait_for_service(timeout_sec=30):
        raise SystemExit("SERVICE_NOT_AVAILABLE")
    req = SynthesizeSpeech.Request()
    req.text = args.text
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=args.timeout)
    r = fut.result()
    if r is None:
        print("CALL_TIMEOUT")
        return
    m = r.model
    print(
        f"success={r.success} err={r.error_code} msg='{r.message[:80]}' "
        f"segments={len(r.audio_segments)} total_dur={r.total_duration_sec:.2f}s "
        f"infer_ms={r.inference_time_ms:.1f} state={m.runtime_state} backend={m.backend}"
    )
    for i, seg in enumerate(r.audio_segments):
        print(
            f"  seg{i}: '{seg.text[:20]}' {seg.duration_sec:.2f}s bytes={len(seg.audio_data)} "
            f"{seg.audio_format}@{seg.sample_rate} infer_ms={seg.inference_time_ms:.1f}"
        )
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
