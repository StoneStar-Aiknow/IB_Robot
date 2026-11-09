"""Synthesize a sentence with the rig's running TTS into a WAV file (a test speaker's voice).

rig_env.sh python3 tools/synth_wav.py "欧拉欧拉，今天可以选什么水果？" owner.wav
"""

from __future__ import annotations

import sys

import rclpy

from ibrobot_msgs.srv import SynthesizeSpeech


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    rclpy.init()
    node = rclpy.create_node("x2_local_voice_synth_wav")
    client = node.create_client(SynthesizeSpeech, "/voice_tts/synthesize")
    if not client.wait_for_service(timeout_sec=10.0):
        raise SystemExit("/voice_tts/synthesize is not available; start the rig first")
    request = SynthesizeSpeech.Request()
    request.text = sys.argv[1]
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=60.0)
    result = future.result()
    if result is None or not result.success or not result.audio_segments:
        raise SystemExit(f"synthesis failed: {getattr(result, 'message', 'timeout')}")
    with open(sys.argv[2], "wb") as stream:
        stream.write(bytes(result.audio_segments[0].audio_data))
    print("wrote", sys.argv[2])
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
