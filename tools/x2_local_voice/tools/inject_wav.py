"""Play WAV files into the rig as vendor speech segments, without a microphone.

    rig_env.sh python3 tools/inject_wav.py owner.wav [other.wav ...] [--gap-s 3]

Each file becomes one BEGIN / PROCESSING / END segment of 100 ms blocks on
``/agent/process_audio_output``, resampled to 16 kHz mono, exactly what the local audio HAL
publishes for live speech. Do not speak into the microphone meanwhile: overlapping segments
from two sources make the ASR drop one of them.
"""

from __future__ import annotations

import argparse
import time
import wave

import numpy as np
import rclpy
from aimdk_msgs.msg import ProcessedAudioOutput

BLOCK = 1600  # 100 ms at 16 kHz
BEGIN, PROCESSING, END = 1, 2, 3


def load_16k(path: str) -> np.ndarray:
    with wave.open(path) as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected mono 16-bit PCM")
        rate = stream.getframerate()
        pcm = np.frombuffer(stream.readframes(stream.getnframes()), "<i2").astype(np.float32)
    count = int(len(pcm) * 16000 / rate)
    return np.interp(np.arange(count) * rate / 16000, np.arange(len(pcm)), pcm).astype("<i2")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", nargs="+")
    parser.add_argument("--gap-s", type=float, default=3.0, help="silence after each file")
    args = parser.parse_args()
    rclpy.init()
    node = rclpy.create_node("x2_local_voice_inject_wav")
    publisher = node.create_publisher(ProcessedAudioOutput, "/agent/process_audio_output", 50)
    # A BEGIN sent before the runtime has matched this publisher is lost, and without its
    # BEGIN the ASR drops the whole segment: wait for the subscriber, then settle.
    deadline = time.monotonic() + 10.0
    while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
        time.sleep(0.1)
    if publisher.get_subscription_count() == 0:
        raise SystemExit("no subscriber on /agent/process_audio_output; is the rig running?")
    time.sleep(1.0)
    sequence = 0
    for path in args.wav:
        pcm = load_16k(path)
        blocks = [pcm[start : start + BLOCK] for start in range(0, len(pcm), BLOCK)]
        for index, block in enumerate(blocks):
            message = ProcessedAudioOutput()
            sequence += 1
            message.header.sequence = sequence
            message.stream_id = 7
            message.audio_vad_state.value = BEGIN if index == 0 else END if index == len(blocks) - 1 else PROCESSING
            message.audio_data = block.tobytes()
            publisher.publish(message)
            time.sleep(0.1)
        print(f"injected {path} ({len(pcm) / 16000:.2f} s)", flush=True)
        time.sleep(args.gap_s)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
