"""Local stand-in for the X2 audio HAL: this PC's microphone and headphones.

Only the hardware edge is replaced; the runtime, ASR, TTS, voice bridge and Agent are
the production nodes.

* Microphone -> ``/agent/process_audio_output`` (aimdk_msgs/ProcessedAudioOutput), the
  vendor's VAD-segmented speech stream the X2 runtime bridges to ``audio.speech``.
  Segmentation uses the project's own Silero VAD module; BEGIN carries a pre-roll
  backfill like the vendor's (~650 ms).
* ``/aima/hal/audio/playback`` (aimdk_msgs/AudioPlayback), what the runtime hands the
  vendor player, -> local speaker via ``pacat``. A change of ``pkg_name`` flushes
  buffered audio, as hal_audio does.

Started by ``tools/x2_local_voice/start.sh`` through ``rig_env.sh`` (AimDK overlay and a
PulseAudio session). See ``docs/aimdk_x2_local_voice_rig_zh.md``.
"""

from __future__ import annotations

import argparse
import collections
import queue
import shutil
import subprocess
import threading
import time

import numpy as np
import rclpy
from aimdk_msgs.msg import AudioPlayback, ProcessedAudioOutput
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from voice_asr_service.vad_module import VADConfig, VADModule, VADState

RATE = 16000
FRAME = 512  # Silero frame used by the project's VAD
BEGIN, PROCESSING, END = 1, 2, 3


class SileroVad:
    """The project's Silero model in one ORT thread (the manifest session spins on every core)."""

    CONTEXT = 64

    def __init__(self, model_path: str) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self._session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
        self.reset()

    def infer(self, frame: np.ndarray) -> float:
        audio = np.concatenate((self._context, np.asarray(frame, np.float32))).reshape(1, -1)
        out, self._state = self._session.run(
            None, {"input": audio, "state": self._state, "sr": np.asarray(RATE, dtype=np.int64)}
        )
        self._context = audio[0, -self.CONTEXT :].copy()
        return float(out.reshape(-1)[0])

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), np.float32)
        self._context = np.zeros(self.CONTEXT, np.float32)

    def close(self) -> None:
        pass


class LocalAudioHal(Node):
    def __init__(self, args):
        super().__init__("local_x2_audio_hal")
        self.args = args
        self._speech_pub = self.create_publisher(
            ProcessedAudioOutput,
            "/agent/process_audio_output",
            QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE),
        )
        self.create_subscription(
            AudioPlayback,
            "/aima/hal/audio/playback",
            self._on_playback,
            QoSProfile(depth=1000, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE),
        )
        self._sequence = 0
        self._play_queue: queue.Queue = queue.Queue()
        self._player: subprocess.Popen | None = None
        self._player_key = None
        self._player_lock = threading.Lock()
        self._played_bytes = 0
        self._stop = threading.Event()
        threading.Thread(target=self._mic_loop, daemon=True, name="mic").start()
        threading.Thread(target=self._play_loop, daemon=True, name="speaker").start()
        self.get_logger().info(
            f"local audio HAL ready: mic={args.source or 'default'} speaker={args.sink or 'default'}"
        )

    # ---------------------------------------------------------------- microphone
    def _publish(self, state: int, pcm: np.ndarray) -> None:
        message = ProcessedAudioOutput()
        message.header.stamp = self.get_clock().now().to_msg()
        self._sequence += 1
        message.header.sequence = self._sequence
        message.stream_id = 1
        message.audio_vad_state.value = state
        message.audio_data = np.clip(pcm * 32768.0, -32768, 32767).astype("<i2").tobytes()
        self._speech_pub.publish(message)

    def _mic_loop(self) -> None:
        vad = VADModule(
            VADConfig(
                sample_rate=RATE,
                frame_size=FRAME,
                pre_roll_ms=self.args.pre_roll_ms,
                min_speech_duration=0.25,
                min_silence_duration=self.args.end_silence_s,
                post_roll_ms=200.0,
            )
        )
        vad.set_runtime(SileroVad(self.args.vad_model))
        vad.initialize()
        command = ["parecord", "--raw", "--format=s16le", f"--rate={RATE}", "--channels=1", "--latency-msec=30"]
        if self.args.source:
            command.append(f"--device={self.args.source}")
        recorder = subprocess.Popen(command, stdout=subprocess.PIPE)
        recent = collections.deque(maxlen=int(self.args.pre_roll_ms / 1000 * RATE / FRAME) + 1)
        pending: list[np.ndarray] = []
        open_segment = False
        level_t, peak = time.monotonic(), 0.0
        try:
            while not self._stop.is_set():
                raw = recorder.stdout.read(FRAME * 2)
                if len(raw) < FRAME * 2:
                    self.get_logger().error("microphone stream ended (parecord exited)")
                    return
                frame = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                peak = max(peak, float(np.abs(frame).max()))
                result = vad.process(frame)
                recent.append(frame)
                speaking = result.state in (VADState.SPEAKING, VADState.ENDING)
                if not open_segment and result.state == VADState.SPEAKING:
                    # Vendor-style backfill: the speech onset plus pre-roll in BEGIN.
                    open_segment = True
                    self._publish(BEGIN, np.concatenate(list(recent)))
                    pending = []
                    self.get_logger().info("speech BEGIN")
                elif open_segment and speaking:
                    pending.append(frame)
                    if len(pending) >= 3:  # ~96 ms, like the vendor's ~100 ms blocks
                        self._publish(PROCESSING, np.concatenate(pending))
                        pending = []
                elif open_segment:
                    self._publish(END, np.concatenate(pending) if pending else np.zeros(0, np.float32))
                    pending, open_segment = [], False
                    self.get_logger().info("speech END")
                if self.args.show_level and time.monotonic() - level_t >= 2.0:
                    self.get_logger().info(f"mic peak {peak:.3f}")
                    level_t, peak = time.monotonic(), 0.0
        finally:
            recorder.kill()

    # ------------------------------------------------------------------- speaker
    def _on_playback(self, message: AudioPlayback) -> None:
        info = message.info
        key = (str(message.pkg_name), int(info.sample_rate), int(info.channels), str(info.sample_format))
        self._play_queue.put((key, bytes(message.data.data)))

    def _restart_player(self, key) -> None:
        with self._player_lock:
            if self._player is not None:
                self._player.kill()
                self._player.wait()
            _pkg, rate, channels, fmt = key
            command = [
                "pacat",
                "--playback",
                "--raw",
                f"--format={fmt.lower()}",
                f"--rate={rate}",
                f"--channels={channels}",
                "--latency-msec=80",
            ]
            if self.args.sink:
                command.append(f"--device={self.args.sink}")
            self._player = subprocess.Popen(command, stdin=subprocess.PIPE)
            self._player_key = key

    def _play_loop(self) -> None:
        stream_bytes = 0
        while not self._stop.is_set():
            try:
                key, data = self._play_queue.get(timeout=0.5)
            except queue.Empty:
                if stream_bytes and self._player_key:
                    seconds = stream_bytes / (2 * self._player_key[1] * self._player_key[2])
                    self.get_logger().info(f"playback idle after {seconds:.1f} s of audio")
                    stream_bytes = 0
                continue
            if not stream_bytes:
                self.get_logger().info(f"playback start pkg={key[0]!r} rate={key[1]} channels={key[2]}")
            stream_bytes += len(data)
            if key != self._player_key or self._player is None or self._player.poll() is not None:
                if self._player_key is not None and key[0] != self._player_key[0]:
                    self.get_logger().info(f"playback source changed to {key[0]!r}: flushing")
                self._restart_player(key)
            try:
                self._player.stdin.write(data)
                self._player.stdin.flush()
                self._played_bytes += len(data)
            except BrokenPipeError:
                self._player = None

    def close(self) -> None:
        self._stop.set()
        with self._player_lock:
            if self._player is not None:
                self._player.kill()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="", help="PulseAudio source (microphone); empty = default")
    parser.add_argument("--sink", default="", help="PulseAudio sink (headphones); empty = default")
    parser.add_argument("--vad-model", required=True, help="silero_vad.onnx (the ASR's own VAD bundle asset)")
    parser.add_argument("--pre-roll-ms", type=float, default=650.0)
    parser.add_argument("--end-silence-s", type=float, default=0.6)
    parser.add_argument("--show-level", action="store_true")
    args = parser.parse_args()
    for tool in ("parecord", "pacat"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not found")
    rclpy.init()
    node = LocalAudioHal(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
