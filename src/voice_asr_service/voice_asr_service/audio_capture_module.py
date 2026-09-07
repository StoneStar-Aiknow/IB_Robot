#!/usr/bin/env python3
"""Buffer audio_common PCM messages for the Voice ASR pipeline."""

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np


class CaptureState(Enum):
    UNINITIALIZED = "uninitialized"
    OPENING = "opening"
    CAPTURING = "capturing"
    PAUSED = "paused"
    CLOSED = "closed"
    ERROR = "error"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 512
    buffer_seconds: float = 5.0
    input_channel: int = 0


class RingBuffer:
    """环形缓冲区，用于存储预录音频"""

    def __init__(self, max_samples: int):
        self.max_samples = max_samples
        self.buffer = np.zeros(max_samples, dtype=np.float32)
        self.write_pos = 0
        self.size = 0
        self.lock = threading.Lock()

    def write(self, data: np.ndarray):
        with self.lock:
            n = len(data)
            if n >= self.max_samples:
                self.buffer[:] = data[-self.max_samples :]
                self.write_pos = self.max_samples
                self.size = self.max_samples
            else:
                end_pos = (self.write_pos + n) % self.max_samples
                if end_pos < self.write_pos:
                    split = self.max_samples - self.write_pos
                    self.buffer[self.write_pos :] = data[:split]
                    self.buffer[:end_pos] = data[split:]
                else:
                    self.buffer[self.write_pos : end_pos] = data
                self.write_pos = end_pos
                self.size = min(self.size + n, self.max_samples)

    def read_all(self) -> np.ndarray:
        with self.lock:
            if self.size == 0:
                return np.array([], dtype=np.float32)
            if self.size < self.max_samples:
                return self.buffer[: self.size].copy()
            # Buffer is full; chronological data starts at write_pos
            return np.concatenate([self.buffer[self.write_pos :], self.buffer[: self.write_pos]])

    def read_last(self, n_samples: int) -> np.ndarray:
        with self.lock:
            if n_samples >= self.size:
                return self.read_all()
            start_pos = (self.write_pos - n_samples) % self.max_samples
            if start_pos < self.write_pos:
                return self.buffer[start_pos : self.write_pos].copy()
            else:
                return np.concatenate([self.buffer[start_pos:], self.buffer[: self.write_pos]])

    def clear(self):
        with self.lock:
            self.buffer.fill(0)
            self.write_pos = 0
            self.size = 0


class AudioCaptureModule:
    """Maintain ASR queue and pre-roll state for the shared ROS audio stream."""

    def __init__(self, config: AudioConfig | None = None):
        self.config = config or AudioConfig()
        self.state = CaptureState.UNINITIALIZED

        self._audio_queue: queue.Queue = queue.Queue(maxsize=100)
        self._ring_buffer = RingBuffer(int(self.config.sample_rate * self.config.buffer_seconds))

        self._pause_event = threading.Event()

        self._on_error_callback: Callable[[str], None] | None = None
        self._pending_shared_audio = np.empty(0, dtype=np.float32)

    def set_error_callback(self, callback: Callable[[str], None]):
        self._on_error_callback = callback

    def initialize(self) -> bool:
        if self.state in [CaptureState.CLOSED, CaptureState.PAUSED, CaptureState.CAPTURING]:
            return True
        self.state = CaptureState.OPENING
        self.state = CaptureState.CLOSED
        return True

    def start_capture(self) -> bool:
        if self.state == CaptureState.CAPTURING:
            return True

        if self.state == CaptureState.PAUSED:
            self._pause_event.clear()
            self.state = CaptureState.CAPTURING
            return True

        if self.state != CaptureState.CLOSED and not self.initialize():
            return False
        self._pause_event.clear()
        self.state = CaptureState.CAPTURING
        return True

    def stop_capture(self):
        self.state = CaptureState.CLOSED

    def pause(self):
        if self.state == CaptureState.CAPTURING:
            self._pause_event.set()
            self.state = CaptureState.PAUSED

    def resume(self):
        if self.state == CaptureState.PAUSED:
            self._pause_event.clear()
            self.state = CaptureState.CAPTURING

    def get_audio_chunk(self, timeout: float = 0.1) -> np.ndarray | None:
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_pre_roll_audio(self, seconds: float = 0.3) -> np.ndarray:
        n_samples = int(self.config.sample_rate * seconds)
        return self._ring_buffer.read_last(n_samples)

    def feed_audio(self, data: bytes | np.ndarray, channels: int = 1) -> bool:
        """Feed one interleaved audio_common PCM frame into the ASR queue."""
        if self._pause_event.is_set():
            return False
        if isinstance(data, bytes | bytearray):
            values = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                frames = values[: values.size - values.size % channels].reshape(-1, channels)
                channel = min(max(int(self.config.input_channel), 0), channels - 1)
                values = frames[:, channel]
        else:
            values = np.asarray(data, dtype=np.float32)
            if values.ndim == 2:
                if values.shape[1] == channels:
                    values = values[:, min(max(int(self.config.input_channel), 0), channels - 1)]
                else:
                    values = values[:, 0]
            values = values.reshape(-1)
        if values.size == 0:
            return False
        if self._pending_shared_audio.size:
            values = np.concatenate((self._pending_shared_audio, values))
        frame_size = max(1, int(self.config.chunk_size))
        complete_size = values.size - values.size % frame_size
        self._pending_shared_audio = values[complete_size:]
        accepted = False
        for start in range(0, complete_size, frame_size):
            chunk = values[start : start + frame_size]
            self._ring_buffer.write(chunk)
            try:
                self._audio_queue.put_nowait(chunk)
                accepted = True
            except queue.Full:
                self._handle_error("Shared audio queue is full; dropping ASR frame")
                break
        return accepted

    def clear_buffer(self):
        self._ring_buffer.clear()
        self._pending_shared_audio = np.empty(0, dtype=np.float32)
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def _handle_error(self, message: str):
        self.state = CaptureState.ERROR
        if self._on_error_callback:
            self._on_error_callback(message)

    def cleanup(self):
        self.stop_capture()
