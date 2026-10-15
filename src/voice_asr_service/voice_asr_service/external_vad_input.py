"""Consume atomic mono S16LE SpeechAudioChunk messages, never cross-topic order.

Only BEGIN can admit a stream. Every failure discards the rest of that segment;
stop and watchdog expiry release the recognizer without publishing partial work.
"""

from __future__ import annotations

import math
import time

import numpy as np

STATE_BEGIN = 1
STATE_PROCESSING = 2
STATE_END = 3


class ExternalVadInput:
    """Own admission, cancellation and deadlines for one external ASR stream."""

    def __init__(
        self,
        *,
        node,
        asr,
        chunk_topic: str,
        sample_rate: int = 16000,
        min_samples: int = 3200,
        max_duration: float = 10.0,
        idle_timeout: float = 2.0,
        publish_partial: bool = True,
        on_final=None,
        on_partial=None,
        on_state=None,
        clock=time.monotonic,
    ) -> None:
        if not chunk_topic or sample_rate != 16000 or min_samples < 0:
            raise ValueError("external input requires a topic, 16000 Hz and non-negative min_samples")
        if not all(math.isfinite(v) and v > 0 for v in (max_duration, idle_timeout)):
            raise ValueError("external input deadlines must be finite and positive")
        self._node, self._asr = node, asr
        self._sample_rate, self._min_samples = sample_rate, min_samples
        self._max_duration, self._idle_timeout = max_duration, idle_timeout
        self._publish_partial = publish_partial
        self._on_final, self._on_partial, self._on_state = on_final, on_partial, on_state
        self._clock = clock
        self._enabled = False
        self._stream_open = False
        self._last_sequence = None
        self._source_id = None
        self._fed = 0
        self._started = self._last_audio = 0.0
        self._last_partial = ""
        from ibrobot_msgs.msg import SpeechAudioChunk

        self._subscription = node.create_subscription(SpeechAudioChunk, chunk_topic, self._on_chunk, 50)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        if not value:
            self.abort()
        self._enabled = bool(value)

    def abort(self) -> None:
        """Discard, never finalize, so cancellation cannot publish a command."""
        was_open = self._stream_open
        self._stream_open = False
        self._fed = 0
        self._last_partial = ""
        try:
            self._asr.reset()
        except Exception as exc:
            self._node.get_logger().error(f"ASR abort failed: {exc}")
        if was_open:
            self._notify_state("listening")

    def check_timeout(self) -> bool:
        if not self._stream_open:
            return False
        now = self._clock()
        if now - self._started >= self._max_duration or now - self._last_audio >= self._idle_timeout:
            self._node.get_logger().warning("External speech deadline expired; discarding segment")
            self.abort()
            return True
        return False

    def _on_chunk(self, message) -> None:
        sequence = int(message.sequence)
        source = message.source_id
        if source != self._source_id:
            if self._stream_open:
                self.abort()
            self._source_id = source
            self._last_sequence = None
        if self._last_sequence is not None and sequence <= self._last_sequence:
            return
        gap = self._last_sequence is not None and sequence != self._last_sequence + 1
        self._last_sequence = sequence
        if not self._enabled:
            return
        state = int(message.state)
        if gap:
            self.abort()
        self.check_timeout()
        try:
            data = bytes(message.audio_data)
            if len(data) % 2 or len(data) > self._sample_rate * 2 * self._max_duration:
                raise ValueError("invalid S16LE chunk length")
            if state not in (0, STATE_BEGIN, STATE_PROCESSING, STATE_END):
                raise ValueError("unknown speech state")
            samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
            if state == STATE_BEGIN:
                if self._stream_open:
                    self.abort()
                if not self._asr.start_streaming():
                    self.abort()
                    return
                self._stream_open = True
                self._fed = 0
                self._started = self._last_audio = self._clock()
                self._last_partial = ""
                self._notify_state("recognizing")
            if not self._stream_open:
                return
            if self._fed + samples.size > self._sample_rate * self._max_duration:
                self.abort()
                return
            if samples.size:
                partial = self._asr.accept_waveform(samples)
                self._fed += samples.size
                self._last_audio = self._clock()
                if partial is not None and partial.text and partial.text != self._last_partial:
                    self._last_partial = partial.text
                    if self._publish_partial and self._on_partial is not None:
                        self._on_partial(partial.text)
            if state == STATE_END:
                # END's PCM is consumed above, before finalization, in one callback.
                result = self._asr.get_final_result()
                enough = self._fed >= self._min_samples
                self._stream_open = False
                self._fed = 0
                self._notify_state("listening")
                if enough and result.text and self._on_final is not None:
                    self._on_final(result.text)
        except Exception as exc:
            self._node.get_logger().error(f"External speech segment dropped: {exc}")
            self.abort()

    def _notify_state(self, state: str) -> None:
        if self._on_state is not None:
            self._on_state(state)
