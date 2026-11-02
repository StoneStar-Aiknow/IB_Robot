"""Consume atomic mono S16LE SpeechAudioChunk messages, never cross-topic order.

Only BEGIN can admit a stream. Every failure discards the rest of that segment;
stop and watchdog expiry release the recognizer without publishing partial work.

``finalize_on_idle`` opts out of that last rule for one specific producer fault:
a source that opens with BEGIN, streams PROCESSING and then simply stops without
ever sending END. The X2 firmware does this (measured 2026-10-23: BEGIN 1 /
END 0 over a 4 s utterance), and discarding threw away complete utterances --
voiceprints appeared for speech that was never transcribed. Silence after speech
is exactly what an idle deadline means, so with the option on the segment is
finalized instead. It stays off by default: where a producer does send END, an
idle expiry really is a stall mid-utterance, and publishing a truncated command
to a robot is worse than publishing nothing. A BEGIN over an open segment still
discards it (see _on_chunk).

Which chunks may join a segment (source restarts, stale and missing sequence
numbers) is decided by ``speech_segments``, the same rules ``speaker_id_node``
applies, so a transcript and its voiceprint come from the same segments.
"""

from __future__ import annotations

import math
import time

import numpy as np

from voice_asr_service.speech_segments import STATE_BEGIN, STATE_END, STATE_PROCESSING, SegmentTracker


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
        finalize_on_idle: bool = False,
        publish_partial: bool = True,
        on_final=None,
        on_segment_final=None,
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
        self._finalize_on_idle = bool(finalize_on_idle)
        self._publish_partial = publish_partial
        self._on_final, self._on_partial, self._on_state = on_final, on_partial, on_state
        # Called with the transcript and its segment, (source_id, BEGIN sequence),
        # right after on_final: consumers that pair it with the segment's
        # voiceprint and wake event (SpeechTranscript) need to know which one.
        self._on_segment_final = on_segment_final
        self._segment: tuple[str, int] | None = None
        self._clock = clock
        self._enabled = False
        self._stream_open = False
        self._segments = SegmentTracker()
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
        # The two deadlines mean different things. Exceeding max_duration is a
        # runaway stream and is always discarded; going idle only means the
        # producer stopped sending, which for a source that never sends END is
        # the end of the utterance.
        if now - self._started >= self._max_duration:
            self._node.get_logger().warning("External speech exceeded max_recording_duration; discarding segment")
            self.abort()
            return True
        if now - self._last_audio >= self._idle_timeout:
            if not self._finalize_on_idle:
                self._node.get_logger().warning("External speech deadline expired; discarding segment")
                self.abort()
                return True
            self._node.get_logger().warning("External speech went idle without END; finalizing segment")
            self._finalize()
            return True
        return False

    def _finalize(self) -> None:
        """Close the open stream and publish its transcript if enough was fed.

        Shared by the END branch and the idle path so the two can never diverge
        on what "finished" means.
        """
        result = self._asr.get_final_result()
        fed, enough = self._fed, self._fed >= self._min_samples
        self._stream_open = False
        self._fed = 0
        self._notify_state("listening")
        if not enough or not result.text:
            # Said once per dropped segment, so "the robot heard nothing" can be
            # told apart from "the robot heard it and threw it away".
            reason = f"only {fed} samples (< {self._min_samples})" if not enough else "no text"
            self._node.get_logger().info(f"External speech segment produced no command: {reason}")
            return
        if self._on_final is not None:
            self._on_final(result.text)
        if self._on_segment_final is not None and self._segment is not None:
            self._on_segment_final(result.text, self._segment)

    def _on_chunk(self, message) -> None:
        # Tracked even while disabled, so re-enabling does not read as a gap.
        admission = self._segments.admit(message.source_id, message.sequence)
        if admission.stale:
            return
        if admission.discard and self._stream_open:
            self._node.get_logger().warning(f"External speech {admission.discard}; discarding segment")
            self.abort()
        if not self._enabled:
            return
        state = int(message.state)
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
                    # Discarded even under finalize_on_idle. On the X2 the open
                    # segment is almost always the wake word's residue: the vendor
                    # opens one for it and then a new one for the command. Kept,
                    # it is transcribed ("零一零七") and opens the session with no
                    # voiceprint, which leaves the addressee gate open (measured
                    # 2026-10-25: 5 of 6 rounds). speaker_id discards it the same way.
                    self._node.get_logger().warning("External speech BEGIN while a segment was open; discarding it")
                    self.abort()
                if not self._asr.start_streaming():
                    self._node.get_logger().warning("ASR refused to start a stream; dropping this segment")
                    self.abort()
                    return
                self._stream_open = True
                self._segment = (str(message.source_id), int(message.sequence))
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
                self._finalize()
        except Exception as exc:
            self._node.get_logger().error(f"External speech segment dropped: {exc}")
            self.abort()

    def _notify_state(self, state: str) -> None:
        if self._on_state is not None:
            self._on_state(state)
