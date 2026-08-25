"""Unified-runtime adapter for the Speech Direction streaming pipeline.

The Speech Direction models retain recurrent state in their backend sessions,
while STFT/OLA, VAD gating, and segment accumulation are host-side state.  No
per-stream device-bank isolation is proved today, so this adapter deliberately
uses one runtime-exclusive stream at a time.
"""

from __future__ import annotations

import inspect
import threading
import time

import numpy as np

from inference_service.unified_runtime import (
    ExecutionContext,
    ExecutionFailure,
    ExecutionFailureFactory,
    ModelRequest,
    ModelResult,
    OutcomeEvidence,
    RecoveryAction,
    RecoveryScope,
    StreamHandle,
    StreamState,
)


class SpeechDirectionStreamingRuntime:
    """Own host-side Speech Direction state behind the unified stream API.

    ``close_backends`` is only enabled by the compatibility constructor used
    when no external ``ModelRuntimeHandle`` is supplied.  Production assembly
    passes ``False`` and puts Sessions/backend resources in separate owned
    assembly entries, so closing this object cannot close a Session.
    """

    state_bank_mode = "runtime_exclusive"
    max_open_streams = 1

    def __init__(
        self,
        pipeline: object,
        *,
        close_backends: bool = False,
        stream_id_prefix: str = "speech-direction",
        failure_factory: ExecutionFailureFactory | None = None,
    ) -> None:
        if pipeline is None:
            raise TypeError("SpeechDirectionStreamingRuntime requires a pipeline")
        if not isinstance(stream_id_prefix, str) or not stream_id_prefix.strip():
            raise ValueError("stream_id_prefix must be non-empty")
        self.pipeline = pipeline
        self.close_backends = bool(close_backends)
        self._stream_id_prefix = stream_id_prefix.strip()
        self._failure_factory = failure_factory or ExecutionFailureFactory()
        self._lock = threading.RLock()
        self._stream: StreamHandle | None = None
        self._closed_stream_ids: set[str] = set()
        self._stream_counter = 0
        self._loaded = False
        self._closed = False
        self._step_active = False

    @property
    def active_stream(self) -> StreamHandle | None:
        with self._lock:
            return self._stream

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._loaded

    def _failure(
        self,
        code: str,
        message: str,
        *,
        operation: str,
        stream_id: str | None = None,
    ) -> ExecutionFailure:
        details: dict[str, object] = {"operation": operation}
        if stream_id is not None:
            details["stream_id"] = stream_id
        return self._failure_factory.create(
            code,
            message,
            evidence=OutcomeEvidence.not_started("admission", **details),
            scope=RecoveryScope.STREAM,
            action=RecoveryAction.NONE,
            details=details,
        )

    def load(self, _context: object = None) -> None:
        """Initialize host state after the handle has loaded backend resources."""

        with self._lock:
            if self._closed:
                raise self._failure("runtime_closed", "Speech Direction streaming runtime is closed", operation="load")
            if self._loaded:
                return

        # The pipeline is constructed with deferred backend reset in the
        # production path.  Reset here, after Session resources are loaded.
        self.pipeline.reset()
        with self._lock:
            self._loaded = True

    def open_stream(self, context: ExecutionContext) -> StreamHandle:
        context.check("stream_admission")
        with self._lock:
            if self._closed:
                raise self._failure(
                    "runtime_closed", "Speech Direction streaming runtime is closed", operation="open_stream"
                )
            if not self._loaded:
                raise self._failure(
                    "runtime_not_ready", "Speech Direction streaming runtime is not loaded", operation="open_stream"
                )
            if self._stream is not None:
                raise self._failure(
                    "stream_capacity_exhausted",
                    "Speech Direction uses a runtime-exclusive state bank",
                    operation="open_stream",
                )
            self._stream_counter += 1
            handle = StreamHandle(f"{self._stream_id_prefix}-{self._stream_counter}")
            self._stream = handle

        try:
            # A newly opened stream always starts with a clean host and device
            # state.  The device state is reset in place; the Session remains
            # loaded and owned by ModelRuntimeHandle.
            self.pipeline.reset()
        except Exception:
            with self._lock:
                self._stream = None
                self._closed_stream_ids.add(handle.stream_id)
            raise
        return handle

    def _resolve_stream(self, stream_handle: StreamHandle | str, operation: str) -> StreamHandle:
        stream_id = stream_handle.stream_id if isinstance(stream_handle, StreamHandle) else stream_handle
        if not isinstance(stream_id, str) or not stream_id:
            raise self._failure("stream_not_found", "invalid Speech Direction stream handle", operation=operation)
        with self._lock:
            active = self._stream
            if active is None:
                code = "stream_closed" if stream_id in self._closed_stream_ids else "stream_not_found"
                raise self._failure(
                    code, f"stream {stream_id!r} is not active", operation=operation, stream_id=stream_id
                )
            if stream_id != active.stream_id or (
                isinstance(stream_handle, StreamHandle) and stream_handle is not active
            ):
                raise self._failure(
                    "stream_not_found",
                    f"stream {stream_id!r} is not owned by this streaming runtime",
                    operation=operation,
                    stream_id=stream_id,
                )
            return active

    def _audio_from_request(self, request: ModelRequest) -> tuple[np.ndarray, int | None]:
        values = request.inputs
        audio = None
        for name in ("audio", "audio_6ch", "observation.audio_6ch", "observation.audio"):
            if name in values:
                audio = values[name]
                break
        if audio is None:
            raise ValueError("Speech Direction stream requests require an 'audio' input")
        capture_start = values.get("capture_start_sample")
        if capture_start is None:
            capture_start = request.metadata.get("capture_start_sample")
        if capture_start is not None and (
            isinstance(capture_start, bool) or not isinstance(capture_start, int) or capture_start < 0
        ):
            raise ValueError("capture_start_sample must be a non-negative integer")
        value = np.asarray(audio, dtype=np.float32)
        expected_hop = getattr(self.pipeline, "hop_size", None)
        if value.ndim != 2 or (isinstance(expected_hop, int) and value.shape[1] != expected_hop):
            raise ValueError(f"audio must have shape (channels, {expected_hop}), got {value.shape}")
        input_channels = getattr(self.pipeline, "input_channels", ())
        if input_channels and max(input_channels) >= value.shape[0]:
            raise ValueError("audio does not contain all configured microphone channels")
        if not np.isfinite(value).all():
            raise ValueError("audio contains NaN or Inf")
        return value, capture_start

    def step(self, stream_handle: StreamHandle, request: ModelRequest, context: ExecutionContext) -> ModelResult:
        active = self._resolve_stream(stream_handle, "step")
        if not isinstance(request, ModelRequest):
            raise TypeError("Speech Direction stream step requires a ModelRequest")
        context.check("stream_step")
        try:
            audio, capture_start = self._audio_from_request(request)
        except (TypeError, ValueError) as exc:
            raise self._failure(
                "invalid_stream_input",
                str(exc),
                operation="step",
                stream_id=active.stream_id,
            ) from exc

        with self._lock:
            if self._step_active:
                raise self._failure(
                    "stream_reentrant",
                    f"stream {active.stream_id!r} already has an active step",
                    operation="step",
                    stream_id=active.stream_id,
                )
            self._step_active = True
        try:
            started = time.perf_counter()
            # Do not catch backend/pipeline exceptions here.  ModelRuntimeHandle
            # needs their started/unknown evidence to gate this exclusive bank.
            hop = self.pipeline.process_block(audio, capture_start_sample=capture_start)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            segment_angle = getattr(hop, "seg_output", None)
            if segment_angle is not None:
                segment_angle = int(segment_angle)
            segment_seq = int(getattr(hop, "seg_seq", 0))
            frame_doa = getattr(hop, "frame_doa", None)
            if frame_doa is not None:
                frame_doa = int(frame_doa)
            vad_state = getattr(self.pipeline, "vad_state", None)
            vad_prob, is_speech = vad_state.get() if vad_state is not None else (0.0, False)
            return ModelResult(
                outputs={
                    "segment_angle": segment_angle,
                    "segment_seq": segment_seq,
                    "frame_doa": frame_doa,
                    "is_gray_hop": bool(getattr(hop, "is_gray_hop", False)),
                    "session_sample": int(getattr(hop, "session_sample", 0)),
                    "vad_prob": float(vad_prob),
                    "is_speech": bool(is_speech),
                },
                latency=elapsed_ms,
                evidence=OutcomeEvidence.completed("backend", state_mutated=True),
                metadata={"stream_id": active.stream_id, "request_id": request.metadata.get("request_id", "")},
            )
        finally:
            with self._lock:
                self._step_active = False

    def reset_stream(self, stream_handle: StreamHandle, context: ExecutionContext) -> None:
        active = self._resolve_stream(stream_handle, "reset_stream")
        context.check("reset_stream")
        with self._lock:
            if self._step_active:
                raise self._failure(
                    "stream_reentrant",
                    f"stream {active.stream_id!r} has an active step",
                    operation="reset_stream",
                    stream_id=active.stream_id,
                )
        self.pipeline.reset()
        # Keep the same stable StreamHandle.  Only its state bank is cleared.
        del active

    def close_stream(self, stream_handle: StreamHandle, context: ExecutionContext) -> None:
        stream_id = stream_handle.stream_id if isinstance(stream_handle, StreamHandle) else stream_handle
        with self._lock:
            if isinstance(stream_id, str) and stream_id in self._closed_stream_ids:
                return
        active = self._resolve_stream(stream_handle, "close_stream")
        context.check("close_stream")
        with self._lock:
            if self._step_active:
                raise self._failure(
                    "stream_reentrant",
                    f"stream {active.stream_id!r} has an active step",
                    operation="close_stream",
                    stream_id=active.stream_id,
                )
        error: BaseException | None = None
        try:
            # Reset before releasing the host bank.  This gives a stateful
            # backend a deterministic final state without closing its Session.
            self.pipeline.reset()
        except BaseException as exc:
            error = exc
        finally:
            with self._lock:
                if self._stream is active:
                    self._stream = None
                    self._closed_stream_ids.add(active.stream_id)
                active._set_state(StreamState.CLOSED)
        if error is not None:
            raise error

    def execute(self, _request: ModelRequest, _context: ExecutionContext) -> object:
        """Prevent request-style callers from bypassing stream admission."""

        raise self._failure(
            "stream_required",
            "Speech Direction requires an open stream",
            operation="execute",
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            active = self._stream
            self._closed = True
            self._stream = None
            if active is not None:
                self._closed_stream_ids.add(active.stream_id)

        # The handle drains active steps before calling this method.  A direct
        # caller still gets best-effort host cleanup.
        close = self.pipeline.close
        try:
            signature = inspect.signature(close)
        except (TypeError, ValueError):
            signature = None
        supports_ownership_flag = (
            signature is None
            or "close_backends" in signature.parameters
            or any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
        )
        if supports_ownership_flag:
            close(close_backends=self.close_backends)
        else:
            # Keep compatibility with a pre-unified pipeline implementation.
            close()


SpeechDirectionStreamingAdapter = SpeechDirectionStreamingRuntime


__all__ = ["SpeechDirectionStreamingAdapter", "SpeechDirectionStreamingRuntime"]
