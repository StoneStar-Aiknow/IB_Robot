"""Tests for the sherpa stream adapter and unified runtime ownership."""

from __future__ import annotations

import numpy as np
import pytest

from inference_service.unified_runtime import (
    ExecutionContext,
    ExecutionContract,
    ModelRequest,
    ModelRuntimeHandle,
    RuntimeAssembly,
)
from voice_asr_service.asr_inference_module import (
    ASRInferenceModule,
    ASRState,
    ModelType,
    _SherpaStreamingRuntime,
    normalize_transcript,
)


class _Result:
    def __init__(self, text: str) -> None:
        self.text = text


class _Stream:
    def __init__(self) -> None:
        self.audio: list[np.ndarray] = []
        self.finished = False

    def accept_waveform(self, _sample_rate: int, audio: np.ndarray) -> None:
        self.audio.append(audio)

    def input_finished(self) -> None:
        self.finished = True


class _Recognizer:
    def __init__(self) -> None:
        self.streams: list[_Stream] = []

    def create_stream(self) -> _Stream:
        stream = _Stream()
        self.streams.append(stream)
        return stream

    def is_ready(self, _stream: _Stream) -> bool:
        return False

    def decode_stream(self, _stream: _Stream) -> None:
        raise AssertionError("fake recognizer should not need decode steps")

    def get_result(self, stream: _Stream) -> _Result:
        return _Result("final" if stream.finished else "partial")


def test_sherpa_stream_is_owned_by_model_runtime_handle() -> None:
    recognizer = _Recognizer()
    runtime = _SherpaStreamingRuntime(recognizer, 16000)
    handle = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=runtime,
            streaming_runtime=runtime,
            session=runtime,
            execution_contract=ExecutionContract(
                state_scope="stream",
                state_bank_mode="per_stream",
                max_open_streams=1,
            ),
            stateful=True,
            resettable=True,
            state_scope="stream",
            state_bank_mode="per_stream",
            max_open_streams=1,
        )
    )
    context = ExecutionContext.create("asr-test")
    handle.load()
    stream = handle.open_stream(context)

    partial = handle.step(stream, ModelRequest(inputs={"audio": np.ones(8, dtype=np.float32)}), context)
    assert partial.outputs == {"text": "partial", "is_final": False}

    handle.reset_stream(stream, context)
    final = handle.step(
        stream,
        ModelRequest(inputs={"audio": np.zeros(0, dtype=np.float32)}, metadata={"final": True}),
        context,
    )
    assert final.outputs == {"text": "final", "is_final": True}

    handle.close_stream(stream, context)
    assert handle.open_stream_count == 0
    handle.close()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("请 打 开 夹 爪", "请打开夹爪"),
        ("  机 械 臂 回 到 零 位 ", "机械臂回到零位"),
        ("打 开 open gripper 好 吗", "打开open gripper好吗"),
        ("open   the gripper", "open the gripper"),
        ("确认，执 行", "确认，执行"),
        ("", ""),
    ],
)
def test_normalize_transcript_drops_separators_between_cjk_only(raw, expected):
    assert normalize_transcript(raw) == expected


def test_final_step_flushes_trailing_speech_with_padding_silence() -> None:
    recognizer = _Recognizer()
    runtime = _SherpaStreamingRuntime(recognizer, 16000, final_tail_padding_seconds=1.5)
    stream = runtime.open_stream(ExecutionContext.create("asr-padding"))
    context = ExecutionContext.create("asr-padding")

    runtime.step(stream, ModelRequest(inputs={"audio": np.ones(512, dtype=np.float32)}), context)
    assert [len(chunk) for chunk in stream.stream.audio] == [512]

    runtime.step(
        stream, ModelRequest(inputs={"audio": np.ones(256, dtype=np.float32)}, metadata={"final": True}), context
    )

    # Padding is appended once, after the last speech and before input_finished.
    assert [len(chunk) for chunk in stream.stream.audio] == [512, 256, 24000]
    assert not stream.stream.audio[-1].any()
    assert stream.stream.finished


def test_failed_final_decode_releases_the_stream_slot_for_the_next_utterance() -> None:
    class _FlakyRecognizer(_Recognizer):
        fail_final = True

        def get_result(self, stream: _Stream) -> _Result:
            if stream.finished and self.fail_final:
                raise RuntimeError("decoder execute failed")
            return super().get_result(stream)

    recognizer = _FlakyRecognizer()
    asr = ASRInferenceModule()
    asr._recognizer = recognizer
    asr._model_type = ModelType.STREAMING
    asr._install_streaming_runtime(_SherpaStreamingRuntime(recognizer, 16000))
    asr.state = ASRState.READY

    assert asr.start_streaming()
    asr.accept_waveform(np.ones(8, dtype=np.float32))
    with pytest.raises(Exception, match="decoder execute failed"):
        asr.end_streaming()

    assert asr.state is ASRState.READY
    assert asr._runtime_handle.open_stream_count == 0

    recognizer.fail_final = False
    assert asr.start_streaming()
    assert asr.end_streaming().text == "final"
    asr.cleanup()


class _UnknownOutcomeRecognizer(_Recognizer):
    """Final decode fails like an async device error: started, outcome unknown."""

    fail_final = True
    reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def get_result(self, stream: _Stream) -> _Result:
        if stream.finished and self.fail_final:
            error = RuntimeError("encoder execute failed")
            error.operation_started = True  # type: ignore[attr-defined]
            error.outcome_known = False  # type: ignore[attr-defined]
            error.state_mutated = True  # type: ignore[attr-defined]
            raise error
        return super().get_result(stream)


def _runtime_exclusive_asr(recognizer: _Recognizer) -> ASRInferenceModule:
    # Same ownership contract as the Ascend 310P runtime: one device-resident
    # state bank, so an unknown step outcome poisons the whole runtime.
    asr = ASRInferenceModule()
    asr._recognizer = recognizer
    asr._model_type = ModelType.STREAMING

    class _ExclusiveRuntime(_SherpaStreamingRuntime):
        def reset(self, _context=None) -> None:
            recognizer.reset()

    asr._install_streaming_runtime(_ExclusiveRuntime(recognizer, 16000), state_bank_mode="runtime_exclusive")
    asr.state = ASRState.READY
    return asr


def test_runtime_exclusive_failed_final_decode_recovers_runtime_for_next_utterance() -> None:
    recognizer = _UnknownOutcomeRecognizer()
    asr = _runtime_exclusive_asr(recognizer)

    assert asr.start_streaming()
    asr.accept_waveform(np.ones(8, dtype=np.float32))
    with pytest.raises(Exception, match="encoder execute failed"):
        asr.end_streaming()

    # The node resets the facade after every utterance; that must also clear
    # the runtime-scope recovery requirement, or every later open_stream()
    # raises recovery_required and realtime capture is stopped for good.
    asr.reset()
    assert asr.state is ASRState.READY
    assert asr._runtime_handle.recovery_requirement is None
    assert recognizer.reset_calls == 1

    recognizer.fail_final = False
    assert asr.start_streaming()
    assert asr.end_streaming().text == "final"
    asr.cleanup()


def test_runtime_exclusive_failed_final_decode_recovers_without_explicit_reset() -> None:
    recognizer = _UnknownOutcomeRecognizer()
    asr = _runtime_exclusive_asr(recognizer)

    assert asr.start_streaming()
    with pytest.raises(Exception, match="encoder execute failed"):
        asr.end_streaming()

    assert recognizer.reset_calls == 1
    recognizer.fail_final = False
    assert asr.start_streaming()
    assert asr.end_streaming().text == "final"
    asr.cleanup()


def test_runtime_reset_failure_reports_error_instead_of_ready() -> None:
    recognizer = _UnknownOutcomeRecognizer()
    asr = _runtime_exclusive_asr(recognizer)

    def _broken_reset(*_args, **_kwargs):
        raise RuntimeError("device reset failed")

    asr._runtime_handle.reset = _broken_reset
    assert asr.start_streaming()
    with pytest.raises(Exception, match="encoder execute failed"):
        asr.end_streaming()
    assert asr.state is ASRState.ERROR

    # A later facade reset retries recovery and keeps reporting the failure.
    asr.reset()

    assert asr.state is ASRState.ERROR
    assert "device reset failed" in asr._last_error
    assert not asr.start_streaming()
    asr.cleanup()


@pytest.mark.parametrize("entry", ["final", "file", "reset"])
def test_failed_stream_close_is_retried_before_next_utterance(entry):
    from inference_service.unified_runtime import ExecutionFailure

    blocked, closes = [True], []

    class Runtime(_SherpaStreamingRuntime):
        def close_stream(self, stream, context):
            closes.append(stream)
            if blocked[0]:
                raise RuntimeError("native close pending")
            return super().close_stream(stream, context)

    asr = ASRInferenceModule()
    asr._recognizer = recognizer = _Recognizer()
    asr._model_type = ModelType.STREAMING
    asr._install_streaming_runtime(Runtime(recognizer, 16000))
    asr.state = ASRState.READY
    try:
        with pytest.raises(ExecutionFailure, match="close pending"):
            if entry == "file":
                asr.recognize_file(np.ones(8, np.float32), enable_vad=False)
            else:
                assert asr.start_streaming()
                asr.reset() if entry == "reset" else asr.end_streaming()
        assert asr._runtime_handle.open_stream_count == 1
        blocked[0] = False
        asr.reset()
        assert asr.state is ASRState.READY and asr._runtime_handle.open_stream_count == 0
        assert asr.start_streaming()
        assert asr.end_streaming().text == "final"
        assert len(closes) == 3
    finally:
        blocked[0] = False
        asr.cleanup()


@pytest.mark.parametrize("closed", [False, True])
def test_failed_initialization_only_reports_owned_cleanup(monkeypatch, closed):
    import importlib.util
    from types import SimpleNamespace

    from voice_asr_service import asr_inference_module

    asr = ASRInferenceModule()
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: object())

    def fail(*_):
        asr._runtime_handle = SimpleNamespace(closed=closed)
        raise RuntimeError("load failed after rollback")

    monkeypatch.setattr(asr_inference_module, "load_inference_manifest", fail)
    with pytest.raises(RuntimeError, match="initialize ASR") as caught:
        asr.initialize("unused", "cpu")
    assert bool(getattr(caught.value, "close_pending", False)) is not closed
    assert (getattr(caught.value, "cleanup_owner", None) is asr) is not closed


@pytest.mark.parametrize("pending", [False, True])
def test_real_handle_load_rollback_controls_initialization_cleanup_evidence(monkeypatch, pending):
    import importlib.util

    from inference_service.backends.errors import BackendLifecycleError
    from voice_asr_service import asr_inference_module

    asr = ASRInferenceModule()
    blocked, closes = [pending], []

    class Runtime(_SherpaStreamingRuntime):
        def load(self, _context=None):
            raise RuntimeError("load failed")

        def close(self):
            closes.append(True)
            if blocked[0]:
                raise BackendLifecycleError("native cleanup pending", close_pending=True)

    monkeypatch.setattr(importlib.util, "find_spec", lambda _: object())

    def install(*_):
        asr._install_streaming_runtime(Runtime(_Recognizer(), 16000))

    monkeypatch.setattr(asr_inference_module, "load_inference_manifest", install)
    try:
        with pytest.raises(RuntimeError, match="initialize ASR") as caught:
            asr.initialize("unused", "cpu")
        assert bool(getattr(caught.value, "close_pending", False)) is pending
        assert (asr._runtime_handle is not None) is pending
        assert closes == [True]
    finally:
        blocked[0] = False
        asr.cleanup()
