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
from voice_asr_service.asr_inference_module import _SherpaStreamingRuntime, normalize_transcript


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
