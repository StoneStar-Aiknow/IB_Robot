"""Tests for the streaming speech-direction compute schedule."""

from __future__ import annotations

import numpy as np

from voice_asr_service.speech_direction.pipeline import DoaState, VadState
from voice_asr_service.speech_direction.pipeline_streaming import (
    StreamingPipelineParams,
    StreamingSpeechDirectionPipeline,
)


class _FakeFullSubNet:
    def __init__(self) -> None:
        self.calls = 0

    def process_4ch(self, value):
        self.calls += 1
        return np.asarray(value, dtype=np.float32)

    def reset(self) -> None:
        pass


class _FakeSilero:
    def __init__(self) -> None:
        self.calls = 0

    def inference(self, value):
        self.calls += 1
        return 1.0

    def reset_state(self) -> None:
        pass


class _FakeSrp:
    def __init__(self) -> None:
        self.angles = np.asarray([90.0, 120.0], dtype=np.float32)
        self.stft_calls = 0
        self.score_calls = 0

    def stft_4ch(self, value):
        self.stft_calls += 1
        return value

    def compute_all_scores(self, value):
        self.score_calls += 1
        return np.asarray([[0.0, 1.0]], dtype=np.float32)


def test_srp_interval_keeps_enhancement_and_vad_continuous() -> None:
    fullnet = _FakeFullSubNet()
    silero = _FakeSilero()
    srp = _FakeSrp()
    params = StreamingPipelineParams(
        processing_samples=256,
        model_batch_samples=512,
        srp_frame_samples=4096,
        srp_hop_samples=512,
        srp_update_interval_hops=2,
        min_segment_samples=512,
        min_accum_samples=512,
        max_accum_samples=16000,
        segment_max_rms_threshold=0.0,
    )
    pipeline = StreamingSpeechDirectionPipeline(
        fullnet,
        silero,
        srp,
        params,
        VadState(),
        DoaState(),
        vad_threshold=0.5,
        rms_threshold=0.0,
    )

    block = np.ones((6, 256), dtype=np.float32)
    for _ in range(24):
        pipeline.process_block(block)

    assert fullnet.calls == 12
    assert silero.calls == 12
    assert srp.stft_calls == 3
    assert srp.score_calls == 3
    assert pipeline._srp_history.shape == (4096, 4)
