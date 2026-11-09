from types import SimpleNamespace

import numpy as np
import pytest

from inference_service.unified_runtime import ExecutionContext, ExecutionFailure, ModelRequest
from voice_asr_service.transducer_ascend import AscendTransducerRuntime, OnlineFbank


def test_online_fbank_is_chunking_invariant_and_matches_kaldi_frame_count():
    audio = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1
    whole = OnlineFbank()
    whole.accept(audio)
    whole.input_finished()
    chunked = OnlineFbank()
    for start in range(0, len(audio), 512):
        chunked.accept(audio[start : start + 512])
    assert chunked.num_frames == (len(audio) - 280) // 160 + 1  # only fully covered windows before the end
    chunked.input_finished()

    # snip_edges=False: (num_samples + shift/2) // shift frames once input is finished.
    assert whole.num_frames == chunked.num_frames == (16000 + 80) // 160
    np.testing.assert_allclose(chunked.frames(0, chunked.num_frames), whole.frames(0, whole.num_frames), atol=1e-5)
    assert whole.frames(0, 1).shape == (1, 80)


def test_reset_keeps_a_runtime_only_module_ready():
    """The Ascend deployment has no sherpa recognizer; reset must not drop it to IDLE."""
    from voice_asr_service.asr_inference_module import ASRInferenceModule, ASRState

    module = ASRInferenceModule()
    # A healthy handle has no pending stream cleanup or runtime recovery.
    module._runtime_handle = SimpleNamespace(reset_required=False, retry_pending_stream_closes=lambda _: None)
    module.state = ASRState.READY

    module.reset()

    assert module.state is ASRState.READY and module.is_ready


@pytest.mark.parametrize("reset_fails", [False, True])
def test_closed_ascend_stream_recovers_only_after_encoder_reset(reset_fails):
    from inference_service.backends.errors import BackendInferenceError
    from voice_asr_service.asr_inference_module import ASRInferenceModule

    calls = []

    def reset():
        calls.append("encoder-reset")
        if reset_fails:
            raise RuntimeError("encoder reset failed")

    def unknown(*_):
        raise BackendInferenceError("encoder outcome unknown", operation_started=True, outcome_known=False)

    runtime = object.__new__(AscendTransducerRuntime)
    runtime._encoder = SimpleNamespace(reset=reset, close=lambda: None)
    runtime.open_stream = lambda _: object()
    runtime.step = unknown
    module = ASRInferenceModule()
    module._install_streaming_runtime(runtime, state_bank_mode="runtime_exclusive")
    owner = module._runtime_handle
    context = ExecutionContext.create("ascend-reset")
    try:
        stream = owner.open_stream(context)
        with pytest.raises(ExecutionFailure):
            owner.step(stream, ModelRequest(), context)
        owner.close_stream(stream, context)
        assert calls == []
        if reset_fails:
            with pytest.raises(ExecutionFailure, match="encoder reset failed"):
                owner.reset(context)
            assert not owner.ready
            assert owner.recovery_requirement is not None
        else:
            owner.reset(context)
            assert owner.ready
            assert owner.recovery_requirement is None
        assert calls == ["encoder-reset"]
    finally:
        module.cleanup()
