import numpy as np

from voice_asr_service.transducer_ascend import OnlineFbank


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
    module._runtime_handle = object()
    module.state = ASRState.READY

    module.reset()

    assert module.state is ASRState.READY and module.is_ready
