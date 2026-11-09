from types import SimpleNamespace

import numpy as np
import pytest

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
    # A healthy handle: reset() only asks whether the runtime needs recovery.
    module._runtime_handle = SimpleNamespace(reset_required=False)
    module.state = ASRState.READY

    module.reset()

    assert module.state is ASRState.READY and module.is_ready


class _Session:
    def __init__(self, fn):
        self.fn = fn

    def run(self, _outputs, inputs):
        return self.fn(inputs)


def test_greedy_search_times_every_token_by_its_encoder_frame():
    """Token start times are what lets a consumer keep one speaker's part of a segment."""
    from voice_asr_service.asr_inference_module import token_text_ends
    from voice_asr_service.transducer_ascend import AscendTransducerRuntime, _TransducerStream

    # Encoder: 4 output frames per 32-frame chunk (subsampling 8 here), frame value = its index.
    class _Encoder:
        input_shape = (1, 45, 80)
        calls = 0

        def infer(self, _features):
            base = _Encoder.calls * 4
            _Encoder.calls += 1
            return np.array([[base + index] for index in range(4)], dtype=np.float32)

    emit = {1: 5, 6: 7}  # encoder frame -> token id
    runtime = AscendTransducerRuntime.__new__(AscendTransducerRuntime)
    runtime._encoder = _Encoder()
    runtime._window, runtime._chunk, runtime._sample_rate = 45, 32, 16000
    runtime._tail = np.zeros(0, dtype=np.float32)
    runtime._tokens = {5: "你", 7: "\u2581OK"}
    runtime._decoder = _Session(lambda inputs: [np.zeros((1, 2), dtype=np.float32)])

    def joiner(inputs):
        logits = np.zeros((1, 8), dtype=np.float32)
        logits[0, emit.get(int(inputs["encoder_out"][0, 0]), 0)] = 1.0
        return [logits]

    runtime._joiner = _Session(joiner)
    stream = _TransducerStream()
    stream.decoder_out = np.zeros((1, 2), dtype=np.float32)
    request = SimpleNamespace(inputs={"audio": np.zeros(16000, dtype=np.float32)}, metadata={"final": True})
    context = SimpleNamespace(check=lambda _label: None)

    out = runtime.step(stream, request, context)

    assert out["text"] == "你OK"
    assert out["tokens"] == ["你", "\u2581OK"]
    # Frame 1 is input frame 0 + 1 * 32/4 = 8 -> 0.08 s; frame 6 is chunk 1 (input 32) + 2 * 8 = 48 -> 0.48 s.
    assert out["timestamps"] == pytest.approx([0.08, 0.48])
    assert token_text_ends(out["tokens"], out["text"]) == [1, 3]


@pytest.mark.parametrize(
    "tokens,text,ends",
    [
        (["灵", "犀", "挥", "手"], "灵犀挥手", [1, 2, 3, 4]),
        (["\u2581HE", "LLO", "\u2581WORLD", "你"], "HELLO WORLD你", [2, 5, 11, 12]),
        (["你", "\u2581OK"], "你OK", [1, 3]),
        (["你", "好"], "你坏", None),
        (["你"], "你好", None),
        ([], "", None),
    ],
)
def test_token_text_ends_follow_the_published_normalization(tokens, text, ends):
    from voice_asr_service.asr_inference_module import token_text_ends

    assert token_text_ends(tokens, text) == ends
