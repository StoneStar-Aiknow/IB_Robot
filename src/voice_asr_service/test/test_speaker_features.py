"""The host's filterbank must be the one the speaker model was trained on.

The model takes features, not audio, so these settings are part of the model's
contract: get them wrong and every embedding is wrong in a way no shape or
dtype check can see. sherpa-onnx computed them before the model became a
bundle, so it is the reference -- when it is installed, this pins the two
together on real speech rather than on a synthetic signal, because a tone
exercises almost none of the mel bins.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from voice_asr_service.speaker_id.features import (
    NUM_MEL_BINS,
    SAMPLE_RATE,
    WINDOW_FRAMES,
    fbank,
    global_mean_normalized,
    num_frames,
    utterance_windows,
)

_AUDIO = Path(__file__).resolve().parent / "speech_direction" / "audio"


def _packaged_model() -> Path | None:
    """The bundle's ONNX artifact, found from wherever the tests were installed."""
    for base in Path(__file__).resolve().parents:
        candidate = base / "models" / "campplus-speaker" / "assets" / "campplus_zh.onnx"
        if candidate.is_file():
            return candidate
    return None


def _speech(seconds: float) -> np.ndarray:
    """One mono 16 kHz channel of real speech, trimmed to ``seconds``."""
    soundfile = pytest.importorskip("soundfile")
    clips = sorted(_AUDIO.glob("*.flac"))
    if not clips:
        pytest.skip("no speech fixtures available")
    data, rate = soundfile.read(str(clips[0]), dtype="float32", always_2d=True)
    channel = np.asarray(data[:, 0], dtype=np.float32)
    if rate != SAMPLE_RATE:
        # Linear resampling is enough: both sides see the identical samples.
        target = int(len(channel) * SAMPLE_RATE / rate)
        channel = np.interp(np.linspace(0, len(channel) - 1, target), np.arange(len(channel)), channel).astype(
            np.float32
        )
    wanted = int(seconds * SAMPLE_RATE)
    if len(channel) < wanted:
        pytest.skip("speech fixture is shorter than the requested clip")
    return channel[:wanted]


def test_the_frame_count_follows_the_ten_millisecond_shift():
    assert num_frames(SAMPLE_RATE) == 100
    assert fbank(np.zeros(SAMPLE_RATE, dtype=np.float32)).shape == (100, NUM_MEL_BINS)


def test_an_empty_waveform_has_no_frames():
    assert fbank(np.zeros(0, dtype=np.float32)).shape == (0, NUM_MEL_BINS)


def test_global_mean_normalization_centres_every_bin():
    features = fbank(_speech(2.0))

    normalized = global_mean_normalized(features)

    assert np.abs(normalized.mean(axis=0)).max() < 1e-4


def pooling_divisor() -> float:
    """How the installed ONNX Runtime divides a pooling window that runs past the input.

    The model implements its context pooling as ``AveragePool`` with a kernel
    baked at export (100 frames, stride 100, ``ceil_mode``). Any utterance whose
    pooled length is not a multiple of that kernel therefore has one partial
    window, and runtimes disagree on its divisor: measured 2026-10-29, ONNX
    Runtime 1.23.2 divides by the full kernel while 1.31.0 divides by the
    elements actually present -- and PyTorch's ``AvgPool1d``, the original
    model's implementation, does the latter. The choice rescales the mask the
    CAM layer applies, so it changes every embedding, and with it the meaning of
    a calibrated similarity threshold.

    Returns the divisor for a 50-element input under a 100-wide kernel: 100 for
    the older reading, 50 for PyTorch's.
    """
    import numpy as np
    import onnxruntime
    from onnx import TensorProto, helper

    node = helper.make_node(
        "AveragePool", ["x"], ["y"], kernel_shape=[100], strides=[100], pads=[0, 0], ceil_mode=1, count_include_pad=1
    )
    graph = helper.make_graph(
        [node],
        "divisor_probe",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 50])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, None])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 8
    session = onnxruntime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    # The window holds 50 ones, so the mean reports the divisor the runtime used.
    mean = float(session.run(None, {"x": np.ones((1, 1, 50), np.float32)})[0].flatten()[0])
    return 50.0 / mean


def test_an_utterance_shorter_than_a_window_is_tiled_not_padded():
    """Silence is not what a speaker sounds like, so the window repeats the voice."""
    features = np.arange(50 * NUM_MEL_BINS, dtype=np.float32).reshape(50, NUM_MEL_BINS)

    windows = utterance_windows(features)

    assert len(windows) == 1
    assert windows[0].shape == (WINDOW_FRAMES, NUM_MEL_BINS)
    # Every frame of the window came from the utterance, none from zeros.
    assert np.isin(windows[0], features).all()
    assert np.array_equal(windows[0][:50], features)


def test_a_long_utterance_is_covered_end_to_end():
    """The last window is aligned to the end, so no speech is dropped."""
    features = np.arange(450 * NUM_MEL_BINS, dtype=np.float32).reshape(450, NUM_MEL_BINS)

    windows = utterance_windows(features)

    assert [w.shape for w in windows] == [(WINDOW_FRAMES, NUM_MEL_BINS)] * 3
    assert np.array_equal(windows[0][0], features[0])
    assert np.array_equal(windows[-1][-1], features[-1])


def test_an_exact_multiple_produces_no_overlapping_window():
    features = np.zeros((2 * WINDOW_FRAMES, NUM_MEL_BINS), dtype=np.float32)

    assert len(utterance_windows(features)) == 2


def test_an_empty_utterance_has_no_windows():
    assert utterance_windows(np.zeros((0, NUM_MEL_BINS), dtype=np.float32)) == []


def test_the_embedder_averages_the_windows_it_was_given():
    """The vector is the mean of the per-window unit vectors, on any deployment.

    Pinning the arithmetic is what lets a CPU anchor be compared with an NPU
    measurement: the runtimes agree per window, so they agree on the mean.
    """
    pytest.importorskip("onnxruntime")
    import onnxruntime

    model = _packaged_model()
    if model is None:
        pytest.skip("speaker embedding bundle is not provisioned")
    from voice_asr_service.speaker_id.embedder import SpeakerEmbedder

    samples = _speech(3.0)
    features = global_mean_normalized(fbank(samples))
    windows = utterance_windows(features)
    assert len(windows) > 1, "a 3 s utterance should span more than one window"
    session = onnxruntime.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    per_window = [
        np.asarray(session.run(None, {"x": w[None].astype(np.float32)})[0][0], dtype=np.float64) for w in windows
    ]
    expected = np.mean([v / np.linalg.norm(v) for v in per_window], axis=0)

    embedder = SpeakerEmbedder(model.parent.parent, "torch_cpu")
    embedder.initialize()
    try:
        produced = np.asarray(embedder.embed((samples * 32768).astype("<i2").tobytes()), dtype=np.float64)
    finally:
        embedder.close()

    cosine = float(produced @ expected / math.sqrt(produced @ produced) / math.sqrt(expected @ expected))
    assert cosine > 0.999999


def test_the_pooling_divisor_is_one_of_the_two_known_readings():
    """Fail loudly on a third behaviour rather than let embeddings drift silently."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")

    divisor = pooling_divisor()

    assert divisor in (50.0, 100.0), (
        f"this ONNX Runtime divides a partial pooling window by {divisor}, which is neither the "
        "full kernel (100) nor the elements present (50); speaker similarity thresholds "
        "calibrated elsewhere cannot be transferred to this host"
    )


def test_the_whole_chain_matches_the_sherpa_extractor_embedding():
    """End to end: our features through the packaged model equal sherpa-onnx's own vector.

    Only comparable when sherpa-onnx's bundled runtime and the installed one read
    a partial pooling window the same way; where they differ the two embeddings
    are both self-consistent but not equal, which is a property of the runtimes
    rather than of this package (see ``pooling_divisor``).
    """
    sherpa_onnx = pytest.importorskip("sherpa_onnx")
    onnxruntime = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    model = _packaged_model()
    if model is None:
        pytest.skip("speaker embedding bundle is not provisioned")
    if pooling_divisor() != 100.0:
        pytest.skip(
            "the installed ONNX Runtime divides a partial pooling window by the elements present, "
            "while sherpa-onnx's bundled runtime divides by the full kernel"
        )
    samples = _speech(3.0)
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model), num_threads=1)
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=samples)
    stream.input_finished()
    expected = np.asarray(extractor.compute(stream), dtype=np.float64)

    session = onnxruntime.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    features = global_mean_normalized(fbank(samples))[None]
    produced = np.asarray(session.run(None, {"x": features})[0][0], dtype=np.float64)

    cosine = float(produced @ expected / math.sqrt(produced @ produced) / math.sqrt(expected @ expected))
    assert cosine > 0.9999


def test_the_bundle_backed_embedder_returns_a_unit_vector_and_its_model_identity():
    """The identity has to come from the manifest, because the numbers cannot carry it.

    Two vectors are comparable exactly when they live in the same vector space,
    and a file name cannot establish that: the same name can hold a different
    model. Published alongside every vector as ``model_id``.
    """
    pytest.importorskip("onnxruntime")
    model = _packaged_model()
    if model is None:
        pytest.skip("speaker embedding bundle is not provisioned")
    from voice_asr_service.speaker_id.embedder import SpeakerEmbedder, voiceprint_model_id

    embedder = SpeakerEmbedder(model.parent.parent, "torch_cpu")
    embedder.initialize()
    try:
        assert embedder.dim == 192
        assert embedder.model_id == voiceprint_model_id("campplus", _campplus_semantic_identity())
        pcm = (_speech(2.0) * 32768).astype("<i2").tobytes()

        vector = embedder.embed(pcm)

        assert len(vector) == 192
        assert math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0, rel_tol=1e-6)
        # Too short to frame at all: no vector rather than a meaningless one.
        assert embedder.embed(b"") == ()
    finally:
        embedder.close()


def test_the_embedder_refuses_a_deployment_that_is_not_an_embedding_model():
    """A wrong deployment name must fail at load, not produce unrelated numbers."""
    pytest.importorskip("onnxruntime")
    model = _packaged_model()
    if model is None:
        pytest.skip("speaker embedding bundle is not provisioned")
    from voice_asr_service.speaker_id.embedder import SpeakerEmbedder

    embedder = SpeakerEmbedder(model.parent.parent, "no_such_deployment")

    with pytest.raises(Exception, match="no_such_deployment|deployment"):
        embedder.initialize()


def _campplus_semantic_identity(**changes):
    from inference_manifest import SemanticIdentity
    from voice_asr_service.package_campplus_bundle import (
        CAMPPLUS_LOGICAL_REVISION,
        CAMPPLUS_POSTPROCESSING,
        CAMPPLUS_PREPROCESSING,
    )

    fields = {
        "logical_model_revision": CAMPPLUS_LOGICAL_REVISION,
        "preprocessing_contract": CAMPPLUS_PREPROCESSING,
        "output_semantics": CAMPPLUS_POSTPROCESSING,
    }
    return SemanticIdentity(**{**fields, **changes})


def test_the_model_id_changes_with_anything_that_moves_a_vector():
    from voice_asr_service.speaker_id.embedder import voiceprint_model_id

    reference = voiceprint_model_id("campplus", _campplus_semantic_identity())

    assert voiceprint_model_id("campplus", _campplus_semantic_identity()) == reference
    for changed in (
        {"logical_model_revision": "campplus@another-checkpoint"},
        {"preprocessing_contract": "kaldi-fbank80-16khz-25ms-10ms-povey-globalmean-v1"},
        {"output_semantics": "speaker-embedding-512-l2-normalized-v1"},
    ):
        assert voiceprint_model_id("campplus", _campplus_semantic_identity(**changed)) != reference


def _campplus_tail_graph(source, destination):
    """The model cut where the Ascend OM is cut: mean and std, before the concat.

    The same two nodes ATC is given as ``--out_nodes``, so what runs on the NPU
    and what runs here are the same half of the graph.
    """
    import onnx

    model = onnx.load(str(source))
    del model.graph.output[:]
    for name in ("/xvector/stats/ReduceMean_output_0", "/xvector/stats/Sqrt_output_0"):
        model.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
    onnx.save(model, str(destination))
    return destination


def test_the_host_head_reproduces_the_graphs_own_tail(tmp_path):
    """The host arithmetic after the NPU must equal the part of the graph it replaces.

    The Ascend deployment runs the model only up to the statistics and leaves the
    concat, dense layer and BatchNorm to the host, so the host half is the only
    thing standing between the NPU's output and a vector the gate compares with
    an anchor. Nothing about it can be checked on the NPU alone: this runs the
    same cut on CPU, where it is the graph's own tail that is the reference.
    """
    onnxruntime = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    model = _packaged_model()
    if model is None:
        pytest.skip("speaker embedding bundle is not provisioned")
    from voice_asr_service.speaker_id.embedder import _load_head

    window = utterance_windows(global_mean_normalized(fbank(_speech(2.0))))[0][None].astype(np.float32)
    whole = onnxruntime.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    expected = np.asarray(whole.run(None, {"x": window})[0][0], dtype=np.float64)

    trunk = onnxruntime.InferenceSession(
        str(_campplus_tail_graph(model, tmp_path / "trunk.onnx")), providers=["CPUExecutionProvider"]
    )
    mean, std = (np.asarray(value, dtype=np.float32).reshape(-1) for value in trunk.run(None, {"x": window}))
    weight, scale, shift = _load_head(model)

    produced = np.asarray(weight @ np.concatenate([mean, std]) * scale + shift, dtype=np.float64)

    assert produced.shape == expected.shape
    assert np.allclose(produced, expected, rtol=0, atol=2e-5), (
        f"host head and graph tail differ by up to {np.abs(produced - expected).max():.3g}"
    )


def test_the_head_refuses_a_batchnorm_parameter_it_cannot_read(tmp_path):
    """A declared parameter that the reader cannot resolve must fail, not default.

    Defaulting a BatchNorm's scale to one and its bias to zero rescales every
    embedding, and a rescaled embedding is still a unit vector of the right
    dimension: no later check can see it. This checkpoint happens to carry an
    identity scale, so only a graph that does not is evidence either way.
    """
    pytest.importorskip("onnx")
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from voice_asr_service.speaker_id.embedder import _load_head

    dim = 4
    dense = np.eye(dim, dtype=np.float32)
    gamma = np.linspace(0.5, 2.0, dim).astype(np.float32)
    beta = np.linspace(-1.0, 1.0, dim).astype(np.float32)
    mean = np.full(dim, 0.25, np.float32)
    variance = np.full(dim, 4.0, np.float32)

    def build(path, *, scale_in_constant: bool):
        nodes, inits = [], [numpy_helper.from_array(dense, "xvector.dense.linear.weight")]
        inits += [numpy_helper.from_array(mean, "bn_mean"), numpy_helper.from_array(variance, "bn_var")]
        if scale_in_constant:
            # Where this model's exporter actually puts them.
            for name, array in (("bn_scale", gamma), ("bn_bias", beta)):
                nodes.append(
                    helper.make_node("Constant", [], [name], value=numpy_helper.from_array(array, name + "_v"))
                )
        else:
            # Declared but unreadable: a dangling name the reader must reject.
            pass
        nodes.append(
            helper.make_node(
                "BatchNormalization", ["y", "bn_scale", "bn_bias", "bn_mean", "bn_var"], ["embedding"], epsilon=1e-5
            )
        )
        graph = helper.make_graph(
            nodes,
            "head_probe",
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, dim])],
            [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, dim])],
            initializer=inits,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
        model.ir_version = 8
        onnx.save(model, str(path))
        return path

    weight, scale, shift = _load_head(build(tmp_path / "readable.onnx", scale_in_constant=True))
    # Folded: scale = gamma / sqrt(var + eps), shift = beta - mean * scale.
    expected_scale = gamma / np.sqrt(variance + 1e-5)
    assert np.allclose(scale, expected_scale)
    assert np.allclose(shift, beta - mean * expected_scale)
    assert np.array_equal(weight, dense)

    with pytest.raises(ValueError, match="scale|bias|cannot resolve"):
        _load_head(build(tmp_path / "unreadable.onnx", scale_in_constant=False))
