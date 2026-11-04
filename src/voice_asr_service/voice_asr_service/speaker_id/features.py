"""Features for the speaker embedding model: Kaldi fbank, normalized and cut into fixed windows.

The model takes features rather than PCM, so the host computes them the way the
model was trained: the shared Kaldi fbank (``voice_asr_service.kaldi_fbank``,
also the transducer's), with sherpa-onnx's speaker extractor settings, which is
how the model was run before it became a bundle. ``global-mean`` normalization
then subtracts the per-utterance mean of every bin.
``voice_asr_service/test/test_speaker_features.py`` pins this against
sherpa-onnx's own extractor when it is installed.
"""

from __future__ import annotations

import numpy as np

from voice_asr_service.kaldi_fbank import NUM_MEL_BINS, SAMPLE_RATE, fbank, num_frames

__all__ = [
    "NUM_MEL_BINS",
    "SAMPLE_RATE",
    "WINDOW_FRAMES",
    "fbank",
    "global_mean_normalized",
    "num_frames",
    "utterance_windows",
]


def global_mean_normalized(features: np.ndarray) -> np.ndarray:
    """Subtract each bin's mean over the utterance (the model's ``global-mean``)."""
    features = np.asarray(features, dtype=np.float32)
    if features.shape[0] == 0:
        return features
    return features - features.mean(axis=0, keepdims=True)


#: Frames the model is fed at a time. The exported graph bakes its context
#: pooling kernel at this length, which is what makes a fixed window the model's
#: natural unit rather than an arbitrary choice: Ascend's ATC only compiles the
#: graph at exactly this input length (every other length, and a dynamic one,
#: fails its AvgPoolV2 fusion pass), and a partial pooling window is where ONNX
#: Runtime versions disagree on the divisor. One window is 2.0 s of audio.
WINDOW_FRAMES = 200


def utterance_windows(features: np.ndarray, window: int = WINDOW_FRAMES) -> list[np.ndarray]:
    """Split one utterance's features into fixed-length windows the model can take.

    An utterance shorter than a window is tiled from its own frames rather than
    zero-padded: silence is not what the speaker sounds like, and measured over
    225 utterances the tiled embedding sits closer to the whole-utterance one
    (median 0.974 against 0.963) with a quarter as many far outliers.

    A longer utterance becomes consecutive windows with the last one aligned to
    its end, so every frame is covered and none is seen twice except in that
    overlap. The caller embeds each window and averages the unit vectors; that
    agreed with the whole-utterance embedding to p5 0.964, where taking only the
    first window reached p5 0.712.

    Both deployments apply this, so an anchor and the utterance judged against
    it are always comparable, and one calibration serves CPU and NPU alike:
    measured on an Ascend 310P, the same windows through the OM and through CPU
    ONNX Runtime agree to cosine 0.999976 at worst.
    """
    features = np.asarray(features, dtype=np.float32)
    window = int(window)
    if window <= 0:
        raise ValueError("window must be positive")
    frames = features.shape[0]
    if frames == 0:
        return []
    if frames < window:
        repeats = -(-window // frames)  # ceil, so the tile is never short
        return [np.tile(features, (repeats, 1))[:window]]
    starts = list(range(0, frames - window + 1, window))
    if starts[-1] + window < frames:
        starts.append(frames - window)
    return [features[start : start + window] for start in starts]
