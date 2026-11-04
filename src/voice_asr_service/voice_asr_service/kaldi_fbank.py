"""Kaldi-compatible log-mel filterbank in numpy, shared by every model that takes fbank features.

Both host paths that turn PCM into model features use it: the streaming
transducer whose encoder runs on Ascend (``transducer_ascend.OnlineFbank``) and
the speaker embedding model (``speaker_id.features``). Both were exported for
sherpa-onnx, whose extractors run kaldi-native-fbank with the same settings:
25 ms Povey windows every 10 ms, DC removal, pre-emphasis 0.97, a 512-point
power spectrum, 80 triangular mel bins between 20 Hz and Nyquist - 400 Hz,
natural log, no dither, and ``snip_edges=false`` (frame ``i`` is centred at
``i * shift + shift / 2``, the waveform reflected at both ends). Samples are
mono floats in [-1, 1].

Implemented here rather than taken from a feature library so that no new
dependency is needed and the CPU and NPU deployments share one host path.
"""

from __future__ import annotations

import math

import numpy as np

SAMPLE_RATE = 16000
NUM_MEL_BINS = 80
FRAME_LENGTH = 400  # 25 ms
FRAME_SHIFT = 160  # 10 ms
_FFT_SIZE = 512
_PREEMPHASIS = 0.97
_LOW_FREQ = 20.0
_HIGH_FREQ_OFFSET = -400.0
_LOG_FLOOR = float(np.finfo(np.float32).eps)


def _mel(frequency: np.ndarray | float) -> np.ndarray | float:
    return 1127.0 * np.log(1.0 + np.asarray(frequency) / 700.0)


def _mel_banks() -> np.ndarray:
    """Kaldi's triangular mel filters over the first ``FFT_SIZE / 2`` bins."""
    num_bins = _FFT_SIZE // 2
    high = 0.5 * SAMPLE_RATE + _HIGH_FREQ_OFFSET
    mel_low, mel_high = _mel(_LOW_FREQ), _mel(high)
    delta = (mel_high - mel_low) / (NUM_MEL_BINS + 1)
    bin_mel = _mel(np.arange(num_bins) * (SAMPLE_RATE / _FFT_SIZE))
    banks = np.zeros((NUM_MEL_BINS, num_bins), dtype=np.float64)
    for index in range(NUM_MEL_BINS):
        left = mel_low + index * delta
        centre = left + delta
        right = centre + delta
        rising = (bin_mel - left) / (centre - left)
        falling = (right - bin_mel) / (right - centre)
        weights = np.minimum(rising, falling)
        inside = (bin_mel > left) & (bin_mel < right)
        banks[index] = np.where(inside, np.maximum(weights, 0.0), 0.0)
    return banks


_BANKS = _mel_banks()
_WINDOW = np.power(0.5 - 0.5 * np.cos(2.0 * math.pi * np.arange(FRAME_LENGTH) / (FRAME_LENGTH - 1)), 0.85)


def num_frames(num_samples: int) -> int:
    """Frames the whole of ``num_samples`` yields (``snip_edges=false``)."""
    return (int(num_samples) + FRAME_SHIFT // 2) // FRAME_SHIFT


def log_mel_frames(samples: np.ndarray, first: int, count: int) -> np.ndarray:
    """Frames ``first`` to ``first + count`` of ``samples``, shaped ``(count, 80)``.

    ``samples`` is everything received so far as float64; a frame reaching past
    either end sees the waveform reflected there, which only happens for the
    frames ``num_frames`` adds once the input is complete.
    """
    if count <= 0:
        return np.zeros((0, NUM_MEL_BINS), dtype=np.float32)
    size = samples.size
    # Frame i covers [i*shift + shift/2 - length/2, +length), reflected at both ends.
    starts = np.arange(first, first + count) * FRAME_SHIFT + FRAME_SHIFT // 2 - FRAME_LENGTH // 2
    positions = starts[:, None] + np.arange(FRAME_LENGTH)[None, :]
    positions = np.where(positions < 0, -positions - 1, positions)
    positions = np.where(positions >= size, 2 * size - 1 - positions, positions)
    positions = np.clip(positions, 0, size - 1)
    windows = samples[positions]
    windows = windows - windows.mean(axis=1, keepdims=True)
    emphasized = windows.copy()
    emphasized[:, 1:] -= _PREEMPHASIS * windows[:, :-1]
    emphasized[:, 0] -= _PREEMPHASIS * windows[:, 0]
    emphasized *= _WINDOW
    spectrum = np.fft.rfft(emphasized, n=_FFT_SIZE, axis=1)
    power = spectrum.real**2 + spectrum.imag**2
    energies = power[:, : _FFT_SIZE // 2] @ _BANKS.T
    return np.log(np.maximum(energies, _LOG_FLOOR)).astype(np.float32)


def fbank(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Log-mel filterbank of a complete utterance, shaped ``(frames, 80)``."""
    if int(sample_rate) != SAMPLE_RATE:
        raise ValueError(f"fbank features need {SAMPLE_RATE} Hz audio, got {sample_rate}")
    wave = np.asarray(samples, dtype=np.float64).reshape(-1)
    return log_mel_frames(wave, 0, num_frames(wave.size))
