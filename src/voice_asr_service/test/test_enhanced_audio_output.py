"""Tests for channel order, clipping, and the enhancement output side path."""

import numpy as np
import pytest

from voice_asr_service.speech_direction.audio_output import encode_enhanced_pcm


def test_pcm_preserves_channel_order_and_clips_instead_of_wrapping():
    samples = np.array([[0.25, -0.5, 1.0, -1.5], [0.0, 0.125, -0.125, 2.0]], dtype=np.float32)
    encoded = np.frombuffer(encode_enhanced_pcm(samples), dtype="<i2").reshape(-1, 4)
    np.testing.assert_array_equal(encoded, [[8192, -16384, 32767, -32768], [0, 4096, -4096, 32767]])


@pytest.mark.parametrize("samples", [np.zeros((4,)), np.zeros((2, 3)), np.full((2, 4), np.nan)])
def test_invalid_pcm_is_rejected(samples):
    with pytest.raises(ValueError):
        encode_enhanced_pcm(samples)
