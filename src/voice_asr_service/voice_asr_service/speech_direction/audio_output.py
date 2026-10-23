"""PCM serialization for the optional four-channel enhancement output."""

import numpy as np


def encode_enhanced_pcm(samples: np.ndarray) -> bytes:
    """Encode (frames, 4) normalized floats as interleaved S16LE without gain."""
    value = np.asarray(samples, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 4:
        raise ValueError("enhanced audio must have shape (frames, 4)")
    if not np.isfinite(value).all():
        raise ValueError("enhanced audio contains NaN or Inf")
    return np.clip(np.rint(value * 32768.0), -32768, 32767).astype("<i2").tobytes()
