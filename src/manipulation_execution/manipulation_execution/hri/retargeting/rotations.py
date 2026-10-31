"""6D rotation encoding used by the PEAR SMPL-X head."""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def decode_rotation_6d(raw) -> np.ndarray:
    """Decode packed 6D rotations into 3x3 matrices, shape ``(..., K, 3, 3)``.

    Every six values are two 3-vectors ``a1, a2``. Gram-Schmidt turns them into
    ``b1, b2`` and ``b3 = b1 x b2``, which are the matrix *columns*. Stacking
    them as rows also yields an orthonormal matrix with det +1 -- just the wrong
    one -- so an orthogonality check cannot tell the two apart; the layout was
    pinned against PEAR's own decoded output instead.
    """
    x = np.asarray(raw, dtype=np.float64)
    x = x.reshape(x.shape[:-1] + (-1, 2, 3))
    a1 = x[..., 0, :]
    a2 = x[..., 1, :]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), _EPS)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), _EPS)
    return np.stack((b1, b2, np.cross(b1, b2)), axis=-1)


def encode_rotation_6d(matrices) -> np.ndarray:
    """Inverse of :func:`decode_rotation_6d`: ``(..., K, 3, 3)`` -> ``(..., K * 6)``."""
    m = np.asarray(matrices, dtype=np.float64)
    pairs = np.swapaxes(m[..., :, :2], -1, -2)  # first two columns, as rows
    return pairs.reshape(m.shape[:-3] + (-1,))
