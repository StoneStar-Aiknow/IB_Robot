import random
import struct
import sys

import pytest

from embodied_common.trajectory_templates import _matmul


def _generic(a, b):
    return [[sum(a[row][k] * b[k][col] for k in range(4)) for col in range(4)] for row in range(4)]


def _bits(matrix):
    return [struct.pack("<d", value) for row in matrix for value in row]


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="sum() uses compensated float summation from Python 3.12")
def test_unrolled_matmul_is_bit_identical_to_the_generic_sum_including_signed_zero():
    # -1 * 0 is -0.0; sum() starts from 0 and returns +0.0, so must the unrolled form.
    flip = [[-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    swap = [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    assert _bits(_matmul(flip, swap)) == _bits(_generic(flip, swap))

    rng = random.Random(0)
    values = (0.0, -0.0, 1.0, -1.0)
    for _ in range(2000):
        a = [[rng.choice(values) if rng.random() < 0.5 else rng.uniform(-1, 1) for _ in range(4)] for _ in range(4)]
        b = [[rng.choice(values) if rng.random() < 0.5 else rng.uniform(-1, 1) for _ in range(4)] for _ in range(4)]
        assert _bits(_matmul(a, b)) == _bits(_generic(a, b))
