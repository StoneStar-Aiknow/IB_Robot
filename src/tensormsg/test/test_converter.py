"""Tests for generic tensor message conversion helpers."""

from types import SimpleNamespace

import numpy as np

from tensormsg.converter import TensorMsgConverter


class _JointCurrentLike:
    def __init__(self):
        self.name = ["joint_a", "joint_b"]
        self.current = [0.25, 0.5]


def test_decode_falls_back_to_joint_current_style_names():
    spec = SimpleNamespace(names=["current.joint_a", "current.joint_b", "current.missing"])

    decoded = TensorMsgConverter.decode(_JointCurrentLike(), spec)

    assert decoded.dtype == np.float32
    assert decoded[:2].tolist() == [0.25, 0.5]
    assert np.isnan(decoded[2])
