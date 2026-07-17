"""Tests for generic tensor message conversion helpers."""

from types import SimpleNamespace

import numpy as np

from tensormsg.converter import TensorMsgConverter


class _JointCurrentLike:
    def __init__(self):
        self.name = ["joint_a", "joint_b"]
        self.current = [0.25, 0.5]


class Image:
    __module__ = "sensor_msgs.msg._image"

    def __init__(self, data: bytes, *, encoding: str):
        self.height = 1
        self.width = 2
        self.encoding = encoding
        self.step = 6
        self.data = data


def test_decode_falls_back_to_joint_current_style_names():
    spec = SimpleNamespace(names=["current.joint_a", "current.joint_b", "current.missing"])

    decoded = TensorMsgConverter.decode(_JointCurrentLike(), spec)

    assert decoded.dtype == np.float32
    assert decoded[:2].tolist() == [0.25, 0.5]
    assert np.isnan(decoded[2])


def test_decode_rgb_image_returns_contiguous_chw_tensor():
    message = Image(bytes([255, 0, 0, 0, 255, 0]), encoding="rgb8")

    decoded = TensorMsgConverter.decode(message)

    assert decoded.shape == (3, 1, 2)
    assert decoded.dtype == np.float32
    assert decoded.flags.c_contiguous
    np.testing.assert_array_equal(decoded[:, 0, 0], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(decoded[:, 0, 1], np.array([0.0, 1.0, 0.0], dtype=np.float32))


def test_decode_bgr_image_converts_to_rgb_chw_tensor():
    message = Image(bytes([0, 0, 255, 0, 255, 0]), encoding="bgr8")

    decoded = TensorMsgConverter.decode(message)

    assert decoded.shape == (3, 1, 2)
    assert decoded.flags.c_contiguous
    np.testing.assert_array_equal(decoded[:, 0, 0], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(decoded[:, 0, 1], np.array([0.0, 1.0, 0.0], dtype=np.float32))
