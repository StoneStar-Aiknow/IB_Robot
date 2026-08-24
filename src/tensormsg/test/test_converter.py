"""Tests for generic tensor message conversion helpers."""

from types import SimpleNamespace

import numpy as np
import pytest

from tensormsg.converter import (
    TensorMsgConverter,
    decoded_frame_to_chw_float,
    decoded_frame_to_hwc_uint8,
    hwc_uint8_to_nv12,
    nv12_to_hwc_uint8,
    ros_image_to_hwc_uint8,
)


class _JointCurrentLike:
    def __init__(self):
        self.name = ["joint_a", "joint_b"]
        self.current = [0.25, 0.5]


class Image:
    __module__ = "sensor_msgs.msg._image"

    def __init__(
        self,
        data: bytes,
        *,
        encoding: str,
        height: int = 1,
        width: int = 2,
        step: int | None = None,
        is_bigendian: bool = False,
    ):
        self.height = height
        self.width = width
        self.encoding = encoding
        self.step = step if step is not None else len(data) // height
        self.is_bigendian = is_bigendian
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


def test_ros_image_to_hwc_uint8_removes_row_padding_for_encoder_input():
    message = Image(
        bytes([255, 0, 0, 0, 255, 0, 91, 92, 0, 0, 255, 255, 255, 255, 93, 94]),
        encoding="rgb8",
        height=2,
        step=8,
    )

    frame = ros_image_to_hwc_uint8(message)

    assert frame.shape == (2, 2, 3)
    assert frame.dtype == np.uint8
    assert frame.flags.c_contiguous
    np.testing.assert_array_equal(frame[1], np.array([[0, 0, 255], [255, 255, 255]], dtype=np.uint8))


@pytest.mark.parametrize(
    ("encoding", "frame"),
    [
        ("mono8", np.array([[17, 34]], dtype=np.uint8)),
        ("rgba8", np.array([[[1, 2, 3, 99], [4, 5, 6, 88]]], dtype=np.uint8)),
        ("bgra8", np.array([[[3, 2, 1, 99], [6, 5, 4, 88]]], dtype=np.uint8)),
    ],
)
def test_decoded_frame_conversion_handles_mono_and_alpha(encoding, frame):
    converted = decoded_frame_to_hwc_uint8(frame, encoding=encoding)

    expected = (
        np.array([[[17, 17, 17], [34, 34, 34]]], dtype=np.uint8)
        if encoding == "mono8"
        else np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    )
    np.testing.assert_array_equal(converted, expected)
    assert converted.flags.c_contiguous


def test_decoded_frame_conversion_supports_bgr_output_and_canonical_chw():
    bgr = np.array([[[3, 2, 1], [6, 5, 4]]], dtype=np.uint8)

    encoder_frame = decoded_frame_to_hwc_uint8(bgr, encoding="bgr8", output_encoding="bgr8")
    canonical = decoded_frame_to_chw_float(bgr, encoding="bgr8")

    np.testing.assert_array_equal(encoder_frame, bgr)
    np.testing.assert_array_equal(canonical[:, 0, 0], np.array([1, 2, 3], dtype=np.float32) / 255.0)
    assert canonical.flags.c_contiguous


def test_decode_big_endian_padded_depth_image():
    rows = np.array([[1000, 0], [2500, 10000]], dtype=">u2").tobytes()
    data = rows[:4] + b"\xaa\xbb" + rows[4:] + b"\xcc\xdd"
    message = Image(data, encoding="16UC1", height=2, step=6, is_bigendian=True)

    decoded = TensorMsgConverter.decode(message)

    assert decoded.shape == (3, 2, 2)
    np.testing.assert_allclose(decoded[:, 0, 0], 0.1)
    assert np.isnan(decoded[:, 0, 1]).all()
    np.testing.assert_allclose(decoded[:, 1, 0], 0.25)
    np.testing.assert_allclose(decoded[:, 1, 1], 1.0)


@pytest.mark.parametrize("resize", [(0, 2), (2, -1), (1,), (1.5, 2)])
def test_image_resize_rejects_invalid_dimensions(resize):
    message = Image(bytes([255, 0, 0, 0, 255, 0]), encoding="rgb8")

    with pytest.raises(ValueError, match="image resize"):
        TensorMsgConverter.decode(message, SimpleNamespace(names=[], image_resize=resize))


def test_ros_image_rejects_short_rows_and_truncated_data():
    with pytest.raises(ValueError, match="smaller than packed row"):
        ros_image_to_hwc_uint8(Image(bytes(5), encoding="rgb8", step=5))

    with pytest.raises(ValueError, match="expected at least"):
        ros_image_to_hwc_uint8(Image(bytes(6), encoding="rgb8", height=2, step=6))


@pytest.mark.parametrize("color_range", ["limited", "full"])
def test_bt709_nv12_round_trip_preserves_color_blocks(color_range):
    colors = np.array(
        [
            [[255, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 0]],
            [[255, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 0]],
            [[0, 0, 255], [0, 0, 255], [255, 255, 255], [255, 255, 255]],
            [[0, 0, 255], [0, 0, 255], [255, 255, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )

    nv12 = hwc_uint8_to_nv12(colors, color_range=color_range)
    decoded = nv12_to_hwc_uint8(nv12, width=4, height=4, color_range=color_range)

    assert nv12.shape == (6, 4)
    assert decoded.flags.c_contiguous
    assert np.max(np.abs(decoded.astype(np.int16) - colors.astype(np.int16))) <= 2


def test_nv12_conversion_supports_bgr_and_padded_stride():
    bgr = np.array(
        [
            [[0, 0, 255], [0, 0, 255]],
            [[0, 0, 255], [0, 0, 255]],
        ],
        dtype=np.uint8,
    )

    nv12 = hwc_uint8_to_nv12(bgr, encoding="bgr8", stride=8)
    decoded = nv12_to_hwc_uint8(nv12.tobytes(), width=2, height=2, stride=8, output_encoding="bgr8")

    assert nv12.shape == (3, 8)
    assert np.all(nv12[:, 2:] == 0)
    assert np.max(np.abs(decoded.astype(np.int16) - bgr.astype(np.int16))) <= 2


def test_nv12_limited_and_full_ranges_use_expected_black_white_levels():
    black_white = np.array(
        [
            [[0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 255, 255]],
            [[0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )

    limited = hwc_uint8_to_nv12(black_white, color_range="limited")
    full = hwc_uint8_to_nv12(black_white, color_range="full")

    assert limited[0, :].tolist() == [16, 16, 235, 235]
    assert full[0, :].tolist() == [0, 0, 255, 255]


@pytest.mark.parametrize(
    ("operation", "match"),
    [
        (lambda: hwc_uint8_to_nv12(np.zeros((3, 4, 3), dtype=np.uint8)), "even dimensions"),
        (lambda: hwc_uint8_to_nv12(np.zeros((2, 4, 3), dtype=np.uint8), stride=3), "smaller than width"),
        (lambda: hwc_uint8_to_nv12(np.zeros((2, 4, 3), dtype=np.uint8), stride=1.5), "positive integer"),
        (lambda: nv12_to_hwc_uint8(bytes(8), width=4, height=2), "expected at least"),
        (lambda: nv12_to_hwc_uint8(bytes(12), width=4, height=2, color_space="bt601"), "bt709"),
        (lambda: nv12_to_hwc_uint8(bytes(12), width=4, height=2, color_range="unknown"), "color_range"),
    ],
)
def test_nv12_conversion_rejects_invalid_surfaces(operation, match):
    with pytest.raises(ValueError, match=match):
        operation()


def test_nv12_conversion_matches_float_reference_within_one_lsb():
    rng = np.random.default_rng(123)
    image = rng.integers(0, 256, size=(4, 6, 3), dtype=np.uint8)
    for encoding in ("rgb8", "bgr8"):
        for color_range in ("limited", "full"):
            source = image if encoding == "rgb8" else image[..., ::-1]
            red = source[..., 0].astype(np.float32)
            green = source[..., 1].astype(np.float32)
            blue = source[..., 2].astype(np.float32)
            if color_range == "limited":
                y_plane = 16.0 + 0.182586 * red + 0.614231 * green + 0.062007 * blue
                u_plane = 128.0 - 0.100644 * red - 0.338572 * green + 0.439216 * blue
                v_plane = 128.0 + 0.439216 * red - 0.398942 * green - 0.040274 * blue
            else:
                y_plane = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                u_plane = 128.0 - 0.114572 * red - 0.385428 * green + 0.5 * blue
                v_plane = 128.0 + 0.5 * red - 0.454153 * green - 0.045847 * blue
            u_subsampled = u_plane.reshape(2, 2, 3, 2).mean(axis=(1, 3))
            v_subsampled = v_plane.reshape(2, 2, 3, 2).mean(axis=(1, 3))
            reference = np.zeros((6, 6), dtype=np.uint8)
            reference[:4] = np.clip(np.rint(y_plane), 0, 255).astype(np.uint8)
            reference[4:, ::2] = np.clip(np.rint(u_subsampled), 0, 255).astype(np.uint8)
            reference[4:, 1::2] = np.clip(np.rint(v_subsampled), 0, 255).astype(np.uint8)
            actual = hwc_uint8_to_nv12(image, encoding=encoding, color_range=color_range)
            assert np.max(np.abs(actual.astype(np.int16) - reference.astype(np.int16))) <= 1


def test_nv12_decode_resize_and_canonical_dds_path_remain_consistent():
    rgb = np.full((4, 4, 3), [64, 128, 192], dtype=np.uint8)
    nv12 = hwc_uint8_to_nv12(rgb)

    decoded = nv12_to_hwc_uint8(nv12, width=4, height=4, resize=(2, 2))
    canonical = decoded_frame_to_chw_float(decoded, encoding="rgb8")
    dds = decoded_frame_to_chw_float(rgb, encoding="rgb8", resize=(2, 2))

    assert canonical.shape == dds.shape == (3, 2, 2)
    assert canonical.flags.c_contiguous
    np.testing.assert_allclose(canonical, dds, atol=2 / 255.0)
