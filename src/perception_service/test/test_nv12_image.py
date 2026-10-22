from array import array
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from inference_service.unified_runtime import ModelResult, OutcomeEvidence, RuntimeLatency
from perception_service.model_contracts import validate_image_message
from perception_service.nv12_image import NV12_FULL_RANGE, Nv12Image
from perception_service.pear_adapter import PearParameterAdapter
from perception_service.yolox_adapter import YoloXAdapter


def nv12_message(width=320, height=240, stride=352):
    storage = np.zeros((height * 3 // 2, stride), dtype=np.uint8)
    storage[:height, :width] = np.arange(width, dtype=np.uint16)[None, :] % 256
    storage[height:, :width:2] = 100
    storage[height:, 1:width:2] = 160
    return SimpleNamespace(
        width=width,
        height=height,
        step=stride,
        encoding=NV12_FULL_RANGE,
        is_bigendian=0,
        data=array("B", storage.tobytes()),
    )


def full_rgb(message):
    storage = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height * 3 // 2, message.step)
    shape = (message.width, message.height)
    blue = cv2.resize(storage[message.height :, : message.width : 2], shape)
    red = cv2.resize(storage[message.height :, 1 : message.width : 2], shape)
    return cv2.cvtColor(cv2.merge((storage[: message.height, : message.width], red, blue)), cv2.COLOR_YCrCb2RGB)


def test_nv12_is_an_explicit_hri_extension_not_a_global_contract_change():
    message = nv12_message()
    with pytest.raises(ValueError, match="unsupported"):
        validate_image_message(message)
    assert Nv12Image(message).shape == (240, 320, 3)


@pytest.mark.parametrize(
    "changes",
    [
        {"width": 319},
        {"height": 239},
        {"step": 319},
        {"encoding": "nv12"},
        {"is_bigendian": 1},
        {"data": array("B", [0])},
        {"width": 0},
    ],
)
def test_nv12_rejects_invalid_geometry_and_range(changes):
    message = nv12_message()
    for name, value in changes.items():
        setattr(message, name, value)
    with pytest.raises(ValueError):
        Nv12Image(message)


def test_yolox_nv12_letterbox_preserves_original_source_ratio():
    message = nv12_message()
    adapter = YoloXAdapter()
    tensor = adapter.preprocess(Nv12Image(message))["observation.image"]
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert np.all(tensor[:, :, 480:] == 114)
    raw = np.zeros((1, 8400, 85), dtype=np.float32)
    raw[0, 0, :6] = [20, 20, np.log(10), np.log(10), 1, 1]
    result = ModelResult(
        outputs={"yolox.raw": raw},
        latency=RuntimeLatency(total_ms=1, backend_ms=1),
        evidence=OutcomeEvidence.completed("adaptation"),
    )
    records = adapter.postprocess(result, image_shape=(240, 320))
    np.testing.assert_allclose(records[0].bbox_xyxy, [60, 60, 100, 100], atol=1e-5)


@pytest.mark.parametrize(
    "box",
    [
        [110, 80, 190, 160],
        [-40, -40, 40, 40],
        [290, 190, 350, 260],
        [-200, -200, -150, -150],
        [600, 600, 700, 700],
    ],
)
def test_pear_roi_matches_full_resolution_crop_without_double_expansion(box):
    message = nv12_message()
    adapter = PearParameterAdapter()
    expected = adapter.preprocess((full_rgb(message), [box]))["pear.input"]
    actual = adapter.preprocess((Nv12Image(message), [box]))["pear.input"]
    np.testing.assert_allclose(actual, expected, atol=1 / 255)


def test_nv12_color_uses_full_range_and_preserves_channel_order():
    message = nv12_message(width=2, height=2, stride=2)
    message.data = array("B", [0, 255, 0, 255, 128, 128])
    converted = Nv12Image(message).small_bgr(2)
    np.testing.assert_array_equal(converted[:, 0], 0)
    np.testing.assert_array_equal(converted[:, 1], 255)


@pytest.mark.parametrize("boxes", [[], [[1, 1, 2, 2], [3, 3, 4, 4]], [[0, 0, 0, 1]], [[0, 0, float("nan"), 1]]])
def test_pear_nv12_rejects_invalid_boxes(boxes):
    with pytest.raises(ValueError):
        PearParameterAdapter().preprocess((Nv12Image(nv12_message()), boxes))
