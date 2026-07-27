from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

from model_utils.pi05_export.quant.quantize_vlm import _resize_calibration_images


def _write_model(path, shape):
    model_input = helper.make_tensor_value_info("observation.images.top", TensorProto.FLOAT, shape)
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)
    graph = helper.make_graph(
        [helper.make_node("Identity", [model_input.name], [output.name])],
        "test",
        [model_input],
        [output],
    )
    onnx.save(helper.make_model(graph), path)
    return model_input.name


def test_resize_calibration_images_matches_static_onnx_shape(tmp_path):
    model_path = tmp_path / "vlm.onnx"
    name = _write_model(model_path, [1, 3, 4, 6])
    feed = {name: np.ones((1, 3, 8, 8), dtype=np.float32)}

    _resize_calibration_images(feed, model_path)

    assert feed[name].shape == (1, 3, 4, 6)
    assert feed[name].dtype == np.float32


def test_resize_calibration_images_keeps_matching_shape(tmp_path):
    model_path = tmp_path / "vlm.onnx"
    name = _write_model(model_path, [1, 3, 4, 6])
    image = np.ones((1, 3, 4, 6), dtype=np.float32)
    feed = {name: image}

    _resize_calibration_images(feed, model_path)

    assert feed[name] is image
