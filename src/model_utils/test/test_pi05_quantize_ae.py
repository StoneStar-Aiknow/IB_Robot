from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from model_utils.pi05_export.quant import quantize_ae


def _write_ae_model(path: Path) -> None:
    keep_weight = numpy_helper.from_array(np.ones((2, 2), dtype=np.float16), name="keep_weight")
    drop_weight = numpy_helper.from_array(np.ones((2, 2), dtype=np.float16), name="drop_weight")
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["noise", keep_weight.name], ["hidden"], name="/keep/MatMul"),
            helper.make_node("MatMul", ["hidden", drop_weight.name], ["velocity"], name="/drop/MatMul"),
        ],
        "ae",
        [helper.make_tensor_value_info("noise", TensorProto.FLOAT16, [1, 2, 2])],
        [helper.make_tensor_value_info("velocity", TensorProto.FLOAT16, [1, 2, 2])],
        [keep_weight, drop_weight],
    )
    onnx.save(helper.make_model(graph), path)


def _write_ae_input_model(path: Path) -> None:
    inputs = [
        helper.make_tensor_value_info("past_kv_tensor", TensorProto.FLOAT16, [1, 2, 1, 1, 2, 2]),
        helper.make_tensor_value_info("prefix_pad_masks", TensorProto.BOOL, [1, 2]),
        helper.make_tensor_value_info("time", TensorProto.FLOAT16, [1]),
        helper.make_tensor_value_info("noise", TensorProto.FLOAT16, [1, 2, 4]),
    ]
    graph = helper.make_graph(
        [helper.make_node("Identity", ["noise"], ["velocity"], name="identity")],
        "ae-inputs",
        inputs,
        [helper.make_tensor_value_info("velocity", TensorProto.FLOAT16, [1, 2, 4])],
    )
    onnx.save(helper.make_model(graph), path)


def test_ae_quantizer_applies_quantize_allowlist(tmp_path, monkeypatch):
    input_path = tmp_path / "ae.onnx"
    output_path = tmp_path / "ae-w8a8.onnx"
    _write_ae_model(input_path)
    captured = {}

    monkeypatch.setattr(quantize_ae, "build_calib_data", lambda **_kwargs: [[np.ones((1, 2, 2), dtype=np.float16)]])

    def fake_quantize(**kwargs):
        captured["disable_names"] = kwargs["disable_names"]
        output_path.write_bytes(input_path.read_bytes())
        return 1

    monkeypatch.setattr(quantize_ae.common, "run_msmodelslim_w8a8", fake_quantize)
    monkeypatch.setattr(
        "sys.argv",
        [
            "quantize_ae",
            "--onnx-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--calib-dir",
            str(tmp_path),
            "--disable-regex",
            "--quantize-regex",
            "^/keep/MatMul$",
            "--quantize-regex-expected",
            "1",
            "--expected-selected-nodes",
            "1",
            "--expected-quantized-nodes",
            "1",
        ],
    )

    assert quantize_ae.main() == 0
    assert captured["disable_names"] == ["/drop/MatMul"]


def test_profiled_ae_quantization_removes_failed_output(tmp_path, monkeypatch):
    input_path = tmp_path / "ae.onnx"
    output_path = tmp_path / "ae-w8a8.onnx"
    metadata = tmp_path / "ae.quant.json"
    policy = tmp_path / "bundle"
    policy.mkdir()
    _write_ae_model(input_path)
    monkeypatch.setattr(quantize_ae, "build_calib_data", lambda **_kwargs: [[np.ones((1, 2, 2), dtype=np.float16)]])

    def fake_quantize(**_kwargs):
        output_path.write_bytes(input_path.read_bytes())
        output_path.with_name(output_path.name + ".data").write_bytes(b"stale")
        return 0

    monkeypatch.setattr(quantize_ae.common, "run_msmodelslim_w8a8", fake_quantize)
    monkeypatch.setattr(
        "sys.argv",
        [
            "quantize_ae",
            "--onnx-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--policy-path",
            str(policy),
            "--calib-dir",
            str(tmp_path),
            "--quantize-regex",
            "^/keep/MatMul$",
            "--expected-quantized-nodes",
            "1",
            "--quant-profile-name",
            "test",
            "--quant-profile-hash",
            "hash",
            "--quant-role",
            "ae",
            "--quant-metadata-path",
            str(metadata),
        ],
    )

    with pytest.raises(RuntimeError, match="expected 1"):
        quantize_ae.main()
    assert not output_path.exists()
    assert not output_path.with_name(output_path.name + ".data").exists()
    assert not metadata.exists()


def test_build_calib_data_expands_each_episode_into_all_trajectory_steps(tmp_path):
    onnx_path = tmp_path / "ae-inputs.onnx"
    _write_ae_input_model(onnx_path)
    sample = tmp_path / "sample_0000"
    sample.mkdir()
    np.save(sample / "ae_in_past_kv.npy", np.zeros((1, 2, 1, 1, 2, 2), dtype=np.float16))
    np.save(sample / "ae_in_prefix_pad_masks.npy", np.array([[True, False]], dtype=bool))
    np.save(sample / "ae_in_noise.npy", np.zeros((1, 2, 4), dtype=np.float16))
    np.save(sample / "ae_in_time_step00.npy", np.array([1.0], dtype=np.float16))
    np.save(sample / "ae_in_time_step01.npy", np.array([0.5], dtype=np.float16))
    np.save(sample / "x_t_step00.npy", np.ones((1, 2, 4), dtype=np.float16))

    feeds = quantize_ae.build_calib_data(
        onnx_path=onnx_path,
        calib_dir=str(tmp_path),
        past_kv_path=None,
        prefix_pad_masks_path=None,
        noise_path=None,
        num_calib=1,
        expected_calibration_steps=2,
    )

    assert len(feeds) == 2
    np.testing.assert_array_equal(feeds[0][3], np.zeros((1, 2, 4), dtype=np.float16))
    np.testing.assert_array_equal(feeds[1][3], np.ones((1, 2, 4), dtype=np.float16))
    assert float(feeds[0][2][0]) == 1.0
    assert float(feeds[1][2][0]) == 0.5

    (sample / "x_t_step00.npy").unlink()
    with pytest.raises(FileNotFoundError, match="missing x_t step 00"):
        quantize_ae.build_calib_data(
            onnx_path=onnx_path,
            calib_dir=str(tmp_path),
            past_kv_path=None,
            prefix_pad_masks_path=None,
            noise_path=None,
            num_calib=1,
            expected_calibration_steps=2,
        )
