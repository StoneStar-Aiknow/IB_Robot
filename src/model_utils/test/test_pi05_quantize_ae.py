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
