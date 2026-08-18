from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from model_utils.pi05_export.quant.smoothquant import (
    prepare_smoothquant_pair,
    verify_smoothquant_outputs,
)

try:
    import onnxruntime as ort
except (ImportError, OSError):
    ORT_AVAILABLE = False
else:
    ORT_AVAILABLE = "CPUExecutionProvider" in ort.get_available_providers()

requires_ort = pytest.mark.skipif(not ORT_AVAILABLE, reason="ONNX Runtime CPUExecutionProvider is unavailable")
QKV_NAMES = ["/attn/q/MatMul", "/attn/k/MatMul", "/attn/v/MatMul"]


def _model(graph) -> onnx.ModelProto:  # noqa: ANN001
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _save_external(model: onnx.ModelProto, path: Path) -> None:
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=path.name + ".data",
        size_threshold=0,
    )


def _write_qkv(
    path: Path,
    *,
    omit: frozenset[str] = frozenset(),
    q_shape: tuple[int, int] = (3, 2),
    share_qv_weight: bool = False,
) -> None:
    arrays = {
        "q_weight": np.array([[1.0, -2.0], [3.0, 0.5], [0.0, 0.0]], dtype=np.float32),
        "k_weight": np.array([[2.0, 1.0], [-1.0, 4.0], [0.0, 0.0]], dtype=np.float32),
        "v_weight": np.array([[0.5, 2.0], [1.5, -3.0], [0.0, 0.0]], dtype=np.float32),
    }
    if q_shape != (3, 2):
        arrays["q_weight"] = np.ones(q_shape, dtype=np.float32)
    initializers = []
    nodes = []
    outputs = []
    for role, node_name in zip(("q", "k", "v"), QKV_NAMES, strict=True):
        if role in omit:
            continue
        weight_name = "q_weight" if share_qv_weight and role == "v" else f"{role}_weight"
        if not any(initializer.name == weight_name for initializer in initializers):
            initializers.append(numpy_helper.from_array(arrays[weight_name], name=weight_name))
        output_name = f"{role}_output"
        nodes.append(helper.make_node("MatMul", ["hidden", weight_name], [output_name], name=node_name))
        outputs.append(output_name)
    nodes.append(helper.make_node("Concat", outputs, ["output"], name="/attn/Concat", axis=-1))
    graph = helper.make_graph(
        nodes,
        "qkv",
        [helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, 2, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 2 * len(outputs)])],
        initializers,
    )
    _save_external(_model(graph), path)


def _write_gemm(path: Path, *, trans_a: int = 0) -> None:
    weight = np.array([[1.0, 2.0, -0.5], [-2.0, 0.25, 3.0]], dtype=np.float32)
    bias = np.array([0.5, -0.25], dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Gemm",
                ["hidden", "weight", "bias"],
                ["output"],
                name="/mlp/Gemm",
                transA=trans_a,
                transB=1,
            )
        ],
        "gemm",
        [helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 2])],
        [numpy_helper.from_array(weight, name="weight"), numpy_helper.from_array(bias, name="bias")],
    )
    _save_external(_model(graph), path)


def _hash_sources(*paths: Path) -> dict[str, str]:
    files = []
    for path in paths:
        files.extend((path, path.with_name(path.name + ".data")))
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def _qkv_calib() -> list[list[np.ndarray]]:
    first = np.array([[[1.0, -2.0, 0.0], [3.0, 4.0, 0.0]]], dtype=np.float32)
    second = np.array([[[-2.0, 1.0, 0.0], [0.5, -3.0, 0.0]]], dtype=np.float32)
    return [[first], [second]]


@requires_ort
def test_shared_qkv_zero_channel_immutability_plan_determinism_and_equivalence(tmp_path):
    donor = tmp_path / "donor.onnx"
    npu = tmp_path / "npu.onnx"
    _write_qkv(donor)
    _write_qkv(npu)
    before = _hash_sources(donor, npu)

    first = prepare_smoothquant_pair(
        donor,
        npu,
        tmp_path / "run-a",
        _qkv_calib(),
        QKV_NAMES,
        0.5,
        1e-5,
        output_prefix="qkv",
    )
    second = prepare_smoothquant_pair(
        donor,
        npu,
        tmp_path / "run-b",
        _qkv_calib(),
        QKV_NAMES,
        0.5,
        1e-5,
        output_prefix="qkv",
    )

    assert _hash_sources(donor, npu) == before
    assert first.group_count == 1
    assert first.node_count == 3
    assert first.plan_digest == second.plan_digest
    assert first.plan_path.read_bytes() == second.plan_path.read_bytes()
    plan = json.loads(first.plan_path.read_text(encoding="utf-8"))
    assert plan["plan_digest"] == first.plan_digest
    assert plan["groups"][0]["nodes"] == QKV_NAMES

    smoothed = onnx.load(first.donor_path, load_external_data=True)
    nodes = {node.name: node for node in smoothed.graph.node}
    muls = [node for node in smoothed.graph.node if node.op_type == "Mul"]
    assert len(muls) == 1
    assert {nodes[name].input[0] for name in QKV_NAMES} == {muls[0].output[0]}
    initializers = {initializer.name: initializer for initializer in smoothed.graph.initializer}
    inverse = numpy_helper.to_array(initializers[muls[0].input[1]])
    assert inverse.shape == (3,)
    assert inverse[-1] == pytest.approx(1.0)

    report = verify_smoothquant_outputs(donor, first.donor_path, _qkv_calib())
    assert report.sample_count == 2
    assert len(report.comparisons) == 2
    assert all(comparison.finite and comparison.allclose for comparison in report.comparisons)
    assert all(comparison.shape == (1, 2, 6) for comparison in report.comparisons)


@requires_ort
def test_selected_subset_reroutes_only_selected_consumers_and_clones_shared_weight(tmp_path):
    donor = tmp_path / "donor.onnx"
    npu = tmp_path / "npu.onnx"
    _write_qkv(donor, share_qv_weight=True)
    _write_qkv(npu, share_qv_weight=True)
    selected = QKV_NAMES[:2]

    result = prepare_smoothquant_pair(
        donor,
        npu,
        tmp_path / "out",
        _qkv_calib(),
        selected,
        0.5,
        1e-5,
    )

    model = onnx.load(result.donor_path, load_external_data=True)
    nodes = {node.name: node for node in model.graph.node}
    assert nodes[selected[0]].input[0] == nodes[selected[1]].input[0]
    assert nodes[QKV_NAMES[2]].input[0] == "hidden"
    assert nodes[selected[0]].input[1] != nodes[QKV_NAMES[2]].input[1]
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    np.testing.assert_array_equal(
        numpy_helper.to_array(initializers[nodes[QKV_NAMES[2]].input[1]]),
        np.array([[1.0, -2.0], [3.0, 0.5], [0.0, 0.0]], dtype=np.float32),
    )
    verify_smoothquant_outputs(donor, result.donor_path, _qkv_calib())


@requires_ort
def test_gemm_transb_scales_physical_weight_input_axis_and_remains_equivalent(tmp_path):
    donor = tmp_path / "donor.onnx"
    npu = tmp_path / "npu.onnx"
    _write_gemm(donor)
    _write_gemm(npu)
    calib = [[np.array([[1.0, -2.0, 0.5], [3.0, 1.0, -4.0]], dtype=np.float32)]]

    result = prepare_smoothquant_pair(
        donor,
        npu,
        tmp_path / "out",
        calib,
        ["/mlp/Gemm"],
        0.5,
        1e-5,
    )

    original = numpy_helper.to_array(
        {initializer.name: initializer for initializer in onnx.load(donor).graph.initializer}["weight"]
    )
    smoothed_model = onnx.load(result.donor_path, load_external_data=True)
    smoothed = numpy_helper.to_array(
        {initializer.name: initializer for initializer in smoothed_model.graph.initializer}["weight"]
    )
    ratios = smoothed / original
    np.testing.assert_allclose(ratios[0], ratios[1], rtol=1e-6, atol=1e-6)
    assert not np.allclose(ratios, 1.0)
    verify_smoothquant_outputs(donor, result.donor_path, calib)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_alpha_is_rejected_before_loading_models(tmp_path, alpha):
    with pytest.raises(ValueError, match="alpha"):
        prepare_smoothquant_pair(
            tmp_path / "missing-donor.onnx",
            tmp_path / "missing-npu.onnx",
            tmp_path,
            [],
            ["node"],
            alpha,
            1e-5,
        )


def test_duplicate_selected_names_are_rejected_before_loading_models(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        prepare_smoothquant_pair(
            tmp_path / "missing-donor.onnx",
            tmp_path / "missing-npu.onnx",
            tmp_path,
            [],
            ["node", "node"],
            0.5,
            1e-5,
        )


def test_missing_npu_node_fails_closed(tmp_path):
    donor = tmp_path / "donor.onnx"
    npu = tmp_path / "npu.onnx"
    _write_qkv(donor)
    _write_qkv(npu, omit=frozenset({"q"}))

    with pytest.raises(ValueError, match="missing"):
        prepare_smoothquant_pair(
            donor,
            npu,
            tmp_path / "out",
            _qkv_calib(),
            [QKV_NAMES[0]],
            0.5,
            1e-5,
        )


def test_mismatched_npu_weight_shape_fails_closed(tmp_path):
    donor = tmp_path / "donor.onnx"
    npu = tmp_path / "npu.onnx"
    _write_qkv(donor)
    _write_qkv(npu, q_shape=(4, 2))

    with pytest.raises(ValueError, match="shape mismatch"):
        prepare_smoothquant_pair(
            donor,
            npu,
            tmp_path / "out",
            _qkv_calib(),
            [QKV_NAMES[0]],
            0.5,
            1e-5,
        )


def test_gemm_transa_is_rejected(tmp_path):
    donor = tmp_path / "donor.onnx"
    npu = tmp_path / "npu.onnx"
    _write_gemm(donor, trans_a=1)
    _write_gemm(npu, trans_a=1)

    with pytest.raises(ValueError, match="unsupported transA"):
        prepare_smoothquant_pair(
            donor,
            npu,
            tmp_path / "out",
            [[np.ones((2, 3), dtype=np.float32)]],
            ["/mlp/Gemm"],
            0.5,
            1e-5,
        )
