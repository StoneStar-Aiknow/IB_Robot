from __future__ import annotations

import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from model_utils.pi05_export.quant import w8a8_common
from model_utils.pi05_export.quant.quantize_vlm import (
    _resize_calibration_images,
    _resolve_disable_regexes,
    build_arg_parser,
    validate_unfused_geglu_route,
)
from model_utils.pi05_export.verify_pi05_split_equivalence import load_real_batches_raw, preprocess_real_batches


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


def test_external_data_ort_session_loads_model_by_path(monkeypatch):
    weight = numpy_helper.from_array(np.ones(1024, dtype=np.float32), name="weight")
    graph = helper.make_graph(
        [helper.make_node("Identity", ["weight"], ["output"])],
        "test",
        [],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1024])],
        [weight],
    )
    model = helper.make_model(graph)
    captured = {}

    def fake_inference_session(path):
        model_path = captured["path"] = Path(path)
        assert model_path.is_file()
        assert model_path.with_name("model.onnx.data").is_file()
        return SimpleNamespace()

    monkeypatch.setattr(onnxruntime, "InferenceSession", fake_inference_session)

    session = w8a8_common._create_external_data_ort_session(model)
    assert captured["path"].exists()
    del session
    gc.collect()
    assert not captured["path"].exists()


def test_external_data_ort_session_uses_scratch_directory(monkeypatch, tmp_path):
    model = helper.make_model(helper.make_graph([], "test", [], []))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    captured = {}

    def fake_inference_session(path):
        captured["path"] = Path(path)
        return SimpleNamespace()

    monkeypatch.setattr(onnxruntime, "InferenceSession", fake_inference_session)

    session = w8a8_common._create_external_data_ort_session(model, scratch)
    assert captured["path"].parent.parent == scratch
    del session


def test_fp16_qdq_compatibility_transform_runs_on_opset17_ort():
    scale = numpy_helper.from_array(np.array(0.1, dtype=np.float16), name="scale")
    zero = numpy_helper.from_array(np.array(0, dtype=np.int8), name="zero")
    graph = helper.make_graph(
        [
            helper.make_node("QuantizeLinear", ["input", "scale", "zero"], ["quantized"]),
            helper.make_node("DequantizeLinear", ["quantized", "scale", "zero"], ["output"]),
        ],
        "test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT16, [2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [2])],
        [scale, zero],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)

    assert w8a8_common._make_fp16_qdq_ort_compatible(model) == (1, 1)

    session = onnxruntime.InferenceSession(model.SerializeToString())
    output = session.run(None, {"input": np.array([0.2, -0.3], dtype=np.float16)})[0]
    assert output.dtype == np.float16
    np.testing.assert_allclose(output, [0.2, -0.3], atol=1e-3)


def test_fp16_qdq_compatibility_transform_reuses_shared_input_cast():
    scale = numpy_helper.from_array(np.array(0.1, dtype=np.float16), name="scale")
    zero = numpy_helper.from_array(np.array(0, dtype=np.int8), name="zero")
    graph = helper.make_graph(
        [
            helper.make_node("QuantizeLinear", ["input", "scale", "zero"], ["quantized_0"]),
            helper.make_node("DequantizeLinear", ["quantized_0", "scale", "zero"], ["output_0"]),
            helper.make_node("QuantizeLinear", ["input", "scale", "zero"], ["quantized_1"]),
            helper.make_node("DequantizeLinear", ["quantized_1", "scale", "zero"], ["output_1"]),
        ],
        "test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT16, [2])],
        [
            helper.make_tensor_value_info("output_0", TensorProto.FLOAT16, [2]),
            helper.make_tensor_value_info("output_1", TensorProto.FLOAT16, [2]),
        ],
        [scale, zero],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)

    assert w8a8_common._make_fp16_qdq_ort_compatible(model) == (2, 2)
    assert sum(node.op_type == "Cast" and list(node.input) == ["input"] for node in model.graph.node) == 1
    onnx.checker.check_model(model)


def test_fp16_qdq_compatibility_transform_supports_quantized_gemm(tmp_path):
    del tmp_path
    initializers = [
        numpy_helper.from_array(np.array(0.1, dtype=np.float16), name="a_scale"),
        numpy_helper.from_array(np.array(0, dtype=np.uint8), name="a_zero"),
        numpy_helper.from_array(np.full((2, 2), 10, dtype=np.int8), name="weight"),
        numpy_helper.from_array(np.array(0.1, dtype=np.float16), name="b_scale"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), name="b_zero"),
        numpy_helper.from_array(np.array(0.01, dtype=np.float16), name="y_scale"),
        numpy_helper.from_array(np.array(0, dtype=np.uint8), name="y_zero"),
    ]
    graph = helper.make_graph(
        [
            helper.make_node(
                "QGemm",
                ["input", "a_scale", "a_zero", "weight", "b_scale", "b_zero", "", "y_scale", "y_zero"],
                ["output"],
                name="gemm",
                domain="com.microsoft",
            )
        ],
        "test",
        [helper.make_tensor_value_info("input", TensorProto.UINT8, [1, 2])],
        [helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 2])],
        initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
        ir_version=8,
    )

    w8a8_common._make_fp16_qdq_ort_compatible(model)

    onnx.checker.check_model(model)
    session = onnxruntime.InferenceSession(model.SerializeToString())
    output = session.run(None, {"input": np.full((1, 2), 10, dtype=np.uint8)})[0]
    np.testing.assert_array_equal(output, [[200, 200]])


def test_unloaded_external_data_is_detected():
    weight = numpy_helper.from_array(np.ones(1, dtype=np.float32), name="weight")
    weight.data_location = TensorProto.EXTERNAL
    location = weight.external_data.add()
    location.key = "location"
    location.value = "weight.data"
    weight.ClearField("raw_data")
    model = helper.make_model(helper.make_graph([], "test", [], [], [weight]))

    assert w8a8_common._has_unloaded_external_data(model)
    model.graph.initializer[0].raw_data = np.ones(1, dtype=np.float32).tobytes()
    assert not w8a8_common._has_unloaded_external_data(model)


def test_large_model_save_patch_rejects_unloaded_external_data(monkeypatch, tmp_path):
    weight = numpy_helper.from_array(np.ones(1, dtype=np.float32), name="weight")
    weight.data_location = TensorProto.EXTERNAL
    location = weight.external_data.add()
    location.key = "location"
    location.value = "weight.data"
    weight.ClearField("raw_data")
    model = helper.make_model(helper.make_graph([], "test", [], [], [weight]))
    sidecar = tmp_path / "model.onnx.data"
    sidecar.write_bytes(b"original")

    original_save = onnx.save
    original_save_model = onnx.save_model
    original_marker = getattr(onnx, "_pi05_large_save_patched", None)
    monkeypatch.setattr(w8a8_common, "_PROTOBUF_INLINE_LIMIT", 0)
    try:
        if original_marker is not None:
            del onnx._pi05_large_save_patched
        w8a8_common.install_large_model_save_patch()
        with pytest.raises(ValueError, match="unloaded external tensors"):
            onnx.save(model, tmp_path / "model.onnx")
        assert sidecar.read_bytes() == b"original"
    finally:
        onnx.save = original_save
        onnx.save_model = original_save_model
        if original_marker is None:
            del onnx._pi05_large_save_patched
        else:
            onnx._pi05_large_save_patched = original_marker


def test_large_model_save_patch_handles_explicit_false(monkeypatch, tmp_path):
    weight = numpy_helper.from_array(np.ones(1024, dtype=np.float32), name="weight")
    model = helper.make_model(helper.make_graph([], "test", [], [], [weight]))
    output = tmp_path / "model.onnx"

    original_save = onnx.save
    original_save_model = onnx.save_model
    original_marker = getattr(onnx, "_pi05_large_save_patched", None)
    monkeypatch.setattr(w8a8_common, "_PROTOBUF_INLINE_LIMIT", 0)
    try:
        if original_marker is not None:
            del onnx._pi05_large_save_patched
        w8a8_common.install_large_model_save_patch()
        onnx.save(model, output, save_as_external_data=False)
        assert output.with_name("model.onnx.data").is_file()
        onnx.load(output)
    finally:
        onnx.save = original_save
        onnx.save_model = original_save_model
        if original_marker is None:
            del onnx._pi05_large_save_patched
        else:
            onnx._pi05_large_save_patched = original_marker


def test_external_pair_save_restores_existing_files_on_install_failure(monkeypatch, tmp_path):
    output = tmp_path / "model.onnx"
    sidecar = tmp_path / "model.onnx.data"
    output.write_bytes(b"old-model")
    sidecar.write_bytes(b"old-data")

    original_replace = w8a8_common.os.replace

    def fail_model_install(source, destination):
        if Path(source).name == output.name and Path(destination) == output:
            raise OSError("injected install failure")
        original_replace(source, destination)

    monkeypatch.setattr(w8a8_common.os, "replace", fail_model_install)

    weight = numpy_helper.from_array(np.ones(1024, dtype=np.float32), name="weight")
    model = helper.make_model(helper.make_graph([], "test", [], [], [weight]))
    with pytest.raises(OSError, match="injected"):
        w8a8_common._save_external_data_pair(onnx.save_model, model, output, sidecar.name, size_threshold=1)

    assert output.read_bytes() == b"old-model"
    assert sidecar.read_bytes() == b"old-data"


def test_external_pair_save_removes_stale_sidecar(tmp_path):
    output = tmp_path / "model.onnx"
    sidecar = tmp_path / "model.onnx.data"
    output.write_bytes(b"old-model")
    sidecar.write_bytes(b"old-data")

    def fake_save(model, path, **kwargs):
        onnx.save_model(model, path, **kwargs)

    model = helper.make_model(helper.make_graph([], "test", [], []))
    w8a8_common._save_external_data_pair(fake_save, model, output, sidecar.name)

    onnx.load(output)
    assert not sidecar.exists()


def test_external_pair_save_writes_loadable_pair(tmp_path):
    weight = numpy_helper.from_array(np.ones(1024, dtype=np.float32), name="weight")
    model = helper.make_model(helper.make_graph([], "test", [], [], [weight]))
    output = tmp_path / "model.onnx"

    w8a8_common.save_onnx_external(model, output)

    loaded = onnx.load(output)
    np.testing.assert_array_equal(numpy_helper.to_array(loaded.graph.initializer[0]), np.ones(1024, dtype=np.float32))


def test_external_pair_save_accepts_ascend_custom_ops(tmp_path):
    weight = numpy_helper.from_array(np.ones(1024, dtype=np.float32), name="weight")
    graph = helper.make_graph(
        [helper.make_node("AscendQuant", ["weight"], ["output"])],
        "test",
        [],
        [helper.make_tensor_value_info("output", TensorProto.INT8, [1024])],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    output = tmp_path / "model.onnx"

    w8a8_common.save_onnx_external(model, output)

    assert onnx.load(output, load_external_data=False).graph.node[0].op_type == "AscendQuant"


def test_external_pair_validation_rejects_truncated_tensor(tmp_path):
    weight = numpy_helper.from_array(np.ones(1, dtype=np.float32), name="weight")
    weight.data_location = TensorProto.EXTERNAL
    for key, value in (("location", "model.onnx.data"), ("offset", "0"), ("length", "0")):
        entry = weight.external_data.add()
        entry.key = key
        entry.value = value
    weight.ClearField("raw_data")
    model = helper.make_model(helper.make_graph([], "test", [], [], [weight]))
    output = tmp_path / "model.onnx"
    onnx.save_model(model, output)
    output.with_name("model.onnx.data").write_bytes(b"")

    with pytest.raises(RuntimeError, match="bounds exceed"):
        w8a8_common._validate_external_data_pair(output, "model.onnx.data")


def test_external_pair_save_accepts_bfloat16_storage(tmp_path):
    weight = TensorProto(name="weight", data_type=TensorProto.BFLOAT16, dims=[1024], raw_data=b"\0" * 2048)
    model = helper.make_model(helper.make_graph([], "test", [], [], [weight]))
    output = tmp_path / "model.onnx"

    w8a8_common.save_onnx_external(model, output)

    assert onnx.load(output).graph.initializer[0].raw_data == b"\0" * 2048


def test_activation_l2_error_does_not_overflow_fp16():
    float_array = np.full(1000, 100, dtype=np.float16)
    quant_array = np.zeros(1000, dtype=np.float16)

    error = w8a8_common._activation_l2_error(float_array, quant_array)

    assert np.isfinite(error)
    assert error == np.linalg.norm(float_array.astype(np.float32))


@pytest.mark.parametrize(
    ("float_array", "quant_array", "message"),
    [
        (np.ones(2), np.ones(3), "shapes do not match"),
        (np.array([np.nan]), np.zeros(1), "finite values"),
        (np.zeros(1), np.array([np.inf]), "finite values"),
    ],
)
def test_activation_l2_error_rejects_invalid_arrays(float_array, quant_array, message):
    with pytest.raises(ValueError, match=message):
        w8a8_common._activation_l2_error(float_array, quant_array)


@pytest.mark.parametrize("amp_num", [2, 3])
def test_amp_rollback_count_rejects_full_or_oversized_rollback(amp_num):
    with pytest.raises(ValueError, match="smaller than the 2 rankable"):
        w8a8_common._validate_amp_rollback_count(amp_num, 2)


def test_amp_calibration_samples_can_be_replaced():
    rollback_module = SimpleNamespace()
    first = [[np.array([1], dtype=np.float32)]]
    second = [[np.array([2], dtype=np.float32)]]

    w8a8_common._set_msmodelslim_amp_calib_samples(rollback_module, first)
    w8a8_common._set_msmodelslim_amp_calib_samples(rollback_module, second)

    assert rollback_module._pi05_calib_samples is second


def test_amp_scratch_directory_can_be_replaced(tmp_path):
    rollback_module = SimpleNamespace()
    first = tmp_path / "first"
    second = tmp_path / "second"

    w8a8_common._set_msmodelslim_amp_scratch_dir(rollback_module, first)
    w8a8_common._set_msmodelslim_amp_scratch_dir(rollback_module, second)

    assert rollback_module._pi05_amp_scratch_dir == second


def test_activation_l2_sums_aggregate_samples():
    sums = np.zeros(2, dtype=np.float64)
    w8a8_common._update_activation_l2_sums(
        sums,
        [np.array([1, 2], dtype=np.float16), np.array([4], dtype=np.float16)],
        [np.array([0, 0], dtype=np.float16), np.array([1], dtype=np.float16)],
    )
    w8a8_common._update_activation_l2_sums(
        sums,
        [np.array([3, 4], dtype=np.float16), np.array([2], dtype=np.float16)],
        [np.array([0, 0], dtype=np.float16), np.array([1], dtype=np.float16)],
    )

    np.testing.assert_allclose(sums / 2, [(np.sqrt(5) + 5) / 2, 2])


def test_amp_rank_samples_cli_default_and_override():
    parser = build_arg_parser()

    assert parser.parse_args(["--onnx-path", "model.onnx"]).amp_rank_samples == 1
    assert parser.parse_args(["--onnx-path", "model.onnx", "--amp-rank-samples", "8"]).amp_rank_samples == 8


def test_smoothquant_cli_defaults_and_override(tmp_path):
    parser = build_arg_parser()

    defaults = parser.parse_args(["--onnx-path", "model.onnx"])
    assert defaults.smoothquant_alpha is None
    assert defaults.smoothquant_epsilon == 1e-5
    assert defaults.smoothquant_output_dir is None

    configured = parser.parse_args(
        [
            "--onnx-path",
            "model.onnx",
            "--smoothquant-alpha",
            "0.3",
            "--smoothquant-epsilon",
            "1e-4",
            "--smoothquant-output-dir",
            str(tmp_path),
        ]
    )
    assert configured.smoothquant_alpha == 0.3
    assert configured.smoothquant_epsilon == 1e-4
    assert configured.smoothquant_output_dir == tmp_path


def test_quantize_regex_restricts_nodes_and_preserves_exclusions():
    quantizable = [
        ("/layers.0/mlp/MatMul", "MatMul"),
        ("/layers.0/mlp/down_proj/MatMul", "MatMul"),
        ("/vision/fc1/MatMul", "MatMul"),
    ]

    disabled = w8a8_common.restrict_quantizable_nodes(
        quantizable,
        ["/vision/fc1/MatMul"],
        [r"/layers\.0/mlp/(?:MatMul|down_proj/MatMul)$", r"/vision/"],
    )

    assert disabled == ["/vision/fc1/MatMul"]


def test_quantize_regex_rejects_no_matches():
    with pytest.raises(ValueError, match="None of the --quantize-regex"):
        w8a8_common.restrict_quantizable_nodes([("/layers.0/mlp/MatMul", "MatMul")], [], [r"/missing/"])


def test_unfused_geglu_route_removes_only_builtin_projection_exclusion():
    fallback = _resolve_disable_regexes(None, unfused_geglu=True)

    assert not any("gate_proj" in pattern for pattern in fallback)
    assert r"self_attn/MatMul(_\d+)?$" in fallback
    assert _resolve_disable_regexes(["custom"], unfused_geglu=True) == ["custom"]


def test_restore_opset_imports_preserves_custom_domains():
    model = helper.make_model(helper.make_graph([], "test", [], []), opset_imports=[helper.make_opsetid("", 11)])

    w8a8_common._restore_opset_imports(model, [("", 17), ("com.microsoft", 1)])

    assert [(entry.domain, entry.version) for entry in model.opset_import] == [("", 17), ("com.microsoft", 1)]


def test_load_real_batches_raw_converts_numeric_fields_only(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            [
                {
                    "task": "pick",
                    "state": [1, 2],
                    "mask": [True, False],
                    "optional": None,
                    "metadata": {"source": "camera"},
                }
            ]
        ),
        encoding="utf-8",
    )

    sample = load_real_batches_raw(str(path))[0]

    assert sample["task"] == "pick"
    np.testing.assert_array_equal(sample["state"], np.array([1, 2], dtype=np.float32))
    np.testing.assert_array_equal(sample["mask"], np.array([True, False]))
    assert sample["optional"] is None
    assert sample["metadata"] == {"source": "camera"}


def test_preprocess_real_batches_overrides_checkpoint_device(monkeypatch, tmp_path):
    import lerobot.policies.factory as factory
    import lerobot.policies.utils as policy_utils

    import inference_service.lerobot_assets as assets

    captured = {}

    def fake_make_pre_post_processors(**kwargs):
        captured["overrides"] = kwargs["preprocessor_overrides"]
        return lambda observation: observation, None

    def fake_prepare(observation, device, task):
        captured["task"] = task
        return {"value": torch.ones(1, device=device), "task": task}

    tokenizer = tmp_path / "tokenizer"
    monkeypatch.setattr(factory, "make_pre_post_processors", fake_make_pre_post_processors)
    monkeypatch.setattr(policy_utils, "prepare_observation_for_inference", fake_prepare)
    monkeypatch.setattr(assets, "resolve_local_semantic_reference", lambda *_args: tokenizer)

    result = preprocess_real_batches(
        [{"observation.state": np.zeros(6, dtype=np.float32), "task": "pick banana"}],
        str(tmp_path),
        object(),
        torch.device("cpu"),
    )

    assert captured["overrides"] == {
        "device_processor": {"device": "cpu"},
        "tokenizer_processor": {"tokenizer_name": tokenizer},
    }
    assert captured["task"] == "pick banana"
    assert list(result[0]) == ["value"]


def test_preprocess_real_batches_filters_non_array_metadata(monkeypatch, tmp_path):
    import lerobot.policies.factory as factory
    import lerobot.policies.utils as policy_utils

    import inference_service.lerobot_assets as assets

    captured = {}

    def fake_prepare(observation, _device, task):
        captured["observation"] = observation
        captured["task"] = task
        return {"value": torch.ones(1)}

    monkeypatch.setattr(factory, "make_pre_post_processors", lambda **_kwargs: (lambda observation: observation, None))
    monkeypatch.setattr(policy_utils, "prepare_observation_for_inference", fake_prepare)
    monkeypatch.setattr(assets, "resolve_local_semantic_reference", lambda *_args: None)

    result = preprocess_real_batches(
        [
            {
                "observation.state": np.zeros(6, dtype=np.float32),
                "task": "pick",
                "optional": None,
                "metadata": {"source": "camera"},
            }
        ],
        str(tmp_path),
        object(),
        torch.device("cpu"),
    )

    assert list(captured["observation"]) == ["observation.state"]
    assert captured["task"] == "pick"
    assert list(result[0]) == ["value"]


def test_route_a_transplants_fused_geglu_matmul(tmp_path):
    donor_path = tmp_path / "donor.onnx"
    npu_path = tmp_path / "npu.onnx"
    output_path = tmp_path / "output.onnx"
    stem = "/layers.0/mlp/MatMul"
    weight_int8 = numpy_helper.from_array(np.ones((2, 4), dtype=np.int8), name="fused_weight_quantized")
    deq_scale = numpy_helper.from_array(np.ones(4, dtype=np.uint64), name="fused_deq_scale")
    donor_graph = helper.make_graph(
        [
            helper.make_node("AscendQuant", ["x"], ["x_quant"], name=f"{stem}_input_quant"),
            helper.make_node(
                "MatMul",
                ["x_quant", weight_int8.name],
                ["fused_int32"],
                name=f"{stem}_quant",
            ),
            helper.make_node(
                "AscendDequant",
                ["fused_int32", deq_scale.name],
                ["fused_fp16"],
                name=f"{stem}_dequant",
            ),
            helper.make_node("Identity", ["fused_fp16"], ["y"], name="donor_output"),
        ],
        "donor",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 4])],
        [weight_int8, deq_scale],
    )
    onnx.save(helper.make_model(donor_graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), donor_path)

    weight_fp16 = numpy_helper.from_array(np.ones((2, 4), dtype=np.float16), name="fused_weight_fp16")
    npu_graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", weight_fp16.name], ["up_gate"], name=stem),
            helper.make_node(
                "NPUGeglu",
                ["up_gate"],
                ["y"],
                name="/layers.0/mlp/NPUGeglu",
                dim=-1,
                approximate=1,
                activate_left=0,
            ),
        ],
        "npu",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 2])],
        [weight_fp16],
    )
    onnx.save(helper.make_model(npu_graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), npu_path)

    w8a8_common.transplant_int8_into_npu_graph(donor_path, npu_path, output_path)

    result = onnx.load(output_path, load_external_data=False)
    nodes = {node.name: node for node in result.graph.node}
    assert stem not in nodes
    assert f"{stem}_quant" in nodes
    assert "/layers.0/mlp/NPUGeglu" in nodes
    assert nodes[f"{stem}_dequant"].output == ["up_gate"]
    assert nodes["/layers.0/mlp/NPUGeglu"].input == ["up_gate"]
    initializers = {initializer.name: initializer for initializer in result.graph.initializer}
    assert initializers[weight_int8.name].data_type == TensorProto.INT8
    assert weight_fp16.name not in initializers


def test_fused_geglu_route_rejects_legacy_donor():
    donor = helper.make_model(
        helper.make_graph(
            [helper.make_node("MatMul", ["x", "w"], ["gate"], name="/layers.0/mlp/gate_proj/MatMul")],
            "donor",
            [],
            [],
        )
    )
    npu = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("MatMul", ["x", "w"], ["up_gate"], name="/layers.0/mlp/MatMul"),
                helper.make_node("NPUGeglu", ["up_gate"], ["y"], name="/layers.0/mlp/NPUGeglu"),
            ],
            "npu",
            [],
            [],
        )
    )

    with pytest.raises(RuntimeError, match="missing MatMul"):
        w8a8_common.validate_fused_geglu_route(donor, npu)


def test_npu_geglu_deployment_requires_fused_site():
    npu = helper.make_model(helper.make_graph([], "npu", [], []))

    with pytest.raises(RuntimeError, match="no NPUGeglu"):
        w8a8_common.validate_npu_geglu_deployment(npu)


def test_npu_geglu_deployment_enforces_expected_site_count():
    weight = numpy_helper.from_array(np.ones((2, 4), dtype=np.float16), name="weight")
    npu = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("MatMul", ["x", weight.name], ["up_gate"], name="/layers.0/mlp/MatMul"),
                helper.make_node("NPUGeglu", ["up_gate"], ["y"], name="/layers.0/mlp/NPUGeglu"),
            ],
            "npu",
            [],
            [],
            [weight],
        )
    )

    assert w8a8_common.validate_npu_geglu_deployment(npu, expected=1) == ["/layers.0/mlp/MatMul"]
    with pytest.raises(RuntimeError, match="1 NPUGeglu nodes, expected 2"):
        w8a8_common.validate_npu_geglu_deployment(npu, expected=2)


def test_unfused_geglu_route_matches_separate_projections():
    gate_weight = numpy_helper.from_array(np.ones((2, 4), dtype=np.float16), name="gate_weight")
    up_weight = numpy_helper.from_array(np.ones((2, 4), dtype=np.float16), name="up_weight")
    nodes = [
        helper.make_node("MatMul", ["x", gate_weight.name], ["gate"], name="/layers.0/mlp/gate_proj/MatMul"),
        helper.make_node("MatMul", ["x", up_weight.name], ["up"], name="/layers.0/mlp/up_proj/MatMul"),
    ]
    donor = helper.make_model(helper.make_graph(nodes, "donor", [], [], [gate_weight, up_weight]))
    npu = helper.make_model(helper.make_graph(nodes, "npu", [], [], [gate_weight, up_weight]))

    assert validate_unfused_geglu_route(donor, npu) == [
        "/layers.0/mlp/gate_proj/MatMul",
        "/layers.0/mlp/up_proj/MatMul",
    ]
