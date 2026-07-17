import json
from pathlib import Path
from unittest.mock import patch

import onnx
import pytest
from onnx import TensorProto, helper

from inference_manifest import load_inference_manifest
from model_utils.export_onnx_atc import _act_input_shape, convert_onnx_to_om, write_ascend_deployment


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _create_policy_bundle(policy_dir: Path) -> dict:
    config = {
        "type": "act",
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [6]},
            "observation.images.front": {"type": "VISUAL", "shape": [3, 16, 24]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [6]}},
        "chunk_size": 4,
    }
    _write_json(policy_dir / "config.json", config)
    _write_json(policy_dir / "policy_preprocessor.json", {"name": "pre", "steps": []})
    _write_json(policy_dir / "policy_postprocessor.json", {"name": "post", "steps": []})
    return config


def _write_act_onnx(path: Path) -> None:
    graph = helper.make_graph(
        [],
        "act",
        [
            helper.make_tensor_value_info("observation.state", TensorProto.FLOAT, [1, 6]),
            helper.make_tensor_value_info("observation.images.front", TensorProto.FLOAT, [1, 3, 16, 24]),
        ],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 4, 6])],
    )
    onnx.save(helper.make_model(graph), path)


def _write_act_abi(path: Path, *, output_name: str = "action") -> None:
    _write_json(
        path,
        {
            "inputs": [
                {"name": "observation.state", "index": 0, "dtype": "float32", "shape": [1, 6]},
                {
                    "name": "observation.images.front",
                    "index": 1,
                    "dtype": "float32",
                    "shape": [1, 3, 16, 24],
                    "layout": "NCHW",
                },
            ],
            "outputs": [{"name": output_name, "index": 0, "dtype": "float32", "shape": [1, 4, 6]}],
        },
    )


def test_write_ascend_deployment_records_complete_manifest(tmp_path):
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    config = _create_policy_bundle(policy_dir)
    onnx_path = policy_dir / "model.onnx"
    _write_act_onnx(onnx_path)
    om_path = policy_dir / "model.om"
    om_path.write_bytes(b"om")
    abi_path = policy_dir / "model.om.abi.json"
    _write_act_abi(abi_path)

    manifest_path = write_ascend_deployment(
        str(policy_dir),
        config,
        str(onnx_path),
        str(om_path),
        "Ascend310B1",
        str(abi_path),
    )
    validated = load_inference_manifest(policy_dir, "ascend")

    assert manifest_path == policy_dir / "inference_manifest.json"
    assert validated.deployment.backend == "ascend"
    assert validated.deployment.target.soc == "Ascend310B1"
    assert validated.deployment.artifacts["policy"].path.startswith("artifacts/ascend/ascend/policy-")
    assert validated.deployment.artifacts["policy"].path.endswith(".om")
    assert [binding.semantic for binding in validated.deployment.bindings["policy"].inputs] == [
        "observation.state",
        "observation.images.front",
    ]
    assert validated.deployment.bindings["policy"].inputs[1].layout == "NCHW"


def test_write_ascend_deployment_maps_compiler_output_name_to_action(tmp_path):
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    config = _create_policy_bundle(policy_dir)
    onnx_path = policy_dir / "model.onnx"
    _write_act_onnx(onnx_path)
    om_path = policy_dir / "model.om"
    om_path.write_bytes(b"om")
    abi_path = policy_dir / "model.om.abi.json"
    runtime_name = "/model/action_head/Add:0:action"
    _write_act_abi(abi_path, output_name=runtime_name)

    write_ascend_deployment(
        str(policy_dir),
        config,
        str(onnx_path),
        str(om_path),
        "Ascend310B1",
        str(abi_path),
    )
    validated = load_inference_manifest(policy_dir, "ascend")
    output = validated.deployment.bindings["policy"].outputs[0]

    assert output.runtime_name == runtime_name
    assert output.semantic == "action"


def test_write_ascend_deployment_rejects_non_act_policy(tmp_path):
    with pytest.raises(ValueError, match="not supported"):
        write_ascend_deployment(
            str(tmp_path),
            {"type": "pi05"},
            str(tmp_path / "model.onnx"),
            str(tmp_path / "model.om"),
            "Ascend310B1",
            str(tmp_path / "model.om.abi.json"),
        )


def test_write_ascend_deployment_requires_compiled_runtime_abi(tmp_path):
    with pytest.raises(ValueError, match="compiler/runtime ABI JSON"):
        write_ascend_deployment(
            str(tmp_path),
            {"type": "act"},
            str(tmp_path / "model.onnx"),
            str(tmp_path / "model.om"),
            "Ascend310B1",
        )


def test_act_input_shape_only_includes_runtime_inputs():
    config = {
        "input_features": {
            "observation.state": {"shape": [6]},
            "observation.state.current": {"shape": [6]},
            "observation.images.front": {"shape": [3, 240, 320]},
        }
    }

    assert _act_input_shape(config) == "observation.state:1,6;observation.images.front:1,3,240,320"


def test_convert_onnx_to_om_uses_config_input_shape():
    with patch("model_utils.export_onnx_atc.subprocess.run") as run:
        run.return_value.returncode = 0

        assert convert_onnx_to_om(
            {"input_features": {"observation.state": {"shape": [6]}}},
            "/tmp/model.onnx",
            "/tmp/model.om",
            "Ascend310B1",
        )

    run.assert_called_once_with(
        [
            "atc",
            "--framework=5",
            "--soc_version=Ascend310B1",
            "--model=/tmp/model.onnx",
            "--output=/tmp/model",
            "--input_shape=observation.state:1,6",
        ],
        check=False,
    )
