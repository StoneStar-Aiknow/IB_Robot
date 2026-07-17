from __future__ import annotations

import json
from pathlib import Path

from inference_manifest import load_inference_manifest
from model_utils.export_onnx_rknn import _rknn_output_paths, write_rknn_deployment


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_write_rknn_deployment_uses_compiler_runtime_layout(tmp_path):
    config = {
        "type": "act",
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [6]},
            "observation.images.top": {"type": "VISUAL", "shape": [3, 16, 24]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [6]}},
    }
    _write_json(tmp_path / "config.json", config)
    _write_json(tmp_path / "policy_preprocessor.json", {"name": "pre", "steps": []})
    _write_json(tmp_path / "policy_postprocessor.json", {"name": "post", "steps": []})
    model = tmp_path / "artifacts" / "policy.rknn"
    model.parent.mkdir()
    model.write_bytes(b"rknn")
    abi = tmp_path / "policy.rknn.abi.json"
    _write_json(
        abi,
        {
            "inputs": [
                {"name": "observation.state", "index": 0, "dtype": "float32", "shape": [1, 6]},
                {
                    "name": "observation.images.top",
                    "index": 1,
                    "dtype": "float32",
                    "shape": [1, 16, 24, 3],
                    "layout": "NHWC",
                },
            ],
            "outputs": [{"name": "action", "index": 0, "dtype": "float32", "shape": [1, 4, 6]}],
        },
    )

    manifest_path = write_rknn_deployment(tmp_path, config, model, abi)
    validated = load_inference_manifest(tmp_path, "rknn")

    assert manifest_path == tmp_path / "inference_manifest.json"
    assert validated.deployment.backend == "rknn"
    assert validated.deployment.bindings["policy"].inputs[1].layout == "NHWC"
    assert validated.deployment.artifacts["policy"].path.startswith("artifacts/rknn/rknn/policy-")


def test_rknn_output_paths_derive_from_onnx_stem(tmp_path):
    onnx_path = tmp_path / "compiled" / "act.onnx"

    output, abi = _rknn_output_paths(onnx_path)

    assert output == str(onnx_path.with_suffix(".rknn"))
    assert abi == f"{output}.abi.json"
