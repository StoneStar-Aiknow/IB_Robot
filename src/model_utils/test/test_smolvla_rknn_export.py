from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_manifest import load_inference_manifest
from model_utils.smolvla_export.export_rknn_modules import write_smolvla_rknn_deployment


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _bundle(root: Path) -> dict:
    config = {
        "type": "smolvla",
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [6]},
            "observation.images.top": {"type": "VISUAL", "shape": [3, 480, 640]},
            "observation.images.wrist": {"type": "VISUAL", "shape": [3, 480, 640]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [6]}},
        "tokenizer_max_length": 4,
        "max_state_dim": 8,
        "chunk_size": 2,
        "max_action_dim": 8,
        "num_steps": 2,
        "add_image_special_tokens": False,
        "empty_cameras": 0,
    }
    _write_json(root / "config.json", config)
    _write_json(root / "policy_preprocessor.json", {"name": "pre", "steps": []})
    _write_json(root / "policy_postprocessor.json", {"name": "post", "steps": []})
    return config


def _abi_files(root: Path, *, legacy_action: bool = False) -> tuple[Path, Path, Path]:
    vision = root / "vision.abi.json"
    prefill = root / "prefill.abi.json"
    action = root / "action.abi.json"
    _write_json(
        vision,
        {
            "inputs": [
                {
                    "name": "pixel_values",
                    "index": 0,
                    "dtype": "float32",
                    "shape": [1, 3, 16, 16],
                    "layout": "NCHW",
                }
            ],
            "outputs": [{"name": "image_embeddings", "index": 0, "dtype": "float32", "shape": [1, 2, 4]}],
        },
    )
    _write_json(
        prefill,
        {
            "inputs": [
                {"name": "prefix_embs", "index": 0, "dtype": "float32", "shape": [1, 9, 4]},
                {"name": "attention_mask", "index": 1, "dtype": "int64", "shape": [1, 9, 9]},
                {"name": "position_ids", "index": 2, "dtype": "int64", "shape": [1, 9]},
            ],
            "outputs": [
                {"name": "past_key_0", "index": 0, "dtype": "float32", "shape": [1, 9, 1, 4]},
                {"name": "past_value_0", "index": 1, "dtype": "float32", "shape": [1, 9, 1, 4]},
            ],
        },
    )
    action_inputs = (
        [
            {"name": "past_kv_tensor", "index": 0, "dtype": "float32", "shape": [1, 2, 1, 9, 1, 4]},
            {"name": "prefix_pad_masks", "index": 1, "dtype": "bool", "shape": [1, 9]},
            {"name": "time", "index": 2, "dtype": "float32", "shape": [1]},
            {"name": "noise", "index": 3, "dtype": "float32", "shape": [1, 2, 8]},
        ]
        if legacy_action
        else [
            {"name": "x_t", "index": 0, "dtype": "float32", "shape": [1, 2, 8]},
            {"name": "timestep", "index": 1, "dtype": "float32", "shape": [1]},
            {"name": "prefix_pad_masks", "index": 2, "dtype": "bool", "shape": [1, 9]},
            {"name": "past_key_0", "index": 3, "dtype": "float32", "shape": [1, 9, 1, 4]},
            {"name": "past_value_0", "index": 4, "dtype": "float32", "shape": [1, 9, 1, 4]},
        ]
    )
    _write_json(
        action,
        {
            "inputs": action_inputs,
            "outputs": [{"name": "v_t", "index": 0, "dtype": "float32", "shape": [1, 2, 8]}],
        },
    )
    return vision, prefill, action


def test_write_smolvla_rknn_deployment_packages_multi_camera_execution(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    config = _bundle(bundle)
    compiler = tmp_path / "compiler"
    compiler.mkdir()
    vision_abi, prefill_abi, action_abi = _abi_files(compiler)
    vision = compiler / "vision.rknn"
    prefill = compiler / "prefill.rknn"
    action = compiler / "action.rknn"
    embedding = compiler / "embedding.pt"
    state_projection = compiler / "state_projection.pt"
    for path in (vision, prefill, action, embedding, state_projection):
        path.write_bytes(path.name.encode())

    manifest_path = write_smolvla_rknn_deployment(
        bundle,
        config,
        vision_rknn=vision,
        vision_abi_path=vision_abi,
        prefill_rknn=prefill,
        prefill_abi_path=prefill_abi,
        action_rknn=action,
        action_abi_path=action_abi,
        embedding_path=embedding,
        state_projection_path=state_projection,
    )
    validated = load_inference_manifest(bundle, "rknn")

    assert manifest_path == bundle / "inference_manifest.json"
    assert validated.deployment.execution == ("vision_top", "vision_wrist", "embedding", "prefill", "action")
    assert set(validated.deployment.artifacts) == {
        "vision_top",
        "vision_wrist",
        "embedding",
        "prefill",
        "action",
        "state_projection",
    }
    assert validated.deployment.artifacts["vision_top"].path == validated.deployment.artifacts["vision_wrist"].path
    assert validated.deployment.artifacts["vision_top"].share_group == "vision"
    assert validated.deployment.artifacts["vision_wrist"].share_group == "vision"
    assert tuple(binding.semantic for binding in validated.deployment.bindings["action"].inputs[-2:]) == (
        "internal.past_key.0",
        "internal.past_value.0",
    )


def test_write_smolvla_rknn_deployment_rejects_legacy_flattened_cache_action(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    config = _bundle(bundle)
    compiler = tmp_path / "compiler"
    compiler.mkdir()
    vision_abi, prefill_abi, action_abi = _abi_files(compiler, legacy_action=True)
    artifacts = []
    for name in ("vision.rknn", "prefill.rknn", "action.rknn", "embedding.pt", "state_projection.pt"):
        path = compiler / name
        path.write_bytes(name.encode())
        artifacts.append(path)

    with pytest.raises(ValueError, match="Legacy flattened-KV"):
        write_smolvla_rknn_deployment(
            bundle,
            config,
            vision_rknn=artifacts[0],
            vision_abi_path=vision_abi,
            prefill_rknn=artifacts[1],
            prefill_abi_path=prefill_abi,
            action_rknn=artifacts[2],
            action_abi_path=action_abi,
            embedding_path=artifacts[3],
            state_projection_path=artifacts[4],
        )
