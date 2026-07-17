from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inference_manifest import BundleFile, canonical_bundle_digest, sha256_file

POLICY_FEATURES = {
    "input_features": {
        "observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.top": {"type": "VISUAL", "shape": [3, 16, 24]},
    },
    "output_features": {"action": {"type": "ACTION", "shape": [6]}},
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def create_policy_bundle(
    root: Path,
    policy_type: str = "act",
    *,
    local_tokenizer: bool = False,
    include_weights: bool = True,
) -> tuple[str, ...]:
    config: dict[str, Any] = {
        "type": policy_type,
        **POLICY_FEATURES,
        "device": "cuda",
    }
    preprocessor_steps: list[dict[str, Any]] = [
        {
            "registry_name": "normalizer_processor",
            "config": {"features": {**POLICY_FEATURES["input_features"], **POLICY_FEATURES["output_features"]}},
            "state_file": "policy_preprocessor_step_0_normalizer_processor.safetensors",
        }
    ]
    if policy_type in {"pi05", "smolvla"}:
        tokenizer_name = (
            "tokenizer"
            if local_tokenizer
            else ("bert-base-uncased" if policy_type == "pi05" else "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        )
        preprocessor_steps.append(
            {
                "registry_name": "tokenizer_processor",
                "config": {"tokenizer_name": tokenizer_name, "max_length": 48},
            }
        )
        if policy_type == "smolvla":
            config["vlm_model_name"] = tokenizer_name

    write_json(root / "config.json", config)
    write_json(root / "policy_preprocessor.json", {"name": "policy_preprocessor", "steps": preprocessor_steps})
    write_json(
        root / "policy_postprocessor.json",
        {
            "name": "policy_postprocessor",
            "steps": [
                {
                    "registry_name": "unnormalizer_processor",
                    "config": {"features": POLICY_FEATURES["output_features"]},
                    "state_file": "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
                }
            ],
        },
    )
    (root / "policy_preprocessor_step_0_normalizer_processor.safetensors").write_bytes(b"preprocessor-state")
    (root / "policy_postprocessor_step_0_unnormalizer_processor.safetensors").write_bytes(b"postprocessor-state")
    if include_weights:
        (root / "model.safetensors").write_bytes(b"native-policy-weights")
    if local_tokenizer:
        tokenizer = root / "tokenizer"
        tokenizer.mkdir()
        write_json(tokenizer / "tokenizer_config.json", {"model_max_length": 48})
        write_json(tokenizer / "tokenizer.json", {"version": "1.0"})

    required = {
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_0_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    }
    if include_weights:
        required.add("model.safetensors")
    if local_tokenizer:
        required.update({"tokenizer/tokenizer_config.json", "tokenizer/tokenizer.json"})
    return tuple(sorted(required))


def make_manifest(
    root: Path,
    bundle_paths: tuple[str, ...],
    *,
    deployment_name: str = "cpu",
    compiled: bool = False,
    backend: str = "rknn",
) -> dict[str, Any]:
    entries = [BundleFile(path=path, sha256=sha256_file(root.joinpath(*path.split("/")))) for path in bundle_paths]
    if compiled:
        artifact_path = "artifacts/policy.rknn"
        artifact_file = root / artifact_path
        artifact_file.parent.mkdir(parents=True, exist_ok=True)
        artifact_file.write_bytes(b"compiled-policy")
        deployment: dict[str, Any] = {
            "backend": backend,
            "target": {"soc": "rk3588", "runtime": "rknn-lite"},
            "artifacts": {
                "policy": {
                    "path": artifact_path,
                    "format": "rknn",
                    "sha256": sha256_file(artifact_file),
                }
            },
            "execution": ["policy"],
            "bindings": {
                "policy": {
                    "inputs": [
                        {
                            "semantic": "observation.state",
                            "runtime_name": "state",
                            "index": 0,
                            "dtype": "float32",
                            "shape": [1, 6],
                        },
                        {
                            "semantic": "observation.images.top",
                            "runtime_name": "image",
                            "index": 1,
                            "dtype": "float32",
                            "layout": "NCHW",
                            "shape": [1, 3, 16, 24],
                        },
                    ],
                    "outputs": [
                        {
                            "semantic": "action",
                            "runtime_name": "actions",
                            "index": 0,
                            "dtype": "float32",
                            "shape": [1, 4, 6],
                        }
                    ],
                }
            },
        }
    else:
        deployment = {"backend": "torch", "device": "cpu"}

    return {
        "schema_version": 1,
        "bundle": {
            "name": f"test-{deployment_name}",
            "files": [entry.model_dump(mode="json") for entry in entries],
            "digest": {"algorithm": "sha256", "value": canonical_bundle_digest(entries)},
        },
        "deployments": {deployment_name: deployment},
    }


def write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    write_json(root / "inference_manifest.json", manifest)
