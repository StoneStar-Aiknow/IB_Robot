from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_manifest import TensorBinding, TorchDeployment, load_inference_manifest, sha256_file
from model_utils.inference_manifest_export import (
    RuntimeABI,
    RuntimeTensor,
    artifact_bindings,
    compiled_deployment,
    copy_policy_metadata_bundle,
    package_deployment_artifact,
    read_runtime_abi,
    read_tcim_abi,
    upsert_deployment,
)
from model_utils.package_compiled_deployment import package_compiled_deployment
from model_utils.package_torch_deployment import package_torch_deployments


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _create_bundle(root: Path, policy_type: str = "act") -> None:
    _write_json(
        root / "config.json",
        {
            "type": policy_type,
            "input_features": {
                "observation.state": {"type": "STATE", "shape": [6]},
                "observation.images.top": {"type": "VISUAL", "shape": [3, 16, 24]},
            },
            "output_features": {"action": {"type": "ACTION", "shape": [6]}},
        },
    )
    _write_json(root / "policy_preprocessor.json", {"name": "pre", "steps": []})
    _write_json(root / "policy_postprocessor.json", {"name": "post", "steps": []})


def _act_abi() -> RuntimeABI:
    return RuntimeABI(
        inputs=(
            RuntimeTensor("state", 0, "float32", (1, 6)),
            RuntimeTensor("image", 1, "float32", (1, 3, 16, 24)),
        ),
        outputs=(RuntimeTensor("action", 0, "float32", (1, 4, 6)),),
    )


def _act_bindings():
    return artifact_bindings(
        _act_abi(),
        input_semantics={"state": "observation.state", "image": "observation.images.top"},
        output_semantics={"action": "action"},
        image_layouts={"observation.images.top": "NCHW"},
    )


def test_upsert_deployment_preserves_existing_deployments_and_rehashes_bundle(tmp_path):
    _create_bundle(tmp_path)
    artifact = tmp_path / "artifacts" / "policy.rknn"
    artifact.parent.mkdir()
    artifact.write_bytes(b"rknn")
    torch_deployment = TorchDeployment(backend="torch", device="cpu")
    with pytest.raises(ValueError, match="model.safetensors"):
        upsert_deployment(tmp_path, "cpu", torch_deployment)

    (tmp_path / "model.safetensors").write_bytes(b"weights")
    upsert_deployment(tmp_path, "cpu", torch_deployment)
    deployment = compiled_deployment(
        tmp_path,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (artifact, "rknn")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )
    validated = upsert_deployment(tmp_path, "rknn", deployment)

    assert set(validated.manifest.deployments) == {"cpu", "rknn"}
    assert "model.safetensors" in {entry.path for entry in validated.manifest.bundle.files}
    assert load_inference_manifest(tmp_path, "cpu").deployment.backend == "torch"


def test_package_torch_deployments_generates_cpu_and_cuda_by_default(tmp_path):
    _create_bundle(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    validated = package_torch_deployments(tmp_path)

    assert [item.deployment_name for item in validated] == ["torch-cpu", "torch-cuda"]
    assert [item.deployment.device for item in validated] == ["cpu", "cuda"]
    manifest = load_inference_manifest(tmp_path, "torch-cpu").manifest
    assert set(manifest.deployments) == {"torch-cpu", "torch-cuda"}


def test_package_torch_deployments_supports_device_selection_and_prefix(tmp_path):
    _create_bundle(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    validated = package_torch_deployments(tmp_path, devices=("cpu",), deployment_prefix="native")

    assert validated[0].deployment_name == "native-cpu"
    assert validated[0].deployment.device == "cpu"


def test_package_deployment_artifact_accepts_canonical_source_path(tmp_path):
    source = tmp_path / "policy.rknn"
    source.write_bytes(b"rknn")
    artifact = tmp_path / "artifacts" / "rknn" / "rk3588" / f"policy-{sha256_file(source)[:12]}.rknn"
    artifact.parent.mkdir(parents=True)
    source.rename(artifact)

    packaged = package_deployment_artifact(
        tmp_path,
        artifact,
        backend="rknn",
        deployment_name="rk3588",
        role="policy",
        force_copy=True,
    )

    assert packaged == artifact


def test_artifact_binding_requires_complete_semantic_and_layout_mapping():
    with pytest.raises(ValueError, match="No semantic mapping"):
        artifact_bindings(
            _act_abi(),
            input_semantics={"state": "observation.state"},
            output_semantics={"action": "action"},
        )
    with pytest.raises(ValueError, match="explicit runtime layout"):
        artifact_bindings(
            _act_abi(),
            input_semantics={"state": "observation.state", "image": "observation.images.top"},
            output_semantics={"action": "action"},
        )


def test_read_runtime_abi_rejects_noncontiguous_indices(tmp_path):
    metadata = tmp_path / "abi.json"
    _write_json(
        metadata,
        {
            "inputs": [{"name": "state", "index": 1, "dtype": "float32", "shape": [1, 6]}],
            "outputs": [{"name": "action", "index": 0, "dtype": "float32", "shape": [1, 4, 6]}],
        },
    )

    with pytest.raises(ValueError, match="contiguous from zero"):
        read_runtime_abi(metadata)


def test_read_runtime_abi_accepts_sparse_output_indices(tmp_path):
    metadata = tmp_path / "abi.json"
    _write_json(
        metadata,
        {
            "inputs": [{"name": "state", "index": 0, "dtype": "float32", "shape": [1, 6]}],
            "outputs": [{"name": "action", "index": 3, "dtype": "float32", "shape": [1, 4, 6]}],
        },
    )

    abi = read_runtime_abi(metadata)

    assert abi.outputs[0].index == 3


def test_runtime_image_layout_is_authoritative():
    abi = RuntimeABI(
        inputs=(RuntimeTensor("image", 0, "float32", (1, 16, 24, 3), "NHWC"),),
        outputs=(RuntimeTensor("action", 0, "float32", (1, 4, 6)),),
    )

    bindings = artifact_bindings(
        abi,
        input_semantics={"image": "observation.images.top"},
        output_semantics={"action": "action"},
    )

    assert bindings.inputs[0].layout == "NHWC"
    with pytest.raises(ValueError, match="runtime ABI reports NHWC"):
        artifact_bindings(
            abi,
            input_semantics={"image": "observation.images.top"},
            output_semantics={"action": "action"},
            image_layouts={"observation.images.top": "NCHW"},
        )


def test_upsert_deployment_restores_previous_manifest_on_strict_validation_failure(tmp_path):
    _create_bundle(tmp_path)
    artifact = tmp_path / "policy.rknn"
    artifact.write_bytes(b"rknn")
    valid = compiled_deployment(
        tmp_path,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (artifact, "rknn")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )
    upsert_deployment(tmp_path, "rknn", valid)
    original = (tmp_path / "inference_manifest.json").read_bytes()
    invalid_bindings = artifact_bindings(
        RuntimeABI(
            inputs=(RuntimeTensor("unknown", 0, "float32", (1, 6)),),
            outputs=(RuntimeTensor("action", 0, "float32", (1, 4, 6)),),
        ),
        input_semantics={"unknown": "observation.unknown"},
        output_semantics={"action": "action"},
    )
    invalid = compiled_deployment(
        tmp_path,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (artifact, "rknn")},
        execution=("policy",),
        bindings={"policy": invalid_bindings},
    )

    with pytest.raises(ValueError, match="unknown LeRobot input feature"):
        upsert_deployment(tmp_path, "rknn", invalid)

    assert (tmp_path / "inference_manifest.json").read_bytes() == original


def test_upsert_deployment_validates_preserved_deployment_artifacts(tmp_path):
    _create_bundle(tmp_path)
    first_artifact = tmp_path / "first.rknn"
    first_artifact.write_bytes(b"first")
    first = compiled_deployment(
        tmp_path,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (first_artifact, "rknn")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )
    upsert_deployment(tmp_path, "first", first)
    original = (tmp_path / "inference_manifest.json").read_bytes()
    first_artifact.write_bytes(b"changed")

    second_artifact = tmp_path / "second.om"
    second_artifact.write_bytes(b"second")
    second = compiled_deployment(
        tmp_path,
        backend="ascend",
        target_soc="Ascend310P3",
        target_runtime="acl",
        artifacts={"policy": (second_artifact, "om")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        upsert_deployment(tmp_path, "second", second)

    assert (tmp_path / "inference_manifest.json").read_bytes() == original


def test_upsert_deployment_rejects_external_semantic_dependencies(tmp_path):
    _create_bundle(tmp_path, "smolvla")
    preprocessor = json.loads((tmp_path / "policy_preprocessor.json").read_text(encoding="utf-8"))
    preprocessor["steps"] = [
        {
            "registry_name": "tokenizer_processor",
            "config": {"tokenizer_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"},
        }
    ]
    _write_json(tmp_path / "policy_preprocessor.json", preprocessor)
    artifact = tmp_path / "policy.rknn"
    artifact.write_bytes(b"rknn")
    deployment = compiled_deployment(
        tmp_path,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (artifact, "rknn")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )

    with pytest.raises(ValueError, match="all semantic dependencies to be local bundle assets"):
        upsert_deployment(tmp_path, "rknn", deployment)

    assert not (tmp_path / "inference_manifest.json").exists()


def test_package_hisilicon_requires_complete_runtime_abi_and_executable_worker(tmp_path):
    _create_bundle(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    model = artifacts / "policy.om"
    worker = artifacts / "worker"
    model.write_bytes(b"om")
    worker.write_text("#!/bin/sh\n", encoding="utf-8")
    worker.chmod(0o755)
    _write_json(
        tmp_path / "policy_abi.json",
        {
            "inputs": [
                {"name": "state", "index": 0, "dtype": "float32", "shape": [1, 6]},
                {"name": "image", "index": 1, "dtype": "float32", "shape": [1, 3, 16, 24]},
            ],
            "outputs": [{"name": "action", "index": 0, "dtype": "float32", "shape": [1, 4, 6]}],
        },
    )
    spec = tmp_path / "hisilicon.json"
    _write_json(
        spec,
        {
            "execution": ["policy"],
            "roles": {
                "policy": {
                    "artifact": "artifacts/policy.om",
                    "format": "om",
                    "abi": "policy_abi.json",
                    "input_semantics": {"state": "observation.state", "image": "observation.images.top"},
                    "output_semantics": {"action": "action"},
                    "image_layouts": {"observation.images.top": "NCHW"},
                }
            },
            "artifacts": {"worker": {"path": "artifacts/worker", "format": "executable"}},
        },
    )

    validated = package_compiled_deployment(
        bundle_root=tmp_path,
        deployment_name="sd3403",
        backend="hisilicon",
        target_soc="sd3403",
        target_runtime="hisilicon-worker",
        spec_path=spec,
    )

    assert validated.deployment.backend == "hisilicon"
    assert set(validated.deployment.artifacts) == {"policy", "worker"}
    assert load_inference_manifest(tmp_path, "sd3403").deployment.target.soc == "sd3403"

    worker.chmod(0o644)
    with pytest.raises(ValueError, match="not executable"):
        package_compiled_deployment(
            bundle_root=tmp_path,
            deployment_name="sd3403",
            backend="hisilicon",
            target_soc="sd3403",
            target_runtime="hisilicon-worker",
            spec_path=spec,
        )


def test_package_compiled_deployment_rejects_backend_target_mismatch(tmp_path):
    _create_bundle(tmp_path)
    artifact = tmp_path / "policy.om"
    artifact.write_bytes(b"om")
    _write_json(
        tmp_path / "policy_abi.json",
        {
            "inputs": [
                {"name": "state", "index": 0, "dtype": "float32", "shape": [1, 6]},
                {
                    "name": "image",
                    "index": 1,
                    "dtype": "float32",
                    "shape": [1, 3, 16, 24],
                    "layout": "NCHW",
                },
            ],
            "outputs": [{"name": "action", "index": 0, "dtype": "float32", "shape": [1, 4, 6]}],
        },
    )
    spec = tmp_path / "package.json"
    _write_json(
        spec,
        {
            "execution": ["policy"],
            "roles": {
                "policy": {
                    "artifact": "policy.om",
                    "format": "om",
                    "abi": "policy_abi.json",
                    "input_semantics": {"state": "observation.state", "image": "observation.images.top"},
                    "output_semantics": {"action": "action"},
                }
            },
        },
    )

    with pytest.raises(ValueError, match="RKNN deployment requires"):
        package_compiled_deployment(
            bundle_root=tmp_path,
            deployment_name="bad",
            backend="rknn",
            target_soc="rk3588",
            target_runtime="acl",
            spec_path=spec,
        )


def test_tensor_binding_type_is_strict():
    with pytest.raises(ValueError):
        TensorBinding(semantic="action", runtime_name="action", index=0, dtype="float32", shape=(1, 0))


def test_read_tcim_abi_uses_model_descriptors(tmp_path):
    metadata = tmp_path / "model.json"
    _write_json(
        metadata,
        {
            "Golden": {"inputs": [], "outputs": []},
            "Model": {
                "inputs": [{"name": "x_t", "shape": [1, 2, 8], "dtype": {"code": "float", "bits": 16}}],
                "outputs": [{"name": "v_t", "shape": [1, 2, 8], "dtype": {"code": "float", "bits": 16}}],
            },
        },
    )

    abi = read_tcim_abi(metadata)

    assert abi.inputs[0].dtype == "float16"
    assert abi.outputs[0].shape == (1, 2, 8)


def test_copy_policy_metadata_bundle_copies_only_required_semantic_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _create_bundle(source)
    (source / "model.safetensors").write_bytes(b"native")
    destination = tmp_path / "compiled"

    copied = copy_policy_metadata_bundle(source, destination)

    assert set(copied) == {"config.json", "policy_preprocessor.json", "policy_postprocessor.json"}
    assert (destination / "config.json").is_file()
    assert not (destination / "model.safetensors").exists()
