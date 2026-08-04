from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_manifest import load_inference_manifest
from model_utils.graspgen_export.write_manifest import (
    GRASPGEN_EXECUTION,
    write_graspgen_ascend_deployment,
)
from model_utils.inference_manifest_export import RuntimeABI, RuntimeTensor


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


_ROLE_ABI: dict[str, RuntimeABI] = {
    "generator_sa1": RuntimeABI(
        inputs=(RuntimeTensor("grouped_features", 0, "float32", (1, 3, 256, 64)),),
        outputs=(RuntimeTensor("PartitionedCall_/ReduceMax_ReduceMax_1:0:features", 0, "float32", (1, 128, 256)),),
    ),
    "generator_sa2": RuntimeABI(
        inputs=(RuntimeTensor("grouped_features", 0, "float32", (1, 131, 64, 128)),),
        outputs=(RuntimeTensor("features", 0, "float32", (1, 256, 64)),),
    ),
    "generator_encoder_head": RuntimeABI(
        inputs=(RuntimeTensor("grouped_features", 0, "float32", (1, 259, 1, 64)),),
        outputs=(
            RuntimeTensor(
                "PartitionedCall_/prediction_head/prediction_head.4/Gemm_add_17:0:object_embedding",
                0,
                "float32",
                (1, 512),
            ),
        ),
    ),
    "discriminator_sa1": RuntimeABI(
        inputs=(RuntimeTensor("grouped_features", 0, "float32", (1, 3, 256, 64)),),
        outputs=(RuntimeTensor("features", 0, "float32", (1, 128, 256)),),
    ),
    "discriminator_sa2": RuntimeABI(
        inputs=(RuntimeTensor("grouped_features", 0, "float32", (1, 131, 64, 128)),),
        outputs=(RuntimeTensor("features", 0, "float32", (1, 256, 64)),),
    ),
    "discriminator_encoder_head": RuntimeABI(
        inputs=(RuntimeTensor("grouped_features", 0, "float32", (1, 259, 1, 64)),),
        outputs=(RuntimeTensor("object_embedding", 0, "float32", (1, 512)),),
    ),
    "denoiser": RuntimeABI(
        inputs=(
            RuntimeTensor("object_embedding", 0, "float32", (1, 512)),
            RuntimeTensor("sample", 1, "float32", (1000, 6)),
            RuntimeTensor("timestep", 2, "float32", (1,)),
        ),
        outputs=(
            RuntimeTensor(
                "PartitionedCall_/prediction_head/prediction_head.4/Gemm_add_130:0:predicted_noise",
                0,
                "float32",
                (1000, 6),
            ),
        ),
    ),
    "discriminator_head": RuntimeABI(
        inputs=(
            RuntimeTensor("object_embedding", 0, "float32", (1, 512)),
            RuntimeTensor("grasp_rt", 1, "float32", (1000, 6)),
        ),
        outputs=(
            RuntimeTensor(
                "PartitionedCall_/prediction_head/prediction_head.4/Gemm_add_24:0:logits",
                0,
                "float32",
                (1000, 1),
            ),
            RuntimeTensor("/Sigmoid:0:confidence", 1, "float32", (1000, 1)),
        ),
    ),
}


def _runtime_abi_json(abi: RuntimeABI) -> dict:
    def tensors(values):
        return [
            {
                "name": tensor.name,
                "index": tensor.index,
                "dtype": tensor.dtype,
                "shape": list(tensor.shape),
            }
            for tensor in values
        ]

    return {"inputs": tensors(abi.inputs), "outputs": tensors(abi.outputs)}


def _seed_bundle(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    onnx_dir = tmp_path / "onnx"
    om_dir = tmp_path / "om"
    om_abi_dir = tmp_path / "om_abi"
    onnx_dir.mkdir()
    om_dir.mkdir()
    om_abi_dir.mkdir()
    for role in GRASPGEN_EXECUTION:
        (onnx_dir / f"{role}.onnx").write_bytes(f"{role}-onnx".encode())
        (om_dir / f"{role}.om").write_bytes(f"{role}-om".encode())
        _write_json(om_abi_dir / f"{role}.om.abi.json", _runtime_abi_json(_ROLE_ABI[role]))
    onnx_manifest = onnx_dir / "graspgen.onnx.json"
    _write_json(
        onnx_manifest,
        {
            "schema_version": 1,
            "model_type": "graspgen",
            "backend": "onnx",
            "artifacts": {role: {"onnx": f"{role}.onnx"} for role in GRASPGEN_EXECUTION},
            "execution": list(GRASPGEN_EXECUTION),
            "backend_config": {
                "grasp_batch_size": 1000,
                "point_count": 2048,
                "grasp_repr": "r3_so3",
                "kappa": 2.02217,
                "diffusion_steps": 10,
                "compositional_scheduler": True,
                "geometry": {
                    "npoints": [256, 64],
                    "radii": [0.02, 0.04],
                    "nsamples": [64, 128],
                    "fps_start_index": 0,
                    "ball_query_order": "input_index",
                },
            },
        },
    )
    return bundle, onnx_manifest, om_dir, om_abi_dir


def test_write_graspgen_ascend_deployment_produces_unified_manifest(tmp_path):
    bundle, onnx_manifest, om_dir, om_abi_dir = _seed_bundle(tmp_path)

    manifest_path = write_graspgen_ascend_deployment(
        bundle,
        deployment_name="ascend",
        om_dir=om_dir,
        om_abi_dir=om_abi_dir,
        soc_version="Ascend310P3",
        onnx_manifest=json.loads(onnx_manifest.read_text(encoding="utf-8")),
        grasp_batch_size=1000,
        point_count=2048,
    )

    assert manifest_path == bundle / "inference_manifest.json"
    validated = load_inference_manifest(bundle, "ascend")
    assert validated.policy.policy_type == "graspgen"
    assert validated.deployment.execution == GRASPGEN_EXECUTION
    assert validated.deployment.target.soc == "Ascend310P3"
    assert validated.deployment.target.runtime == "acl"
    assert {(link.semantic, link.producer, link.consumer) for link in validated.deployment.device_links} == {
        (
            "internal.graspgen.generator_embedding",
            "generator_encoder_head",
            "denoiser",
        ),
        (
            "internal.graspgen.discriminator_embedding",
            "discriminator_encoder_head",
            "discriminator_head",
        ),
    }
    assert validated.deployment.bindings["generator_encoder_head"].outputs[0].semantic == (
        "internal.graspgen.generator_embedding"
    )
    assert validated.deployment.bindings["denoiser"].inputs[0].semantic == ("internal.graspgen.generator_embedding")
    head_outputs = {b.semantic for b in validated.deployment.bindings["discriminator_head"].outputs}
    assert head_outputs == {"grasp.poses", "grasp.confidence"}
    assert (
        validated.deployment.bindings["generator_sa1"].outputs[0].runtime_name
        == "PartitionedCall_/ReduceMax_ReduceMax_1:0:features"
    )
    assert validated.deployment.bindings["discriminator_head"].outputs[1].shape == (1000, 1)
    config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    assert config["type"] == "graspgen"
    assert config["kappa"] == 2.02217
    assert config["geometry"]["npoints"] == [256, 64]


def test_write_graspgen_ascend_deployment_rejects_wrong_runtime_tensor_count(tmp_path):
    bundle, onnx_manifest, om_dir, om_abi_dir = _seed_bundle(tmp_path)
    bad_abi = RuntimeABI(
        inputs=(
            RuntimeTensor("object_embedding", 0, "float32", (1, 512)),
            RuntimeTensor("sample", 1, "float32", (1000, 6)),
        ),
        outputs=(RuntimeTensor("features", 0, "float32", (1, 128, 256)),),
    )
    _write_json(om_abi_dir / "denoiser.om.abi.json", _runtime_abi_json(bad_abi))

    with pytest.raises(ValueError, match="runtime ABI has 2 inputs; expected 3"):
        write_graspgen_ascend_deployment(
            bundle,
            deployment_name="ascend",
            om_dir=om_dir,
            om_abi_dir=om_abi_dir,
            soc_version="Ascend310P3",
            onnx_manifest=json.loads(onnx_manifest.read_text(encoding="utf-8")),
            grasp_batch_size=1000,
            point_count=2048,
        )


def test_write_graspgen_ascend_deployment_inspects_missing_runtime_abis(tmp_path, monkeypatch):
    bundle, onnx_manifest, om_dir, om_abi_dir = _seed_bundle(tmp_path)
    for abi_path in om_abi_dir.iterdir():
        abi_path.unlink()
    calls = []

    def fake_write_acl_om_abi(om_path, output_path, *, device_id, acl_config_path):
        role = Path(om_path).stem
        calls.append((role, device_id, acl_config_path))
        _write_json(Path(output_path), _runtime_abi_json(_ROLE_ABI[role]))
        return Path(output_path)

    monkeypatch.setattr(
        "model_utils.graspgen_export.write_manifest.write_acl_om_abi",
        fake_write_acl_om_abi,
    )

    write_graspgen_ascend_deployment(
        bundle,
        deployment_name="ascend",
        om_dir=om_dir,
        om_abi_dir=om_abi_dir,
        soc_version="Ascend310P3",
        onnx_manifest=json.loads(onnx_manifest.read_text(encoding="utf-8")),
        grasp_batch_size=1000,
        point_count=2048,
        abi_device_id=2,
        acl_config_path="acl.json",
    )

    assert calls == [(role, 2, "acl.json") for role in GRASPGEN_EXECUTION]


def test_write_graspgen_ascend_deployment_can_require_existing_runtime_abis(tmp_path):
    bundle, onnx_manifest, om_dir, om_abi_dir = _seed_bundle(tmp_path)
    (om_abi_dir / "generator_sa1.om.abi.json").unlink()

    with pytest.raises(FileNotFoundError, match="generator_sa1"):
        write_graspgen_ascend_deployment(
            bundle,
            deployment_name="ascend",
            om_dir=om_dir,
            om_abi_dir=om_abi_dir,
            soc_version="Ascend310P3",
            onnx_manifest=json.loads(onnx_manifest.read_text(encoding="utf-8")),
            grasp_batch_size=1000,
            point_count=2048,
            inspect_missing_abi=False,
        )
