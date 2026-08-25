# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""GraspGen packages into a grasp-domain bundle, not a LeRobot policy bundle.

The previous packager wrote ``config.json``, ``policy_preprocessor.json`` and
``policy_postprocessor.json`` so that a non-policy model would survive the policy loader's
LeRobot asset checks, and it wrote them before any of the eight OM ABIs had been read - so
a missing artifact left a bundle that looked packaged and was not. These tests hold the
replacement to the generic runtime shape already used by staged models: the current
compatibility descriptor, the model's own constants in ``assets/adapter.json``, and
nothing written at all until every role has been resolved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import (
    GRASPGEN_BATCH,
    GRASPGEN_DEPLOYMENT,
    GRASPGEN_POINTS,
    GRASPGEN_ROLE_ABI,
    package_graspgen_export,
    runtime_abi_json,
    write_json,
)

from inference_manifest import load_inference_manifest
from model_utils.graspgen_contract import (
    GRASPGEN_CONTRACT_VERSION,
    GRASPGEN_EXECUTION,
    graspgen_geometry,
)
from model_utils.inference_manifest_export import RuntimeABI, RuntimeTensor
from perception_service.graspgen_adapter import GraspGenAdapter

_LEROBOT_ASSETS = ("config.json", "policy_preprocessor.json", "policy_postprocessor.json")


def test_the_bundle_declares_a_perception_model_and_no_lerobot_assets(graspgen_export):
    manifest_path = package_graspgen_export(graspgen_export)

    bundle = graspgen_export.bundle
    assert manifest_path == bundle / "inference_manifest.json"
    validated = load_inference_manifest(bundle, GRASPGEN_DEPLOYMENT)
    assert validated.manifest.model.interface == "tensor_model"
    assert validated.manifest.model.model_type == "graspgen"
    assert validated.manifest.model.operation == "generate_grasps"
    assert validated.policy is None
    for asset in _LEROBOT_ASSETS:
        assert not (bundle / asset).exists()


def test_the_service_contract_is_the_only_thing_the_descriptor_declares(graspgen_bundle):
    """The poses are integrated on the host, so no OM slot may claim to produce them."""
    validated = load_inference_manifest(graspgen_bundle, GRASPGEN_DEPLOYMENT)

    model = validated.manifest.model
    assert [tensor.semantic for tensor in model.inputs] == ["observation.object_points"]
    assert [tensor.semantic for tensor in model.outputs] == ["grasp.poses", "grasp.confidence"]
    head_outputs = {binding.semantic for binding in validated.deployment.bindings["discriminator_head"].outputs}
    assert head_outputs == {"host.graspgen.grasp_logits", "host.graspgen.grasp_scores"}


def test_the_roles_bind_in_the_shared_execution_order_over_two_device_links(graspgen_bundle):
    deployment = load_inference_manifest(graspgen_bundle, GRASPGEN_DEPLOYMENT).deployment

    assert deployment.execution == GRASPGEN_EXECUTION
    assert deployment.target.soc == "Ascend310P3"
    assert deployment.target.runtime == "acl"
    assert {(link.semantic, link.producer, link.consumer) for link in deployment.device_links} == {
        ("internal.graspgen.generator_embedding", "generator_encoder_head", "denoiser"),
        ("internal.graspgen.discriminator_embedding", "discriminator_encoder_head", "discriminator_head"),
    }
    assert deployment.bindings["denoiser"].inputs[0].semantic == "internal.graspgen.generator_embedding"
    assert (
        deployment.bindings["generator_sa1"].outputs[0].runtime_name
        == "PartitionedCall_/ReduceMax_ReduceMax_1:0:features"
    )
    assert deployment.bindings["generator_sa1"].inputs[0].layout == "NCHW"
    assert deployment.artifacts["denoiser"].path == f"artifacts/ascend/{GRASPGEN_DEPLOYMENT}/denoiser.om"


def test_the_model_constants_land_in_the_adapter_asset_the_adapter_reads_back(graspgen_bundle):
    config = GraspGenAdapter.from_bundle(graspgen_bundle).config

    assert config.kappa == 2.02217
    assert config.diffusion_steps == 10
    assert config.grasp_batch_size == GRASPGEN_BATCH
    assert config.point_count == GRASPGEN_POINTS
    assert (config.npoints, config.radii, config.nsamples) == ((256, 64), (0.02, 0.04), (64, 128))

    manifest = load_inference_manifest(graspgen_bundle, GRASPGEN_DEPLOYMENT).manifest
    assert {entry.path for entry in manifest.bundle.files} == {
        "assets/adapter.json",
        "assets/discriminator_checkpoint.pth",
        "assets/generator_checkpoint.pth",
        "assets/graspgen_config.yml",
    }
    assert set(manifest.deployments) == {"torch_cuda", GRASPGEN_DEPLOYMENT}


def test_the_packaged_geometry_drops_the_encoder_heads_null_stage(graspgen_bundle):
    """The ONNX manifest lists one entry per exported stage; the bundle lists sampled ones."""
    assets = json.loads((graspgen_bundle / "assets" / "adapter.json").read_text(encoding="utf-8"))

    assert assets["geometry"] == graspgen_geometry()
    assert None not in assets["geometry"]["npoints"]


def test_repackaging_an_unchanged_export_keeps_the_bundle_and_deployment_revisions(graspgen_export):
    package_graspgen_export(graspgen_export)
    first = load_inference_manifest(graspgen_export.bundle, GRASPGEN_DEPLOYMENT)
    package_graspgen_export(graspgen_export)
    second = load_inference_manifest(graspgen_export.bundle, GRASPGEN_DEPLOYMENT)

    assert second.manifest.bundle.uuid == first.manifest.bundle.uuid
    assert second.manifest.bundle.revision == first.manifest.bundle.revision == 1
    assert second.deployment.uuid == first.deployment.uuid
    assert second.deployment.revision == first.deployment.revision == 1


def test_a_role_whose_abi_disagrees_with_its_semantics_is_rejected(graspgen_export):
    bad_abi = RuntimeABI(
        inputs=(
            RuntimeTensor("object_embedding", 0, "float32", (1, 512)),
            RuntimeTensor("sample", 1, "float32", (GRASPGEN_BATCH, 6)),
        ),
        outputs=(RuntimeTensor("features", 0, "float32", (1, 128, 256)),),
    )
    write_json(graspgen_export.om_abi_dir / "denoiser.om.abi.json", runtime_abi_json(bad_abi))

    with pytest.raises(ValueError, match="runtime ABI has 2 inputs; expected 3"):
        package_graspgen_export(graspgen_export)

    assert list(graspgen_export.bundle.iterdir()) == []


def test_a_missing_role_leaves_the_bundle_exactly_as_it_was(graspgen_export):
    """Resolution runs to completion before the first byte is written."""
    (graspgen_export.om_dir / "discriminator_head.om").unlink()

    with pytest.raises(FileNotFoundError, match="discriminator_head"):
        package_graspgen_export(graspgen_export)

    assert list(graspgen_export.bundle.iterdir()) == []


def test_missing_runtime_abis_are_introspected_from_the_compiled_oms(graspgen_export, monkeypatch):
    for abi_path in graspgen_export.om_abi_dir.iterdir():
        abi_path.unlink()
    calls = []

    def fake_write_acl_om_abi(om_path, output_path, *, device_id, acl_config_path):
        role = Path(om_path).stem
        calls.append((role, device_id, acl_config_path))
        write_json(Path(output_path), runtime_abi_json(GRASPGEN_ROLE_ABI[role]))
        return Path(output_path)

    monkeypatch.setattr(
        "perception_service.package_graspgen_ascend_bundle.write_acl_om_abi",
        fake_write_acl_om_abi,
    )

    package_graspgen_export(graspgen_export, abi_device_id=2, acl_config_path="acl.json")

    assert calls == [(role, 2, "acl.json") for role in GRASPGEN_EXECUTION]


def test_introspection_can_be_refused_so_a_host_without_a_device_fails_loudly(graspgen_export):
    (graspgen_export.om_abi_dir / "generator_sa1.om.abi.json").unlink()

    with pytest.raises(FileNotFoundError, match="generator_sa1"):
        package_graspgen_export(graspgen_export, inspect_missing_abi=False)

    assert list(graspgen_export.bundle.iterdir()) == []


def test_an_export_from_another_contract_is_refused_before_anything_is_read(graspgen_export):
    graspgen_export.onnx_manifest["contract_version"] = GRASPGEN_CONTRACT_VERSION + 1

    with pytest.raises(ValueError, match="contract_version"):
        package_graspgen_export(graspgen_export)

    assert list(graspgen_export.bundle.iterdir()) == []
