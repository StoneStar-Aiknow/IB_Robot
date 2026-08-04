from __future__ import annotations

import builtins
import copy
import json
import os
import subprocess
import sys

import pytest

from inference_manifest import (
    BundleFile,
    CompiledDeployment,
    ManifestIntegrityError,
    ManifestPathError,
    ManifestValidationError,
    SemanticIdentity,
    canonical_bundle_digest,
    canonical_manifest_bytes,
    canonical_semantic_identity_json,
    deployment_fingerprint,
    load_inference_manifest,
    normalize_bundle_path,
    normalize_unique_paths,
    semantic_identity_fingerprint,
    write_inference_manifest,
)
from inference_manifest.models import DeviceLink
from tests.manifest_fixtures import (
    TEST_BUNDLE_UUID,
    create_non_policy_bundle,
    create_policy_bundle,
    make_manifest,
    make_non_policy_manifest,
    write_manifest,
)


@pytest.mark.parametrize(
    "path",
    [
        "/absolute/file",
        "../outside",
        "folder/../outside",
        "./config.json",
        "folder//file",
        "folder\\file",
        "C:/models/file",
    ],
)
def test_bundle_paths_reject_unsafe_forms(path):
    with pytest.raises(ManifestPathError):
        normalize_bundle_path(path)


def test_duplicate_normalized_paths_are_rejected():
    with pytest.raises(ManifestPathError, match="duplicate normalized path 'config.json'"):
        normalize_unique_paths(["config.json", "config.json"], "bundle.files")


def test_symlink_escape_and_broken_link_are_rejected(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (bundle / "escape.bin").symlink_to(outside)
    (bundle / "broken.bin").symlink_to(tmp_path / "missing.bin")

    from inference_manifest import resolve_bundle_file

    with pytest.raises(ManifestPathError, match="escapes the bundle root"):
        resolve_bundle_file(bundle, "escape.bin")
    with pytest.raises(ManifestPathError, match="broken symlink"):
        resolve_bundle_file(bundle, "broken.bin")


def test_schema_version_unknown_fields_aliases_and_duplicate_json_keys(tmp_path):
    paths = create_policy_bundle(tmp_path)
    manifest = make_manifest(tmp_path, paths)

    manifest["schema_version"] = 3
    write_manifest(tmp_path, manifest)
    with pytest.raises(ManifestValidationError, match=r"Unsupported schema_version 3.*inference_manifest.json"):
        load_inference_manifest(tmp_path, "cpu")

    manifest = make_manifest(tmp_path, paths)
    manifest["unexpected"] = True
    write_manifest(tmp_path, manifest)
    with pytest.raises(ManifestValidationError, match=r"schema validation failed.*unexpected"):
        load_inference_manifest(tmp_path, "cpu")

    manifest = make_manifest(tmp_path, paths)
    manifest["deployments"]["cpu"] = {"backend": "ascend_om", "device": "cpu"}
    write_manifest(tmp_path, manifest)
    with pytest.raises(ManifestValidationError, match="ascend_om"):
        load_inference_manifest(tmp_path, "cpu")

    (tmp_path / "inference_manifest.json").write_text(
        '{"schema_version":2,"schema_version":2}',
        encoding="utf-8",
    )
    with pytest.raises(ManifestValidationError, match="duplicate JSON key 'schema_version'"):
        load_inference_manifest(tmp_path, "cpu")


def test_schema_v1_requires_regeneration(tmp_path):
    paths = tuple(path for path in create_policy_bundle(tmp_path) if path != "model.safetensors")
    manifest = make_manifest(tmp_path, paths, deployment_name="rk3588", compiled=True)
    manifest["schema_version"] = 1
    manifest["bundle"].pop("uuid")
    manifest["bundle"].pop("revision")
    manifest["bundle"]["digest"].pop("scope")
    for entry in manifest["bundle"]["files"]:
        entry["sha256"] = "1" * 64
    deployment = manifest["deployments"]["rk3588"]
    deployment.pop("uuid")
    deployment.pop("revision")
    for artifact in deployment["artifacts"].values():
        artifact["sha256"] = "2" * 64
    write_manifest(tmp_path, manifest)

    with pytest.raises(ManifestValidationError, match="unsupported.*rerun the owning exporter or packager"):
        load_inference_manifest(tmp_path, "rk3588")


def test_missing_manifest_fails_before_bundle_metadata_loading(tmp_path):
    with pytest.raises(ManifestValidationError, match=r"Unable to read JSON file.*inference_manifest.json"):
        load_inference_manifest(tmp_path, "cpu")


def test_legacy_and_explicit_policy_manifests_retain_strict_lerobot_validation(tmp_path):
    paths = create_policy_bundle(tmp_path)
    legacy = make_manifest(tmp_path, paths)
    write_manifest(tmp_path, legacy)

    legacy_validated = load_inference_manifest(tmp_path, "cpu")
    assert legacy_validated.manifest.model.kind == "policy"
    assert legacy_validated.manifest.model.family == "lerobot"
    assert legacy_validated.policy is not None

    explicit = copy.deepcopy(legacy)
    explicit["model"] = {"kind": "policy", "family": "act", "inputs": [], "outputs": []}
    write_manifest(tmp_path, explicit)
    explicit_validated = load_inference_manifest(tmp_path, "cpu")
    assert explicit_validated.manifest.model.family == "act"
    assert explicit_validated.policy.policy_type == "act"
    assert explicit_validated.fingerprint == legacy_validated.fingerprint

    mismatched = copy.deepcopy(explicit)
    mismatched["model"]["family"] = "smolvla"
    write_manifest(tmp_path, mismatched)
    with pytest.raises(ManifestValidationError, match=r"family 'smolvla'.*policy type 'act'"):
        load_inference_manifest(tmp_path, "cpu")

    write_manifest(tmp_path, explicit)
    (tmp_path / "policy_preprocessor.json").unlink()
    with pytest.raises(ManifestPathError, match="policy_preprocessor.json"):
        load_inference_manifest(tmp_path, "cpu")


def test_unknown_model_kind_is_rejected(tmp_path):
    paths = create_non_policy_bundle(tmp_path)
    manifest = make_non_policy_manifest(tmp_path, paths)
    manifest["model"]["kind"] = "detector"
    write_manifest(tmp_path, manifest)

    with pytest.raises(ManifestValidationError, match=r"model.kind.*detector"):
        load_inference_manifest(tmp_path, "ascend")


def _semantic_identity():
    return {
        "logical_model_revision": "google/siglip2-base-patch16-224@v1",
        "preprocessing_contract": "siglip2-dual-encoder-v1",
        "output_semantics": "normalized-joint-image-text-embedding-v1",
        "embedding": {
            "embedding_space_id": "siglip2-base-patch16-224:v1",
            "dimension": 768,
            "normalization": "l2",
            "image_preprocessing": "siglip2-image-224-v1",
            "text_preprocessing": "siglip2-tokenizer-64-v1",
        },
    }


def test_perception_manifest_accepts_complete_semantic_embedding_identity(tmp_path):
    paths = create_non_policy_bundle(tmp_path)
    manifest = make_non_policy_manifest(tmp_path, paths)
    manifest["model"]["semantic_identity"] = _semantic_identity()
    write_manifest(tmp_path, manifest)

    identity = load_inference_manifest(tmp_path, "ascend").manifest.model.semantic_identity

    assert identity is not None
    assert identity.embedding is not None
    assert identity.embedding.embedding_space_id == "siglip2-base-patch16-224:v1"
    assert identity.embedding.dimension == 768


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda embedding: embedding.pop("text_preprocessing"), "text_preprocessing"),
        (lambda embedding: embedding.update(dimension=0), "dimension"),
        (lambda embedding: embedding.update(extra="invalid"), "extra"),
    ],
)
def test_semantic_embedding_identity_is_strict_and_complete(tmp_path, mutation, expected):
    paths = create_non_policy_bundle(tmp_path)
    manifest = make_non_policy_manifest(tmp_path, paths)
    identity = _semantic_identity()
    mutation(identity["embedding"])
    manifest["model"]["semantic_identity"] = identity
    write_manifest(tmp_path, manifest)

    with pytest.raises(ManifestValidationError, match=expected):
        load_inference_manifest(tmp_path, "ascend")


def test_semantic_identity_json_and_fingerprint_are_canonical_and_provenance_independent():
    identity = SemanticIdentity.model_validate(_semantic_identity())
    reordered = SemanticIdentity.model_validate(
        {
            "output_semantics": identity.output_semantics,
            "embedding": identity.embedding.model_dump(),
            "preprocessing_contract": identity.preprocessing_contract,
            "logical_model_revision": identity.logical_model_revision,
        }
    )

    expected_json = (
        '{"embedding":{"dimension":768,"embedding_space_id":"siglip2-base-patch16-224:v1",'
        '"image_preprocessing":"siglip2-image-224-v1","normalization":"l2",'
        '"text_preprocessing":"siglip2-tokenizer-64-v1"},'
        '"logical_model_revision":"google/siglip2-base-patch16-224@v1",'
        '"output_semantics":"normalized-joint-image-text-embedding-v1",'
        '"preprocessing_contract":"siglip2-dual-encoder-v1"}'
    )
    assert canonical_semantic_identity_json(identity) == expected_json
    assert canonical_semantic_identity_json(reordered) == expected_json
    assert semantic_identity_fingerprint(identity) == semantic_identity_fingerprint(reordered)
    assert len(semantic_identity_fingerprint(identity)) == 64


@pytest.mark.parametrize("output_semantic", ["tag_logits", "image_embedding", "masks", "boxes"])
def test_compiled_non_policy_bundle_accepts_declared_semantic_outputs(tmp_path, output_semantic):
    paths = create_non_policy_bundle(tmp_path)
    manifest = make_non_policy_manifest(tmp_path, paths, output_semantic=output_semantic)
    write_manifest(tmp_path, manifest)

    validated = load_inference_manifest(tmp_path, "ascend")

    assert validated.policy is None
    assert validated.manifest.model.kind == "perception"
    assert validated.manifest.model.family == "ram_plus"
    assert validated.manifest.model.outputs[0].semantic == output_semantic
    assert validated.deployment.bindings["model"].outputs[0].semantic == output_semantic
    assert not (tmp_path / "config.json").exists()
    assert not (tmp_path / "policy_preprocessor.json").exists()
    assert not (tmp_path / "policy_postprocessor.json").exists()
    assert not (tmp_path / "model.safetensors").exists()


def test_torch_non_policy_bundle_does_not_require_lerobot_files(tmp_path):
    paths = create_non_policy_bundle(tmp_path)
    manifest = make_non_policy_manifest(tmp_path, paths, deployment_name="cpu")
    manifest["deployments"]["cpu"] = {
        "uuid": manifest["deployments"]["cpu"]["uuid"],
        "revision": 1,
        "backend": "torch",
        "device": "cpu",
    }
    write_manifest(tmp_path, manifest)

    validated = load_inference_manifest(tmp_path, "cpu")

    assert validated.deployment.backend == "torch"
    assert validated.policy is None


def test_compiled_policy_bundle_still_requires_action_output(tmp_path):
    paths = tuple(path for path in create_policy_bundle(tmp_path) if path != "model.safetensors")
    manifest = make_manifest(tmp_path, paths, deployment_name="rk3588", compiled=True)
    manifest["deployments"]["rk3588"]["bindings"]["policy"]["outputs"][0]["semantic"] = "tag_logits"
    write_manifest(tmp_path, manifest)

    with pytest.raises(ManifestValidationError, match="policy deployment must declare an action output"):
        load_inference_manifest(tmp_path, "rk3588")


@pytest.mark.parametrize(
    ("direction", "field", "value", "message"),
    [
        ("outputs", "semantic", "scores", "declared semantic output bindings"),
        ("outputs", "dtype", "float16", "tag_logits.*dtype expected float32, actual float16"),
        ("outputs", "shape", [1, 1000], "tag_logits.*shape expected.*4585.*actual.*1000"),
        ("inputs", "layout", "NHWC", "observation.image.*layout expected NCHW, actual NHWC"),
    ],
)
def test_non_policy_binding_must_match_declared_semantic_contract(tmp_path, direction, field, value, message):
    paths = create_non_policy_bundle(tmp_path)
    manifest = make_non_policy_manifest(tmp_path, paths)
    manifest["deployments"]["ascend"]["bindings"]["model"][direction][0][field] = value
    write_manifest(tmp_path, manifest)

    with pytest.raises(ManifestValidationError, match=message):
        load_inference_manifest(tmp_path, "ascend")


def test_non_policy_manifest_requires_complete_semantic_contract(tmp_path):
    paths = create_non_policy_bundle(tmp_path)
    manifest = make_non_policy_manifest(tmp_path, paths)
    manifest["model"]["outputs"] = []
    write_manifest(tmp_path, manifest)

    with pytest.raises(ManifestValidationError, match=r"model.outputs.*should be non-empty"):
        load_inference_manifest(tmp_path, "ascend")


def test_non_policy_bundle_integrity_artifacts_and_paths_are_validated(tmp_path):
    paths = create_non_policy_bundle(tmp_path)
    manifest = make_non_policy_manifest(tmp_path, paths)

    invalid_digest = copy.deepcopy(manifest)
    invalid_digest["bundle"]["digest"]["value"] = "0" * 64
    write_manifest(tmp_path, invalid_digest)
    with pytest.raises(ManifestIntegrityError, match="Bundle digest mismatch"):
        load_inference_manifest(tmp_path, "ascend")

    missing_artifact = copy.deepcopy(manifest)
    (tmp_path / "artifacts" / "ram_plus.om").unlink()
    write_manifest(tmp_path, missing_artifact)
    with pytest.raises(ManifestPathError, match="does not exist"):
        load_inference_manifest(tmp_path, "ascend")

    escaped_artifact = copy.deepcopy(manifest)
    escaped_artifact["deployments"]["ascend"]["artifacts"]["model"]["path"] = "../ram_plus.om"
    write_manifest(tmp_path, escaped_artifact)
    with pytest.raises(ManifestValidationError, match="parent traversal"):
        load_inference_manifest(tmp_path, "ascend")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("uuid", "not-a-uuid"),
        ("uuid", "00000000-0000-0000-0000-000000000000"),
        ("revision", 0),
        ("revision", True),
        ("revision", "1"),
    ],
)
def test_bundle_identity_requires_canonical_uuid_and_positive_integer_revision(tmp_path, field, value):
    paths = create_policy_bundle(tmp_path)
    manifest = make_manifest(tmp_path, paths)
    manifest["bundle"][field] = value
    write_manifest(tmp_path, manifest)

    with pytest.raises(ManifestValidationError):
        load_inference_manifest(tmp_path, "cpu")


@pytest.mark.parametrize("alias", ["ascend_om", "ascend_om_3403", "3403", "om"])
def test_removed_backend_aliases_are_rejected(tmp_path, alias):
    paths = create_policy_bundle(tmp_path)
    manifest = make_manifest(tmp_path, paths)
    manifest["deployments"]["cpu"] = {"backend": alias, "device": "cpu"}
    write_manifest(tmp_path, manifest)

    with pytest.raises(ManifestValidationError, match=alias):
        load_inference_manifest(tmp_path, "cpu")


def test_bundle_content_changes_do_not_require_runtime_rehashing(tmp_path):
    paths = create_policy_bundle(tmp_path)
    manifest = make_manifest(tmp_path, paths)
    write_manifest(tmp_path, manifest)
    before = load_inference_manifest(tmp_path, "cpu")
    config = tmp_path / "config.json"
    config.write_text(config.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    after = load_inference_manifest(tmp_path, "cpu")
    assert after.fingerprint == before.fingerprint


def test_artifact_content_changes_are_left_to_backend_validation(tmp_path):
    paths = tuple(path for path in create_policy_bundle(tmp_path) if path != "model.safetensors")
    manifest = make_manifest(tmp_path, paths, deployment_name="rk3588", compiled=True)
    write_manifest(tmp_path, manifest)
    artifact = manifest["deployments"]["rk3588"]["artifacts"]["policy"]
    before = load_inference_manifest(tmp_path, "rk3588")
    (tmp_path / artifact["path"]).write_bytes(b"changed-compiled-policy")
    after = load_inference_manifest(tmp_path, "rk3588")
    assert after.fingerprint == before.fingerprint


def test_canonical_bundle_digest_is_order_independent():
    entries = [
        BundleFile(path="z.json"),
        BundleFile(path="a.json"),
    ]

    assert canonical_bundle_digest(TEST_BUNDLE_UUID, 1, "test", entries) == canonical_bundle_digest(
        TEST_BUNDLE_UUID, 1, "test", reversed(entries)
    )
    assert canonical_bundle_digest(TEST_BUNDLE_UUID, 1, "test", entries) != canonical_bundle_digest(
        TEST_BUNDLE_UUID, 2, "test", entries
    )


def test_deployment_fingerprint_is_stable_and_tracks_identity_changes():
    deployment = CompiledDeployment.model_validate_json(
        json.dumps(
            {
                "backend": "rknn",
                "uuid": "123e4567-e89b-42d3-a456-426614174001",
                "revision": 1,
                "target": {"soc": "rk3588", "runtime": "rknn-lite"},
                "artifacts": {"policy": {"path": "policy.rknn", "format": "rknn"}},
                "execution": ["policy"],
                "bindings": {
                    "policy": {
                        "inputs": [
                            {
                                "semantic": "observation.state",
                                "runtime_name": "state",
                                "dtype": "float32",
                                "shape": [1, 6],
                            }
                        ],
                        "outputs": [
                            {
                                "semantic": "action",
                                "runtime_name": "actions",
                                "dtype": "float32",
                                "shape": [1, -1, 6],
                            }
                        ],
                    }
                },
            }
        )
    )

    baseline = deployment_fingerprint(2, "4" * 64, "rk3588", deployment)
    assert baseline == deployment_fingerprint(2, "4" * 64, "rk3588", deployment)
    assert baseline != deployment_fingerprint(2, "5" * 64, "rk3588", deployment)
    assert baseline != deployment_fingerprint(2, "4" * 64, "other", deployment)

    changed = deployment.model_copy(update={"target": deployment.target.model_copy(update={"runtime": "rknn-lite-2"})})
    assert baseline != deployment_fingerprint(2, "4" * 64, "rk3588", changed)
    revised = deployment.model_copy(update={"revision": 2})
    assert baseline != deployment_fingerprint(2, "4" * 64, "rk3588", revised)


def test_device_link_source_defaults_preserve_canonical_identity():
    implicit = DeviceLink(
        semantic="internal.cache",
        producer="prefill",
        consumer="decode",
        transport="device_pointer",
        owner="producer",
    )
    explicit = implicit.model_copy(update={"producer_binding": "output"})
    input_sourced = implicit.model_copy(update={"producer_binding": "input"})

    assert implicit.producer_binding == "output"
    assert implicit.model_dump(mode="json", exclude_defaults=True) == explicit.model_dump(
        mode="json", exclude_defaults=True
    )
    assert "producer_binding" not in implicit.model_dump(mode="json", exclude_defaults=True)
    assert input_sourced.model_dump(mode="json", exclude_defaults=True)["producer_binding"] == "input"

    with pytest.raises(ValueError, match="owner='producer'"):
        DeviceLink(
            semantic="internal.cache",
            producer="prefill",
            consumer="decode",
            producer_binding="input",
            transport="device_pointer",
            owner="consumer",
        )


def test_missing_deployment_reports_available_names(tmp_path):
    paths = create_policy_bundle(tmp_path)
    write_manifest(tmp_path, make_manifest(tmp_path, paths))

    with pytest.raises(ManifestValidationError, match=r"'missing'.*available deployments: \['cpu'\]"):
        load_inference_manifest(tmp_path, "missing")


def test_compiled_execution_bindings_and_feature_compatibility(tmp_path):
    paths = tuple(path for path in create_policy_bundle(tmp_path) if path != "model.safetensors")
    manifest = make_manifest(tmp_path, paths, deployment_name="rk3588", compiled=True)
    write_manifest(tmp_path, manifest)

    validated = load_inference_manifest(tmp_path, "rk3588")
    assert validated.deployment.backend == "rknn"
    assert validated.policy.policy_type == "act"

    invalid_roles = copy.deepcopy(manifest)
    invalid_roles["deployments"]["rk3588"]["execution"] = ["missing"]
    write_manifest(tmp_path, invalid_roles)
    with pytest.raises(ManifestValidationError, match="artifact roles must contain every execution role"):
        load_inference_manifest(tmp_path, "rk3588")

    invalid_layout = copy.deepcopy(manifest)
    del invalid_layout["deployments"]["rk3588"]["bindings"]["policy"]["inputs"][1]["layout"]
    write_manifest(tmp_path, invalid_layout)
    with pytest.raises(ManifestValidationError, match=r"layout.*required"):
        load_inference_manifest(tmp_path, "rk3588")

    non_image_layout = copy.deepcopy(manifest)
    non_image_layout["deployments"]["rk3588"]["bindings"]["policy"]["inputs"][0]["layout"] = "NCHW"
    write_manifest(tmp_path, non_image_layout)
    with pytest.raises(ManifestValidationError, match=r"not be valid"):
        load_inference_manifest(tmp_path, "rk3588")

    invalid_shape = copy.deepcopy(manifest)
    invalid_shape["deployments"]["rk3588"]["bindings"]["policy"]["inputs"][0]["shape"] = [1, 5]
    write_manifest(tmp_path, invalid_shape)
    with pytest.raises(ManifestValidationError, match=r"observation.state.*incompatible"):
        load_inference_manifest(tmp_path, "rk3588")


def test_compiled_deployment_allows_verified_auxiliary_artifacts_and_sparse_output_indices(tmp_path):
    paths = tuple(path for path in create_policy_bundle(tmp_path) if path != "model.safetensors")
    manifest = make_manifest(tmp_path, paths, deployment_name="hisilicon", compiled=True, backend="hisilicon")
    deployment = manifest["deployments"]["hisilicon"]
    deployment["target"] = {"soc": "sd3403", "runtime": "hisilicon-worker"}
    deployment["artifacts"]["policy"]["format"] = "om"
    deployment["bindings"]["policy"]["outputs"][0]["index"] = 1

    worker_path = tmp_path / "artifacts" / "worker"
    worker_path.write_bytes(b"worker")
    deployment["artifacts"]["worker"] = {
        "path": "artifacts/worker",
        "format": "executable",
    }
    write_manifest(tmp_path, manifest)

    validated = load_inference_manifest(tmp_path, "hisilicon")

    assert validated.deployment.execution == ("policy",)
    assert set(validated.deployment.artifacts) == {"policy", "worker"}
    assert validated.deployment.bindings["policy"].outputs[0].index == 1

    worker_path.write_bytes(b"changed-worker")
    assert load_inference_manifest(tmp_path, "hisilicon").deployment.backend == "hisilicon"


@pytest.mark.parametrize("policy_type", ["pi05", "smolvla"])
def test_vla_bindings_allow_declared_state_and_action_padding(tmp_path, policy_type):
    paths = tuple(
        path for path in create_policy_bundle(tmp_path, policy_type=policy_type) if path != "model.safetensors"
    )
    manifest = make_manifest(tmp_path, paths, deployment_name="rk3588", compiled=True)
    bindings = manifest["deployments"]["rk3588"]["bindings"]["policy"]
    bindings["inputs"][0]["shape"] = [1, 32]
    bindings["outputs"][0]["shape"] = [1, 4, 32]
    write_manifest(tmp_path, manifest)

    validated = load_inference_manifest(tmp_path, "rk3588")

    assert validated.policy.policy_type == policy_type


def test_graspgen_compiled_deployment_accepts_grasp_outputs_without_action(tmp_path):
    paths = tuple(
        path for path in create_policy_bundle(tmp_path, policy_type="graspgen") if path != "model.safetensors"
    )
    manifest = make_manifest(
        tmp_path,
        paths,
        deployment_name="ascend",
        compiled=True,
        backend="ascend",
        policy_type="graspgen",
    )
    deployment = manifest["deployments"]["ascend"]
    deployment["target"] = {"soc": "Ascend310P3", "runtime": "acl"}
    deployment["artifacts"]["policy"]["format"] = "om"
    write_manifest(tmp_path, manifest)

    validated = load_inference_manifest(tmp_path, "ascend")

    assert validated.policy.policy_type == "graspgen"
    assert "grasp.poses" in validated.policy.output_features
    assert "grasp.confidence" in validated.policy.output_features
    outputs = {b.semantic for b in validated.deployment.bindings["policy"].outputs}
    assert outputs == {"grasp.poses", "grasp.confidence"}
    assert validated.deployment.execution == ("policy",)


@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
def test_vla_visual_bindings_allow_compiled_resize_dimensions(tmp_path, layout):
    paths = tuple(path for path in create_policy_bundle(tmp_path, policy_type="smolvla") if path != "model.safetensors")
    manifest = make_manifest(tmp_path, paths, deployment_name="rk3588", compiled=True)
    image_binding = manifest["deployments"]["rk3588"]["bindings"]["policy"]["inputs"][1]
    image_binding["layout"] = layout
    image_binding["shape"] = [1, 3, 512, 512] if layout == "NCHW" else [1, 512, 512, 3]
    write_manifest(tmp_path, manifest)

    validated = load_inference_manifest(tmp_path, "rk3588")

    assert validated.policy.policy_type == "smolvla"


def test_rank_four_layout_validation_is_independent_of_semantic_prefix(tmp_path):
    paths = tuple(path for path in create_policy_bundle(tmp_path) if path != "model.safetensors")
    manifest = make_manifest(tmp_path, paths, deployment_name="rk3588", compiled=True)
    image_binding = manifest["deployments"]["rk3588"]["bindings"]["policy"]["inputs"][1]
    image_binding["semantic"] = "depth.frame"
    write_manifest(tmp_path, manifest)

    validated = load_inference_manifest(tmp_path, "rk3588")
    assert validated.deployment.bindings["policy"].inputs[1].layout == "NCHW"

    del image_binding["layout"]
    write_manifest(tmp_path, manifest)
    with pytest.raises(ManifestValidationError, match=r"layout.*required"):
        load_inference_manifest(tmp_path, "rk3588")


def test_selected_bundle_and_artifact_paths_cannot_duplicate(tmp_path):
    paths = tuple(path for path in create_policy_bundle(tmp_path) if path != "model.safetensors")
    manifest = make_manifest(tmp_path, paths, deployment_name="rk3588", compiled=True)
    artifact = manifest["deployments"]["rk3588"]["artifacts"]["policy"]
    manifest["bundle"]["files"].append({"path": artifact["path"]})
    entries = [BundleFile.model_validate(entry) for entry in manifest["bundle"]["files"]]
    manifest["bundle"]["digest"]["value"] = canonical_bundle_digest(
        manifest["bundle"]["uuid"],
        manifest["bundle"]["revision"],
        manifest["bundle"]["name"],
        entries,
    )
    write_manifest(tmp_path, manifest)

    with pytest.raises(ManifestValidationError, match="distinct paths"):
        load_inference_manifest(tmp_path, "rk3588")


def test_canonical_writer_is_deterministic_and_replaces_atomically(tmp_path):
    paths = create_policy_bundle(tmp_path)
    manifest = make_manifest(tmp_path, paths)
    destination = tmp_path / "nested" / "inference_manifest.json"

    first = write_inference_manifest(destination, manifest)
    first_content = first.read_bytes()
    assert first_content == canonical_manifest_bytes(manifest)

    reordered = {"deployments": manifest["deployments"], "bundle": manifest["bundle"], "schema_version": 2}
    write_inference_manifest(destination, reordered)
    assert destination.read_bytes() == first_content
    assert not list(destination.parent.glob("*.tmp"))

    unsafe = copy.deepcopy(manifest)
    unsafe["bundle"]["files"][0]["path"] = "../config.json"
    with pytest.raises(ManifestValidationError, match="parent traversal"):
        write_inference_manifest(destination, unsafe)


def test_manifest_package_and_loading_are_dependency_neutral(monkeypatch, tmp_path):
    forbidden_roots = {
        "acl",
        "geometry_msgs",
        "inference_service",
        "model_utils",
        "rclpy",
        "rknn",
        "rknnlite",
        "robot_config",
        "sensor_msgs",
        "std_msgs",
        "tcim",
        "torch",
        "torch_npu",
    }
    code = f"""
import sys
import inference_manifest

forbidden = {forbidden_roots!r}
loaded = sorted(name for name in sys.modules if name.split('.', 1)[0] in forbidden)
if loaded:
    raise SystemExit(f"forbidden imports loaded by inference_manifest: {{loaded}}")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        env=os.environ.copy(),
        text=True,
    )
    assert result.returncode == 0, result.stderr

    paths = tuple(path for path in create_policy_bundle(tmp_path) if path != "model.safetensors")
    write_manifest(tmp_path, make_manifest(tmp_path, paths, deployment_name="rk3588", compiled=True))
    attempted: list[str] = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", maxsplit=1)[0] in forbidden_roots:
            attempted.append(name)
            raise AssertionError(f"forbidden dependency import attempted: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    load_inference_manifest(tmp_path, "rk3588")

    assert attempted == []
