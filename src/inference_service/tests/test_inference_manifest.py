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
    canonical_bundle_digest,
    canonical_manifest_bytes,
    deployment_fingerprint,
    load_inference_manifest,
    normalize_bundle_path,
    normalize_unique_paths,
    sha256_file,
    write_inference_manifest,
)
from inference_manifest.models import DeviceLink
from tests.manifest_fixtures import create_policy_bundle, make_manifest, write_manifest


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

    manifest["schema_version"] = 2
    write_manifest(tmp_path, manifest)
    with pytest.raises(ManifestValidationError, match=r"Unsupported schema_version 2.*inference_manifest.json"):
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
        '{"schema_version":1,"schema_version":1}',
        encoding="utf-8",
    )
    with pytest.raises(ManifestValidationError, match="duplicate JSON key 'schema_version'"):
        load_inference_manifest(tmp_path, "cpu")


@pytest.mark.parametrize("alias", ["ascend_om", "ascend_om_3403", "3403", "om"])
def test_removed_backend_aliases_are_rejected(tmp_path, alias):
    paths = create_policy_bundle(tmp_path)
    manifest = make_manifest(tmp_path, paths)
    manifest["deployments"]["cpu"] = {"backend": alias, "device": "cpu"}
    write_manifest(tmp_path, manifest)

    with pytest.raises(ManifestValidationError, match=alias):
        load_inference_manifest(tmp_path, "cpu")


def test_hash_mismatch_reports_digests_and_exporter_guidance(tmp_path):
    paths = create_policy_bundle(tmp_path)
    manifest = make_manifest(tmp_path, paths)
    write_manifest(tmp_path, manifest)
    expected = next(entry["sha256"] for entry in manifest["bundle"]["files"] if entry["path"] == "config.json")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    actual = sha256_file(tmp_path / "config.json")

    with pytest.raises(ManifestIntegrityError) as error:
        load_inference_manifest(tmp_path, "cpu")

    message = str(error.value)
    assert "config.json" in message
    assert expected in message
    assert actual in message
    assert "Rerun the owning exporter or packaging workflow" in message


def test_artifact_hash_mismatch_reports_role_and_exporter_guidance(tmp_path):
    paths = tuple(path for path in create_policy_bundle(tmp_path) if path != "model.safetensors")
    manifest = make_manifest(tmp_path, paths, deployment_name="rk3588", compiled=True)
    write_manifest(tmp_path, manifest)
    artifact = manifest["deployments"]["rk3588"]["artifacts"]["policy"]
    (tmp_path / artifact["path"]).write_bytes(b"changed-compiled-policy")
    actual = sha256_file(tmp_path / artifact["path"])

    with pytest.raises(ManifestIntegrityError) as error:
        load_inference_manifest(tmp_path, "rk3588")

    message = str(error.value)
    assert "role 'policy'" in message
    assert artifact["sha256"] in message
    assert actual in message
    assert "Rerun the owning exporter or packaging workflow" in message


def test_canonical_bundle_digest_is_order_independent():
    entries = [
        BundleFile(path="z.json", sha256="1" * 64),
        BundleFile(path="a.json", sha256="2" * 64),
    ]

    assert canonical_bundle_digest(entries) == canonical_bundle_digest(reversed(entries))


def test_deployment_fingerprint_is_stable_and_tracks_identity_changes():
    deployment = CompiledDeployment.model_validate_json(
        json.dumps(
            {
                "backend": "rknn",
                "target": {"soc": "rk3588", "runtime": "rknn-lite"},
                "artifacts": {"policy": {"path": "policy.rknn", "format": "rknn", "sha256": "3" * 64}},
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

    baseline = deployment_fingerprint(1, "4" * 64, "rk3588", deployment)
    assert baseline == deployment_fingerprint(1, "4" * 64, "rk3588", deployment)
    assert baseline != deployment_fingerprint(1, "5" * 64, "rk3588", deployment)
    assert baseline != deployment_fingerprint(1, "4" * 64, "other", deployment)

    changed = deployment.model_copy(update={"target": deployment.target.model_copy(update={"runtime": "rknn-lite-2"})})
    assert baseline != deployment_fingerprint(1, "4" * 64, "rk3588", changed)


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
        "sha256": sha256_file(worker_path),
    }
    write_manifest(tmp_path, manifest)

    validated = load_inference_manifest(tmp_path, "hisilicon")

    assert validated.deployment.execution == ("policy",)
    assert set(validated.deployment.artifacts) == {"policy", "worker"}
    assert validated.deployment.bindings["policy"].outputs[0].index == 1

    worker_path.write_bytes(b"changed-worker")
    with pytest.raises(ManifestIntegrityError, match="role 'worker'"):
        load_inference_manifest(tmp_path, "hisilicon")


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


def test_selected_bundle_and_artifact_paths_cannot_duplicate(tmp_path):
    paths = tuple(path for path in create_policy_bundle(tmp_path) if path != "model.safetensors")
    manifest = make_manifest(tmp_path, paths, deployment_name="rk3588", compiled=True)
    artifact = manifest["deployments"]["rk3588"]["artifacts"]["policy"]
    manifest["bundle"]["files"].append({"path": artifact["path"], "sha256": artifact["sha256"]})
    entries = [BundleFile.model_validate(entry) for entry in manifest["bundle"]["files"]]
    manifest["bundle"]["digest"]["value"] = canonical_bundle_digest(entries)
    write_manifest(tmp_path, manifest)

    with pytest.raises(ManifestPathError, match="duplicate normalized path 'artifacts/policy.rknn'"):
        load_inference_manifest(tmp_path, "rk3588")


def test_canonical_writer_is_deterministic_and_replaces_atomically(tmp_path):
    paths = create_policy_bundle(tmp_path)
    manifest = make_manifest(tmp_path, paths)
    destination = tmp_path / "nested" / "inference_manifest.json"

    first = write_inference_manifest(destination, manifest)
    first_content = first.read_bytes()
    assert first_content == canonical_manifest_bytes(manifest)

    reordered = {"deployments": manifest["deployments"], "bundle": manifest["bundle"], "schema_version": 1}
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
