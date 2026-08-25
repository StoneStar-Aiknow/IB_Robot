import json

from inference_manifest import load_inference_manifest
from perception_service.package_perception_bundles import _specs, package_bundle


def _write_required_assets(root, spec) -> None:
    for relative in spec.required_paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")


def test_packager_preserves_lineage_and_bumps_bundle_for_asset_replacement(tmp_path) -> None:
    spec = _specs()["sam2"]
    root = tmp_path / spec.name
    _write_required_assets(root, spec)

    package_bundle(root, spec)
    first = load_inference_manifest(root, "torch_cpu").manifest
    first_bytes = (root / "inference_manifest.json").read_bytes()

    package_bundle(root, spec)
    second = load_inference_manifest(root, "torch_cpu").manifest

    assert (root / "inference_manifest.json").read_bytes() == first_bytes
    assert second.bundle.uuid == first.bundle.uuid
    assert second.bundle.revision == first.bundle.revision
    assert second.deployments == first.deployments

    checkpoint = root / spec.required_paths[0]
    checkpoint.write_text("replacement", encoding="utf-8")
    package_bundle(root, spec)
    third = load_inference_manifest(root, "torch_cpu").manifest

    assert third.bundle.uuid == first.bundle.uuid
    assert third.bundle.revision == first.bundle.revision + 1
    assert third.deployments == first.deployments
    digests = json.loads((root / "assets" / "artifact-digests.json").read_text(encoding="utf-8"))
    assert spec.required_paths[0] in digests


def test_ram_plus_packager_promotes_available_ascend_deployments(tmp_path) -> None:
    spec = _specs()["ram_plus"]
    root = tmp_path / spec.name
    _write_required_assets(root, spec)
    for compiled in spec.compiled:
        candidate = root.parent / "_work" / root.name / compiled.source
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_bytes(compiled.deployment.encode())

    package_bundle(root, spec)
    first = load_inference_manifest(root, "ascend_310p").manifest
    for compiled in spec.compiled:
        deployment = first.deployments[compiled.deployment]
        assert (root / compiled.artifact).read_bytes() == compiled.deployment.encode()
        assert deployment.target.soc == compiled.target_soc
        assert deployment.artifacts["model"].path == compiled.artifact
        assert deployment.bindings["model"].inputs[0].semantic == "observation.image"
        assert deployment.bindings["model"].outputs[0].semantic == "tag_logits"

    package_bundle(root, spec)
    second = load_inference_manifest(root, "ascend_310p").manifest
    assert second.deployments == first.deployments

    compiled = spec.compiled[0]
    deployment = first.deployments[compiled.deployment]
    candidate = root.parent / "_work" / root.name / compiled.source
    candidate.write_bytes(b"replacement-om")
    package_bundle(root, spec)
    third = load_inference_manifest(root, compiled.deployment).manifest
    assert third.deployments[compiled.deployment].uuid == deployment.uuid
    assert third.deployments[compiled.deployment].revision == deployment.revision + 1


def test_ram_plus_packager_omits_unavailable_ascend_deployment(tmp_path) -> None:
    spec = _specs()["ram_plus"]
    root = tmp_path / spec.name
    _write_required_assets(root, spec)

    package_bundle(root, spec)

    manifest = load_inference_manifest(root, "torch_cpu").manifest
    assert all(compiled.deployment not in manifest.deployments for compiled in spec.compiled)


def test_grounded_sam2_bundle_uses_source_bound_architecture_config() -> None:
    spec = _specs()["grounded_sam2"]

    assert (spec.model.model_type, spec.model.operation) == ("grounding_dino", "detect")
    assert (spec.adapter["model_type"], spec.adapter["operation"]) == ("grounding_dino", "detect")
    assert "gdino_config" not in spec.adapter
    assert all(not path.endswith(".py") for path in spec.required_paths)


def test_grounded_sam2_packager_removes_legacy_copied_config(tmp_path) -> None:
    spec = _specs()["grounded_sam2"]
    root = tmp_path / spec.name
    _write_required_assets(root, spec)
    legacy_config = root / "assets" / "GroundingDINO_SwinT_OGC.py"
    legacy_config.write_text("num_queries = 900", encoding="utf-8")

    package_bundle(root, spec)

    assert not legacy_config.exists()
    manifest = load_inference_manifest(root, "torch_cpu").manifest
    assert all(not entry.path.endswith(".py") for entry in manifest.bundle.files)
