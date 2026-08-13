from inference_manifest import load_inference_manifest
from perception_service.package_ascend_perception_bundles import _specs, package_bundle
from perception_service.semantic_model_adapters import SigLIP2ImageAdapter


def _write_sources(root, spec):
    for deployment in spec.deployments:
        for artifact in deployment.artifacts:
            path = root / artifact.source
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(artifact.role.encode())
    for source, _destination in spec.assets:
        path = root / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"asset")


def test_compiled_ascend_bundle_specs_validate(tmp_path):
    for family, spec in _specs().items():
        _write_sources(tmp_path, spec)
        manifest_path = package_bundle(tmp_path, spec)
        assert manifest_path.is_file()
        for deployment in spec.deployments:
            validated = load_inference_manifest(manifest_path.parent, deployment.name)
            assert validated.deployment.target.soc == deployment.soc
            assert set(validated.deployment.artifacts) == {artifact.role for artifact in deployment.artifacts}
        if family == "grounding_dino":
            assert len(validated.deployment.execution) == 12
            assert validated.deployment.device_links
            assert (manifest_path.parent / "assets/bert-base-uncased/vocab.txt").read_bytes() == b"asset"


def test_siglip2_ascend_bundle_declares_loadable_embedding_identity(tmp_path, monkeypatch):
    spec = _specs()["siglip2"]
    _write_sources(tmp_path, spec)
    manifest_path = package_bundle(tmp_path, spec)
    validated = load_inference_manifest(manifest_path.parent, "ascend_310b")
    monkeypatch.setattr(
        "perception_service.semantic_model_adapters._load_siglip2_tokenizer",
        lambda _path: object(),
    )

    adapter = SigLIP2ImageAdapter.from_bundle(
        manifest_path.parent,
        validated.manifest.model.semantic_identity,
        model=validated.manifest.model,
        deployment=validated.deployment,
    )

    assert adapter.dimension == 1152
    assert validated.deployment.bindings["vision"].inputs[0].runtime_name == "image"
    assert validated.deployment.bindings["text"].inputs[0].runtime_name == "input_ids"
    assert validated.deployment.bindings["vision"].outputs[0].runtime_name is None
    assert validated.deployment.bindings["text"].outputs[0].runtime_name is None


def test_compiled_ascend_bundle_revisions_follow_artifact_content(tmp_path):
    spec = _specs()["sam2"]
    _write_sources(tmp_path, spec)
    manifest_path = package_bundle(tmp_path, spec)
    first = load_inference_manifest(manifest_path.parent, "ascend_310p").manifest

    package_bundle(tmp_path, spec)
    unchanged = load_inference_manifest(manifest_path.parent, "ascend_310p").manifest
    assert unchanged.bundle.revision == first.bundle.revision
    assert unchanged.deployments["ascend_310p"].revision == first.deployments["ascend_310p"].revision

    changed_artifact = tmp_path / spec.deployments[0].artifacts[0].source
    changed_artifact.write_bytes(b"changed encoder")
    package_bundle(tmp_path, spec)
    changed = load_inference_manifest(manifest_path.parent, "ascend_310p").manifest
    assert changed.bundle.revision == first.bundle.revision + 1
    assert changed.deployments["ascend_310p"].revision == first.deployments["ascend_310p"].revision + 1
    assert changed.deployments["ascend_310b"].revision == first.deployments["ascend_310b"].revision
