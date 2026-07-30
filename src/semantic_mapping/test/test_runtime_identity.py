from types import SimpleNamespace

import pytest

from semantic_mapping.runtime_identity import MappingRunPin, MappingRunPinMismatch, RuntimeDiagnostic, SemanticIdentity


def _identity(name, *, embedding=False):
    value = {
        "logical_model_revision": f"{name}@v1",
        "preprocessing_contract": f"{name}-pre-v1",
        "output_semantics": f"{name}-output-v1",
    }
    if embedding:
        value["embedding"] = {
            "embedding_space_id": "siglip2-space:v1",
            "dimension": 2,
            "normalization": "l2",
            "image_preprocessing": "image-v1",
            "text_preprocessing": "text-v1",
        }
    return SemanticIdentity.from_dict(value)


def _info(role, identity, *, deployment="cuda", fingerprint="b", generation=7):
    embedding = identity.embedding
    return SimpleNamespace(
        ready=True,
        failure_reason="",
        message="",
        instance_id=role,
        model_name=role,
        model_version="1",
        manifest_fingerprint="a" * 64,
        deployment_name=deployment,
        deployment_fingerprint=fingerprint * 64,
        backend=deployment,
        runtime_version="1.0",
        semantic_identity_json=identity.canonical_json,
        semantic_identity_fingerprint=identity.fingerprint,
        embedding_space_id="" if embedding is None else embedding.embedding_space_id,
        embedding_dimension=0 if embedding is None else embedding.dimension,
        normalization="" if embedding is None else embedding.normalization,
        configuration_generation=generation,
    )


def _pin():
    identities = {
        "sam2": _identity("sam2"),
        "ram_plus": _identity("ram-plus"),
        "siglip2_image": _identity("siglip2", embedding=True),
    }
    return MappingRunPin("run-1", 7, {role: role for role in identities}, identities), identities


def test_runtime_diagnostic_requires_canonical_identity_and_matching_fingerprint():
    identity = _identity("sam2")
    diagnostic = RuntimeDiagnostic.from_runtime_info(_info("sam2", identity))
    assert diagnostic.semantic_identity == identity

    malformed = _info("sam2", identity)
    malformed.semantic_identity_json = identity.canonical_json.replace(":", ": ", 1)
    with pytest.raises(ValueError, match="canonical"):
        RuntimeDiagnostic.from_runtime_info(malformed)


def test_runtime_diagnostic_canonical_identity_preserves_unicode():
    identity = SemanticIdentity.from_dict(
        {
            "logical_model_revision": "杯子@v1",
            "preprocessing_contract": "图像-v1",
            "output_semantics": "标签-v1",
        }
    )

    assert "杯子" in identity.canonical_json
    assert RuntimeDiagnostic.from_runtime_info(_info("sam2", identity)).semantic_identity == identity


def test_run_pin_accepts_changed_deployment_for_equal_semantic_identity():
    pin, identities = _pin()
    diagnostics = {
        "sam2": (_info("sam2", identities["sam2"], deployment="cpu", fingerprint="c"),),
        "ram_plus": (_info("ram_plus", identities["ram_plus"]),),
        "siglip2_image": (
            _info("siglip2_image", identities["siglip2_image"]),
            _info("siglip2_image", identities["siglip2_image"], deployment="cpu", fingerprint="c"),
        ),
    }
    provenance = pin.validate_frame(diagnostics)
    assert provenance["sam2"].backend == "cpu"
    assert provenance["siglip2_image"].deployment_fingerprint == "c" * 64


def test_run_pin_rejects_generation_and_embedding_space_changes():
    pin, identities = _pin()
    diagnostics = {
        "sam2": (_info("sam2", identities["sam2"], generation=8),),
        "ram_plus": (_info("ram_plus", identities["ram_plus"]),),
        "siglip2_image": (_info("siglip2_image", identities["siglip2_image"]),),
    }
    with pytest.raises(MappingRunPinMismatch, match="configuration generation"):
        pin.validate_frame(diagnostics)

    incompatible = _identity("other-siglip2", embedding=True)
    diagnostics["sam2"] = (_info("sam2", identities["sam2"]),)
    diagnostics["siglip2_image"] = (_info("siglip2_image", incompatible),)
    with pytest.raises(MappingRunPinMismatch, match="semantic identity"):
        pin.validate_frame(diagnostics)
