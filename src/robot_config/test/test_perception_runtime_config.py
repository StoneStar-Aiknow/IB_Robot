"""Tests for typed semantic perception service configuration and launch generation."""

import json
from pathlib import Path

import pytest

from inference_manifest import BundleFile, canonical_bundle_digest
from robot_config.launch_builders.perception_models import generate_perception_model_nodes
from robot_config.perception_runtime_config import PerceptionRuntimeConfigError, parse_perception_runtime_config


def _bundle(root: Path, family: str, *, deployment: str = "torch_cpu") -> Path:
    root.mkdir(parents=True)
    marker = root / "assets" / "identity.txt"
    marker.parent.mkdir()
    marker.write_text(family, encoding="utf-8")
    entry = BundleFile(path="assets/identity.txt")
    manifest = {
        "schema_version": 2,
        "bundle": {
            "uuid": "123e4567-e89b-42d3-a456-426614174000",
            "revision": 1,
            "name": f"test-{family}",
            "files": [entry.model_dump(mode="json")],
            "digest": {
                "algorithm": "sha256",
                "scope": "structure",
                "value": canonical_bundle_digest("123e4567-e89b-42d3-a456-426614174000", 1, f"test-{family}", [entry]),
            },
        },
        "model": {
            "kind": "perception",
            "family": family,
            "inputs": [{"semantic": "features", "dtype": "float32", "shape": [1]}],
            "outputs": [{"semantic": "scores", "dtype": "float32", "shape": [1]}],
            "semantic_identity": {
                "logical_model_revision": f"{family}@v1",
                "preprocessing_contract": "test-pre-v1",
                "output_semantics": "test-output-v1",
            },
        },
        "deployments": {
            deployment: {
                "uuid": "123e4567-e89b-42d3-a456-426614174001",
                "revision": 1,
                "backend": "torch",
                "device": "cpu",
            }
        },
    }
    (root / "inference_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _service(bundle: Path, endpoint: str, *, instance_id: str = "depth_front", required: bool = True) -> dict:
    return {
        "id": instance_id,
        "enabled": True,
        "required": required,
        "bundle_path": str(bundle),
        "deployment": "torch_cpu",
        "adapter_class": "example_depth.plugin:DepthServicePlugin",
        "service_type": "example_depth_msgs/srv/EstimateDepth",
        "endpoint": endpoint,
        "runtime_options": {"device_id": 0},
    }


def _config(bundle: Path) -> dict:
    return {"perception_services": {"services": [_service(bundle, "/depth/front")]}}


def _node_parameters(node) -> dict:
    values = vars(node)["_Node__parameters"][0]
    normalized = {}
    for key, value in values.items():
        normalized_key = "".join(getattr(part, "text", str(part)) for part in key)
        if isinstance(value, tuple):
            value = "".join(getattr(part, "text", str(part)) for part in value).removesuffix("\n...\n")
        if isinstance(value, str):
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
        normalized[normalized_key] = value
    return normalized


def test_parse_perception_runtime_config_is_typed_and_immutable(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "depth", "metric_depth")
    runtime = parse_perception_runtime_config(_config(bundle))
    service = runtime.services[0]

    assert service.enabled and service.required
    assert service.instance_id == "depth_front"
    assert service.bundle_path == bundle
    assert service.deployment == "torch_cpu"
    assert service.validated_manifest.deployment_name == "torch_cpu"
    assert 0 < runtime.configuration_generation({"depth": "depth_front"}) < 2**63
    with pytest.raises(TypeError):
        service.runtime_options["device_id"] = 1  # type: ignore[index]


@pytest.mark.parametrize("legacy", ["model_backend", "backend", "device"])
def test_raw_backend_selection_is_rejected(tmp_path: Path, legacy: str) -> None:
    config = _config(_bundle(tmp_path / "depth", "metric_depth"))
    config["perception_services"][legacy] = "cuda"

    with pytest.raises(PerceptionRuntimeConfigError, match="deployment explicitly"):
        parse_perception_runtime_config(config)


def test_unknown_family_is_accepted_and_missing_deployment_fails_before_launch(tmp_path: Path) -> None:
    unknown = _config(_bundle(tmp_path / "unknown", "future_multimodal_family"))
    assert parse_perception_runtime_config(unknown).services[0].validated_manifest.manifest.model.family == (
        "future_multimodal_family"
    )

    missing = _config(_bundle(tmp_path / "missing", "metric_depth", deployment="other"))
    with pytest.raises(PerceptionRuntimeConfigError, match="Deployment 'torch_cpu' is not present"):
        parse_perception_runtime_config(missing)


def test_required_disabled_invalid_options_and_endpoint_conflicts_are_rejected(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "depth", "metric_depth")
    disabled = _config(bundle)
    disabled["perception_services"]["services"][0].update({"enabled": False, "required": True})
    with pytest.raises(PerceptionRuntimeConfigError, match="cannot be required when disabled"):
        parse_perception_runtime_config(disabled)

    invalid_options = _config(bundle)
    invalid_options["perception_services"]["services"][0]["runtime_options"] = {"threshold": float("nan")}
    with pytest.raises(PerceptionRuntimeConfigError, match="JSON-compatible finite values"):
        parse_perception_runtime_config(invalid_options)

    conflict = _config(bundle)
    conflict["perception_services"]["services"].append(
        _service(
            _bundle(tmp_path / "flow", "optical_flow"),
            "/depth/front",
            instance_id="flow_front",
        )
    )
    with pytest.raises(PerceptionRuntimeConfigError, match="endpoint conflict"):
        parse_perception_runtime_config(conflict)


def test_launch_builder_emits_named_selection_and_keeps_optional_service_independent(tmp_path: Path) -> None:
    config = _config(_bundle(tmp_path / "depth", "metric_depth"))
    config["perception_services"]["services"].append({"id": "optional_vlm", "enabled": False, "required": False})

    nodes = generate_perception_model_nodes(config)

    assert len(nodes) == 1
    node = nodes[0]
    assert vars(node)["_Node__node_executable"] == "model_service_node"
    assert vars(node)["_Node__node_name"] == "model_service_depth_front"
    parameters = _node_parameters(node)
    assert parameters["instance_id"] == "depth_front"
    assert parameters["bundle_path"] == str(tmp_path / "depth")
    assert parameters["deployment"] == "torch_cpu"
    assert parameters["adapter_class"] == "example_depth.plugin:DepthServicePlugin"
    assert parameters["service_type"] == "example_depth_msgs/srv/EstimateDepth"
    assert parameters["service_endpoint"] == "/depth/front"
    assert json.loads(parameters["runtime_options_json"])["device_id"] == 0
    assert parameters["require_semantic_identity"] is False
    assert "backend" not in parameters


def test_service_list_rejects_duplicate_ids_and_malformed_plugin_identity(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "depth", "metric_depth")
    duplicate = _config(bundle)
    duplicate["perception_services"]["services"].append(_service(bundle, "/depth/other", instance_id="depth_front"))
    with pytest.raises(PerceptionRuntimeConfigError, match="Duplicate model service id"):
        parse_perception_runtime_config(duplicate)

    malformed = _config(bundle)
    malformed["perception_services"]["services"][0]["adapter_class"] = "not an import"
    with pytest.raises(PerceptionRuntimeConfigError, match="canonical module:Class"):
        parse_perception_runtime_config(malformed)
