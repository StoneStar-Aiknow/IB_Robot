"""Launch-builder test for the perception_models entry (inference_service's contribution).

Migrated from ``robot_config/test/test_perception_runtime_config.py``
(openspec change ``migrate-business-launch-builders``, task A.5/A.8): the
test whose subject is ``generate_perception_model_nodes`` followed the module
to its owning package; the parse-level tests stay in robot_config with
``robot_config.perception_runtime_config``. Assertions are verbatim; only the
import path changed.
"""

import json
from pathlib import Path

from inference_manifest import BundleFile, canonical_bundle_digest
from inference_service.launch_builders.perception_models import generate_perception_model_nodes


def _bundle(root: Path, family: str, *, deployment: str = "torch_cpu") -> Path:
    root.mkdir(parents=True)
    marker = root / "assets" / "identity.txt"
    marker.parent.mkdir()
    marker.write_text(family, encoding="utf-8")
    entry = BundleFile(path="assets/identity.txt")
    manifest = {
        "schema_version": 3,
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
            "interface": "tensor_model",
            "model_type": family,
            "operation": "infer",
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
                "execution_contract": {
                    "state_scope": "request",
                    "execution_structure": "direct",
                    "cancellation_granularity": "request_boundary",
                },
                "runtime_profile": {
                    "backend": "torch",
                    "target": {"runtime": "torch"},
                    "profile": {"device": "cpu"},
                },
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
