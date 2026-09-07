"""Small schema-v3 perception bundles for semantic-mapping tests."""

import json
from pathlib import Path

from inference_manifest import BundleFile, canonical_bundle_digest

_SERVICES = (
    ("semantic_sam2_masks", "sam2", "ibrobot_msgs/srv/GenerateMasks", "SAM2GenerateMasksPlugin", True),
    ("semantic_ram_plus_tags", "ram_plus", "ibrobot_msgs/srv/RecognizeTags", "RAMPlusRecognizeTagsPlugin", True),
    ("semantic_siglip2_image", "siglip2", "ibrobot_msgs/srv/EncodeEmbeddings", "SigLIP2EncodeEmbeddingsPlugin", True),
    ("semantic_siglip2_text", "siglip2", "ibrobot_msgs/srv/EncodeText", "SigLIP2EncodeTextPlugin", False),
    (
        "semantic_gdino_confirmation",
        "grounding_dino",
        "ibrobot_msgs/srv/GroundingDetect",
        "GroundingDetectPlugin",
        False,
    ),
)


def configure_perception_bundles(robot: dict, root: Path) -> None:
    services = robot["perception_services"]["services"]
    for index, (service_id, family, service_type, plugin, required) in enumerate(_SERVICES, start=1):
        bundle = root / service_id
        bundle.mkdir(parents=True)
        marker = bundle / "assets" / "identity.txt"
        marker.parent.mkdir()
        marker.write_text(family, encoding="utf-8")
        entry = BundleFile(path="assets/identity.txt")
        bundle_uuid = f"123e4567-e89b-42d3-a456-42661417400{index}"
        manifest = {
            "schema_version": 3,
            "bundle": {
                "uuid": bundle_uuid,
                "revision": 1,
                "name": f"test-{family}",
                "files": [entry.model_dump(mode="json")],
                "digest": {
                    "algorithm": "sha256",
                    "scope": "structure",
                    "value": canonical_bundle_digest(bundle_uuid, 1, f"test-{family}", [entry]),
                },
            },
            "model": {
                "interface": "tensor_model",
                "model_type": family,
                "operation": {
                    "sam2": "automatic",
                    "ram_plus": "recognize_tags",
                    "siglip2": "encode",
                    "grounding_dino": "detect",
                }[family],
                "inputs": [{"semantic": "features", "dtype": "float32", "shape": [1]}],
                "outputs": [{"semantic": "scores", "dtype": "float32", "shape": [1]}],
                "semantic_identity": {
                    "logical_model_revision": f"{family}@v1",
                    "preprocessing_contract": f"{family}-pre-v1",
                    "output_semantics": f"{family}-output-v1",
                    **(
                        {
                            "embedding": {
                                "embedding_space_id": "siglip2-test-space:v1",
                                "dimension": 4,
                                "normalization": "l2",
                                "image_preprocessing": "siglip2-image-v1",
                                "text_preprocessing": "siglip2-text-v1",
                            }
                        }
                        if family == "siglip2"
                        else {}
                    ),
                },
            },
            "deployments": {
                "torch_cpu": {
                    "uuid": f"123e4567-e89b-42d3-a456-42661417401{index}",
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
        (bundle / "inference_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        service = next(item for item in services if item["id"] == service_id)
        service.update(
            {
                "enabled": True,
                "required": required,
                "bundle_path": str(bundle),
                "deployment": "torch_cpu",
                "adapter_class": f"perception_service.model_service_plugins:{plugin}",
                "service_type": service_type,
                "endpoint": f"/semantic_perception/{service_id}",
                "runtime_options": {},
            }
        )
