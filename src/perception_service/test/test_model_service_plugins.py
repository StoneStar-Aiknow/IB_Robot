import json
from types import SimpleNamespace

import numpy as np
import pytest
from sensor_msgs.msg import Image

from inference_manifest import BundleFile, CompiledDeployment, canonical_bundle_digest, load_inference_manifest
from inference_service.backends import BackendCapabilities
from inference_service.backends.lifecycle import PartialLoadRollback
from inference_service.backends.types import RuntimeContext
from inference_service.generic_runtime import NamedTensorRequest
from inference_service.model_sessions import ModelSession
from perception_service.model_service_node import _runtime_info
from perception_service.model_service_plugin import ModelServicePlugin
from perception_service.model_service_plugins import (
    GroundingDetectPlugin,
    RAMPlusRecognizeTagsPlugin,
    SAM2GenerateMasksPlugin,
    SigLIP2EncodeEmbeddingsPlugin,
    SigLIP2EncodeTextPlugin,
    _new_session,
    _SessionPlugin,
)
from perception_service.ram_plus_adapter import RAMPlusAdapter
from perception_service.semantic_model_adapters import (
    GroundingDINOAdapter,
    SAM2Adapter,
    SigLIP2ImageAdapter,
    SigLIP2TextAdapter,
)
from semantic_mapping.runtime_identity import RuntimeDiagnostic

FAMILIES = {
    "sam2": (SAM2GenerateMasksPlugin, SAM2Adapter),
    "ram_plus": (RAMPlusRecognizeTagsPlugin, RAMPlusAdapter),
    "siglip2_image": (SigLIP2EncodeEmbeddingsPlugin, SigLIP2ImageAdapter),
    "siglip2_text": (SigLIP2EncodeTextPlugin, SigLIP2TextAdapter),
    "grounding_dino": (GroundingDetectPlugin, GroundingDINOAdapter),
}


def _descriptors(role):
    if role == "sam2":
        return (
            [{"semantic": "observation.image", "dtype": "uint8", "shape": [-1, -1, 3]}],
            [
                {"semantic": "masks", "dtype": "uint8", "shape": [-1, -1, -1]},
                {"semantic": "boxes", "dtype": "float32", "shape": [-1, 4]},
                {"semantic": "scores", "dtype": "float32", "shape": [-1]},
                {"semantic": "stability_scores", "dtype": "float32", "shape": [-1]},
            ],
        )
    if role == "ram_plus":
        return (
            [{"semantic": "observation.image", "dtype": "float32", "shape": [1, 3, 384, 384], "layout": "NCHW"}],
            [{"semantic": "tag_logits", "dtype": "float32", "shape": [1, 4585]}],
        )
    if role == "siglip2_image":
        return (
            [
                {"semantic": "masked_images", "dtype": "float32", "shape": [-1, 3, 384, 384], "layout": "NCHW"},
                {"semantic": "text_tokens", "dtype": "int64", "shape": [-1, 64]},
                {"semantic": "text_attention_mask", "dtype": "int64", "shape": [-1, 64]},
            ],
            [
                {"semantic": "image_embeddings", "dtype": "float32", "shape": [-1, 4]},
                {"semantic": "text_embeddings", "dtype": "float32", "shape": [-1, 4]},
            ],
        )
    if role == "siglip2_text":
        return (
            [
                {"semantic": "masked_images", "dtype": "float32", "shape": [-1, 3, 384, 384], "layout": "NCHW"},
                {"semantic": "text_tokens", "dtype": "int64", "shape": [-1, 64]},
                {"semantic": "text_attention_mask", "dtype": "int64", "shape": [-1, 64]},
            ],
            [
                {"semantic": "image_embeddings", "dtype": "float32", "shape": [-1, 4]},
                {"semantic": "text_embeddings", "dtype": "float32", "shape": [-1, 4]},
            ],
        )
    return (
        [
            {"semantic": "observation.image", "dtype": "uint8", "shape": [-1, -1, 3]},
            {"semantic": "text_prompt", "dtype": "uint8", "shape": [-1]},
            {"semantic": "box_threshold", "dtype": "float32", "shape": [1]},
            {"semantic": "text_threshold", "dtype": "float32", "shape": [1]},
        ],
        [
            {"semantic": "boxes", "dtype": "float32", "shape": [-1, 4]},
            {"semantic": "scores", "dtype": "float32", "shape": [-1]},
            {"semantic": "masks", "dtype": "uint8", "shape": [-1, -1, -1]},
            {"semantic": "label_indices", "dtype": "int32", "shape": [-1]},
        ],
    )


def _write_bundle(root, role, *, embedding=True, adapter_preprocessing=None):
    plugin, adapter_type = FAMILIES[role]
    family = plugin.family
    assets = root / "assets"
    assets.mkdir(parents=True)
    identity = adapter_type.identity
    (assets / "adapter.json").write_text(
        json.dumps(
            {
                "family": family,
                "preprocessing": adapter_preprocessing or identity.preprocessing,
                "postprocessing": identity.postprocessing,
            }
        ),
        encoding="utf-8",
    )
    if role == "ram_plus":
        (assets / "ram_tag_list.txt").write_text(
            "\n".join(["cup", "table", *(f"tag-{index}" for index in range(2, 4585))]) + "\n",
            encoding="utf-8",
        )
        thresholds = np.full(4585, 0.99, dtype=np.float32)
        thresholds[:2] = [0.4, 0.8]
        np.savetxt(assets / "ram_tag_list_threshold.txt", thresholds)
    paths = ["assets/adapter.json"]
    if role == "ram_plus":
        paths += ["assets/ram_tag_list.txt", "assets/ram_tag_list_threshold.txt"]
    files = [BundleFile(path=path) for path in paths]
    inputs, outputs = _descriptors(role)
    semantic_identity = {
        "logical_model_revision": f"{family}@v1",
        "preprocessing_contract": identity.preprocessing,
        "output_semantics": identity.postprocessing,
    }
    if role.startswith("siglip2") and embedding:
        semantic_identity["embedding"] = {
            "embedding_space_id": "siglip2-test-space",
            "dimension": 4,
            "normalization": "l2",
            "image_preprocessing": "siglip2-image-384-v1",
            "text_preprocessing": "siglip2-text-64-v1",
        }
    bundle_uuid = "8fa9838a-2e15-4cf4-a9d5-4fb876c10eb7"
    name = f"{role}-test"
    manifest = {
        "schema_version": 2,
        "bundle": {
            "uuid": bundle_uuid,
            "revision": 1,
            "name": name,
            "files": [entry.model_dump(mode="json") for entry in files],
            "digest": {
                "algorithm": "sha256",
                "scope": "structure",
                "value": canonical_bundle_digest(bundle_uuid, 1, name, files),
            },
        },
        "model": {
            "kind": "perception",
            "family": family,
            "inputs": inputs,
            "outputs": outputs,
            "semantic_identity": semantic_identity,
        },
        "deployments": {
            "torch_cpu": {
                "uuid": "26547f4a-1d02-4ea1-b4dc-c887ca557a68",
                "revision": 1,
                "backend": "torch",
                "device": "cpu",
            }
        },
    }
    (root / "inference_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return load_inference_manifest(root, "torch_cpu")


class _FakeSession(ModelSession):
    next_id = 0

    def __init__(self, adapter):
        type(self).next_id += 1
        super().__init__(f"fake-{self.next_id}", BackendCapabilities())
        self.adapter = adapter
        self.close_count = 0
        self.requests = []
        self.loaded_runtime = SimpleNamespace(__version__="test-runtime-2.1")

    def _load(self, context: RuntimeContext, _rollback: PartialLoadRollback) -> None:
        assert context.deployment_name == "torch_cpu"

    def _execute(self, request: NamedTensorRequest):
        self.requests.append(request)
        if isinstance(self.adapter, SAM2Adapter):
            height, width = request.inputs["observation.image"].shape[:2]
            return {
                "masks": np.ones((1, height, width), dtype=np.uint8),
                "boxes": np.asarray([[0, 0, width, height]], dtype=np.float32),
                "scores": np.asarray([0.9], dtype=np.float32),
                "stability_scores": np.asarray([0.8], dtype=np.float32),
            }
        if isinstance(self.adapter, RAMPlusAdapter):
            logits = np.full((1, 4585), -80.0, dtype=np.float32)
            logits[0, :2] = [1.0, 2.0]
            return {"tag_logits": logits}
        if isinstance(self.adapter, SigLIP2ImageAdapter):
            count = len(request.inputs["masked_images"])
            text_count = len(request.inputs["text_tokens"])
            return {
                "image_embeddings": np.tile([3.0, 4.0, 0.0, 0.0], (count, 1)).astype(np.float32),
                "text_embeddings": np.tile([0.0, 1.0, 0.0, 0.0], (text_count, 1)).astype(np.float32),
            }
        if isinstance(self.adapter, SigLIP2TextAdapter):
            count = len(request.inputs["text_tokens"])
            return {
                "image_embeddings": np.empty((0, 4), dtype=np.float32),
                "text_embeddings": np.tile([0.0, 0.0, 3.0, 4.0], (count, 1)).astype(np.float32),
            }
        return {
            "boxes": np.asarray([[0, 0, 2, 2]], dtype=np.float32),
            "scores": np.asarray([0.9], dtype=np.float32),
            "masks": np.ones((1, 4, 4), dtype=np.uint8),
            "label_indices": np.asarray([0], dtype=np.int32),
        }

    def _close(self) -> None:
        self.close_count += 1


class _Bridge:
    @staticmethod
    def imgmsg_to_cv2(image, desired_encoding):
        channels = 3 if desired_encoding == "rgb8" else 1
        shape = (image.height, image.width, channels) if channels == 3 else (image.height, image.width)
        return np.frombuffer(image.data, dtype=np.uint8).reshape(shape)

    @staticmethod
    def cv2_to_imgmsg(value, encoding):
        array = np.asarray(value, dtype=np.uint8)
        return Image(
            height=array.shape[0], width=array.shape[1], encoding=encoding, step=array.shape[1], data=array.tobytes()
        )


def _host():
    return SimpleNamespace(bridge=_Bridge())


def _image(encoding="rgb8"):
    return Image(height=4, width=4, encoding=encoding, step=12, data=np.zeros((4, 4, 3), dtype=np.uint8).tobytes())


@pytest.fixture(autouse=True)
def fake_sessions(monkeypatch):
    class FakeTokenizer:
        def __call__(self, texts, **_kwargs):
            tokens = np.zeros((len(texts), 64), dtype=np.int64)
            attention = np.zeros_like(tokens)
            for index, text in enumerate(texts):
                width = min(len(text), 64)
                tokens[index, :width] = np.arange(1, width + 1)
                attention[index, :width] = 1
            return {"input_ids": tokens, "attention_mask": attention}

    monkeypatch.setattr(
        "perception_service.semantic_model_adapters._load_siglip2_tokenizer", lambda _path: FakeTokenizer()
    )
    monkeypatch.setattr(
        _SessionPlugin,
        "_session_factory",
        staticmethod(lambda _family, adapter, _validated, _options: _FakeSession(adapter)),
    )


@pytest.mark.parametrize("role", FAMILIES)
def test_concrete_plugins_are_typed_model_session_hosts_with_health_and_idempotent_close(tmp_path, role) -> None:
    validated = _write_bundle(tmp_path / role, role)
    plugin_type, _adapter_type = FAMILIES[role]
    plugin = plugin_type(_host(), validated, {})

    assert isinstance(plugin, ModelServicePlugin)
    assert isinstance(plugin.session, ModelSession)
    assert plugin.runtime_status().ready
    assert plugin.runtime_status().metadata["runtime_version"] == "test-runtime-2.1"
    assert plugin.service_type.startswith("ibrobot_msgs/srv/")
    plugin.close()
    plugin.close()
    assert plugin.session.close_count == 1
    assert not plugin.runtime_status().ready


def test_plugin_status_projects_parseable_runtime_provenance(tmp_path) -> None:
    validated = _write_bundle(tmp_path / "sam2", "sam2")
    plugin = SAM2GenerateMasksPlugin(_host(), validated, {})

    info = _runtime_info("sam2", validated, plugin.runtime_status(), configuration_generation=7)
    diagnostic = RuntimeDiagnostic.from_runtime_info(info)

    assert diagnostic.provenance.runtime_version == "test-runtime-2.1"
    assert diagnostic.configuration_generation == 7
    plugin.close()


def test_plugin_fails_closed_when_loaded_runtime_exposes_no_version(tmp_path) -> None:
    validated = _write_bundle(tmp_path / "sam2", "sam2")
    plugin = SAM2GenerateMasksPlugin(_host(), validated, {})
    plugin.session.loaded_runtime = SimpleNamespace()

    status = plugin.runtime_status()

    assert not status.ready
    assert status.metadata == {}
    assert status.failure_reason == "loaded model runtime did not expose a version"
    plugin.close()


def test_siglip_image_and_text_use_independent_sessions_and_execute_named_requests(tmp_path) -> None:
    image_manifest = _write_bundle(tmp_path / "image", "siglip2_image")
    text_manifest = _write_bundle(tmp_path / "text", "siglip2_text")
    image_identity = image_manifest.manifest.model.semantic_identity.embedding
    text_identity = text_manifest.manifest.model.semantic_identity.embedding
    image_plugin = SigLIP2EncodeEmbeddingsPlugin(_host(), image_manifest, {})
    text_plugin = SigLIP2EncodeTextPlugin(_host(), text_manifest, {})
    image = _image()
    mask = Image(height=4, width=4, encoding="mono8", step=4, data=np.ones((4, 4), dtype=np.uint8).tobytes())
    mask.header.stamp = image.header.stamp
    image_response = SimpleNamespace(results=[])
    text_response = SimpleNamespace(results=[])

    image_plugin.handle(SimpleNamespace(image=image, masks=[mask], candidate_labels=["cup"]), image_response)
    text_plugin.handle(SimpleNamespace(texts=["cup", "table"]), text_response)

    assert image_plugin.session is not text_plugin.session
    assert image_identity == text_identity
    assert image_identity.embedding_space_id == "siglip2-test-space"
    assert image_identity.dimension == 4
    assert image_identity.normalization == "l2"
    assert image_identity.image_preprocessing == "siglip2-image-384-v1"
    assert image_identity.text_preprocessing == "siglip2-text-64-v1"
    assert set(image_plugin.session.requests[0].inputs) == {
        "masked_images",
        "text_attention_mask",
        "text_tokens",
    }
    assert set(text_plugin.session.requests[0].inputs) == {
        "masked_images",
        "text_attention_mask",
        "text_tokens",
    }
    assert np.linalg.norm(image_response.results[0].embedding) == pytest.approx(1.0)
    assert image_response.results[0].matched_label == "cup"
    assert len(text_response.results) == 2
    image_plugin.close()
    assert text_plugin.runtime_status().ready
    text_plugin.close()


def test_plugins_reject_raw_selection_identity_drift_and_missing_siglip_metadata(tmp_path) -> None:
    sam = _write_bundle(tmp_path / "sam", "sam2")
    with pytest.raises(ValueError, match="raw backend/device/fallback"):
        SAM2GenerateMasksPlugin(_host(), sam, {"backend": "cpu"})

    drift = _write_bundle(tmp_path / "drift", "sam2", adapter_preprocessing="wrong")
    with pytest.raises(ValueError, match="adapter identity mismatch"):
        SAM2GenerateMasksPlugin(_host(), drift, {})

    missing = _write_bundle(tmp_path / "missing", "siglip2_text", embedding=False)
    with pytest.raises(ValueError, match="embedding metadata"):
        SigLIP2EncodeTextPlugin(_host(), missing, {})


def test_unfinished_compiled_family_fails_before_session_creation_without_fallback() -> None:
    deployment = CompiledDeployment.model_validate(
        {
            "uuid": "040cde1f-3081-4544-938b-a6b5dba3dd6c",
            "revision": 1,
            "backend": "ascend",
            "target": {"soc": "Ascend310P3", "runtime": "acl"},
            "artifacts": {"model": {"path": "artifacts/model.om", "format": "om"}},
            "execution": ("model",),
            "bindings": {
                "model": {
                    "inputs": (
                        {
                            "semantic": "observation.image",
                            "runtime_name": "image",
                            "index": 0,
                            "dtype": "uint8",
                            "shape": (1,),
                        },
                    ),
                    "outputs": (
                        {
                            "semantic": "masks",
                            "runtime_name": "masks",
                            "index": 0,
                            "dtype": "uint8",
                            "shape": (1,),
                        },
                    ),
                }
            },
        }
    )
    validated = SimpleNamespace(deployment=deployment)

    with pytest.raises(RuntimeError, match="ABI is not finalized"):
        _new_session("sam2", SAM2Adapter(), validated, {})
