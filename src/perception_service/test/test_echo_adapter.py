import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inference_manifest import BundleFile, canonical_bundle_digest, load_inference_manifest
from inference_service.model_service_node import _instantiate_plugin, _validate_service_contract
from perception_service.echo_adapter import EchoAdapter, EchoServicePlugin


def _bundle(root: Path) -> Path:
    root.mkdir()
    marker = root / "identity.txt"
    marker.write_text("dummy-echo", encoding="utf-8")
    entry = BundleFile(path=marker.name)
    manifest = {
        "schema_version": 2,
        "bundle": {
            "uuid": "123e4567-e89b-42d3-a456-426614174000",
            "revision": 1,
            "name": "dummy-echo",
            "files": [entry.model_dump(mode="json")],
            "digest": {
                "algorithm": "sha256",
                "scope": "structure",
                "value": canonical_bundle_digest("123e4567-e89b-42d3-a456-426614174000", 1, "dummy-echo", [entry]),
            },
        },
        "model": {
            "kind": "perception",
            "family": "dummy_echo",
            "inputs": [{"semantic": "echo.input", "dtype": "float32", "shape": [2]}],
            "outputs": [{"semantic": "echo.output", "dtype": "float32", "shape": [2]}],
        },
        "deployments": {
            "cpu": {
                "uuid": "123e4567-e89b-42d3-a456-426614174001",
                "revision": 1,
                "backend": "torch",
                "device": "cpu",
            }
        },
    }
    (root / "inference_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_echo_plugin_preserves_typed_request_and_runtime_identity(tmp_path: Path):
    validated = load_inference_manifest(_bundle(tmp_path / "echo"), "cpu")
    plugin = EchoServicePlugin(None, validated, {})
    request = SimpleNamespace(request_id="request-1", value=[1.25, -2.5])
    response = SimpleNamespace(value=[])

    message = plugin.handle(request, response)

    assert message == "echoed 2 values"
    assert response.value == [1.25, -2.5]
    assert plugin.runtime_status().ready
    assert plugin.session.health().failure_count == 0
    assert plugin.runtime_status().metadata["deployment"] == "cpu"
    assert plugin.runtime_status().metadata["backend"] == "torch"
    plugin.close()
    plugin.close()
    assert plugin.runtime_status().state == "closed"


def test_echo_plugin_declares_a_generated_service_contract(tmp_path: Path):
    from ibrobot_msgs.srv import EchoModel

    validated = load_inference_manifest(_bundle(tmp_path / "echo"), "cpu")
    _validate_service_contract(EchoModel)
    plugin = _instantiate_plugin(
        EchoServicePlugin,
        "ibrobot_msgs/srv/EchoModel",
        None,
        validated,
        {},
    )

    plugin.close()


def test_echo_adapter_and_plugin_fail_closed(tmp_path: Path):
    adapter = EchoAdapter()
    with pytest.raises(ValueError, match="two finite values"):
        adapter.preprocess([np.nan, 1.0])

    validated = load_inference_manifest(_bundle(tmp_path / "echo"), "cpu")
    with pytest.raises(ValueError, match="does not accept runtime options"):
        EchoServicePlugin(None, validated, {"device": "cuda"})
