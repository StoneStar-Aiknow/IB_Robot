import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inference_manifest import BundleFile, canonical_bundle_digest, load_inference_manifest
from inference_service.backends import STATIC_BACKEND_DESCRIPTORS, BackendRegistry, ResourceDomainAdmissions
from inference_service.model_service_node import _instantiate_plugin, _validate_service_contract
from inference_service.runtime_composition import RuntimeDependencies
from inference_service.unified_runtime import (
    RegistrySet,
    RuntimeAssemblerRegistry,
    RuntimeProviders,
    SessionBuilderRegistry,
)
from perception_service.echo_adapter import EchoAdapter, EchoServicePlugin


def _bundle(root: Path) -> Path:
    root.mkdir()
    marker = root / "identity.txt"
    marker.write_text("dummy-echo", encoding="utf-8")
    entry = BundleFile(path=marker.name)
    manifest = {
        "schema_version": 3,
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
            "interface": "tensor_model",
            "model_type": "dummy_echo",
            "operation": "echo",
            "inputs": [{"semantic": "echo.input", "dtype": "float32", "shape": [2]}],
            "outputs": [{"semantic": "echo.output", "dtype": "float32", "shape": [2]}],
        },
        "deployments": {
            "cpu": {
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


@pytest.fixture
def runtime_dependencies():
    registry_set = RegistrySet(
        BackendRegistry(STATIC_BACKEND_DESCRIPTORS),
        SessionBuilderRegistry(),
        RuntimeAssemblerRegistry(),
    ).freeze()
    providers = RuntimeProviders.create(object(), ResourceDomainAdmissions())
    yield RuntimeDependencies(registry_set, providers)
    providers.close()


def _plugin(validated, options, dependencies):
    return EchoServicePlugin(
        None,
        validated,
        options,
        registry_set=dependencies.registry_set,
        providers=dependencies.providers,
    )


def test_echo_plugin_preserves_typed_request_and_runtime_identity(tmp_path: Path, runtime_dependencies):
    validated = load_inference_manifest(_bundle(tmp_path / "echo"), "cpu")
    plugin = _plugin(validated, {}, runtime_dependencies)
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


def test_echo_plugin_declares_a_generated_service_contract(tmp_path: Path, runtime_dependencies):
    from ibrobot_msgs.srv import EchoModel

    validated = load_inference_manifest(_bundle(tmp_path / "echo"), "cpu")
    _validate_service_contract(EchoModel)
    plugin = _instantiate_plugin(
        EchoServicePlugin,
        "ibrobot_msgs/srv/EchoModel",
        None,
        validated,
        {},
        registry_set=runtime_dependencies.registry_set,
        providers=runtime_dependencies.providers,
    )

    plugin.close()


def test_echo_adapter_and_plugin_fail_closed(tmp_path: Path, runtime_dependencies):
    adapter = EchoAdapter()
    with pytest.raises(ValueError, match="two finite values"):
        adapter.preprocess([np.nan, 1.0])

    validated = load_inference_manifest(_bundle(tmp_path / "echo"), "cpu")
    with pytest.raises(ValueError, match="does not accept runtime options"):
        _plugin(validated, {"device": "cuda"}, runtime_dependencies)
