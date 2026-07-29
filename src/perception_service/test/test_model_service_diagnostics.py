from types import SimpleNamespace

import pytest

from perception_service.model_service_node import (
    ModelServiceNode,
    _instantiate_plugin,
    _runtime_info,
    _validate_service_contract,
)
from perception_service.model_service_plugin import ModelServicePlugin, PluginRuntimeStatus


def _validated_manifest():
    digest = "a" * 64
    return SimpleNamespace(
        manifest=SimpleNamespace(
            bundle=SimpleNamespace(name="depth-model", revision=3, digest=SimpleNamespace(value=digest))
        ),
        deployment_name="compiled-target",
        fingerprint="b" * 64,
        deployment=SimpleNamespace(backend="compiled"),
    )


def test_runtime_info_reports_family_neutral_manifest_and_deployment_identity():
    info = _runtime_info(
        "depth-primary",
        _validated_manifest(),
        PluginRuntimeStatus(
            state="ready",
            ready=True,
            metadata={"runtime_version": "1.2", "backend": "spoofed", "ready": False},
        ),
    )

    assert info.instance_id == "depth-primary"
    assert info.model_name == "depth-model"
    assert info.model_version == "3"
    assert info.manifest_fingerprint == "a" * 64
    assert info.deployment_name == "compiled-target"
    assert info.deployment_fingerprint == "b" * 64
    assert info.backend == "compiled"
    assert info.runtime_state == "ready"
    assert info.ready
    assert not info.failure_reason
    assert info.runtime_version == "1.2"


def test_runtime_info_reports_initialization_failure_without_model_assumptions():
    info = _runtime_info(
        "depth-primary",
        None,
        PluginRuntimeStatus(state="failed", ready=False, failure_reason="adapter failed to load"),
    )

    assert info.runtime_state == "failed"
    assert not info.ready
    assert info.failure_reason == "adapter failed to load"
    assert info.message == "adapter failed to load"
    assert not info.manifest_fingerprint
    assert not info.deployment_fingerprint


@pytest.mark.parametrize("required", [False, True])
def test_plugin_initialization_honors_required_failure_policy(monkeypatch, required):
    class Logger:
        def error(self, _message):
            return None

    parameters = {
        "required": required,
        "runtime_options_json": "{}",
        "bundle_path": "/missing",
        "deployment": "cpu",
        "adapter_class": "missing:Plugin",
    }
    host = SimpleNamespace(
        validated_manifest=None,
        plugin=None,
        failure_reason="not initialized",
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        get_logger=lambda: Logger(),
    )
    monkeypatch.setattr(
        "perception_service.model_service_node.load_inference_manifest",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("bundle unavailable")),
    )

    if required:
        with pytest.raises(RuntimeError, match="bundle unavailable"):
            ModelServiceNode._initialize_plugin(host, "test_msgs/srv/Echo")
    else:
        ModelServiceNode._initialize_plugin(host, "test_msgs/srv/Echo")
        assert host.plugin is None
        assert host.failure_reason == "bundle unavailable"


class _Plugin(ModelServicePlugin):
    service_type = "test_msgs/srv/Echo"

    def __init__(self, host, validated, options):
        self.values = host, validated, options

    def handle(self, request, response) -> str:
        return "ok"

    def runtime_status(self) -> PluginRuntimeStatus:
        return PluginRuntimeStatus(state="ready", ready=True)

    def close(self) -> None:
        return None


def test_plugin_instantiation_requires_protocol_and_matching_service_type():
    plugin = _instantiate_plugin(_Plugin, _Plugin.service_type, "host", "manifest", {"value": 1})
    assert plugin.values == ("host", "manifest", {"value": 1})

    with pytest.raises(RuntimeError, match="must implement ModelServicePlugin"):
        _instantiate_plugin(object, _Plugin.service_type, None, None, {})
    with pytest.raises(RuntimeError, match="declares service type"):
        _instantiate_plugin(_Plugin, "test_msgs/srv/Other", None, None, {})


def test_typed_service_response_requires_common_diagnostic_envelope():
    class CompleteResponse:
        @staticmethod
        def get_fields_and_field_types():
            return {
                "value": "float[]",
                "model": "msg",
                "inference_time_ms": "float",
                "success": "bool",
                "message": "string",
            }

    class CompleteService:
        Response = CompleteResponse

    class IncompleteResponse:
        @staticmethod
        def get_fields_and_field_types():
            return {"success": "bool", "message": "string"}

    class IncompleteService:
        Response = IncompleteResponse

    _validate_service_contract(CompleteService)
    with pytest.raises(RuntimeError, match="common diagnostic fields"):
        _validate_service_contract(IncompleteService)
