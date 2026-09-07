"""Generic host for strongly typed model-service plugins."""

from __future__ import annotations

import importlib
import inspect
import json
import time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from ibrobot_msgs.msg import ModelRuntimeInfo
from inference_manifest import (
    canonical_semantic_identity_json,
    load_inference_manifest,
    semantic_identity_fingerprint,
)
from inference_service.runtime_composition import (
    build_model_service_runtime_dependencies,
    require_runtime_dependencies,
)
from inference_service.unified_runtime import RegistrySet, RuntimeProviders

from .model_service_plugin import ModelServiceError, ModelServicePlugin, PluginRuntimeStatus


def _load_class(path: str):
    try:
        module_name, class_name = path.split(":", 1)
        return getattr(importlib.import_module(module_name), class_name)
    except (AttributeError, ImportError, ValueError) as exc:
        raise RuntimeError(f"cannot load model service plugin {path!r}: {exc}") from exc


def _load_service_type(type_name: str):
    try:
        package, marker, name = type_name.split("/", 2)
        if marker != "srv":
            raise ValueError("middle component must be 'srv'")
        return getattr(importlib.import_module(f"{package}.srv"), name)
    except (AttributeError, ImportError, ValueError) as exc:
        raise RuntimeError(f"cannot load ROS service type {type_name!r}: {exc}") from exc


def _validate_service_contract(service_type) -> None:
    response_type = getattr(service_type, "Response", None)
    fields_method = getattr(response_type, "get_fields_and_field_types", None)
    if not callable(fields_method):
        raise RuntimeError("configured ROS service type has no generated Response contract")
    required = {"model", "inference_time_ms", "success", "message"}
    missing = sorted(required - set(fields_method()))
    if missing:
        raise RuntimeError(f"model service response is missing common diagnostic fields: {missing}")


def _instantiate_plugin(
    plugin_class,
    service_type_name: str,
    host,
    validated,
    options,
    *,
    registry_set: RegistrySet | None = None,
    providers: RuntimeProviders | None = None,
) -> ModelServicePlugin:
    if not isinstance(plugin_class, type) or not issubclass(plugin_class, ModelServicePlugin):
        raise RuntimeError("configured plugin must implement ModelServicePlugin")
    if plugin_class.service_type != service_type_name:
        raise RuntimeError(
            f"plugin declares service type {plugin_class.service_type!r}, configured {service_type_name!r}"
        )
    try:
        signature = inspect.signature(plugin_class)
    except (TypeError, ValueError):
        signature = None
    kwargs = {}
    if signature is None:
        kwargs = {"registry_set": registry_set, "providers": providers}
    else:
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
        )
        if accepts_kwargs or "registry_set" in signature.parameters:
            kwargs["registry_set"] = registry_set
        if accepts_kwargs or "providers" in signature.parameters:
            kwargs["providers"] = providers
    return plugin_class(host, validated, options, **kwargs)


def _runtime_info(
    instance_id: str,
    validated,
    status: PluginRuntimeStatus,
    configuration_generation: int = 0,
) -> ModelRuntimeInfo:
    result = ModelRuntimeInfo()
    result.instance_id = instance_id
    result.configuration_generation = configuration_generation
    if validated is not None:
        manifest = validated.manifest
        result.model_name = manifest.bundle.name
        result.model_version = str(manifest.bundle.revision)
        result.manifest_fingerprint = manifest.bundle.digest.value
        result.deployment_name = validated.deployment_name
        result.deployment_fingerprint = validated.fingerprint
        result.backend = validated.deployment.backend
        identity = manifest.model.semantic_identity
        if identity is not None:
            result.semantic_identity_json = canonical_semantic_identity_json(identity)
            result.semantic_identity_fingerprint = semantic_identity_fingerprint(identity)
            if identity.embedding is not None:
                result.embedding_space_id = identity.embedding.embedding_space_id
                result.embedding_dimension = identity.embedding.dimension
                result.normalization = identity.embedding.normalization
    result.runtime_state = status.state
    result.failure_reason = status.failure_reason
    result.ready = status.ready
    result.message = status.failure_reason
    if "runtime_version" in status.metadata:
        result.runtime_version = str(status.metadata["runtime_version"])
    return result


class ModelServiceNode(Node):
    """Load one configured plugin without knowing its model family or service shape."""

    def __init__(
        self,
        *,
        registry_set: RegistrySet | None = None,
        providers: RuntimeProviders | None = None,
    ):
        super().__init__("model_service")
        registry_set, providers = require_runtime_dependencies(
            registry_set,
            providers,
            owner=type(self).__name__,
        )
        self._registry_set = registry_set
        self._providers = providers
        for name, default in (
            ("instance_id", ""),
            ("bundle_path", ""),
            ("deployment", ""),
            ("adapter_class", ""),
            ("service_type", ""),
            ("service_endpoint", ""),
            ("runtime_options_json", "{}"),
        ):
            self.declare_parameter(name, default)
        self.declare_parameter("required", False)
        self.declare_parameter("require_semantic_identity", False)
        self.declare_parameter("configuration_generation", 0)
        self.validated_manifest = None
        self.plugin = None
        self.failure_reason = "model service has not initialized"

        service_type_name = str(self.get_parameter("service_type").value)
        endpoint = str(self.get_parameter("service_endpoint").value)
        service_type = _load_service_type(service_type_name)
        _validate_service_contract(service_type)
        self._initialize_plugin(service_type_name)
        self.create_service(
            service_type,
            endpoint,
            self._dispatch,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    def _initialize_plugin(self, service_type_name: str) -> None:
        try:
            options = json.loads(str(self.get_parameter("runtime_options_json").value))
            if not isinstance(options, dict):
                raise ValueError("runtime_options_json must contain a JSON object")
            validated_manifest = load_inference_manifest(
                str(self.get_parameter("bundle_path").value),
                str(self.get_parameter("deployment").value),
            )
            if (
                bool(self.get_parameter("require_semantic_identity").value)
                and validated_manifest.manifest.model.semantic_identity is None
            ):
                raise RuntimeError("selected model manifest does not declare semantic_identity")
            self.validated_manifest = validated_manifest
            plugin_class = _load_class(str(self.get_parameter("adapter_class").value))
            self.plugin = _instantiate_plugin(
                plugin_class,
                service_type_name,
                self,
                self.validated_manifest,
                options,
                registry_set=getattr(self, "_registry_set", None),
                providers=getattr(self, "_providers", None),
            )
            self.failure_reason = ""
        except Exception as exc:  # Keep the typed service alive to report non-readiness.
            self.failure_reason = str(exc)
            self.get_logger().error(f"Model service initialization failed: {exc}")
            if bool(self.get_parameter("required").value):
                raise

    def _dispatch(self, request, response):
        started = time.perf_counter()
        if self.plugin is None:
            response.success = False
            response.message = self.failure_reason
            if hasattr(response, "error_code"):
                response.error_code = "MODEL_NOT_READY"
        else:
            status = None
            try:
                status = self.plugin.runtime_status()
                if not status.ready:
                    raise RuntimeError(status.failure_reason or f"model runtime is {status.state}")
                response.message = self.plugin.handle(request, response)
                response.success = True
                self.failure_reason = ""
            except Exception as exc:
                response.success = False
                response.message = str(exc)
                if hasattr(response, "error_code"):
                    response.error_code = (
                        "MODEL_NOT_READY" if status is not None and not status.ready else "INFERENCE_FAILED"
                    )
                if isinstance(exc, ModelServiceError):
                    for name, value in exc.response_fields.items():
                        if hasattr(response, name):
                            setattr(response, name, value)
                elif hasattr(response, "error_code"):
                    response.error_code = "INFERENCE_FAILED"
        response.model = self.runtime_info()
        response.inference_time_ms = (time.perf_counter() - started) * 1000.0
        return response

    def runtime_info(self) -> ModelRuntimeInfo:
        if self.plugin is None or self.failure_reason:
            status = PluginRuntimeStatus(state="failed", ready=False, failure_reason=self.failure_reason)
        else:
            try:
                status = self.plugin.runtime_status()
            except Exception as exc:
                self.failure_reason = str(exc)
                status = PluginRuntimeStatus(state="failed", ready=False, failure_reason=self.failure_reason)
        return _runtime_info(
            str(self.get_parameter("instance_id").value),
            self.validated_manifest,
            status,
            int(self.get_parameter("configuration_generation").value),
        )

    def destroy_node(self):
        if self.plugin is not None:
            self.plugin.close()
        self.plugin = None
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    dependencies = build_model_service_runtime_dependencies()
    node = None
    try:
        node = ModelServiceNode(
            registry_set=dependencies.registry_set,
            providers=dependencies.providers,
        )
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        dependencies.providers.close()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
