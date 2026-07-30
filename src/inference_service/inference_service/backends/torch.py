"""Native LeRobot execution on deployment-selected Torch devices."""

from __future__ import annotations

import gc
import importlib
import inspect
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext, suppress
from typing import Any

from inference_manifest import TorchDeployment
from inference_service.backends.errors import BackendInferenceError, BackendLoadError
from inference_service.backends.lifecycle import LifecycleBackend, PartialLoadRollback
from inference_service.backends.types import (
    BackendAdmissionEvidence,
    BackendCapabilities,
    BackendResult,
    InferenceRequest,
    RuntimeContext,
)
from inference_service.lerobot_assets import (
    TOKENIZER_REFERENCE_KEYS,
    VLM_REFERENCE_KEYS,
    resolve_local_semantic_reference,
)

_ACTION_METHODS = frozenset({"predict_action_chunk", "select_action"})
_NOISE_KEYS = ("_noise", "action.noise", "noise")
_MODEL_DTYPES = {
    "native": None,
    "fp16": "half",
    "bf16": "bfloat16",
    "fp32": "float",
}


class TorchBackend(LifecycleBackend):
    """Load and execute one native LeRobot policy without rewriting its bundle."""

    def __init__(self, device_name: str, *, expose_hardware_identity: bool = False) -> None:
        super().__init__(
            "torch",
            BackendCapabilities(
                thread_safe=False,
                max_in_flight_per_instance=1,
                supports_multiple_instances=True,
                hardware_resource_id=f"torch:{device_name}" if expose_hardware_identity else None,
                resource_domain=f"torch:{device_name}",
                max_in_flight_per_resource_domain=1,
                supports_cancellation=False,
                admission_evidence=BackendAdmissionEvidence(
                    sdk_initialization=True,
                    multi_instance_execution=True,
                    failure_isolation=True,
                    independent_close=True,
                ),
            ),
        )
        self._configured_device_name = device_name
        self._torch: Any | None = None
        self._device: Any | None = None
        self._device_name: str | None = None
        self._policy: Any | None = None
        self._policy_config: Any | None = None
        self._preprocessor: Any | None = None
        self._postprocessor: Any | None = None
        self._context: RuntimeContext | None = None
        self._model_dtype = "native"

    @property
    def policy(self) -> Any | None:
        """Loaded policy for processor/pipeline integration and optional instrumentation."""

        return self._policy

    @property
    def policy_config(self) -> Any | None:
        return self._policy_config

    @property
    def preprocessor(self) -> Any | None:
        return self._preprocessor

    @property
    def postprocessor(self) -> Any | None:
        return self._postprocessor

    def _load(self, context: RuntimeContext, rollback: PartialLoadRollback) -> None:
        deployment = context.deployment
        if not isinstance(deployment, TorchDeployment):
            raise BackendLoadError(
                f"TorchBackend requires a native Torch deployment, got {type(deployment).__name__}",
                code="invalid_deployment",
            )
        if deployment.device != self._configured_device_name:
            raise BackendLoadError(
                f"TorchBackend was created for device {self._configured_device_name!r}, got {deployment.device!r}",
                code="deployment_context_mismatch",
            )
        model_dtype = self._validate_runtime_options(context.runtime_options)

        torch_module = self._import_required("torch", "PyTorch")
        self._validate_device(torch_module, deployment.device)
        if deployment.device == "npu":
            self._import_required("torch_npu", "torch_npu")
            self._validate_npu(torch_module)

        try:
            device = torch_module.device(deployment.device)
        except Exception as exc:
            raise BackendLoadError(
                f"cannot construct Torch device {deployment.device!r}: {exc}",
                code="invalid_device",
            ) from exc

        pretrained_config_type = self._import_attribute(
            "lerobot.configs.policies", "PreTrainedConfig", "LeRobot policy configuration"
        )
        factory_module = self._import_required("lerobot.policies.factory", "LeRobot policy factory")
        get_policy_class = self._require_attribute(factory_module, "get_policy_class", "LeRobot policy factory")
        make_processors = self._require_attribute(
            factory_module, "make_pre_post_processors", "LeRobot processor factory"
        )

        bundle_path = str(context.validated_manifest.bundle_root)
        policy_type = context.policy.policy_type
        policy_config = pretrained_config_type.from_pretrained(bundle_path, local_files_only=True)
        try:
            policy_config.device = deployment.device
        except (AttributeError, TypeError) as exc:
            raise BackendLoadError(
                "loaded LeRobot policy config does not permit supported in-memory runtime device placement",
                code="incompatible_policy_config",
            ) from exc
        try:
            local_vlm_path = resolve_local_semantic_reference(
                context.validated_manifest.bundle_root,
                "config.json",
                VLM_REFERENCE_KEYS,
            )
            local_tokenizer_path = resolve_local_semantic_reference(
                context.validated_manifest.bundle_root,
                "policy_preprocessor.json",
                TOKENIZER_REFERENCE_KEYS,
            )
        except ValueError as exc:
            raise BackendLoadError(str(exc), code="invalid_policy_assets") from exc
        if local_vlm_path is not None:
            try:
                policy_config.vlm_model_name = local_vlm_path
            except (AttributeError, TypeError) as exc:
                raise BackendLoadError(
                    "loaded LeRobot policy config does not permit in-memory VLM asset resolution",
                    code="incompatible_policy_config",
                ) from exc
        policy_class = get_policy_class(policy_type)
        policy = policy_class.from_pretrained(
            bundle_path,
            config=policy_config,
            local_files_only=True,
        )

        self._torch = torch_module
        self._device = device
        self._device_name = deployment.device
        self._policy = policy
        self._policy_config = policy_config
        self._context = context
        rollback.defer(self._release_loaded_objects)

        moved_policy = policy.to(device)
        if moved_policy is not None:
            self._policy = moved_policy
            policy = moved_policy
        self._cast_model(policy, model_dtype)
        evaluated_policy = policy.eval()
        if evaluated_policy is not None:
            self._policy = evaluated_policy
            policy = evaluated_policy
        self._model_dtype = model_dtype

        preprocessor_overrides = {"device_processor": {"device": str(device)}}
        if local_tokenizer_path is not None:
            preprocessor_overrides["tokenizer_processor"] = {"tokenizer_name": local_tokenizer_path}
        postprocessor_overrides = {"device_processor": {"device": str(device)}}
        preprocessor, postprocessor = make_processors(
            policy_cfg=policy_config,
            pretrained_path=bundle_path,
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides=postprocessor_overrides,
        )
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor

        resettable = callable(getattr(policy, "reset", None))
        self._update_loaded_capabilities(
            resettable=resettable,
            stateful=resettable,
            supports_attention=self._observes_attention(policy),
            supports_cancellation=False,
        )

    def _infer(self, request: InferenceRequest) -> BackendResult:
        policy = self._policy
        torch_module = self._torch
        if policy is None or torch_module is None or self._device is None or self._context is None:
            raise BackendInferenceError("TorchBackend is not fully loaded", code="runtime_not_loaded")

        batch = {key: self._move_input(value) for key, value in request.inputs.items()}
        action_method = request.metadata.get("action_method", "predict_action_chunk")
        if not isinstance(action_method, str) or action_method not in _ACTION_METHODS:
            raise BackendInferenceError(
                f"unsupported native action method {action_method!r}; expected one of {sorted(_ACTION_METHODS)}",
                code="invalid_action_method",
            )
        callback = getattr(policy, action_method, None)
        if not callable(callback):
            raise BackendInferenceError(
                f"loaded policy does not implement {action_method}",
                code="unsupported_action_method",
            )

        kwargs: dict[str, object] = {}
        noise = self._pop_noise(batch)
        if noise is not None and self._accepts_keyword(callback, "noise"):
            kwargs["noise"] = self._place_noise(noise)

        inference_context = getattr(torch_module, "inference_mode", None)
        manager = inference_context() if callable(inference_context) else nullcontext()
        start = time.perf_counter()
        with manager:
            action = callback(batch, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        action = self._remove_batch_dimension(action)
        actual_chunk_size = self._chunk_size(action, action_method)
        return BackendResult(
            action=action,
            actual_chunk_size=actual_chunk_size,
            backend_latency_ms=latency_ms,
            metadata={
                "request_id": request.request_id,
                "policy_type": self._context.policy.policy_type,
                "device": self._device_name,
                "action_method": action_method,
                "deployment_name": self._context.deployment_name,
                "deployment_fingerprint": self._context.deployment_fingerprint,
                "external_noise": noise is not None and "noise" in kwargs,
                "model_dtype": self._model_dtype,
            },
        )

    def _reset(self) -> None:
        reset = getattr(self._policy, "reset", None)
        if not callable(reset):
            super()._reset()
        reset()

    def _close(self) -> None:
        self._release_loaded_objects()

    def _release_loaded_objects(self) -> None:
        torch_module = self._torch
        device_name = self._device_name
        self._policy = None
        self._policy_config = None
        self._preprocessor = None
        self._postprocessor = None
        self._context = None
        self._device = None
        self._device_name = None
        self._model_dtype = "native"
        self._torch = None
        gc.collect()
        if torch_module is None or device_name is None:
            return
        cache_owner = getattr(torch_module, device_name, None)
        empty_cache = getattr(cache_owner, "empty_cache", None)
        if callable(empty_cache):
            with suppress(Exception):
                empty_cache()

    @staticmethod
    def _import_required(module_name: str, description: str) -> Any:
        try:
            return importlib.import_module(module_name)
        except (ImportError, OSError) as exc:
            raise BackendLoadError(
                f"{description} dependency {module_name!r} is unavailable: {exc}",
                code="missing_dependency",
            ) from exc

    @classmethod
    def _import_attribute(cls, module_name: str, attribute: str, description: str) -> Any:
        module = cls._import_required(module_name, description)
        return cls._require_attribute(module, attribute, description)

    @staticmethod
    def _require_attribute(module: Any, attribute: str, description: str) -> Any:
        try:
            value = getattr(module, attribute)
        except AttributeError as exc:
            raise BackendLoadError(
                f"{description} does not expose required attribute {attribute!r}",
                code="incompatible_dependency",
            ) from exc
        if not callable(value):
            raise BackendLoadError(
                f"{description} attribute {attribute!r} is not callable",
                code="incompatible_dependency",
            )
        return value

    @staticmethod
    def _validate_device(torch_module: Any, device: str) -> None:
        if device == "cpu" or device == "npu":
            return
        if device == "cuda":
            available = callable(getattr(getattr(torch_module, "cuda", None), "is_available", None)) and bool(
                torch_module.cuda.is_available()
            )
        elif device == "mps":
            mps = getattr(getattr(torch_module, "backends", None), "mps", None)
            available = callable(getattr(mps, "is_available", None)) and bool(mps.is_available())
        else:
            available = False
        if not available:
            raise BackendLoadError(
                f"Torch deployment device {device!r} is not available on this host",
                code="device_unavailable",
            )

    @staticmethod
    def _validate_npu(torch_module: Any) -> None:
        npu = getattr(torch_module, "npu", None)
        is_available = getattr(npu, "is_available", None)
        if not callable(is_available) or not is_available():
            raise BackendLoadError(
                "Torch deployment device 'npu' is not available after importing torch_npu",
                code="device_unavailable",
            )

    @staticmethod
    def _validate_runtime_options(options: Mapping[str, object]) -> str:
        unknown = sorted(set(options) - {"model_dtype"})
        if unknown:
            raise BackendLoadError(f"unknown Torch runtime options: {unknown}", code="invalid_runtime_options")
        model_dtype = options.get("model_dtype", "native")
        if not isinstance(model_dtype, str) or model_dtype not in _MODEL_DTYPES:
            raise BackendLoadError(
                f"unsupported Torch model_dtype {model_dtype!r}; expected one of {sorted(_MODEL_DTYPES)}",
                code="invalid_runtime_options",
            )
        return model_dtype

    @staticmethod
    def _cast_model(policy: object, model_dtype: str) -> None:
        method_name = _MODEL_DTYPES[model_dtype]
        if method_name is None:
            return
        model = getattr(policy, "model", None)
        cast = getattr(model, method_name, None)
        if not callable(cast):
            raise BackendLoadError(
                f"loaded LeRobot policy model does not support model_dtype {model_dtype!r}",
                code="unsupported_model_dtype",
            )
        converted = cast()
        if converted is not None:
            try:
                policy.model = converted
            except (AttributeError, TypeError) as exc:
                raise BackendLoadError(
                    "loaded LeRobot policy does not permit replacing its converted model",
                    code="incompatible_policy_model",
                ) from exc

    def _move_input(self, value: object) -> object:
        if value is None or isinstance(value, str | bytes | Mapping):
            return value
        if (
            isinstance(value, Sequence)
            and not isinstance(value, str | bytes)
            and (not value or isinstance(value[0], str | bytes | Mapping))
        ):
            return value

        torch_module = self._torch
        is_tensor = getattr(torch_module, "is_tensor", None)
        if callable(is_tensor) and is_tensor(value):
            tensor = value
        else:
            as_tensor = getattr(torch_module, "as_tensor", None)
            if not callable(as_tensor):
                return value
            try:
                tensor = as_tensor(value)
            except (TypeError, ValueError):
                return value
        move = getattr(tensor, "to", None)
        return move(self._device) if callable(move) else tensor

    @staticmethod
    def _accepts_keyword(callback: Any, keyword: str) -> bool:
        try:
            parameters = inspect.signature(callback).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD or parameter.name == keyword for parameter in parameters
        )

    @staticmethod
    def _pop_noise(batch: dict[str, object]) -> object | None:
        noise: object | None = None
        for key in _NOISE_KEYS:
            if key not in batch:
                continue
            value = batch.pop(key)
            if noise is None:
                noise = value
        return noise

    def _place_noise(self, noise: object) -> object:
        action_projection = getattr(getattr(self._policy, "model", None), "action_in_proj", None)
        parameter = getattr(action_projection, "weight", None)
        parameters = getattr(self._policy, "parameters", None)
        if parameter is None and callable(parameters):
            try:
                parameter = next(iter(parameters()))
            except (StopIteration, TypeError):
                parameter = None
        if parameter is not None:
            move = getattr(noise, "to", None)
            if callable(move):
                return move(device=getattr(parameter, "device", self._device), dtype=getattr(parameter, "dtype", None))
        move = getattr(noise, "to", None)
        return move(self._device) if callable(move) else noise

    @staticmethod
    def _remove_batch_dimension(action: object) -> object:
        shape = getattr(action, "shape", ())
        squeeze = getattr(action, "squeeze", None)
        if len(shape) >= 2 and shape[0] == 1 and callable(squeeze):
            return squeeze(0)
        return action

    @staticmethod
    def _chunk_size(action: object, action_method: str) -> int:
        if action_method == "select_action":
            return 1
        shape = getattr(action, "shape", ())
        if len(shape) < 2 or int(shape[-2]) < 1:
            raise BackendInferenceError(
                f"predict_action_chunk returned invalid action shape {tuple(shape)}",
                code="invalid_action_shape",
            )
        return int(shape[-2])

    @staticmethod
    def _observes_attention(policy: object) -> bool:
        for owner in (policy, getattr(policy, "model", None)):
            if owner is None:
                continue
            declared = getattr(owner, "supports_attention", None)
            if isinstance(declared, bool):
                return declared
            if any(
                hasattr(owner, attribute) for attribute in ("last_attn_weights", "get_attention_maps", "get_attentions")
            ):
                return True
            decoder_layers = getattr(getattr(owner, "decoder", None), "layers", ())
            if any(hasattr(layer, "multihead_attn") for layer in decoder_layers):
                return True
        return False


def create_backend(context: RuntimeContext) -> TorchBackend:
    """Lazy registry factory for native Torch execution."""

    deployment = context.deployment
    if not isinstance(deployment, TorchDeployment):
        raise BackendLoadError(
            f"TorchBackend requires a native Torch deployment, got {type(deployment).__name__}",
            code="invalid_deployment",
        )
    return TorchBackend(deployment.device, expose_hardware_identity=context.priority_scheduling)
