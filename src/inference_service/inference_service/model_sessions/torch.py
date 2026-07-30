"""Generic callable Torch model execution without LeRobot dependencies."""

from __future__ import annotations

import gc
import importlib
from collections.abc import Callable, Mapping
from contextlib import nullcontext, suppress
from typing import Any

from inference_manifest import TorchDeployment
from inference_service.backends.errors import BackendError, BackendInferenceError, BackendLoadError
from inference_service.backends.lifecycle import PartialLoadRollback
from inference_service.backends.types import BackendAdmissionEvidence, BackendCapabilities, RuntimeContext
from inference_service.generic_runtime import NamedTensorRequest
from inference_service.model_sessions.base import ModelSession


class TorchModelSession(ModelSession):
    """Run a caller-owned callable module on a validated CPU or CUDA deployment."""

    def __init__(
        self,
        module_loader: Callable[[RuntimeContext], object],
        *,
        torch_loader: Callable[[], Any] | None = None,
    ) -> None:
        if not callable(module_loader):
            raise TypeError("module_loader must be callable")
        super().__init__(
            "model-session:torch",
            BackendCapabilities(
                max_in_flight_per_instance=1,
                supports_multiple_instances=True,
                admission_evidence=BackendAdmissionEvidence(
                    sdk_initialization=True,
                    multi_instance_execution=True,
                    failure_isolation=True,
                    independent_close=True,
                ),
            ),
        )
        self._module_loader = module_loader
        self._torch_loader = torch_loader or self._import_torch
        self._torch: Any | None = None
        self._device: Any | None = None
        self._device_name: str | None = None
        self._module: object | None = None

    @property
    def runtime_version(self) -> str:
        return self._runtime_version(self._torch)

    def _load(self, context: RuntimeContext, rollback: PartialLoadRollback) -> None:
        deployment = context.deployment
        if not isinstance(deployment, TorchDeployment) or deployment.device not in {"cpu", "cuda"}:
            raise BackendLoadError(
                "TorchModelSession requires a validated CPU or CUDA Torch deployment", code="invalid_deployment"
            )
        unknown_options = sorted(context.runtime_options)
        if unknown_options:
            raise BackendLoadError(
                f"unknown Torch model-session options: {unknown_options}", code="invalid_runtime_options"
            )

        torch_module = self._torch_loader()
        if deployment.device == "cuda":
            is_available = getattr(getattr(torch_module, "cuda", None), "is_available", None)
            if not callable(is_available) or not is_available():
                raise BackendLoadError("Torch CUDA device is unavailable", code="device_unavailable")
        try:
            device = torch_module.device(deployment.device)
            module = self._module_loader(context)
        except BackendError:
            raise
        except Exception as exc:
            raise BackendLoadError(f"unable to load callable Torch module: {exc}") from exc
        if not callable(module):
            raise BackendLoadError("Torch module loader did not return a callable", code="invalid_model")

        self._torch = torch_module
        self._device = device
        self._device_name = deployment.device
        self._module = module
        rollback.defer(self._release)
        move = getattr(module, "to", None)
        if callable(move):
            moved = move(device)
            if moved is not None:
                self._module = moved
                module = moved
        evaluate = getattr(module, "eval", None)
        if callable(evaluate):
            evaluated = evaluate()
            if evaluated is not None:
                self._module = evaluated

    def _execute(self, request: NamedTensorRequest) -> Mapping[str, object]:
        module = self._module
        torch_module = self._torch
        if module is None or torch_module is None:
            raise BackendInferenceError("Torch model session is not loaded", code="runtime_not_loaded")
        inputs = {name: self._move(value) for name, value in request.inputs.items()}
        inference_mode = getattr(torch_module, "inference_mode", None)
        manager = inference_mode() if callable(inference_mode) else nullcontext()
        with manager:
            outputs = module(inputs)
        descriptors = self._require_context().validated_manifest.manifest.model.outputs
        if not isinstance(outputs, Mapping):
            if len(descriptors) != 1:
                raise BackendInferenceError(
                    "callable Torch module must return a named mapping for multiple outputs",
                    code="invalid_model_output",
                )
            outputs = {descriptors[0].semantic: outputs}
        return {name: self._to_host(value) for name, value in outputs.items()}

    def _close(self) -> None:
        self._release()

    def _release(self) -> None:
        torch_module = self._torch
        device_name = self._device_name
        self._module = None
        self._device = None
        self._device_name = None
        self._torch = None
        gc.collect()
        if torch_module is not None and device_name == "cuda":
            empty_cache = getattr(getattr(torch_module, "cuda", None), "empty_cache", None)
            if callable(empty_cache):
                with suppress(Exception):
                    empty_cache()

    def _move(self, value: object) -> object:
        torch_module = self._torch
        is_tensor = getattr(torch_module, "is_tensor", None)
        tensor = value if callable(is_tensor) and is_tensor(value) else torch_module.as_tensor(value)
        move = getattr(tensor, "to", None)
        return move(self._device) if callable(move) else tensor

    @staticmethod
    def _to_host(value: object) -> object:
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        numpy = getattr(value, "numpy", None)
        return numpy() if callable(numpy) else value

    @staticmethod
    def _import_torch() -> Any:
        try:
            return importlib.import_module("torch")
        except (ImportError, OSError) as exc:
            raise BackendLoadError(
                f"PyTorch dependency 'torch' is unavailable: {exc}", code="missing_dependency"
            ) from exc
