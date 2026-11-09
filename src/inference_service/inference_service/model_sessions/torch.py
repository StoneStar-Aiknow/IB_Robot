"""Generic callable Torch model execution without LeRobot dependencies."""

from __future__ import annotations

import gc
import importlib
from collections.abc import Callable, Mapping
from contextlib import nullcontext, suppress
from typing import Any

from inference_manifest import TorchRuntimeProfile
from inference_service.backends.errors import (
    BackendError,
    BackendInferenceError,
    BackendLifecycleError,
    BackendLoadError,
)
from inference_service.backends.types import BackendAdmissionEvidence, BackendCapabilities, RuntimeContext
from inference_service.model_sessions.base import ModelSession
from inference_service.model_sessions.torch_completion import (
    drain_torch_device_before_release,
    resolve_torch_device,
    torch_execution_scope,
)
from inference_service.unified_runtime import ExecutionContext, LoadRollback, ModelRequest
from inference_service.unified_runtime.errors import CancellationRequested, DeadlineExceeded
from inference_service.unified_runtime.lifecycle import check_lifecycle


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
        self._release_drain_failures = 0

    @property
    def runtime_version(self) -> str:
        return self._runtime_version(self._torch)

    def _load(self, context: RuntimeContext, rollback: LoadRollback) -> None:
        profile = context.backend_profile
        if (
            context.backend != "torch"
            or context.interface != "tensor_model"
            or not isinstance(profile, TorchRuntimeProfile)
            or profile.device not in {"cpu", "cuda", "npu"}
        ):
            raise BackendLoadError(
                "TorchModelSession requires a tensor_model deployment with a typed cpu/cuda/npu Torch profile",
                code="invalid_deployment",
            )
        unknown_options = sorted(context.runtime_options)
        if unknown_options:
            raise BackendLoadError(
                f"unknown Torch model-session options: {unknown_options}", code="invalid_runtime_options"
            )

        torch_module = self._torch_loader()
        device_name = profile.device
        if device_name == "npu":
            # Importing torch_npu is what registers the device, so it has to
            # happen before `is_available` can answer; the accelerator is
            # otherwise reported as absent.
            try:
                importlib.import_module("torch_npu")
            except ImportError as exc:
                raise BackendLoadError(
                    f"torch_npu is required for a Torch npu deployment: {exc}",
                    code="missing_dependency",
                ) from exc
        if device_name in {"cuda", "npu"}:
            accelerator = getattr(torch_module, device_name, None)
            is_available = getattr(accelerator, "is_available", None)
            if not callable(is_available) or not is_available():
                raise BackendLoadError(f"Torch {device_name} device is unavailable", code="device_unavailable")
        try:
            device = resolve_torch_device(torch_module, device_name)
        except BackendError:
            raise
        except Exception as exc:
            raise BackendLoadError(
                f"unable to resolve Torch device {device_name!r}: {exc}", code="device_unavailable"
            ) from exc
        self._torch = torch_module
        self._device = device
        self._device_name = device_name
        # Register the device drain before construction can enqueue native work.
        rollback.defer(self._release)
        try:
            self._module = module = self._module_loader(context)
        except BackendError:
            raise
        except Exception as exc:
            raise BackendLoadError(f"unable to load callable Torch module: {exc}") from exc
        if not callable(module):
            raise BackendLoadError("Torch module loader did not return a callable", code="invalid_model")
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

    def _execute(self, request: ModelRequest, context: ExecutionContext) -> Mapping[str, object]:
        with torch_execution_scope(
            self._torch,
            self._device_name,
            self._device,
            complete_on_success=self._require_context().managed_execution,
            stateful=self.capabilities.stateful,
        ):
            return self._execute_torch(request, context)

    def complete_failure(self, error: Exception) -> None:
        """Confirm direct device work when the enclosing pipeline discards a result."""
        if not self._require_context().managed_execution:
            from .torch_completion import complete_torch_failure

            complete_torch_failure(
                error,
                self._torch,
                self._device_name,
                self._device,
                discarded_result=True,
                stateful=self.capabilities.stateful,
            )

    def _check_completed_context(self, context: ExecutionContext, phase: str) -> None:
        from .torch_completion import check_torch_completed_context

        check_torch_completed_context(
            context,
            phase,
            self._torch,
            self._device_name,
            self._device,
            managed=self._require_context().managed_execution,
            stateful=self.capabilities.stateful,
        )

    def _execute_torch(self, request: ModelRequest, context: ExecutionContext) -> Mapping[str, object]:
        del context
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
        if torch_module is not None:
            # A failed device drain keeps completion unknown, so the references
            # are retained for a retry. A device stuck in a sticky error state
            # never completes; after bounded drain failures the release abandons
            # the wait, drops the references, and lets close terminate. Drain
            # failures during the load's own rollback path do not count: the
            # load error handling retries the release before any public close,
            # and those attempts must not consume the bound.
            try:
                check_lifecycle()
                drain_torch_device_before_release(
                    torch_module,
                    device_name,
                    self._device,
                    failed_attempts=self._release_drain_failures,
                    owner="Torch model session release",
                )
            except BackendLifecycleError:
                if not self._loading:
                    self._release_drain_failures += 1
                raise
            except (DeadlineExceeded, CancellationRequested) as exc:
                raise BackendLifecycleError(str(exc), close_pending=True) from exc
        self._module = None
        self._device = None
        self._device_name = None
        self._torch = None
        gc.collect()
        # Resolve the cache owner from the device name rather than branching per
        # accelerator, as LeRobotTorchModelSession._close does.
        if torch_module is not None and device_name in {"cuda", "npu"}:
            empty_cache = getattr(getattr(torch_module, device_name, None), "empty_cache", None)
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
