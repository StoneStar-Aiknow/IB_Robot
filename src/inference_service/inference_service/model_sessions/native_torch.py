"""Registered Torch resources for domain APIs that do not return named tensors.

This adapter deliberately does not subclass the tensor-validating ModelSession.
The registered model builder owns the domain input/output contract; the common
factory and Handle own assembly, loading failure retention and public lifecycle.
"""

from collections.abc import Callable, Mapping

from inference_manifest import ValidatedManifest, load_inference_manifest

from ..backends.errors import BackendLifecycleError
from ..backends.types import BackendCapabilities
from ..unified_runtime.contracts import OutcomeEvidence, OutcomeState
from ..unified_runtime.errors import CancellationRequested, DeadlineExceeded, clear_exception_frames
from ..unified_runtime.lifecycle import check_lifecycle, check_startup
from .torch_completion import (
    check_torch_completed_context,
    complete_torch_failure,
    drain_torch_device,
    resolve_torch_device,
    torch_device_scope,
    torch_execution_scope,
)


class NativeTorchResource:
    """Keep a domain SDK object private behind the ordinary executor protocol."""

    capabilities = BackendCapabilities()

    def __init__(
        self,
        spec,
        factory: Callable[[], object],
        execute: Callable[[object, Mapping[str, object]], object],
    ):
        """Load via factory(); wrap execute(model, inputs) in outputs["value"].

        The callback transfers output storage to the caller. Neither the SDK
        nor the callback may mutate it after return, even on another request or
        Close. Copy NumPy buffers or detach().clone() Torch buffers inside the
        callback when the SDK reuses them; a view/detach alone is not a copy.
        The callback owns any device-copy completion under the execution scope.
        """
        if not isinstance(spec, ValidatedManifest):
            raise TypeError("native Torch resources require a validated bundle manifest")
        if spec.deployment.backend != "torch":
            raise ValueError("native Torch resources require a Torch deployment")
        contract = spec.deployment.execution_contract
        if contract.stateful or contract.state_scope != "request" or contract.execution_structure != "direct":
            raise ValueError("native Torch resources require a stateless request-direct deployment")
        if contract.cancellation_granularity != "request_boundary":
            raise ValueError("native Torch callbacks support only request_boundary cancellation")
        if not callable(factory) or not callable(execute):
            raise TypeError("native Torch factory and execute must be callable")
        self.spec, self.factory, self.invoke = spec, factory, execute
        self.value = None
        self.managed_execution = None
        self._sdk_closed = False
        self._release_uncertain = None
        self._torch = self._device = None
        profile = spec.runtime_profile
        self._device_name = getattr(profile, "profile", profile).device

    @property
    def runtime_version(self):
        return "" if self._torch is None else str(self._torch.__version__)

    @property
    def runtime_device_index(self):
        return None if self._device is None else self._device.index

    def drain_uncertain_operations(self):
        if self._torch is not None:
            drain_torch_device(self._torch, self._device_name, self._device)

    def load(self, context):
        import torch

        self.managed_execution = context.managed_execution
        check_startup()
        current = load_inference_manifest(self.spec.bundle_root, self.spec.deployment_name)
        if current.fingerprint != self.spec.fingerprint:
            raise ValueError("native deployment changed after validation")
        self._device = resolve_torch_device(
            torch, self._device_name, torch_device_index=context.runtime_options.get("torch_device_index")
        )
        self._torch = torch
        try:
            with torch_device_scope(torch, self._device_name, self._device):
                check_startup()
                self.value = self.factory()
                self.drain_uncertain_operations()
            check_startup()
        except Exception as exc:
            clear_exception_frames(exc)
            raise

    def execute(self, request, context):
        try:
            context.check("native_model")
        except (DeadlineExceeded, CancellationRequested) as exc:
            exc.evidence = OutcomeEvidence.not_started("native_model")
            raise
        with (
            self._torch.no_grad(),
            torch_execution_scope(
                self._torch,
                self._device_name,
                self._device,
                complete_on_success=self.managed_execution,
                force_completion_on_error=True,
                stateful=False,
            ),
        ):
            result = self.invoke(self.value, dict(request.inputs))
        check_torch_completed_context(
            context,
            "native_model",
            self._torch,
            self._device_name,
            self._device,
            managed=self.managed_execution,
            stateful=False,
        )
        return {"value": result}

    def complete_failure(self, error):
        """Fence direct results discarded by the enclosing Handle."""
        if not self.managed_execution:
            complete_torch_failure(
                error, self._torch, self._device_name, self._device, discarded_result=True, stateful=False
            )

    def close(self):
        if self._release_uncertain is not None:
            raise BackendLifecycleError(self._release_uncertain, code="release_outcome_unknown", close_pending=True)
        try:
            check_lifecycle()
            if self._torch is None:
                return
            self.drain_uncertain_operations()
            check_lifecycle()
            with torch_device_scope(self._torch, self._device_name, self._device):
                if not self._sdk_closed:
                    close = getattr(self.value, "close", None)
                    if callable(close):
                        try:
                            close()
                        except BaseException as exc:
                            # Match RKNN/HMM: a partial native free is not retryable.
                            if getattr(getattr(exc, "evidence", None), "state", None) is not OutcomeState.NOT_STARTED:
                                self._release_uncertain = (
                                    f"native Torch close outcome unknown; process restart required: {exc}"
                                )
                            raise
                    self._sdk_closed = True
                check_lifecycle()
                self.drain_uncertain_operations()
                self.value = None
            self._torch = self._device = None
        except BaseException as exc:
            raise BackendLifecycleError(str(exc), code="close_failed", close_pending=True) from exc


def build_native_torch_resource(context, *, factory, execute):
    """Register this builder for an explicit model identity; defer SDK creation."""
    return NativeTorchResource(context.validated_manifest, factory, execute)
