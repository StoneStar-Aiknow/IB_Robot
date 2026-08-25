"""Private transition helpers for sessions not yet expressed as native runtimes.

This module is deliberately not part of the public inference API.  It lets
legacy ``ModelSession`` implementations participate in the unified factory and
handle while their host orchestration is moved into explicit stages.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from datetime import datetime
from types import SimpleNamespace

from inference_service.backends import BackendHealth, BackendState, RuntimeContext
from inference_service.generic_runtime import NamedTensorRequest
from inference_service.pipeline import GenericModelPipeline
from inference_service.pipeline.state import PipelineState
from inference_service.unified_runtime import (
    Deadline,
    ExecutionContext,
    ExecutionContract,
    ModelRequest,
    ModelResult,
    ModelRuntimeFactory,
    ModelRuntimeHandle,
    ModelRuntimeKey,
    OutcomeEvidence,
    RegistrySet,
    RequestDirectAssembler,
    RequestIterativeAssembler,
    RuntimeAssemblerRegistry,
    RuntimeAssembly,
    RuntimeDescriptor,
    RuntimeLatency,
    RuntimeProviders,
    SessionBuilderKey,
    SessionBuilderRegistry,
)


class _SessionExecutorAdapter:
    """Adapt one legacy ModelSession without exposing its request contract."""

    _owns_lifecycle_components = True

    def __init__(self, session: object, load_context: object) -> None:
        self._session = session
        self._load_context = load_context
        self._load_failed = False

    def load(self, _context: object) -> None:
        try:
            self._session.load(self._load_context)
        except Exception:
            self._load_failed = True
            with suppress(Exception):
                self._session.close()
            raise

    def execute(self, request: ModelRequest, context: ExecutionContext) -> ModelResult:
        context.check("request")

        def cancellation() -> None:
            self.cancel(context.request_id, context.deadline.expires_at)

        context.cancellation_token.add_callback(cancellation)
        try:
            result = self._session.infer(
                NamedTensorRequest(
                    request_id=context.request_id,
                    inputs=request.inputs,
                    deadline=context.deadline.expires_at,
                    metadata=request.metadata,
                )
            )
            context.check("result")
            if isinstance(result, ModelResult):
                return result
            outputs = getattr(result, "outputs", None)
            if not isinstance(outputs, Mapping):
                raise TypeError(f"legacy session returned unsupported result {type(result).__name__}")
            return ModelResult(
                outputs=outputs,
                latency=RuntimeLatency(0.0, 0.0),
                evidence=OutcomeEvidence.completed("backend"),
                metadata=getattr(result, "metadata", {}),
            )
        finally:
            context.cancellation_token.remove_callback(cancellation)

    def reset(self, context: ExecutionContext | None = None) -> None:
        deadline = None if context is None else context.deadline.expires_at
        reset = self._session.reset
        try:
            reset(deadline=deadline)
        except TypeError:
            reset()

    def cancel(self, request_id: str, deadline: datetime | None = None) -> None:
        cancel = getattr(self._session, "cancel", None)
        if callable(cancel):
            cancel(request_id, deadline=deadline)

    def health(self) -> BackendHealth:
        return self._session.health()

    def close(self) -> None:
        if self._load_failed:
            self._load_failed = False
            return
        self._session.close()


class _SessionMarker:
    pass


class _UnifiedPipelineView(GenericModelPipeline):
    """Compatibility-shaped view backed entirely by a ModelRuntimeHandle."""

    def __init__(self, handle: ModelRuntimeHandle, context: RuntimeContext, pipeline_id: str) -> None:
        self._runtime_handle = handle
        self._runtime_context = context
        self._pipeline_id = pipeline_id

    @property
    def state(self) -> PipelineState:
        state = self._runtime_handle.state
        if state.value == "reset_required":
            return PipelineState.DEGRADED
        try:
            return PipelineState(state.value)
        except ValueError:
            return PipelineState.FAILED

    @property
    def pipeline_id(self) -> str:
        return self._pipeline_id

    @property
    def runtime_context(self) -> RuntimeContext:
        return self._runtime_context

    def load(self) -> None:
        self._runtime_handle.load(self._runtime_context)

    def execute(self, request: NamedTensorRequest):
        return self._runtime_handle.execute(
            ModelRequest(request.inputs, request.metadata),
            ExecutionContext(request.request_id, request.deadline),
        )

    def reset(self, deadline: datetime | None = None) -> None:
        self._runtime_handle.reset(deadline=Deadline.at(deadline))

    def diagnostics(self):
        runtime = self._runtime_handle.diagnostics()
        health = runtime.health
        backend_health = BackendHealth(
            state=BackendState.READY if health.ready else BackendState.FAILED,
            ready=health.ready,
            reason_code=health.reason_code,
            message=health.message,
            recoverable=health.recoverable,
            failure_count=health.failure_count,
        )
        manifest = self._runtime_context.validated_manifest.manifest
        deployment = self._runtime_context.deployment
        identity = SimpleNamespace(
            bundle=manifest.bundle.name,
            bundle_uuid=manifest.bundle.uuid,
            bundle_revision=manifest.bundle.revision,
            deployment=self._runtime_context.deployment_name,
            deployment_uuid=deployment.uuid,
            deployment_revision=deployment.revision,
            deployment_fingerprint=self._runtime_context.deployment_fingerprint,
            backend=self._runtime_context.backend,
        )
        return SimpleNamespace(
            pipeline_id=self._pipeline_id,
            deployment=identity,
            state=self.state,
            executor_health=backend_health,
            active_requests=runtime.active_executions,
            request_timeout=None,
        )

    health = diagnostics

    def close(self) -> None:
        self._runtime_handle.close()


def build_session_runtime_handle(
    session: object,
    context: RuntimeContext,
    providers: RuntimeProviders,
    *,
    execution_structure: str = "direct",
    orchestration_visibility: str | None = None,
    cancellation_granularity: str | None = None,
    runtime_id: str | None = None,
) -> ModelRuntimeHandle:
    """Build a unified handle around a legacy ModelSession.

    ``session`` remains owned by the private adapter for now.  The factory and
    handle still enforce typed identity, contract validation, lifecycle, and
    normalized ``ModelResult`` publication.
    """

    if not isinstance(providers, RuntimeProviders):
        raise TypeError("build_session_runtime_handle requires RuntimeProviders")
    if execution_structure not in {"direct", "iterative"}:
        raise ValueError("execution_structure must be direct or iterative")
    if execution_structure == "iterative" and orchestration_visibility not in {"executor", "session"}:
        raise ValueError("iterative session runtime requires executor or session visibility")
    if execution_structure == "direct" and orchestration_visibility is not None:
        raise ValueError("direct session runtime cannot declare orchestration visibility")

    profile = context.backend_profile
    if profile is None:
        raise ValueError("session runtime requires a typed backend profile")
    identity = context.identity
    contract = ExecutionContract(
        state_scope="request",
        execution_structure=execution_structure,
        orchestration_visibility=orchestration_visibility,
        cancellation_granularity=cancellation_granularity
        or ("checkpoint" if execution_structure == "iterative" else "request_boundary"),
    )
    backend = context.backend
    runtime_key = ModelRuntimeKey(
        identity.interface,
        identity.model_type,
        identity.operation,
        backend,
        contract.name,
        contract.orchestration_visibility,
    )
    session_key = SessionBuilderKey(identity.interface, identity.model_type, identity.operation, backend)
    session_registry = SessionBuilderRegistry()
    session_registry.register(session_key, lambda _role_context: _SessionMarker())
    assembler_registry = RuntimeAssemblerRegistry()
    adapter = _SessionExecutorAdapter(session, context)
    assembler = RequestIterativeAssembler() if execution_structure == "iterative" else RequestDirectAssembler()

    def assemble(*, contract: object, **_kwargs: object) -> RuntimeAssembly:
        assembly = assembler.assemble(executor=adapter, contract=contract)
        assembly.session = session
        assembly.stateful = bool(getattr(getattr(session, "capabilities", None), "stateful", False))
        assembly.resettable = bool(getattr(getattr(session, "capabilities", None), "resettable", False))
        assembly.runtime_id = runtime_id
        assembly.owned_components = (adapter,)
        return assembly

    assembler_registry.register(
        RuntimeDescriptor(
            key=runtime_key,
            session_builder_key=session_key,
            profile_type=type(profile),
            assembler=assemble,
            execution_contract=contract.name,
            declared_capabilities={
                "stateful": bool(getattr(getattr(session, "capabilities", None), "stateful", False)),
                "execution_contract": contract.name,
            },
        )
    )
    registry_set = RegistrySet(SimpleNamespace(names=(backend,)), session_registry, assembler_registry)
    registry_set.freeze()
    typed_spec = None
    validated = getattr(context, "validated_manifest", None)
    to_runtime_spec = getattr(validated, "to_runtime_spec", None)
    if callable(to_runtime_spec):
        candidate = to_runtime_spec()
        selected = getattr(getattr(validated, "deployment", None), "execution_contract", None)
        if getattr(selected, "name", None) == contract.name:
            typed_spec = candidate
    if typed_spec is not None:
        return ModelRuntimeFactory.create(typed_spec, registry_set, providers)

    spec_values = {
        "identity": identity,
        "deployment": context.deployment,
        "execution_contract": contract,
        "runtime_profile": profile,
        "target_runtime": context.target_runtime,
        "runtime_abi": context.runtime_abi,
        "_load_context": context,
    }
    validated_deployment = getattr(context, "validated_manifest", None)
    if validated_deployment is not None:
        spec_values["validated_deployment"] = validated_deployment
    spec = SimpleNamespace(**spec_values)
    return ModelRuntimeFactory.create(spec, registry_set, providers)


__all__ = ["build_session_runtime_handle"]
