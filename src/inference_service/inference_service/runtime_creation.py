"""Public model construction boundary, independent of the business host.

When scheduling is enabled, online construction obtains Global ownership before
calling a session builder. Otherwise models execute locally. A handle captures
the selected policy, including when it executes later on another thread.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager, nullcontext
from inspect import signature
from uuid import uuid4

from inference_service.backends import runtime_context_defaults
from inference_service.scheduler.runtime_environment import RuntimeEnvironment
from inference_service.scheduler.runtime_protocol import DEFAULT_RUNTIME_ENDPOINT
from inference_service.unified_runtime.construction import RuntimeConstruction, RuntimePurpose
from inference_service.unified_runtime.registry import RuntimeDependencyError

_host_configuration_lock = threading.Lock()
_SESSION_ADAPTERS = {"full": "model_runtime", "role": "model_runtime/role", "stream": "model_runtime/stream"}


@contextmanager
def runtime_creation(
    *,
    purpose=RuntimePurpose.ONLINE,
    providers=None,
    validated_manifest=None,
    runtime_options=None,
    node=None,
    targets=None,
    endpoint=None,
    boot_id=None,
    context_provider=None,
    execution_scope=None,
    clean=None,
    timeout=None,
    request_timeout_ms=5000,
    configured_target_id=None,
    scheduler_enabled=None,
    session_execution="full",
):
    """Construct and load models inside one explicit execution-purpose scope.

    A launch composition may supply validated targets. Library/in-process
    callers supply a manifest and their process providers; the same public
    boundary derives placement from Global's authoritative resource catalogue
    when scheduling is enabled. Scheduling defaults to disabled; targets describe
    placement and cannot enable it. This API never starts Global
    and an enabled scheduler never falls back to uncoordinated execution.

    configured_target_id selects an exact catalogue target ID, such as
    perception/depth_front, not the service ID depth_front used by config compilation.

    When both node and providers are supplied, the node configures the process
    environment before its first creation; later conflicting configuration is
    rejected. Local constructions remain owned by providers, or by the explicit
    node when no providers are supplied. Explicit online targets are owned by
    their node even when providers supply configuration and dependencies.

    session_execution declares full Session, role or Session stream assembly.
    The public loader derives it from Factory options; explicit scopes must
    declare the same choice so discovery and the actual assembler share identity.

    Before online loading, callers must pause submissions to the same device
    and drain accepted work. Initialization retries transient busy responses;
    it has no queue position or progress guarantee under continuous traffic.
    """
    purpose = RuntimePurpose(purpose)
    if session_execution not in _SESSION_ADAPTERS:
        raise ValueError("session_execution must be full, role or stream")
    adapter = _SESSION_ADAPTERS[session_execution]
    provider_environment = getattr(providers, "runtime_environment", None)
    configuration = provider_environment
    if scheduler_enabled is not None and type(scheduler_enabled) is not bool:
        raise ValueError("runtime scheduler enable must be boolean")
    if purpose is RuntimePurpose.ONLINE and node is not None and configuration is None:
        # Explicit hosts consume exactly the same connection parameters as
        # provider-owned hosts, without starting a second control transport.
        with _host_configuration_lock:
            configuration = getattr(node, "_runtime_environment", None)
            if configuration is None:
                configuration = RuntimeEnvironment(
                    endpoint=endpoint if endpoint is not None else DEFAULT_RUNTIME_ENDPOINT,
                    scheduler_enabled=False if scheduler_enabled is None else scheduler_enabled,
                )
                node._runtime_environment = configuration
    from inference_service.unified_runtime.lifecycle import lifecycle_budget

    with (
        lifecycle_budget(timeout) if timeout is not None else nullcontext(),
        configuration.initialization(node if purpose is RuntimePurpose.ONLINE else None, timeout=timeout)
        if configuration is not None
        else nullcontext(),
        lifecycle_budget(timeout if timeout is not None else getattr(configuration, "initialization_timeout", 60.0)),
    ):
        if timeout is None:
            timeout = getattr(configuration, "initialization_timeout", 60.0)
        if endpoint is None:
            endpoint = getattr(configuration, "endpoint", DEFAULT_RUNTIME_ENDPOINT)
        if purpose is RuntimePurpose.ONLINE and configuration is not None:
            if scheduler_enabled is not None and scheduler_enabled != configuration.scheduler_enabled:
                owner = "process" if provider_environment is not None else "host"
                raise ValueError(f"runtime scheduler override conflicts with {owner} configuration")
            scheduler_enabled = configuration.scheduler_enabled
            if scheduler_enabled and endpoint != configuration.endpoint:
                raise ValueError("runtime scheduler endpoint override conflicts with configured authority")
        if scheduler_enabled is None:
            scheduler_enabled = False
        if type(scheduler_enabled) is not bool:
            raise ValueError("runtime scheduler enable must be boolean")
        if purpose is RuntimePurpose.ONLINE and not scheduler_enabled:
            purpose = RuntimePurpose.LOCAL
            targets = None  # Placement is dormant when online scheduling is off.
        if purpose is not RuntimePurpose.ONLINE:
            if targets:
                raise ValueError("an isolated runtime purpose cannot specify managed targets")
            # Isolation removes Global admission, not process ownership. Retain
            # handles before load so provider shutdown also cleans failed loads.
            owner_environment = provider_environment

            def retire_local(handle):
                if owner_environment is not None:
                    owner_environment.retire(handle, ())
                elif (
                    node is not None
                    and not construction.active
                    and all(owner.closed for owner in construction.handles)
                    and construction in node._runtime_constructions
                ):
                    node._runtime_constructions.remove(construction)

            construction = RuntimeConstruction(
                purpose,
                retain=owner_environment.retain if owner_environment is not None else None,
                retire=retire_local if owner_environment is not None or node is not None else None,
            )
            with (
                runtime_context_defaults(priority_scheduling=False, managed_execution=False),
                _owned_construction(construction, node if owner_environment is None else None),
            ):
                yield construction
            return
        from inference_manifest import validate_deployment_artifact_integrity
        from inference_service.scheduler.runtime_binding import bind_runtime, runtime_load_options
        from inference_service.scheduler.runtime_initialization import (
            runtime_initialization,
            settle_initializations,
            validate_initialization_targets,
        )

        if validated_manifest is not None:
            validate_deployment_artifact_integrity(validated_manifest)

        # Explicit targets use the caller's node for cleanup. Only discovery
        # makes the providers' environment the control and cleanup owner.
        owner_environment = None
        if targets is None:
            owner_environment = provider_environment
            if owner_environment is None or validated_manifest is None:
                raise RuntimeDependencyError(
                    "online model creation requires a validated manifest and process runtime environment",
                    code="runtime_management_required",
                )
            node, target = owner_environment.prepare(
                validated_manifest,
                runtime_options or {},
                timeout,
                request_timeout_ms,
                configured_target_id=configured_target_id,
                adapter=adapter,
            )
            targets = [target]
            endpoint = owner_environment.endpoint
        if node is None or not targets:
            raise RuntimeDependencyError(
                "online runtime requires a control context and targets", code="runtime_management_required"
            )
        if validated_manifest is not None:
            validate_initialization_targets(targets, validated_manifest, runtime_options or {}, adapter=adapter)
        boot_id = boot_id or str(uuid4())
        initialization_token = str(uuid4())

        def validate(context, *, session_execution=None):
            from inference_manifest import ValidatedManifest

            actual = context if isinstance(context, ValidatedManifest) else getattr(context, "validated_manifest", None)
            if actual is None:
                raise RuntimeDependencyError(
                    "online session construction requires its actual validated manifest",
                    code="runtime_identity_required",
                )
            actual_adapter = adapter if session_execution is None else _SESSION_ADAPTERS[session_execution]
            validate_initialization_targets(
                targets, actual, getattr(context, "runtime_options", {}), adapter=actual_adapter
            )

        def bind(handle):
            if len(construction.handles) != 1:
                raise RuntimeDependencyError(
                    "one model construction must produce one lifecycle owner; composite roles share its handle",
                    code="runtime_lifecycle_ambiguous",
                )
            if len(targets) != 1:
                raise ValueError("a model runtime requires exactly one target")
            binding = bind_runtime(
                node,
                handle,
                targets[0],
                endpoint=endpoint,
                context_provider=context_provider,
                execution_scope=execution_scope,
                boot_id=boot_id,
                lifecycle_timeout=timeout,
            )
            return binding

        def initialization_clean():
            if construction.failed:
                return construction.rollback_clean or (
                    not construction.handles and not construction.cleanup_owners and clean is not None and clean()
                )
            if construction.bindings:
                return True
            return clean() if clean is not None else True

        def retire_explicit(handle):
            # Public Close has proved local cleanup and Global retirement.
            # Reuse the spin owner's quiescent transport collection boundary.
            for binding in construction.bindings:
                binding.adapter.close()
                if binding.adapter not in node._runtime_retired_adapters:
                    node._runtime_retired_adapters.append(binding.adapter)
            if construction in node._runtime_constructions:
                node._runtime_constructions.remove(construction)
            from inference_service.host_lifecycle import collect_node_runtime_transport

            collect_node_runtime_transport(node)

        @contextmanager
        def initialization():
            with (
                runtime_initialization(
                    node,
                    targets,
                    endpoint,
                    boot_id=boot_id,
                    clean=initialization_clean,
                    retire=lambda: construction.failed or not construction.bindings,
                    cleanup=lambda: construction.retry_rollback(),
                    timeout=timeout,
                    token=initialization_token,
                ),
                runtime_context_defaults(**runtime_load_options(targets)),
            ):
                yield

        construction = RuntimeConstruction(
            purpose,
            initialization=initialization(),
            bind=bind,
            publish=lambda bindings: (owner_environment or RuntimeEnvironment).publish(bindings, timeout),
            retain=owner_environment.retain if owner_environment is not None else None,
            retire=(lambda handle: owner_environment.retire(handle, construction.bindings))
            if owner_environment is not None
            else retire_explicit,
            validate=validate,
            settle=lambda: settle_initializations(node, token=initialization_token),
        )
        with _owned_construction(construction, node if owner_environment is None else None):
            yield construction


@contextmanager
def _owned_construction(construction, node):
    """Retain explicit-node ownership for both local and scheduled construction."""
    if node is not None and not hasattr(node, "_runtime_constructions"):
        node._runtime_constructions = []
        node._runtime_retired_adapters = []
    try:
        with construction:
            if node is not None:
                node._runtime_constructions.append(construction)
            yield construction
    finally:
        if (
            node is not None
            and (construction.rollback_clean or construction.completed)
            and construction in node._runtime_constructions
            and all(handle.closed for handle in construction.handles)
            and not construction.cleanup_owners
        ):
            node._runtime_constructions.remove(construction)


def runtime_purpose(*, runtime_target="hardware", offline_execution=False, distributed=False):
    """Translate existing launch-purpose declarations, never infer from missing metadata."""
    if type(offline_execution) is not bool or type(distributed) is not bool:
        raise ValueError("runtime purpose flags must be boolean")
    if runtime_target not in {"hardware", "simulation", "benchmark"}:
        raise ValueError(f"invalid runtime target: {runtime_target!r}")
    if offline_execution:
        return RuntimePurpose.ISOLATED_OFFLINE
    if distributed:
        return RuntimePurpose.DISTRIBUTED
    return {
        "hardware": RuntimePurpose.ONLINE,
        "simulation": RuntimePurpose.SIMULATION,
        "benchmark": RuntimePurpose.BENCHMARK,
    }[runtime_target]


def load_model_runtime(
    *,
    validated_manifest,
    providers,
    purpose=RuntimePurpose.ONLINE,
    runtime_options=None,
    request_timeout_ms=5000,
    timeout=None,
    configured_target_id=None,
    **assembly_options,
):
    """Public construction and load for a model used without a serving host."""
    from inference_service.unified_runtime.factory import ModelRuntimeFactory

    # Python argument binding happens before the Factory's validation scope.
    # Reject missing/unknown assembly arguments before acquiring any device.
    signature(ModelRuntimeFactory.create_session).bind(
        validated_manifest, providers=providers, runtime_options=runtime_options, **assembly_options
    )
    if timeout is None:
        timeout = getattr(providers.runtime_environment, "initialization_timeout", 60.0)
    with runtime_creation(
        purpose=purpose,
        providers=providers,
        validated_manifest=validated_manifest,
        runtime_options=runtime_options,
        request_timeout_ms=request_timeout_ms,
        timeout=timeout,
        configured_target_id=configured_target_id,
        session_execution="role"
        if assembly_options.get("role_execution")
        else "stream"
        if assembly_options.get("streaming_execution")
        else "full",
    ):
        handle = ModelRuntimeFactory.load_session(
            validated_manifest,
            providers=providers,
            runtime_options=runtime_options,
            timeout=timeout,
            **assembly_options,
        )
    return handle


def load_streaming_model_runtime(**options):
    """Load a registered streaming Session through the common Factory path.

    The Session supplies open/step/reset/close stream operations. Its validated
    deployment defines the stream contract; assembly and failed-load ownership
    are the same as ordinary and domain Sessions.
    """
    return load_model_runtime(streaming_execution=True, **options)
