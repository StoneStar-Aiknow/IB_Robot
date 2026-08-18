"""Perception-owned builders registered with the shared model-session registry."""

from __future__ import annotations

import importlib
import json

from inference_manifest import CompiledDeployment, TorchDeployment
from inference_service.backends import RuntimeContext
from inference_service.model_sessions import (
    MODEL_SESSION_BUILDER_REGISTRY,
    AscendOmModelSession,
    ModelSession,
    TorchModelSession,
)

from .graspgen_session import GraspGenAscendSession
from .sam2_automatic_ascend_session import SAM2AutomaticAscendSession
from .siglip2_ascend_session import SigLIP2AscendSession


def _load_torch_module(family: str, context: RuntimeContext):
    identity_path = context.validated_manifest.bundle_root / "assets" / "adapter.json"
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        module_name, function_name = identity["torch_module_loader"].split(":", 1)
        loader = getattr(importlib.import_module(module_name), function_name)
    except (AttributeError, ImportError, KeyError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"{family} Torch bundle has no loadable family module integration in {identity_path}: {exc}"
        ) from exc
    module = loader(context)
    if not callable(module):
        raise TypeError(f"{family} Torch family module loader did not return a callable")
    return module


def _ascend_device_id(family: str, adapter, deployment, options, allowed: set[str]) -> int:
    if deployment.backend != "ascend" or not adapter.compiled_abi_finalized:
        raise RuntimeError(f"{family} compiled adapter ABI is not finalized; deployment remains not-ready")
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(f"unknown Ascend runtime options: {unknown}")
    return int(options.get("device_id", 0))


def build_perception_session(context: RuntimeContext, *, adapter) -> ModelSession:
    family = context.model.family
    deployment = context.deployment
    options = context.runtime_options
    if isinstance(deployment, CompiledDeployment):
        device_id = _ascend_device_id(family, adapter, deployment, options, {"acl_config_path", "device_id"})
        if family == "siglip2":
            return SigLIP2AscendSession(device_id=device_id)
        if family == "sam2" and adapter.identity.operation == "automatic":
            return SAM2AutomaticAscendSession(device_id=device_id)
        return AscendOmModelSession(device_id=device_id)
    if isinstance(deployment, TorchDeployment):
        if options:
            raise ValueError(
                f"Torch family plugins do not select backend/device or accept runtime options: {sorted(options)}"
            )
        return TorchModelSession(lambda runtime_context: _load_torch_module(family, runtime_context))
    raise RuntimeError(f"{family} deployment type is unsupported and has no fallback")


def build_graspgen_session(context: RuntimeContext, *, adapter) -> ModelSession:
    """Build a Torch CUDA or host-orchestrated Ascend GraspGen session."""

    family = context.model.family
    deployment = context.deployment
    options = context.runtime_options
    if isinstance(deployment, TorchDeployment):
        if options:
            raise ValueError(f"GraspGen Torch deployment does not accept runtime options: {sorted(options)}")
        return TorchModelSession(lambda runtime_context: _load_torch_module(family, runtime_context))
    if not isinstance(deployment, CompiledDeployment):
        raise RuntimeError(f"{family} requires a Torch CUDA or compiled Ascend deployment")
    device_id = _ascend_device_id(
        family,
        adapter,
        deployment,
        options,
        {"acl_config_path", "device_id", "random_seed"},
    )
    return GraspGenAscendSession(device_id=device_id, config=adapter.config)


def register_perception_session_builders() -> None:
    for family, operation in (
        ("ram_plus", ""),
        ("sam2", "automatic"),
        ("sam2", "prompt"),
        ("siglip2", ""),
        ("grounding_dino", "combined"),
        ("grounding_dino", "raw"),
    ):
        for backend in ("torch", "ascend"):
            if MODEL_SESSION_BUILDER_REGISTRY.get("perception", family, operation, backend) is None:
                MODEL_SESSION_BUILDER_REGISTRY.register(
                    "perception",
                    family,
                    operation,
                    backend,
                    build_perception_session,
                )

    for backend in ("torch", "ascend"):
        if MODEL_SESSION_BUILDER_REGISTRY.get("perception", "graspgen", "", backend) is None:
            MODEL_SESSION_BUILDER_REGISTRY.register(
                "perception",
                "graspgen",
                "",
                backend,
                build_graspgen_session,
            )


register_perception_session_builders()


__all__ = ["build_graspgen_session", "build_perception_session", "register_perception_session_builders"]
