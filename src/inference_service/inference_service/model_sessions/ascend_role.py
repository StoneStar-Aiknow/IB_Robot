"""Reusable single-role Ascend execution for model-family pipelines.

Some model families (for example flow-matching TTS) execute one compiled OM
role several times with host-side scheduling between calls.  The regular
``AscendOmModelSession`` intentionally models one complete deployment graph;
this class provides the same shared ACL lease and manifest-bound ABI checks
for a single role without exposing private model-session state.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from contextlib import suppress

import numpy as np

from inference_manifest import CompiledDeployment, ValidatedManifest
from inference_service.backends.ascend.acl_runtime import (
    ACL_RUNTIME_MANAGER,
    AclRuntimeLease,
    AclRuntimeManager,
)
from inference_service.backends.ascend.model import AclModel, numpy_dtype
from inference_service.backends.errors import BackendInferenceError, BackendLoadError


class AscendOmRoleSession:
    """Load and repeatedly execute one manifest-declared Ascend OM role.

    The session owns one model resource and one ACL context lease.  Multiple
    instances share the process-wide ACL runtime manager, so model-family
    adapters can keep a text encoder and a flow decoder alive together without
    reinitializing ACL or reaching into ``AscendOmModelSession`` internals.
    """

    def __init__(
        self,
        validated_manifest: ValidatedManifest,
        role: str,
        *,
        device_id: int = 0,
        acl_config_path: str | None = None,
        runtime_manager: AclRuntimeManager = ACL_RUNTIME_MANAGER,
        model_factory=AclModel,
    ) -> None:
        if type(device_id) is not int or device_id < 0:
            raise ValueError("device_id must be a non-negative integer")
        deployment = validated_manifest.deployment
        if not isinstance(deployment, CompiledDeployment) or deployment.backend != "ascend":
            raise BackendLoadError(
                "AscendOmRoleSession requires a validated Ascend compiled deployment",
                code="invalid_deployment",
            )
        if role not in deployment.execution:
            raise BackendLoadError(f"Ascend role {role!r} is not in deployment execution", code="unknown_role")
        if deployment.artifacts[role].format != "om":
            raise BackendLoadError("Ascend role artifacts must use format 'om'", code="invalid_artifact_format")
        if not (deployment.target.runtime.startswith("acl") or deployment.target.runtime.startswith("ascend")):
            raise BackendLoadError(
                f"Ascend target runtime {deployment.target.runtime!r} is not ACL-compatible",
                code="incompatible_backend_target",
            )
        if acl_config_path is not None and not isinstance(acl_config_path, str):
            raise ValueError("acl_config_path must be a string or None")
        self._validated_manifest = validated_manifest
        self._role = role
        self._device_id = device_id
        self._acl_config_path = acl_config_path
        self._runtime_manager = runtime_manager
        self._model_factory = model_factory
        self._lease: AclRuntimeLease | None = None
        self._model = None
        self._lock = threading.RLock()
        self._closed = False

    @property
    def role(self) -> str:
        return self._role

    @property
    def runtime_version(self) -> str:
        if self._lease is None:
            return ""
        acl = self._lease.acl
        version = getattr(acl, "__version__", None)
        if version:
            return str(version)
        getter = getattr(acl, "get_version", None)
        if callable(getter):
            value = getter()
            if isinstance(value, tuple):
                value = ".".join(str(item) for item in value)
            return str(value or "")
        return ""

    def load(self) -> None:
        with self._lock:
            if self._lease is not None or self._model is not None:
                raise BackendLoadError("Ascend role session is already loaded", code="invalid_load_state")
            deployment = self._validated_manifest.deployment
            artifact = deployment.artifacts[self._role]
            path = self._validated_manifest.bundle_root / artifact.path
            if not path.is_file():
                raise BackendLoadError(
                    f"Ascend artifact {self._role!r} is unavailable: {path}", code="invalid_artifact"
                )
            lease: AclRuntimeLease | None = None
            model = None
            try:
                lease = self._runtime_manager.acquire(self._device_id, self._acl_config_path)
                model = self._model_factory(lease, self._role, path, deployment.bindings[self._role])
                model.load_descriptor()
                model.prepare_datasets()
            except Exception as exc:
                if model is not None:
                    with suppress(Exception):
                        model.close()
                if lease is not None:
                    with suppress(Exception):
                        lease.close()
                if isinstance(exc, BackendLoadError):
                    raise
                raise BackendLoadError(f"failed to load Ascend role {self._role!r}: {exc}") from exc
            self._lease = lease
            self._model = model
            self._closed = False

    def infer(self, inputs: Mapping[str, object]) -> dict[str, np.ndarray]:
        """Execute one role using semantic names and return bound outputs."""

        with self._lock:
            if self._model is None or self._lease is None or self._closed:
                raise BackendInferenceError("Ascend role session is not loaded", code="runtime_not_loaded")
            bindings = self._validated_manifest.deployment.bindings[self._role]
            expected = {binding.semantic: binding for binding in bindings.inputs}
            missing = sorted(set(expected) - set(inputs))
            unexpected = sorted(set(inputs) - set(expected))
            if missing or unexpected:
                raise BackendInferenceError(
                    f"Ascend role {self._role!r} input semantics mismatch: missing={missing}, unexpected={unexpected}",
                    code="input_semantic_mismatch",
                )
            indexed = {}
            for binding in bindings.inputs:
                value = np.asarray(inputs[binding.semantic])
                expected_dtype = numpy_dtype(binding.dtype)
                if value.dtype != expected_dtype:
                    raise BackendInferenceError(
                        f"Ascend role {self._role!r} input {binding.semantic!r} dtype {value.dtype} "
                        f"does not match {expected_dtype}",
                        code="input_dtype_mismatch",
                    )
                if not self._shape_matches(binding.shape, value.shape):
                    raise BackendInferenceError(
                        f"Ascend role {self._role!r} input {binding.semantic!r} shape {value.shape} "
                        f"does not match {binding.shape}",
                        code="input_shape_mismatch",
                    )
                indexed[int(binding.index)] = value
            runtime_outputs = self._model.execute(indexed)
            return {
                binding.semantic: np.asarray(runtime_outputs[int(binding.index)])
                for binding in bindings.outputs
                if binding.index is not None and int(binding.index) in runtime_outputs
            }

    @staticmethod
    def _shape_matches(expected: tuple[int, ...], actual: tuple[int, ...]) -> bool:
        return len(expected) == len(actual) and all(
            expected_dimension == -1 or expected_dimension == actual_dimension
            for expected_dimension, actual_dimension in zip(expected, actual, strict=True)
        )

    def close(self) -> None:
        with self._lock:
            model, lease = self._model, self._lease
            self._model = None
            self._lease = None
            self._closed = True
            errors: list[Exception] = []
            if model is not None:
                try:
                    model.close()
                except Exception as exc:
                    errors.append(exc)
            if lease is not None:
                try:
                    lease.close()
                except Exception as exc:
                    errors.append(exc)
            if errors:
                raise RuntimeError("; ".join(str(error) for error in errors))
