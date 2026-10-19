"""Runtime ownership of immutable skill catalog bundles.

The owner deliberately contains no ROS types or execution leases.  It
serializes catalog activation and keeps the registry's version identity in one
place so the ROS node can expose the same state through status, reload, and
snapshot services.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Protocol

from skill_catalog.models import (
    SkillDiagnostic,
    SkillRegistryError,
    SkillRuntimeBundle,
    SkillSnapshot,
)
from skill_catalog.registry import RegistryActivation, SkillRegistry

SKILL_REQUEST_ID_CONFLICT = "SKILL_REQUEST_ID_CONFLICT"
SKILL_RELOAD_IN_PROGRESS = "SKILL_RELOAD_IN_PROGRESS"
SKILL_SNAPSHOT_DIGEST_MISMATCH = "SKILL_SNAPSHOT_DIGEST_MISMATCH"


class _Lock(Protocol):
    def __enter__(self): ...

    def __exit__(self, exc_type, exc_value, traceback): ...


@dataclass(frozen=True)
class ReloadResult:
    """Stable result returned by one reload request."""

    success: bool
    request_id: str
    old_generation: int = 0
    generation: int = 0
    registry_epoch: str = ""
    registry_digest: str = ""
    capability_digest: str = ""
    source_release_digest: str = ""
    provenance_digest: str = ""
    changed_skills: tuple[str, ...] = ()
    error_code: str = ""
    message: str = ""
    diagnostics: tuple[SkillDiagnostic, ...] = ()


@dataclass(frozen=True)
class PreparedReload:
    """Validated candidate waiting for one atomic node-level activation."""

    request_id: str
    force: bool
    snapshot: SkillSnapshot
    old_bundle: SkillRuntimeBundle | None


class SkillRegistryOwner:
    """Own the active catalog bundle and its process-local lifecycle."""

    def __init__(
        self,
        compile_snapshot: Callable[[], SkillSnapshot],
        *,
        registry_epoch: str | None = None,
        request_history_capacity: int = 64,
        max_unretained_history: int = 2,
        state_lock: _Lock | None = None,
    ) -> None:
        if request_history_capacity <= 0:
            raise ValueError("request_history_capacity must be positive")
        self._compile_snapshot = compile_snapshot
        self._registry = SkillRegistry(
            registry_epoch=registry_epoch,
            max_unretained_history=max_unretained_history,
        )
        self._lock = state_lock or RLock()
        self._request_history_capacity = request_history_capacity
        self._request_history: OrderedDict[str, tuple[bool, ReloadResult]] = OrderedDict()
        self._current: SkillRuntimeBundle | None = None
        self._state = "STARTING"
        self._error_code = "SKILL_REGISTRY_NOT_READY"
        self._message = "skill registry has not been activated"
        self._reload_request_id = ""

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def error_code(self) -> str:
        with self._lock:
            return self._error_code

    @property
    def message(self) -> str:
        with self._lock:
            return self._message

    @property
    def registry_epoch(self) -> str:
        return self._registry.registry_epoch

    @property
    def current(self) -> SkillRuntimeBundle | None:
        with self._lock:
            return self._current

    @property
    def retained_generations(self) -> tuple[int, ...]:
        with self._lock:
            return self._registry.retained_generations

    def activate_initial(self, snapshot: SkillSnapshot) -> RegistryActivation:
        """Activate the initial snapshot, leaving the coordinator ready."""
        with self._lock:
            if self._current is not None:
                raise SkillRegistryError("skill registry is already activated")
            activation = self._registry.activate(snapshot)
            self._current = activation.bundle
            self._state = "READY"
            self._error_code = ""
            self._message = ""
            return activation

    def reload(self, request_id: str, *, force: bool = False) -> ReloadResult:
        """Compile and atomically activate a snapshot for one request ID."""
        result, prepared = self.prepare_reload(request_id, force=force)
        return result if result is not None else self.activate_reload(prepared)

    def prepare_reload(
        self, request_id: str, *, force: bool = False
    ) -> tuple[ReloadResult | None, PreparedReload | None]:
        """Mark reload in progress and compile a candidate without holding the coordinator lock."""
        normalized_request_id = str(request_id).strip()
        if not normalized_request_id:
            return (
                ReloadResult(
                    success=False,
                    request_id="",
                    error_code="INVALID_ARGUMENT",
                    message="request_id must be non-empty",
                ),
                None,
            )

        with self._lock:
            previous_request = self._request_history.get(normalized_request_id)
            if previous_request is not None:
                previous_force, previous_result = previous_request
                if previous_force != bool(force):
                    return (
                        ReloadResult(
                            success=False,
                            request_id=normalized_request_id,
                            error_code=SKILL_REQUEST_ID_CONFLICT,
                            message="request_id was already used with different request fields",
                        ),
                        None,
                    )
                return previous_result, None
            if self._reload_request_id:
                if self._reload_request_id == normalized_request_id:
                    return (
                        ReloadResult(
                            success=False,
                            request_id=normalized_request_id,
                            error_code=SKILL_RELOAD_IN_PROGRESS,
                            message="skill catalog reload request is still in progress",
                        ),
                        None,
                    )
                return (
                    ReloadResult(
                        success=False,
                        request_id=normalized_request_id,
                        error_code=SKILL_RELOAD_IN_PROGRESS,
                        message="another skill catalog reload is in progress",
                    ),
                    None,
                )

            old_bundle = self._current
            self._reload_request_id = normalized_request_id
            self._state = "RELOADING"
            self._error_code = SKILL_RELOAD_IN_PROGRESS
            self._message = "skill catalog reload in progress"
        try:
            snapshot = self._compile_snapshot()
        except Exception as exc:
            with self._lock:
                diagnostics = tuple(getattr(exc, "diagnostics", ()))
                error_code = getattr(exc, "code", "SKILL_SCHEMA_INVALID")
                self._state = "FAILED" if old_bundle is None else "READY"
                self._error_code = error_code
                self._message = str(exc)
                self._reload_request_id = ""
                result = ReloadResult(
                    success=False,
                    request_id=normalized_request_id,
                    old_generation=old_bundle.generation if old_bundle else 0,
                    generation=old_bundle.generation if old_bundle else 0,
                    registry_epoch=self._registry.registry_epoch,
                    error_code=error_code,
                    message=str(exc),
                    diagnostics=diagnostics,
                )
                self._remember_request(normalized_request_id, bool(force), result)
                return result, None
        return None, PreparedReload(normalized_request_id, bool(force), snapshot, old_bundle)

    def activate_reload(self, prepared: PreparedReload | None) -> ReloadResult:
        """Activate one prepared candidate while the node holds its state transaction lock."""
        if prepared is None:
            raise ValueError("prepared reload is required")
        with self._lock:
            if self._reload_request_id != prepared.request_id:
                previous_request = self._request_history.get(prepared.request_id)
                if previous_request is not None and previous_request[0] == prepared.force:
                    return previous_request[1]
                return ReloadResult(
                    success=False,
                    request_id=prepared.request_id,
                    error_code=SKILL_REQUEST_ID_CONFLICT,
                    message="prepared reload is no longer active",
                )
            activation = self._registry.activate(prepared.snapshot)
            self._current = activation.bundle
            self._state = "READY"
            self._error_code = ""
            self._message = ""
            self._reload_request_id = ""
            result = _reload_result(
                prepared.request_id,
                prepared.old_bundle,
                activation,
            )
            self._remember_request(prepared.request_id, prepared.force, result)
            return result

    def get_snapshot(self, *, registry_epoch: str = "", generation: int = 0) -> SkillRuntimeBundle:
        """Return the exact requested bundle or raise a stable catalog error."""
        with self._lock:
            return self._registry.get(registry_epoch=registry_epoch, generation=generation)

    def retain(self, generation: int) -> SkillRuntimeBundle:
        """Retain one exact generation for an admitted execution scope."""
        with self._lock:
            return self._registry.retain(generation)

    def release(self, generation: int) -> None:
        """Release one generation retained by an execution scope."""
        with self._lock:
            self._registry.release(generation)

    def _remember_request(self, request_id: str, force: bool, result: ReloadResult) -> None:
        self._request_history[request_id] = (force, result)
        self._request_history.move_to_end(request_id)
        while len(self._request_history) > self._request_history_capacity:
            self._request_history.popitem(last=False)


def _reload_result(
    request_id: str,
    old_bundle: SkillRuntimeBundle | None,
    activation: RegistryActivation,
) -> ReloadResult:
    bundle = activation.bundle
    snapshot = bundle.snapshot
    return ReloadResult(
        success=True,
        request_id=request_id,
        old_generation=old_bundle.generation if old_bundle else 0,
        generation=bundle.generation,
        registry_epoch=bundle.registry_epoch,
        registry_digest=snapshot.registry_digest,
        capability_digest=snapshot.capability_digest,
        source_release_digest=str(snapshot.provenance.get("source_release_digest", "")),
        provenance_digest=snapshot.provenance_digest,
        changed_skills=activation.changed_skills if activation.changed else (),
    )
