"""Explicit execution purpose and the construction boundary shared by all runtimes.

This module has no transport dependencies. Infrastructure supplies initialization
ownership and a policy binder; business callers only receive a completed handle.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from functools import wraps

from .lifecycle import check_startup
from .registry import RuntimeDependencyError


class RuntimePurpose(str, Enum):
    ONLINE = "online"
    LOCAL = "local"
    SIMULATION = "simulation"
    ISOLATED_OFFLINE = "isolated_offline"
    BENCHMARK = "benchmark"
    DISTRIBUTED = "distributed"


_construction: ContextVar[RuntimeConstruction | None] = ContextVar("runtime_construction", default=None)
_operation: ContextVar[RuntimeConstruction | None] = ContextVar("runtime_operation", default=None)
_cleanup: ContextVar[RuntimeConstruction | None] = ContextVar("runtime_cleanup", default=None)


def runtime_operation(method):
    """Scope backend access to a handle's authorized local operation.

    The wrapped handle entry still validates its dispatch token and lifecycle.
    Sessions consume this transport-neutral scope, never scheduler objects.
    """

    @wraps(method)
    def invoke(owner, *args, **kwargs):
        token = _operation.set(owner._construction)
        try:
            return method(owner, *args, **kwargs)
        finally:
            _operation.reset(token)

    return invoke


def current_runtime_construction() -> RuntimeConstruction | None:
    return _construction.get()


def require_runtime_construction() -> RuntimeConstruction:
    construction = current_runtime_construction()
    if construction is None or not construction.active:
        raise RuntimeDependencyError(
            "model construction requires the public runtime_creation context; "
            "online models must acquire global initialization ownership before entering a backend",
            code="runtime_construction_required",
        )
    return construction


class RuntimeConstruction:
    """One construction transaction; its authority stays with the created handles.

    Online transactions require infrastructure to provide both a live
    initialization context and an execution-policy binder. Explicit simulation
    and isolated purposes use the same boundary without a global owner.
    """

    def __init__(
        self,
        purpose,
        *,
        initialization=None,
        bind=None,
        publish=None,
        retain=None,
        retire=None,
        validate=None,
        settle=None,
    ):
        self.purpose = RuntimePurpose(purpose)
        if self.purpose is RuntimePurpose.ONLINE and (initialization is None or not callable(bind)):
            raise RuntimeDependencyError(
                "online runtime construction requires initialization ownership and an execution policy",
                code="runtime_management_required",
            )
        if self.purpose is not RuntimePurpose.ONLINE and (initialization is not None or bind is not None):
            raise ValueError("an isolated runtime purpose cannot carry a managed initialization or binding")
        self.initialization = initialization
        self.bind = bind
        self.publish = publish
        self.retain = retain
        self.retire = retire
        self.validate = validate
        self.settle_initialization = settle or (lambda: None)
        self.rollback_clean = False
        self.failed = False
        self.active = False
        self.completed = False
        self.handles = []
        self.cleanup_owners = []
        self.bindings = []
        self._token = None
        self._acquisitions = 0
        self._clean_failure = None

    @contextmanager
    def validation(self):
        """Record positive non-start evidence only before any resource builder ran."""
        self.require_loading()
        try:
            yield
        except BaseException as error:
            if not self._acquisitions and not self.handles:
                self._clean_failure = error
            raise

    def begin_acquisition(self):
        self.require_loading()
        with self.validation():
            check_startup()
        self._acquisitions += 1
        self._clean_failure = None

    @property
    def acquisition_count(self):
        return self._acquisitions

    def prove_cleanup(self, error, *, since):
        """Called by a factory only after all of its acquired resources were released."""
        if since == 0:
            self._clean_failure = error

    def cleanup_proven(self, error):
        return error is self._clean_failure

    def __enter__(self):
        check_startup()
        if self.active or self.completed or self._token is not None:
            raise RuntimeError("runtime construction cannot be reused")
        if current_runtime_construction() is not None:
            raise RuntimeError("nested runtime construction would split initialization ownership")
        if self.initialization is not None:
            self.initialization.__enter__()
        self.active = True
        self._token = _construction.set(self)
        return self

    def attach(self, handle):
        self.require_loading()
        if self.retain is not None:
            self.retain(handle)
        self.handles.append(handle)

    def require_loading(self):
        if not self.active or current_runtime_construction() is not self:
            raise RuntimeDependencyError(
                "runtime loading must finish inside its construction ownership boundary",
                code="runtime_initialization_required",
            )

    def require_cleanup(self):
        if _cleanup.get() is not self:
            self.require_loading()

    def retry_rollback(self):
        """Infrastructure calls this only while the original initialization is owned.

        This scope permits cleanup, never new loading or model execution.
        The settlement serializes retries and verifies the Global incarnation.
        """
        if not self.failed or self.completed:
            raise RuntimeError("only a failed construction can retry rollback")
        token = _cleanup.set(self)
        owned = bool(self.handles or self.cleanup_owners)
        try:
            while self.cleanup_owners:
                self.cleanup_owners[-1].close()
                self.cleanup_owners.pop()
            for handle in reversed(self.handles):
                handle._abort_construction()
            self.rollback_clean = (owned or self._clean_failure is not None) and all(
                handle.closed for handle in self.handles
            )
            if not self.rollback_clean:
                raise RuntimeError("initialization cleanup evidence is still pending")
        finally:
            _cleanup.reset(token)

    def validate_context(self, context, *, session_execution=None):
        self.require_loading()
        if self.validate is not None:
            self.validate(context, session_execution=session_execution)

    def require_execution(self):
        if self.purpose is RuntimePurpose.ONLINE and not self.completed:
            raise RuntimeDependencyError(
                "online runtime has not completed initialization and policy binding",
                code="runtime_management_required",
            )

    def require_model_operation(self):
        if self.purpose is not RuntimePurpose.ONLINE:
            return
        if (self.active and current_runtime_construction() is self) or _operation.get() is self:
            return
        raise RuntimeDependencyError(
            "online model resources must be called through their owning runtime handle",
            code="runtime_operation_required",
        )

    def __exit__(self, exc_type, exc, traceback):
        try:
            try:
                if exc is None:
                    check_startup()
                if exc is None and self.cleanup_owners:
                    raise RuntimeDependencyError(
                        "construction cleanup is still pending", code="runtime_cleanup_pending"
                    )
                if exc is None and self.bind is not None:
                    if not self.handles:
                        raise RuntimeDependencyError(
                            "online construction must produce an owned runtime handle",
                            code="runtime_initialization_incomplete",
                        )
                    for handle in self.handles:
                        if handle.local_ready:
                            self.bindings.append(self.bind(handle))
                            if not handle.execution_policy_bound:
                                raise RuntimeDependencyError(
                                    "online construction did not install an execution policy",
                                    code="runtime_management_required",
                                )
                        elif not handle.closed:
                            raise RuntimeDependencyError(
                                "online runtime must finish loading before construction returns",
                                code="runtime_initialization_incomplete",
                            )
                    check_startup()
            except BaseException as binding_error:
                if not self.handles and (not self._acquisitions or self._clean_failure is not None):
                    # A caught factory error can become an incomplete-construction
                    # error here. Preserve its cleanup proof; a new acquisition
                    # clears that proof, and pending owners still block rollback.
                    self._clean_failure = binding_error
                exc_type, exc, traceback = type(binding_error), binding_error, binding_error.__traceback__
            if exc is not None:
                self.failed = True
                errors = []
                if self.purpose is RuntimePurpose.ONLINE:
                    # Execution is still prohibited, and initialization owns
                    # the device. Roll back before releasing that permit.
                    for handle in reversed(self.handles):
                        try:
                            handle._abort_construction()
                        except BaseException as cleanup_error:
                            errors.append(cleanup_error)
                # Local factories already attempt cleanup. Their positive
                # evidence also allows the explicit node to retire this scope;
                # live handles and pending cleanup owners must remain retained.
                self.rollback_clean = (
                    not errors
                    and not self.cleanup_owners
                    and all(handle.closed for handle in self.handles)
                    and (bool(self.handles) or self.cleanup_proven(exc))
                )
            if self.initialization is not None:
                try:
                    self.initialization.__exit__(exc_type, exc, traceback)
                except BaseException:
                    self.failed = True
                    # Keep bound handles and the receipt for a later Close.
                    raise
            if exc is not None:
                if self.rollback_clean and self.retire is not None:
                    for handle in self.handles:
                        self.retire(handle)
                raise exc.with_traceback(traceback)
            if self.publish is not None:
                try:
                    self.publish(self.bindings)
                    check_startup()
                except BaseException:
                    self.failed = True
                    # Initialization has already been settled. Only the bound
                    # policy may now release device resources, through Global.
                    # An uncertain Close stays retained for provider shutdown.
                    cleanup_errors = []
                    for handle in reversed(self.handles):
                        try:
                            handle.close()
                        except Exception as cleanup_error:
                            cleanup_errors.append(cleanup_error)
                    self.rollback_clean = not cleanup_errors and all(handle.closed for handle in self.handles)
                    raise
            self.completed = True
        finally:
            self.active = False
            _construction.reset(self._token)


__all__ = ["RuntimeConstruction", "RuntimePurpose", "current_runtime_construction", "require_runtime_construction"]
