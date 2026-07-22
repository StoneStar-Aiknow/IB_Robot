"""Backend request admission with process-wide resource-domain coordination."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from inference_service.backends.errors import BackendAdmissionError
from inference_service.backends.types import BackendCapabilities


@dataclass
class _DomainGate:
    limit: int
    semaphore: threading.BoundedSemaphore
    exclusive_lock: threading.Lock
    references: int = 0


@dataclass
class _InstanceRecord:
    supports_multiple_instances: bool
    references: int = 0


class ResourceDomainAdmissions:
    """Own shared gates keyed by a backend-declared process resource domain."""

    _shared: ClassVar[ResourceDomainAdmissions | None] = None
    _shared_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._domains: dict[str, _DomainGate] = {}
        self._instances: dict[str, _InstanceRecord] = {}

    @classmethod
    def shared(cls) -> ResourceDomainAdmissions:
        with cls._shared_lock:
            if cls._shared is None:
                cls._shared = cls()
            return cls._shared

    def register(self, domain: str, limit: int) -> ResourceDomainLease:
        if not domain:
            raise BackendAdmissionError("resource domain must be non-empty", code="invalid_resource_domain")
        if limit < 1:
            raise BackendAdmissionError("resource domain limit must be at least one", code="invalid_admission_limit")

        with self._lock:
            gate = self._domains.get(domain)
            if gate is None:
                gate = _DomainGate(
                    limit=limit,
                    semaphore=threading.BoundedSemaphore(limit),
                    exclusive_lock=threading.Lock(),
                )
                self._domains[domain] = gate
            elif gate.limit != limit:
                raise BackendAdmissionError(
                    f"resource domain {domain!r} was registered with conflicting limits {gate.limit} and {limit}",
                    code="resource_domain_limit_conflict",
                )
            gate.references += 1
        return ResourceDomainLease(self, domain, gate)

    def register_instance(self, backend_name: str, supports_multiple_instances: bool) -> BackendInstanceLease:
        if not backend_name:
            raise BackendAdmissionError("backend name must be non-empty", code="invalid_backend_name")
        with self._lock:
            record = self._instances.get(backend_name)
            if record is None:
                record = _InstanceRecord(supports_multiple_instances=supports_multiple_instances)
                self._instances[backend_name] = record
            elif not record.supports_multiple_instances or not supports_multiple_instances:
                raise BackendAdmissionError(
                    f"backend {backend_name!r} does not support multiple live instances",
                    code="multiple_instances_unsupported",
                )
            record.references += 1
        return BackendInstanceLease(self, backend_name, record)

    def _release(self, domain: str, gate: _DomainGate) -> None:
        with self._lock:
            current = self._domains.get(domain)
            if current is not gate:
                return
            gate.references -= 1
            if gate.references == 0:
                del self._domains[domain]

    def _release_instance(self, backend_name: str, record: _InstanceRecord) -> None:
        with self._lock:
            current = self._instances.get(backend_name)
            if current is not record:
                return
            record.references -= 1
            if record.references == 0:
                del self._instances[backend_name]


class BackendInstanceLease:
    def __init__(self, owner: ResourceDomainAdmissions, backend_name: str, record: _InstanceRecord) -> None:
        self._owner = owner
        self._backend_name = backend_name
        self._record = record
        self._closed = False
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._owner._release_instance(self._backend_name, self._record)


class ResourceDomainLease:
    def __init__(self, owner: ResourceDomainAdmissions, domain: str, gate: _DomainGate) -> None:
        self._owner = owner
        self._domain = domain
        self._gate = gate
        self._closed = False
        self._lock = threading.Lock()

    @property
    def limit(self) -> int:
        return self._gate.limit

    def acquire(self, deadline: datetime | None = None) -> None:
        _acquire_before_deadline(
            self._gate.semaphore,
            deadline,
            "request deadline expired while waiting for shared resource-domain admission",
        )

    def release(self) -> None:
        self._gate.semaphore.release()

    @contextmanager
    def exclusive(self, deadline: datetime | None = None) -> Iterator[None]:
        _acquire_before_deadline(
            self._gate.exclusive_lock,
            deadline,
            "request deadline expired while waiting for exclusive resource-domain admission",
        )
        acquired = 0
        try:
            for _ in range(self._gate.limit):
                self.acquire(deadline)
                acquired += 1
            yield
        finally:
            for _ in range(acquired):
                self.release()
            self._gate.exclusive_lock.release()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._owner._release(self._domain, self._gate)


class BackendAdmission:
    """Combine an independent instance gate with an optional shared domain gate."""

    def __init__(
        self,
        backend_name: str,
        capabilities: BackendCapabilities,
        *,
        domains: ResourceDomainAdmissions | None = None,
    ) -> None:
        manager = domains or ResourceDomainAdmissions.shared()
        self._instance_lease = manager.register_instance(backend_name, capabilities.supports_multiple_instances)
        self._instance_limit = capabilities.max_in_flight_per_instance
        self._instance_gate = threading.BoundedSemaphore(self._instance_limit)
        self._instance_exclusive_lock = threading.Lock()
        self._closed = False
        self._close_lock = threading.Lock()
        self._domain_lease: ResourceDomainLease | None = None
        try:
            if capabilities.resource_domain is not None:
                self._domain_lease = manager.register(
                    capabilities.resource_domain,
                    capabilities.resource_domain_limit,
                )
        except Exception:
            self._instance_lease.close()
            raise

    @contextmanager
    def admit(self, deadline: datetime | None = None) -> Iterator[None]:
        self._assert_open()
        _acquire_before_deadline(
            self._instance_gate,
            deadline,
            "request deadline expired while waiting for backend instance admission",
        )
        domain_acquired = False
        try:
            self._assert_open()
            if self._domain_lease is not None:
                self._domain_lease.acquire(deadline)
                domain_acquired = True
                self._assert_open()
            yield
        finally:
            if domain_acquired:
                self._domain_lease.release()
            self._instance_gate.release()

    @contextmanager
    def exclusive(self, deadline: datetime | None = None) -> Iterator[None]:
        self._assert_open()
        _acquire_before_deadline(
            self._instance_exclusive_lock,
            deadline,
            "request deadline expired while waiting for exclusive backend instance admission",
        )
        acquired = 0
        try:
            for _ in range(self._instance_limit):
                _acquire_before_deadline(
                    self._instance_gate,
                    deadline,
                    "request deadline expired while waiting for exclusive backend instance admission",
                )
                acquired += 1
            if self._domain_lease is None:
                self._assert_open()
                yield
            else:
                with self._domain_lease.exclusive(deadline):
                    self._assert_open()
                    yield
        finally:
            for _ in range(acquired):
                self._instance_gate.release()
            self._instance_exclusive_lock.release()

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        if self._domain_lease is not None:
            self._domain_lease.close()
        self._instance_lease.close()

    def _assert_open(self) -> None:
        with self._close_lock:
            if self._closed:
                raise BackendAdmissionError("backend admission is closed", code="admission_closed")


def _acquire_before_deadline(
    semaphore: threading.BoundedSemaphore | threading.Lock,
    deadline: datetime | None,
    timeout_message: str,
) -> None:
    if deadline is None:
        semaphore.acquire()
        return

    now = datetime.now(deadline.tzinfo) if deadline.tzinfo is not None else datetime.now()
    remaining = (deadline - now).total_seconds()
    if remaining <= 0 or not semaphore.acquire(timeout=remaining):
        raise BackendAdmissionError(timeout_message, code="deadline_exceeded")
