"""Process-wide Ascend ACL initialization and per-instance context leases."""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable
from typing import Any

from inference_service.backends.errors import BackendLifecycleError, BackendLoadError
from inference_service.unified_runtime.lifecycle import check_lifecycle, lifecycle_lock

ACL_STREAM_FAST_LAUNCH = 1
_MAX_HARDWARE_PRIORITY_LEVELS = 8
ASCEND_MAX_PRIORITY = 7


def check_acl_ret(ret: object, operation: str) -> None:
    if ret != 0:
        raise RuntimeError(f"{operation} failed with ACL error code {ret}")


class AclRuntimeLease:
    """One backend instance's ACL device and context ownership."""

    def __init__(self, manager: AclRuntimeManager, acl: Any, device_id: int, context: object) -> None:
        self._manager = manager
        self.acl = acl
        self.device_id = device_id
        self.context = context
        self._closed = False
        self._device_released = False
        self._lock = threading.Lock()

    def bind_current_thread(self) -> None:
        check_lifecycle()
        check_acl_ret(self.acl.rt.set_context(self.context), "acl.rt.set_context")

    def close(self) -> None:
        try:
            with lifecycle_lock(self._lock):
                if self._closed:
                    return
                self._manager._release(self)
                self._closed = True
        except Exception as exc:
            raise BackendLifecycleError(str(exc), close_pending=True) from exc


class AclPriorityStreamPool:
    """Own one reusable ACL stream for every hardware priority level."""

    def __init__(
        self,
        lease: AclRuntimeLease,
        least_priority: int,
        greatest_priority: int,
        streams: dict[int, object],
    ) -> None:
        self._lease = lease
        self.least_priority = least_priority
        self.greatest_priority = greatest_priority
        self._streams = streams
        self._closed = False

    @classmethod
    def create(cls, lease: AclRuntimeLease, *, cleanup=None) -> AclPriorityStreamPool:
        lease.bind_current_thread()
        query_priority_range = getattr(lease.acl.rt, "device_get_stream_priority_range", None)
        if not callable(query_priority_range):
            raise BackendLoadError(
                "Ascend ACL runtime does not expose device_get_stream_priority_range",
                code="hardware_priority_unsupported",
            )
        result = query_priority_range()
        if not isinstance(result, tuple) or len(result) != 3:
            raise BackendLoadError(
                "Ascend ACL returned an invalid stream priority range",
                code="invalid_hardware_priority_range",
            )
        least_priority, greatest_priority, ret = result
        check_acl_ret(ret, "acl.rt.device_get_stream_priority_range")
        if (
            type(least_priority) is not int
            or type(greatest_priority) is not int
            or greatest_priority < 0
            or least_priority < greatest_priority
        ):
            raise BackendLoadError(
                f"Ascend ACL returned invalid stream priorities: least={least_priority!r}, "
                f"greatest={greatest_priority!r}",
                code="invalid_hardware_priority_range",
            )
        level_count = least_priority - greatest_priority + 1
        if level_count > _MAX_HARDWARE_PRIORITY_LEVELS:
            raise BackendLoadError(
                f"Ascend ACL reported {level_count} stream priority levels; maximum supported is "
                f"{_MAX_HARDWARE_PRIORITY_LEVELS}",
                code="invalid_hardware_priority_range",
            )
        if greatest_priority != 0 or least_priority != ASCEND_MAX_PRIORITY:
            raise BackendLoadError(
                "Ascend priority scheduling requires the complete hardware range [0, 7], "
                f"runtime reported [{greatest_priority}, {least_priority}]",
                code="hardware_priority_unsupported",
            )

        create_stream = getattr(lease.acl.rt, "create_stream_with_config", None)
        if not callable(create_stream):
            raise BackendLoadError(
                "Ascend ACL runtime does not expose create_stream_with_config",
                code="hardware_priority_unsupported",
            )
        streams: dict[int, object] = {}
        pool = cls(lease, least_priority, greatest_priority, streams)
        if cleanup is not None:
            cleanup.defer(pool.close)
        try:
            for priority in range(greatest_priority, least_priority + 1):
                stream, ret = create_stream(priority, ACL_STREAM_FAST_LAUNCH)
                check_acl_ret(ret, f"acl.rt.create_stream_with_config(priority={priority})")
                if stream is None:
                    raise RuntimeError(f"acl.rt.create_stream_with_config(priority={priority}) returned no stream")
                streams[priority] = stream
        except Exception as create_error:
            if cleanup is not None:
                raise
            try:
                pool.close()
            except Exception as cleanup_error:
                create_error.cleanup_owner = pool
                create_error.close_pending = True
                create_error.rollback_errors = (str(cleanup_error),)
            raise
        return pool

    @property
    def level_count(self) -> int:
        return self.least_priority - self.greatest_priority + 1

    def select(self, priority: int) -> object:
        if self._closed:
            raise RuntimeError("Ascend priority stream pool is closed")
        try:
            return self._streams[priority]
        except KeyError as exc:
            raise ValueError(f"Ascend hardware does not expose priority {priority}") from exc

    def supports(self, priority: int) -> bool:
        return not self._closed and priority in self._streams

    def close(self) -> None:
        if self._closed:
            return
        self._lease.bind_current_thread()
        for priority, stream in reversed(tuple(self._streams.items())):
            check_lifecycle()
            check_acl_ret(
                self._lease.acl.rt.synchronize_stream(stream),
                f"acl.rt.synchronize_stream(priority={priority})",
            )
            check_lifecycle()
            check_acl_ret(
                self._lease.acl.rt.destroy_stream(stream),
                f"acl.rt.destroy_stream(priority={priority})",
            )
            del self._streams[priority]
        self._closed = True


class AclRuntimeManager:
    """Reference-count process initialization and device ownership across pipelines."""

    def __init__(self, module_loader: Callable[[], Any] | None = None) -> None:
        self._module_loader = module_loader or self._import_acl
        self._lock = threading.RLock()
        self._acl: Any | None = None
        self._references = 0
        self._device_references: dict[int, int] = {}
        self._pending_leases: list[AclRuntimeLease] = []

    def acquire(self, device_id: int) -> AclRuntimeLease:
        with lifecycle_lock(self._lock):
            self._retry_pending()
            acl = self._acl or self._module_loader()
            if self._acl is None:
                check_acl_ret(acl.init(), "acl.init")
                self._acl = acl
            lease = AclRuntimeLease(self, acl, device_id, None)
            lease._device_released = True
            self._references += 1
            try:
                if self._device_references.get(device_id, 0) == 0:
                    check_acl_ret(acl.rt.set_device(device_id), "acl.rt.set_device")
                self._device_references[device_id] = self._device_references.get(device_id, 0) + 1
                lease._device_released = False
                lease.context, ret = acl.rt.create_context(device_id)
                check_acl_ret(ret, "acl.rt.create_context")
            except Exception as exc:
                self._pending_leases.append(lease)
                try:
                    self._retry_pending()
                except Exception as cleanup_error:
                    error = BackendLoadError(f"{exc}; ACL acquisition cleanup pending: {cleanup_error}")
                    error.cleanup_owner = self
                    raise error from exc
                raise
            return lease

    def _retry_pending(self):
        while self._pending_leases:
            self._pending_leases[-1].close()
            self._pending_leases.pop()

    def close(self):
        """Retry failed acquisitions before process providers can be released."""
        with lifecycle_lock(self._lock):
            self._retry_pending()
            if self._references:
                raise BackendLifecycleError("ACL leases are still owned by runtime handles", close_pending=True)

    def _release(self, lease: AclRuntimeLease) -> None:
        with lifecycle_lock(self._lock):
            acl = lease.acl
            if lease.context is not None:
                check_lifecycle()
                check_acl_ret(acl.rt.set_context(lease.context), "acl.rt.set_context")
                check_lifecycle()
                check_acl_ret(acl.rt.destroy_context(lease.context), "acl.rt.destroy_context")
                lease.context = None
            if not lease._device_released:
                device_references = self._device_references[lease.device_id]
                if device_references == 1:
                    check_lifecycle()
                    check_acl_ret(acl.rt.reset_device(lease.device_id), "acl.rt.reset_device")
                    del self._device_references[lease.device_id]
                else:
                    self._device_references[lease.device_id] = device_references - 1
                lease._device_released = True
            if self._references == 1:
                check_lifecycle()
                check_acl_ret(acl.finalize(), "acl.finalize")
                self._acl = None
            self._references -= 1

    @staticmethod
    def _import_acl() -> Any:
        try:
            return importlib.import_module("acl")
        except (ImportError, OSError) as exc:
            raise BackendLoadError(
                f"Ascend ACL dependency 'acl' is unavailable: {exc}",
                code="missing_dependency",
            ) from exc
