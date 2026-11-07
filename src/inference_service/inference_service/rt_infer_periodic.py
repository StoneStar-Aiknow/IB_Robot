"""Client-side publisher for RT-Infer periodic admission.

The launch owner creates an unarmed mapping and arms it once after the model
roles report actual warmup completion. Clients only open and publish releases.
Closing a model session
closes its handle; it never disarms or removes owner-owned state.
"""

from __future__ import annotations

import ctypes
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path

RELEASE_MONOTONIC_NS_KEY = "rt_infer_release_monotonic_ns"


class PeriodicAdmissionError(RuntimeError):
    """The periodic admission mapping is unavailable or inconsistent."""


def _boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _identity_alive(identity: Mapping[str, object]) -> bool:
    try:
        stat = (Path("/proc") / str(identity.get("pid")) / "stat").read_text(encoding="utf-8")
        fields = stat.rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return False
    return bool(fields) and fields[0] != "Z" and str(fields[19]) == str(identity.get("start_ticks"))


class PeriodicAdmission:
    """Open one owner-armed mapping and publish PEAR request releases."""

    def __init__(
        self,
        *,
        control_path: Path,
        session_path: Path,
        state_path: Path,
        prefix: Path,
        period_ns: int,
        budget_ns: int,
    ) -> None:
        self._control_path = control_path
        self._session_path = session_path
        self._state_path = state_path
        self._prefix = prefix
        self._period_ns = period_ns
        self._budget_ns = budget_ns
        self._library = None
        self._handle = None
        self._epoch_ns = 0
        self._last_release_ns = 0

    @classmethod
    def from_launch_environment(cls) -> PeriodicAdmission | None:
        raw_path = os.environ.get("RT_INFER_PERIODIC_PATH")
        if not raw_path:
            return None
        raw_prefix = os.environ.get("RTINFER_PREFIX")
        if not raw_prefix:
            raise PeriodicAdmissionError("RTINFER_PREFIX is required for periodic admission")
        try:
            period_hz = int(os.environ["RT_INFER_PERIODIC_PERIOD_HZ"])
            budget_ns = int(os.environ["RT_INFER_PERIODIC_BUDGET_NS"])
        except (KeyError, ValueError) as exc:
            raise PeriodicAdmissionError("periodic admission period/budget environment is invalid") from exc
        if period_hz <= 0 or budget_ns <= 0:
            raise PeriodicAdmissionError("periodic admission period and budget must be positive")
        period_ns = (1_000_000_000 + period_hz - 1) // period_hz
        control_path = Path(raw_path)
        session_path = Path(
            os.environ.get("RT_INFER_PERIODIC_SESSION", str(control_path.parent / "periodic-session.json"))
        )
        state_path = Path(os.environ.get("RTINFER_STATE_FILE", "/run/ibrobot-rtinfer/state.json"))
        return cls(
            control_path=control_path,
            session_path=session_path,
            state_path=state_path,
            prefix=Path(raw_prefix),
            period_ns=period_ns,
            budget_ns=budget_ns,
        )

    def arm(self) -> None:
        """Validate the owner session and open its mapping once."""
        if self._handle is not None:
            return
        session = self._read_json(self._session_path, "periodic session")
        state = self._read_json(self._state_path, "vendor state")
        scheduler = dict(state.get("scheduler") or {})
        expected_scheduler = dict(session.get("scheduler") or {})
        checks = (
            (session.get("boot_id") == _boot_id(), "boot id"),
            (_identity_alive(session.get("owner") or {}), "owner liveness"),
            (session.get("prefix") == str(self._prefix), "prefix"),
            (
                session.get("launch_token") == os.environ.get("RTINFER_LAUNCH_TOKEN"),
                "launch token",
            ),
            (session.get("control_path") == str(self._control_path), "control path"),
            (session.get("period_ns") == self._period_ns, "period"),
            (session.get("budget_ns") == self._budget_ns, "budget"),
            (state.get("status") == "running", "scheduler status"),
            (state.get("prefix") == str(self._prefix), "vendor prefix"),
            (state.get("boot_id") == session.get("boot_id"), "vendor boot id"),
            (scheduler == expected_scheduler and bool(scheduler), "scheduler identity"),
            (_identity_alive(scheduler), "scheduler liveness"),
        )
        failed = [label for valid, label in checks if not valid]
        if failed:
            raise PeriodicAdmissionError("periodic session does not match the live scheduler: " + ", ".join(failed))
        epoch_ns = session.get("epoch_ns")
        if type(epoch_ns) is not int or epoch_ns < 0:
            raise PeriodicAdmissionError("periodic session epoch is invalid")
        bridge = self._prefix / "lib" / "periodic_control_bridge.so"
        if not bridge.is_file():
            raise PeriodicAdmissionError(f"periodic control bridge is unavailable: {bridge}")
        library = ctypes.CDLL(str(bridge), use_errno=True)
        library.periodic_shared_open.restype = ctypes.c_void_p
        library.periodic_shared_open.argtypes = [ctypes.c_char_p, ctypes.c_int]
        library.periodic_shared_release.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        library.periodic_shared_release.restype = ctypes.c_int
        library.periodic_shared_close.argtypes = [ctypes.c_void_p]
        library.periodic_shared_close.restype = None
        handle = library.periodic_shared_open(str(self._control_path).encode(), 1)
        if not handle:
            raise PeriodicAdmissionError(f"opening periodic control mapping failed: errno {ctypes.get_errno()}")
        self._library = library
        self._handle = handle
        self._epoch_ns = epoch_ns
        self._binding = {
            key: session.get(key)
            for key in (
                "boot_id",
                "prefix",
                "launch_token",
                "control_path",
                "period_ns",
                "budget_ns",
                "owner",
                "scheduler",
            )
        }

    def publish_release(self, metadata: Mapping[str, object] | None = None, *, received_ns: int | None = None) -> int:
        """Publish the request's release time without delaying the request.

        Callers may provide an explicit monotonic release timestamp in
        ``ModelRequest.metadata``.  Otherwise the timestamp is the admission
        instant at this method.  Neither path changes the request deadline or
        sleeps to align it to a synthetic grid.
        """
        if self._handle is None or self._library is None:
            raise PeriodicAdmissionError("publish before arm")
        value = None if metadata is None else metadata.get(RELEASE_MONOTONIC_NS_KEY)
        if value is None:
            release_ns = time.monotonic_ns() if received_ns is None else received_ns
        elif type(value) is int and 0 < value < 2**64:
            release_ns = value
        else:
            raise PeriodicAdmissionError(f"{RELEASE_MONOTONIC_NS_KEY} must be a positive integer")
        session = self._read_json(self._session_path, "periodic session")
        if any(session.get(key) != value for key, value in self._binding.items()):
            raise PeriodicAdmissionError("Periodic session identity/configuration changed")
        phase = session.get("phase", "active")
        if phase in ("initializing", "stopping"):
            return release_ns
        self._epoch_ns = session.get("epoch_ns")
        if phase != "active" or type(self._epoch_ns) is not int or self._epoch_ns <= 0:
            raise PeriodicAdmissionError("Invalid active periodic session")
        # Before the owner epoch, requests are cold-start work. Do not invent
        # a future release or block them waiting for the measurement grid.
        if time.monotonic_ns() < self._epoch_ns:
            return release_ns
        if self._epoch_ns and release_ns < self._epoch_ns:
            raise PeriodicAdmissionError("periodic release precedes the owner epoch")
        if release_ns < self._last_release_ns:
            raise PeriodicAdmissionError("periodic releases must be nondecreasing")
        result = self._library.periodic_shared_release(self._handle, release_ns)
        if result:
            if self._read_json(self._session_path, "periodic session").get("phase") == "stopping":
                return release_ns
            raise PeriodicAdmissionError(
                f"publish periodic release {release_ns} failed: errno {-result}; "
                "check owner epoch and releases from other PEAR sessions"
            )
        self._last_release_ns = release_ns
        return release_ns

    def close(self) -> None:
        """Close this process handle without changing owner-owned mapping state."""
        handle, self._handle = self._handle, None
        library, self._library = self._library, None
        if handle is not None and library is not None:
            library.periodic_shared_close(handle)

    @staticmethod
    def _read_json(path: Path, label: str) -> dict:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PeriodicAdmissionError(f"{label} is unavailable or invalid: {path}") from exc
        if not isinstance(value, dict):
            raise PeriodicAdmissionError(f"{label} must be a JSON object: {path}")
        return value


__all__ = ["PeriodicAdmission", "PeriodicAdmissionError", "RELEASE_MONOTONIC_NS_KEY"]
