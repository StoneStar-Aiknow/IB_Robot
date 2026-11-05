"""RT-Infer scheduler supervisor process for a live launch.

The launch owner (``robot_config.launch_builders.rt_infer_scheduler``) starts
this process next to the model nodes when it started the device scheduler.
Supervision contract (review #3/#E):

* **Fixed startup identity**: the owner passes the startup identity and boot;
  the supervisor never adopts the first observed daemon. Every poll
  re-verifies the SAME identity. A swapped daemon (new pid), a swapped state
  file, a dead process, a state file that disappears or turns unreadable —
  each is a failure (exit 2), because after this launch has taken over the
  device, "some scheduler is running" is not enough: it must be OURS.
* **Missing/corrupt state is fail-closed**, never "healthy".
* Exit 0 never happens during normal operation (the process runs until the
  launch tears it down); any exit is either the launch's shutdown or a
  failure the owner maps to a launch Shutdown.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path

import rclpy
from rclpy.node import Node

from robot_config.rt_infer_bundle import expected_toolkit_home

STATE_FILE = Path("/run/ibrobot-rtinfer/state.json")

EXIT_SCHEDULER_UNHEALTHY = 2
EXIT_TEARDOWN_INCOMPLETE = 3


def _vendor_state() -> dict:
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


def _identity_alive(identity: dict) -> bool:
    try:
        stat = (Path("/proc") / str(identity.get("pid")) / "stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return False
    if len(stat) <= 19 or stat[0] == "Z":
        return False
    return str(stat[19]) == str(identity.get("start_ticks"))


def _same_identity(actual: dict, expected: dict) -> bool:
    return (
        bool(expected)
        and actual.get("pid") == expected.get("pid")
        and str(actual.get("start_ticks")) == str(expected.get("start_ticks"))
    )


def _state_matches_expected(state: dict, prefix: str, identity: dict, boot_id: str) -> bool:
    scheduler = state.get("scheduler")
    return (
        isinstance(scheduler, dict)
        and state.get("boot_id") == boot_id
        and state.get("status") == "running"
        and state.get("prefix") == prefix
        and _same_identity(scheduler, identity)
        and _identity_alive(scheduler)
    )


def _client_token_path(pid: int) -> Path:
    return STATE_FILE.parent / f"client-{pid}.token"


def _own_live_clients(token: str) -> tuple[list[dict], list[int]]:
    own: list[dict] = []
    unsafe: list[int] = []
    for identity in list(_vendor_state().get("clients") or []):
        if not _identity_alive(identity):
            continue
        pid = int(identity.get("pid"))
        try:
            recorded = json.loads(_client_token_path(pid).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            unsafe.append(pid)
            continue
        if (
            recorded.get("token") != token
            or recorded.get("pid") != pid
            or str(recorded.get("start_ticks")) != str(identity.get("start_ticks"))
            or recorded.get("boot_id") != Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        ):
            unsafe.append(pid)
            continue
        own.append(identity)
    return own, unsafe


def _signal_own_clients(token: str, signal_number: int = signal.SIGINT) -> bool:
    own, unsafe = _own_live_clients(token)
    if unsafe:
        return False
    for identity in own:
        pid = int(identity["pid"])
        try:
            pidfd = os.pidfd_open(pid)
        except ProcessLookupError:
            continue
        except OSError:
            return False
        try:
            # Keep pidfd open across verification and signal. If the process
            # exits/reuses its PID after verification, the fd still names the
            # old process and pidfd_send_signal returns ESRCH; it cannot kill
            # the new holder of that numeric PID.
            if not _identity_alive(identity):
                continue
            recorded = json.loads(_client_token_path(pid).read_text(encoding="utf-8"))
            if recorded.get("token") != token or not _same_identity(recorded, identity):
                return False
            if recorded.get("closing") is True:
                continue
            with suppress(ProcessLookupError):
                signal.pidfd_send_signal(pidfd, signal_number)
        except (OSError, ValueError):
            return False
        finally:
            os.close(pidfd)
    return True


def _own_wrappers_alive(token: str, boot_id: str) -> bool:
    for path in STATE_FILE.parent.glob("client-*.token"):
        with suppress(OSError, ValueError):
            record = json.loads(path.read_text())
            wrapper = record.get("wrapper") or {}
            if record.get("token") == token and wrapper and wrapper.get("boot_id") != boot_id:
                raise RuntimeError("Owned wrapper has an unverifiable boot identity")
            if record.get("token") == token and record.get("boot_id") == boot_id and _identity_alive(wrapper):
                return True
    return False


def _cleanup_periodic(prefix: str, control_path: str) -> None:
    if not control_path:
        return
    import ctypes

    path = Path(control_path)
    bridge = Path(prefix) / "lib" / "periodic_control_bridge.so"
    try:
        library = ctypes.CDLL(str(bridge), use_errno=True)
        library.periodic_shared_open.restype = ctypes.c_void_p
        library.periodic_shared_open.argtypes = [ctypes.c_char_p, ctypes.c_int]
        library.periodic_shared_disarm.argtypes = [ctypes.c_void_p]
        library.periodic_shared_disarm.restype = None
        library.periodic_shared_close.argtypes = [ctypes.c_void_p]
        library.periodic_shared_close.restype = None
        handle = library.periodic_shared_open(str(path).encode(), 1)
        if handle:
            library.periodic_shared_disarm(handle)
            library.periodic_shared_close(handle)
    except OSError:
        pass
    (path.parent / "periodic-session.json").unlink(missing_ok=True)
    path.unlink(missing_ok=True)


def _periodic_owner_document(control_path, token, prefix, identity, boot_id):
    path = Path(control_path).parent / "periodic-session.json"
    document = json.loads(path.read_text())
    lease = json.loads((STATE_FILE.parent / "launch-owner.lock").read_text())
    state = _vendor_state()
    if not (
        document.get("boot_id") == boot_id == state.get("boot_id") == lease.get("boot_id")
        and document.get("prefix") == prefix == state.get("prefix")
        and document.get("launch_token") == token == lease.get("token")
        and document.get("control_path") == control_path
        and _same_identity(document.get("scheduler") or {}, identity)
        and _same_identity(state.get("scheduler") or {}, identity)
        and _same_identity(document.get("owner") or {}, lease.get("identity") or {})
        and _identity_alive(lease.get("identity") or {})
    ):
        raise RuntimeError("Periodic mapping does not belong to this live launch lease")
    return path, document


def _write_periodic_document(path, document):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, sort_keys=True) + "\n")
    temporary.replace(path)


def _coordinate_periodic(control_path, token, prefix, identity, boot_id, *, stopping=False):
    """Only the owner coordinator arms; clients never wait for this transition."""
    if not control_path:
        return False
    import ctypes
    import fcntl

    with (STATE_FILE.parent / "periodic-owner.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        path, document = _periodic_owner_document(control_path, token, prefix, identity, boot_id)
        phase = document.get("phase")
        if phase == "stopping" or (not stopping and phase == "active"):
            return False
        if phase not in ("initializing", "active"):
            raise RuntimeError(f"Invalid periodic lifecycle phase: {phase!r}")
        if not stopping:
            expected = set(document["expected_roles"])
            if not expected or "pear" not in expected:
                raise RuntimeError("Periodic owner has no expected PEAR role")
            ready = set()
            for client in _vendor_state().get("clients", []):
                if not _identity_alive(client):
                    continue
                try:
                    record = json.loads(_client_token_path(client["pid"]).read_text())
                except (OSError, ValueError):
                    continue
                if (
                    record.get("token") == token
                    and record.get("boot_id") == boot_id
                    and _same_identity(record, client)
                    and record.get("ready") is True
                ):
                    ready.add(record.get("role"))
            if not expected <= ready:
                return False
        library = ctypes.CDLL(str(Path(prefix) / "lib/periodic_control_bridge.so"), use_errno=True)
        library.periodic_shared_open.argtypes = [ctypes.c_char_p, ctypes.c_int]
        library.periodic_shared_open.restype = ctypes.c_void_p
        library.periodic_shared_arm.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        library.periodic_shared_arm.restype = ctypes.c_int
        library.periodic_shared_disarm.argtypes = [ctypes.c_void_p]
        library.periodic_shared_disarm.restype = None
        library.periodic_shared_close.argtypes = [ctypes.c_void_p]
        library.periodic_shared_close.restype = None
        handle = library.periodic_shared_open(control_path.encode(), 1)
        if not handle:
            raise RuntimeError(f"Cannot open owned periodic mapping: errno {ctypes.get_errno()}")
        try:
            if stopping:
                document["phase"] = "stopping"
                _write_periodic_document(path, document)
                library.periodic_shared_disarm(handle)
            else:
                epoch = time.monotonic_ns() + document["period_ns"]
                result = library.periodic_shared_arm(handle, epoch, document["period_ns"], document["budget_ns"])
                if result:
                    raise RuntimeError(f"Owner arm failed: errno {-result}")
                document.update(epoch_ns=epoch, phase="active", activated_ns=time.monotonic_ns())
                _write_periodic_document(path, document)
        finally:
            library.periodic_shared_close(handle)
    return True


def _teardown(
    *,
    token: str,
    prefix: str,
    control_tool: str,
    timeout_sec: float,
    periodic_control_path: str,
    expected_identity: dict,
    expected_boot_id: str,
) -> bool:
    """Drain this launch's clients and stop only its scheduler."""

    def owned_state() -> bool:
        state = _vendor_state()
        try:
            lease = json.loads((STATE_FILE.parent / "launch-owner.lock").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return (
            lease.get("token") == token
            and lease.get("boot_id") == expected_boot_id
            and _identity_alive(lease.get("identity") or {})
            and state.get("boot_id") == expected_boot_id
            and state.get("prefix") == prefix
            and _same_identity(state.get("scheduler") or {}, expected_identity)
        )

    if not owned_state():
        return False

    def event(stage):
        print(
            "RT_INFER_OWNER_EVENT " + json.dumps({"stage": stage, "token": token, "monotonic_ns": time.monotonic_ns()}),
            flush=True,
        )

    # Release the reservation before waiting for background kernels, otherwise
    # an idle PEAR role can prevent those kernels from ever draining.
    if periodic_control_path:
        _coordinate_periodic(periodic_control_path, token, prefix, expected_identity, expected_boot_id, stopping=True)
    event("periodic_disarmed")
    deadline = time.monotonic() + timeout_sec
    if not _signal_own_clients(token):
        return False
    while time.monotonic() < deadline:
        if not owned_state():
            return False
        own, unsafe = _own_live_clients(token)
        if unsafe:
            return False
        if not own and not _own_wrappers_alive(token, expected_boot_id):
            break
        time.sleep(0.1)
    else:
        # Do not report a forced kernel/client termination as a clean drain.
        # Retain the mapping and vendor state for explicit recovery.
        return False
    remaining = [identity for identity in list(_vendor_state().get("clients") or []) if _identity_alive(identity)]
    if remaining or not owned_state():
        return False
    event("clients_and_wrappers_drained")
    environment = os.environ.copy()
    toolkit_home = expected_toolkit_home(Path(prefix))
    if toolkit_home is not None:
        environment["ASCEND_TOOLKIT_HOME"] = toolkit_home
    result = subprocess.run(
        ["python3", control_tool, "--prefix", prefix, "stop"],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
        env=environment,
    )
    event("vendor_stop_returned")
    if result.returncode != 0 or not owned_state() or _vendor_state().get("status") != "stopped":
        print(f"RT-Infer vendor stop was not clean: {result.stderr or result.stdout}", flush=True)
        return False
    _cleanup_periodic(prefix, periodic_control_path)
    for record_path in STATE_FILE.parent.glob("client-*.token"):
        with suppress(OSError, ValueError):
            record = json.loads(record_path.read_text())
            if record.get("token") == token and not _identity_alive(record):
                record_path.unlink()
    return True


def main() -> int:
    rclpy.init()
    node = None
    try:
        node = Node("rt_infer_scheduler_supervisor")
        poll_interval = float(node.declare_parameter("poll_interval_sec", 1.0).value)
        expected_prefix = str(node.declare_parameter("expected_prefix", "").value)
        expected_boot_id = str(node.declare_parameter("expected_boot_id", "").value)
        token = str(node.declare_parameter("launch_token", "").value)
        control_path = str(node.declare_parameter("periodic_control_path", "").value)
        expected_identity = {
            "pid": int(node.declare_parameter("expected_scheduler_pid", 0).value),
            "start_ticks": str(node.declare_parameter("expected_scheduler_start_ticks", 0).value),
        }
        pinned = expected_identity
        while rclpy.ok():
            state = _vendor_state()
            if not state:
                node.get_logger().error("RT-Infer vendor state is missing or unreadable; failing closed")
                return EXIT_SCHEDULER_UNHEALTHY
            identity = dict(state.get("scheduler") or {})
            status = state.get("status")
            healthy = _state_matches_expected(state, expected_prefix, pinned, expected_boot_id)
            if not healthy:
                node.get_logger().error(
                    f"RT-Infer scheduler is not verifiably healthy (status={status!r}); failing closed"
                )
                return EXIT_SCHEDULER_UNHEALTHY
            if not _same_identity(identity, pinned):
                node.get_logger().error(
                    f"RT-Infer scheduler identity changed at runtime (was pid {pinned.get('pid')}, "
                    f"now pid {identity.get('pid')}); a swapped daemon is a failure"
                )
                return EXIT_SCHEDULER_UNHEALTHY
            if _coordinate_periodic(control_path, token, expected_prefix, pinned, expected_boot_id):
                node.get_logger().info("Periodic admission active after all participating model roles warmed up")
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
