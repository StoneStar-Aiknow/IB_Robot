"""Resident ROS bridge shared by successive ``robot-skill`` invocations.

Every ``robot-skill`` call used to create a fresh rclpy context: a new DDS
participant that the running Gateway had to discover before the first
service response, about 1 s (up to 3 s) with Fast DDS.  This module keeps one
:class:`~robot_skill_cli.ros_bridge.RosBridge` alive in a per-user background
process and lets each CLI process drive it over a private Unix socket, so the
discovery cost is paid once instead of on every command.

The embodied pipeline launch owns the daemon (``robot-skill-daemon
--config-path <snapshot> --replace --idle-timeout 0``): it pre-opens and warms
the bridge for that robot so even the first command finds it ready, and it
exits with the launch.  ``robot-skill`` only connects to a running daemon by
default and never leaves a background process behind; starting one on demand
is an explicit opt-in (``IBROBOT_ROBOT_SKILL_DAEMON=spawn``) for sessions
without the pipeline.

Only the ROS transport moves into the resident process.  Argument parsing,
catalog loading, signal handling, stop-state files and stdout stay in the CLI
process, exactly as before.  The client implements the same methods as
``RosBridge``; goal futures, goal handles and feedback callbacks are proxied
and ROS result messages are returned as attribute namespaces.

Safety properties:

* The socket lives in a ``0700`` directory owned by the user; every peer must
  pass an ``SO_PEERCRED`` uid check (no credentials means no access).
* The socket name is derived from the variables that select the ROS graph
  (``ROS_DOMAIN_ID``, ``RMW_IMPLEMENTATION``, SROS2 settings ...) and from the
  installed code, so a client never reaches a daemon on another domain, RMW,
  security identity or build.
* One ROS participant is shared per Gateway transport; the caller token is
  applied per client session.  The daemon removes caller tokens and other
  per-invocation secrets from its own environment, accepts only the known
  bridge parameters and exposes only an explicit set of bridge methods.
* Stopping or replacing a daemon drains it: it stops accepting clients at once
  but keeps serving connected ones, so a running command can still observe and
  cancel its goal.
* If the connection to the daemon is lost anyway, the client fails its pending
  goal futures and sends the follow-up task-id calls (status, cancel, execution
  lookup) through an in-process ``RosBridge``, so the command's existing
  transport-failure path still cancels and converges on the goal.
* Any failure to reach or start the daemon falls back to an in-process
  ``RosBridge``; behavior then matches the previous per-process bridge.
* ``IBROBOT_ROBOT_SKILL_DAEMON`` selects the mode: unset or ``1`` connects
  to a running daemon only, ``spawn`` also starts one on demand, ``0``
  disables the daemon entirely.
"""

from __future__ import annotations

import argparse
import array
import contextlib
import errno
import hashlib
import importlib.util
import itertools
import json
import os
import queue
import socket
import stat
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROTOCOL_VERSION = 3
# Daemon mode: unset/"1" connect only, "spawn" connect or start, "0" off.
DISABLE_ENV = "IBROBOT_ROBOT_SKILL_DAEMON"
_OFF_VALUES = frozenset({"0", "false", "no", "off"})
_SPAWN_VALUES = frozenset({"spawn", "auto"})
IDLE_TIMEOUT_ENV = "IBROBOT_ROBOT_SKILL_DAEMON_IDLE_SEC"
DEFAULT_IDLE_TIMEOUT_SEC = 900.0
DIRECTORY_ENV = "IBROBOT_ROBOT_SKILL_DAEMON_DIR"
SPAWN_WAIT_SEC = 5.0
# A new DDS participant can see the Gateway before the Gateway has discovered
# its reply reader; a request sent in that window loses its response (seen as
# a 30 s RESULT_TIMEOUT with Fast DDS).  A fresh bridge first sends short,
# read-only status probes until one is answered.
WARM_UP_ATTEMPT_SEC = 1.0
WARM_UP_TOTAL_SEC = 15.0
# A launch-started daemon may come up before the Gateway; keep probing longer.
PREWARM_TOTAL_SEC = 120.0
# How long an ``open`` waits for a running warm-up before using the bridge
# anyway (for example while the Gateway is down).
OPEN_WARM_WAIT_SEC = 5.0
# Warm-up budget of the in-process bridge a client builds after losing the daemon.
FALLBACK_WARM_UP_SEC = 5.0
# A stopped or replaced daemon keeps serving connected clients this long.
DRAIN_MAX_SEC = 600.0
# On SIGINT/SIGTERM (launch shutdown) the Gateway goes away too; drain briefly.
SIGNAL_DRAIN_SEC = 3.0
# Outbound messages buffered per client before a stalled client is dropped.
SEND_QUEUE_MAX = 4096
REQUEST_WORKERS = 8
LOG_MAX_BYTES = 1 << 20

# Environment that selects which ROS graph, and under which identity, a
# participant joins.  A daemon only serves clients whose values are identical,
# because they are part of the socket name.  Participant-local transport
# tuning that does not change the graph (for example
# FASTDDS_BUILTIN_TRANSPORTS, which the embodied launch forces to UDPv4 for
# every node it starts) is deliberately excluded, so a launch-started daemon
# and a CLI from the user's shell share one socket.
TRANSPORT_ENV = (
    "ROS_DOMAIN_ID",
    "RMW_IMPLEMENTATION",
    "ROS_LOCALHOST_ONLY",
    "ROS_AUTOMATIC_DISCOVERY_RANGE",
    "ROS_STATIC_PEERS",
    "ROS_DISCOVERY_SERVER",
    "CYCLONEDDS_URI",
    "FASTRTPS_DEFAULT_PROFILES_FILE",
    "FASTDDS_DEFAULT_PROFILES_FILE",
    "ROS_SECURITY_ENABLE",
    "ROS_SECURITY_STRATEGY",
    "ROS_SECURITY_KEYSTORE",
    "ROS_SECURITY_ENCLAVE_OVERRIDE",
)
# Per-invocation state that must never live in the long-lived daemon.
_PRIVATE_ENV_PREFIXES = ("IBROBOT_STOP_", "IBROBOT_HERMES_", "IBROBOT_EXECUTION_TOKEN_")

# RosBridge constructor parameters a client may choose (see cli.bridge_kwargs).
# Caller identity is not among them: it is sent per session.
BRIDGE_PARAMETERS = frozenset(
    {
        "status_service",
        "snapshot_service",
        "reload_service",
        "validate_skill_service",
        "skill_action",
        "primitive_action",
        "validate_primitive_service",
        "plan_service",
        "validate_plan_service",
        "confirm_plan_service",
        "execute_plan_action",
        "execute_program_action",
        "get_execution_service",
        "get_execution_events_service",
        "cancel_execution_service",
        "caller_scope",
        "robot_id",
        "start_visual_game_service",
        "get_visual_game_result_service",
    }
)
# RosBridge methods a client may invoke remotely.
REMOTE_METHODS = frozenset(
    {
        "get_status",
        "validate_skill",
        "get_skill_snapshot",
        "reload_skill_catalog",
        "plan_agent_command",
        "prepare_agent_plan",
        "validate_agent_plan",
        "confirm_agent_plan",
        "wait_for_execute_plan_server",
        "wait_for_program_server",
        "send_program_goal",
        "get_execution",
        "get_execution_events",
        "cancel_execution",
        "wait_for_agent_plan_interfaces",
        "wait_for_public_request_interfaces",
        "wait_for_visual_game_interfaces",
        "send_agent_plan_goal",
        "cancel_agent_plan",
        "get_agent_plan_result",
        "wait_for_skill_server",
        "send_skill_goal",
        "cancel_task",
        "start_visual_game",
        "get_visual_game_result",
        "cancel_goal",
    }
)
# Calls addressed by task/request id, which an in-process bridge can still
# make after the daemon connection is lost (the stop path needs them).
RECOVERABLE_METHODS = frozenset(
    {
        "get_status",
        "cancel_task",
        "cancel_agent_plan",
        "get_agent_plan_result",
        "get_execution",
        "get_execution_events",
        "cancel_execution",
    }
)


class DaemonUnavailable(RuntimeError):
    """The resident bridge cannot be used; the caller falls back in-process."""


# --------------------------------------------------------------------------- #
# Wire encoding
# --------------------------------------------------------------------------- #


def _encode_plain(value: Any) -> Any:
    """Encode a JSON-compatible value, turning ROS messages into mappings."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, dict):
        return {str(key): _encode_plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_encode_plain(item) for item in value]
    if isinstance(value, bytes | bytearray):
        return list(value)
    if isinstance(value, array.array):
        return value.tolist()
    if hasattr(value, "get_fields_and_field_types"):
        return {"__msg__": {name: _encode_plain(getattr(value, name)) for name in value.get_fields_and_field_types()}}
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _encode_plain(tolist())
    raise TypeError(f"cannot encode {type(value).__name__} for the resident bridge")


def _decode_plain(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_plain(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"__msg__"}:
            return SimpleNamespace(**{name: _decode_plain(item) for name, item in value["__msg__"].items()})
        return {key: _decode_plain(item) for key, item in value.items()}
    return value


def _encode_error(exc: BaseException) -> dict[str, Any]:
    from robot_skill_cli.ros_bridge import BridgeError

    if isinstance(exc, BridgeError):
        return {"type": "BridgeError", "code": exc.code, "message": str(exc), "exit_code": exc.exit_code}
    kind = type(exc).__name__
    if kind not in {"ValueError", "TypeError", "KeyError", "TimeoutError"}:
        kind = "RuntimeError"
    return {"type": kind, "message": str(exc)}


def _decode_error(payload: dict[str, Any]) -> BaseException:
    from robot_skill_cli.ros_bridge import BridgeError

    message = str(payload.get("message", ""))
    if payload.get("type") == "BridgeError":
        return BridgeError(str(payload.get("code", "ROS_UNAVAILABLE")), message, exit_code=int(payload["exit_code"]))
    return {
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "TimeoutError": TimeoutError,
    }.get(str(payload.get("type")), RuntimeError)(message)


def _disconnected() -> BaseException:
    from robot_skill_cli.output import EXIT_ROS_UNAVAILABLE
    from robot_skill_cli.ros_bridge import BridgeError

    return BridgeError("ROS_UNAVAILABLE", "resident ROS bridge disconnected", exit_code=EXIT_ROS_UNAVAILABLE)


def _frame(message: dict[str, Any]) -> bytes:
    return (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")


def _send(sock: socket.socket, lock: threading.Lock, message: dict[str, Any]) -> None:
    data = _frame(message)
    with lock:
        sock.sendall(data)


def _read_lines(sock: socket.socket):
    buffer = b""
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            return
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------- #
# Socket location
# --------------------------------------------------------------------------- #


def _package_stamp(package: str) -> list[Any]:
    """Summarize a package's Python sources so a rebuild selects a new daemon."""
    spec = importlib.util.find_spec(package)
    locations = list(spec.submodule_search_locations or []) if spec is not None else []
    if not locations:
        return [package, None]
    newest = count = size = 0
    pending = [locations[0]]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                if entry.is_dir():
                    if entry.name != "__pycache__":
                        pending.append(entry.path)
                elif entry.name.endswith(".py"):
                    info = entry.stat()
                    newest = max(newest, info.st_mtime_ns)
                    count += 1
                    size += info.st_size
    return [locations[0], newest, count, size]


def _code_fingerprint() -> list[Any]:
    # The daemon keeps running the code and message definitions it imported,
    # so the interpreter environment and the resolved package sources are its
    # identity.  Search-path strings are not: the launch and a Hermes wrapper
    # spell PYTHONPATH differently while resolving the same installed files.
    return [
        sys.prefix,
        sys.version,
        *(_package_stamp(package) for package in ("robot_skill_cli", "embodied_common", "ibrobot_msgs")),
    ]


def _runtime_directory() -> Path:
    # Do not follow XDG_RUNTIME_DIR: SSH sessions, env-cleared shells and the
    # robot launch do not all export it, and both ends must agree on the path.
    override = os.environ.get(DIRECTORY_ENV)
    user_runtime = Path(f"/run/user/{os.getuid()}")
    if override:
        root = Path(override)
    elif user_runtime.is_dir() and user_runtime.stat().st_uid == os.getuid():
        root = user_runtime / "ibrobot"
    else:
        root = Path(f"/tmp/ibrobot-{os.getuid()}")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = os.lstat(root)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise DaemonUnavailable(f"unsafe resident bridge directory: {root}")
    return root


def socket_path() -> Path:
    identity = {
        "protocol": PROTOCOL_VERSION,
        "uid": os.getuid(),
        "transport": {name: os.environ.get(name) for name in TRANSPORT_ENV},
        "code": _code_fingerprint(),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    return _runtime_directory() / f"robot-skill-{digest}.sock"


def daemon_enabled() -> bool:
    return os.environ.get(DISABLE_ENV, "1").strip().lower() not in _OFF_VALUES


def daemon_spawn_allowed() -> bool:
    """Return whether ``robot-skill`` may start a daemon itself.

    Off by default: the embodied launch owns the daemon's lifetime, and a
    command (often an Agent tool call) must not leave a detached process
    behind unless the operator opted in.
    """
    return os.environ.get(DISABLE_ENV, "").strip().lower() in _SPAWN_VALUES


def _peer_uid(sock: socket.socket) -> int | None:
    """Return the peer's uid, or ``None`` when it cannot be established."""
    so_peercred = getattr(socket, "SO_PEERCRED", None)
    if so_peercred is None:
        return None
    try:
        raw = sock.getsockopt(socket.SOL_SOCKET, so_peercred, struct.calcsize("3i"))
        return struct.unpack("3i", raw)[1]
    except (OSError, struct.error):
        return None


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class RemoteFuture:
    """Client view of a daemon-side rclpy future."""

    def __init__(self, client: _Connection, future_id: int) -> None:
        self._client = client
        self.future_id = future_id
        self._event = threading.Event()
        self._value: Any = None
        self._error: BaseException | None = None

    def _resolve(self, value: Any = None, error: BaseException | None = None) -> None:
        self._value = value
        self._error = error
        self._event.set()

    def done(self) -> bool:
        return self._event.is_set()

    def result(self) -> Any:
        if not self._event.is_set():
            return None
        if self._error is not None:
            raise self._error
        return self._value

    def wait(self, timeout_sec: float) -> bool:
        return self._event.wait(timeout_sec)

    def cancel(self) -> None:
        with contextlib.suppress(Exception):
            self._client.request({"op": "future_cancel", "future": self.future_id})


class RemoteGoalHandle:
    """Client view of an rclpy ``ClientGoalHandle``."""

    def __init__(self, client: _Connection, handle_id: int, *, accepted: bool, status: int) -> None:
        self._client = client
        self.handle_id = handle_id
        self.accepted = accepted
        self.status = status

    def get_result_async(self) -> RemoteFuture:
        return self._client.request({"op": "handle_call", "handle": self.handle_id, "method": "get_result_async"})

    def cancel_goal_async(self) -> RemoteFuture:
        return self._client.request({"op": "handle_call", "handle": self.handle_id, "method": "cancel_goal_async"})


class _Connection:
    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._send_lock = threading.Lock()
        self._ids = itertools.count(1)
        self._pending: dict[int, list[Any]] = {}
        self._futures: dict[int, RemoteFuture] = {}
        self._early_futures: dict[int, dict[str, Any]] = {}
        self._callbacks: dict[int, Any] = {}
        self._state_lock = threading.Lock()
        self._closed = threading.Event()
        self._reader = threading.Thread(target=self._read, name="robot-skill-bridge-client", daemon=True)
        self._reader.start()

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def _read(self) -> None:
        try:
            for message in _read_lines(self._sock):
                self._dispatch(message)
        except (OSError, ValueError):
            pass
        finally:
            self._closed.set()
            self._fail_all()

    def _fail_all(self) -> None:
        # Every outstanding future fails, so callers take their existing
        # transport-failure paths (which cancel and converge by task id).
        error = _disconnected()
        with self._state_lock:
            pending = list(self._pending.values())
            futures = list(self._futures.values())
            self._early_futures.clear()
        for slot in pending:
            slot[1] = {"ok": False, "error": _encode_error(error)}
            slot[0].set()
        for future in futures:
            if not future.done():
                future._resolve(error=error)

    def _dispatch(self, message: dict[str, Any]) -> None:
        event = message.get("event")
        if event == "future":
            with self._state_lock:
                future = self._futures.pop(int(message["future"]), None)
                if future is None:
                    self._early_futures[int(message["future"])] = message
                    return
            self._settle(future, message)
        elif event == "callback":
            callback = self._callbacks.get(int(message["callback"]))
            if callback is not None:
                with contextlib.suppress(Exception):
                    callback(_decode_plain(message.get("value")))
        elif "id" in message:
            with self._state_lock:
                slot = self._pending.get(int(message["id"]))
            if slot is not None:
                slot[1] = message
                slot[0].set()

    def _settle(self, future: RemoteFuture, message: dict[str, Any]) -> None:
        if message.get("ok"):
            future._resolve(value=self._decode_value(message.get("value")))
        else:
            future._resolve(error=_decode_error(message.get("error") or {}))

    def _decode_value(self, value: Any) -> Any:
        if isinstance(value, dict) and set(value) == {"__ref__"}:
            ref = value["__ref__"]
            if ref["kind"] == "future":
                future = RemoteFuture(self, int(ref["id"]))
                with self._state_lock:
                    early = self._early_futures.pop(future.future_id, None)
                    if early is None:
                        self._futures[future.future_id] = future
                if early is not None:
                    self._settle(future, early)
                elif self._closed.is_set():
                    future._resolve(error=_disconnected())
                return future
            if ref["kind"] == "handle":
                return RemoteGoalHandle(
                    self, int(ref["id"]), accepted=bool(ref["accepted"]), status=int(ref.get("status", 0))
                )
        return _decode_plain(value)

    def _encode_argument(self, value: Any) -> Any:
        if isinstance(value, RemoteFuture):
            return {"__ref__": {"kind": "future", "id": value.future_id}}
        if isinstance(value, RemoteGoalHandle):
            return {"__ref__": {"kind": "handle", "id": value.handle_id}}
        if callable(value):
            callback_id = next(self._ids)
            self._callbacks[callback_id] = value
            return {"__ref__": {"kind": "callback", "id": callback_id}}
        return _encode_plain(value)

    def request(self, message: dict[str, Any], *, timeout_sec: float | None = None) -> Any:
        request_id = next(self._ids)
        slot: list[Any] = [threading.Event(), None]
        with self._state_lock:
            self._pending[request_id] = slot
        try:
            deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
            try:
                _send(self._sock, self._send_lock, {**message, "id": request_id})
            except OSError:
                self._closed.set()
            while not slot[0].wait(0.2):
                if self._closed.is_set() and not slot[0].is_set():
                    raise _disconnected()
                if deadline is not None and time.monotonic() >= deadline:
                    raise DaemonUnavailable(f"resident bridge did not answer {message.get('op')}")
        finally:
            with self._state_lock:
                self._pending.pop(request_id, None)
        reply = slot[1]
        if not reply.get("ok"):
            raise _decode_error(reply.get("error") or {})
        return self._decode_value(reply.get("value"))

    def call(self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        return self.request(
            {
                "op": "call",
                "method": method,
                "args": [self._encode_argument(item) for item in args],
                "kwargs": {key: self._encode_argument(item) for key, item in kwargs.items()},
            }
        )

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self._sock.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(OSError):
            self._sock.close()
        self._reader.join(timeout=1.0)


def _connect(path: Path, *, timeout_sec: float = 1.0) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout_sec)
    try:
        sock.connect(str(path))
    except OSError:
        sock.close()
        raise
    sock.settimeout(None)
    return sock


def _daemon_environment(environ: dict[str, str] | None = None) -> dict[str, str]:
    source = os.environ if environ is None else environ
    return {
        name: value
        for name, value in source.items()
        if not any(name.startswith(prefix) for prefix in _PRIVATE_ENV_PREFIXES)
    }


def _scrub_private_environment() -> None:
    """Drop per-invocation secrets the daemon may have inherited (e.g. from launch)."""
    for name in list(os.environ):
        if any(name.startswith(prefix) for prefix in _PRIVATE_ENV_PREFIXES):
            del os.environ[name]


def _open_log(path: Path):
    log_path = path.with_suffix(".log")
    with contextlib.suppress(OSError):
        if log_path.stat().st_size > LOG_MAX_BYTES:
            os.replace(log_path, log_path.with_suffix(".log.1"))
    return open(log_path, "ab")


def _spawn(path: Path) -> None:
    with _open_log(path) as log:
        subprocess.Popen(
            [sys.executable, "-m", "robot_skill_cli.bridge_daemon", "--socket", str(path)],
            env=_daemon_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=log,
            close_fds=True,
            start_new_session=True,
            cwd="/",
        )


@contextlib.contextmanager
def _socket_lock(path: Path):
    """Serialize spawning, replacing and binding the daemon for one socket."""
    import fcntl

    with open(path.with_suffix(".lock"), "a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield


def connect_or_spawn(*, spawn: bool = True) -> socket.socket:
    path = socket_path()
    try:
        return _connect(path)
    except OSError as exc:
        if not spawn:
            raise DaemonUnavailable(str(exc)) from exc
    with _socket_lock(path):
        try:
            return _connect(path)
        except OSError:
            pass
        _spawn(path)
        deadline = time.monotonic() + SPAWN_WAIT_SEC
        while time.monotonic() < deadline:
            try:
                return _connect(path)
            except OSError:
                time.sleep(0.02)
    raise DaemonUnavailable(f"resident bridge did not start: {path}")


def _not_started():
    from robot_skill_cli.output import EXIT_ROS_UNAVAILABLE
    from robot_skill_cli.ros_bridge import BridgeError

    return BridgeError("ROS_UNAVAILABLE", "ROS bridge is not started", exit_code=EXIT_ROS_UNAVAILABLE)


class ResidentBridge:
    """Drop-in ``RosBridge`` that prefers the resident daemon.

    ``start()`` connects to the daemon (starting it only when
    :func:`daemon_spawn_allowed`) and opens a bridge with the same parameters
    there.  If that is not possible, an in-process
    ``RosBridge`` is used instead, so callers see identical behavior apart from
    the startup cost.  If the daemon connection is lost later, task-id calls
    continue through an in-process bridge built on demand.
    """

    def __init__(self, **bridge_kwargs: Any) -> None:
        self._bridge_kwargs = bridge_kwargs
        shared = dict(bridge_kwargs)
        token_env = str(shared.pop("caller_token_env", "") or "")
        self._caller_token = str(shared.pop("caller_token", "") or "") or (
            os.environ.get(token_env, "") if token_env else ""
        )
        self._shared_kwargs = shared
        self._connection: _Connection | None = None
        self._local = None
        self._fallback = None
        self._fallback_lock = threading.Lock()
        self.start_error = ""
        self.resident = False

    def start(self) -> bool:
        if daemon_enabled():
            try:
                connection = _Connection(connect_or_spawn(spawn=daemon_spawn_allowed()))
                try:
                    started = connection.request(
                        {
                            "op": "open",
                            "version": PROTOCOL_VERSION,
                            "bridge": _encode_plain(self._shared_kwargs),
                            "caller_token": self._caller_token,
                        },
                        timeout_sec=30.0,
                    )
                except BaseException:
                    connection.close()
                    raise
                if started["started"]:
                    self._connection = connection
                    self.resident = True
                    return True
                connection.close()
                # The daemon runs the same code and environment; a second,
                # in-process attempt would fail the same way.
                self.start_error = str(started.get("start_error", ""))
                return False
            except (DaemonUnavailable, OSError, RuntimeError, ValueError) as exc:
                self.start_error = ""
                if os.environ.get("IBROBOT_ROBOT_SKILL_DAEMON_DEBUG") == "1":
                    print(f"robot-skill: resident bridge unavailable: {exc}", file=sys.stderr)
        from robot_skill_cli.ros_bridge import RosBridge

        self._local = RosBridge(**self._bridge_kwargs)
        started = self._local.start()
        self.start_error = self._local.start_error
        return started

    def wait_future(self, future, *, timeout_sec: float, interrupt_event: threading.Event | None = None) -> bool:
        if self._local is not None:
            return self._local.wait_future(future, timeout_sec=timeout_sec, interrupt_event=interrupt_event)
        deadline = time.monotonic() + timeout_sec
        while True:
            if future.done():
                return True
            if interrupt_event is not None and interrupt_event.is_set():
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            if isinstance(future, RemoteFuture):
                future.wait(min(0.01, remaining))
            else:
                time.sleep(min(0.01, remaining))

    def cancel_goal(self, goal_handle, result_future, *, timeout_sec: float) -> bool:
        if self._local is not None:
            return self._local.cancel_goal(goal_handle, result_future, timeout_sec=timeout_sec)
        # A finished future needs no cancel.  After a lost daemon connection
        # the remote result future has already failed, so the caller reads no
        # terminal result and cancels by task id through the fallback bridge.
        if result_future.done():
            return True
        return self._forward("cancel_goal", (goal_handle, result_future), {"timeout_sec": timeout_sec})

    def close(self) -> None:
        if self._local is not None:
            self._local.close()
            self._local = None
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._fallback is not None:
            self._fallback.close()
            self._fallback = None

    def _fallback_bridge(self):
        """Build (once) the in-process bridge used after the daemon connection is lost."""
        with self._fallback_lock:
            if self._fallback is None:
                from robot_skill_cli.ros_bridge import RosBridge

                bridge = RosBridge(**self._bridge_kwargs)
                if not bridge.start():
                    raise _disconnected()
                BridgeDaemon._warm_up(bridge, FALLBACK_WARM_UP_SEC)
                self._fallback = bridge
            return self._fallback

    def _forward(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        from robot_skill_cli.ros_bridge import BridgeError

        connection = self._connection
        if connection is None:
            raise _not_started()
        if not connection.closed:
            try:
                return connection.call(name, args, kwargs)
            except BridgeError as exc:
                if not connection.closed or name not in RECOVERABLE_METHODS:
                    raise
                del exc
        if name not in RECOVERABLE_METHODS:
            raise _disconnected()
        return getattr(self._fallback_bridge(), name)(*args, **kwargs)

    def __getattr__(self, name: str):
        if name.startswith("_") or name not in REMOTE_METHODS:
            raise AttributeError(name)
        local = self.__dict__.get("_local")
        if local is not None:
            return getattr(local, name)

        def _forward(*args, **kwargs):
            return self._forward(name, args, kwargs)

        return _forward


# --------------------------------------------------------------------------- #
# Daemon
# --------------------------------------------------------------------------- #


class _ClientSession:
    """Objects one client may reference; dropped (not cancelled) on disconnect."""

    def __init__(self, server: BridgeDaemon, sock: socket.socket) -> None:
        self.server = server
        self.sock = sock
        self.bridge = None
        self.ids = itertools.count(1)
        self.futures: dict[int, Any] = {}
        self.handles: dict[int, Any] = {}
        self.lock = threading.Lock()
        self.alive = True
        # ROS callbacks only enqueue; a dedicated writer talks to the socket,
        # so a client that stops reading cannot stall the shared executor.
        self.outbox: queue.Queue = queue.Queue(maxsize=SEND_QUEUE_MAX)
        self.writer = threading.Thread(target=self._write, name="robot-skill-daemon-writer", daemon=True)
        self.writer.start()

    def _drop(self) -> None:
        self.alive = False
        with contextlib.suppress(OSError):
            self.sock.shutdown(socket.SHUT_RDWR)

    def send(self, message: dict[str, Any]) -> None:
        if not self.alive:
            return
        try:
            self.outbox.put_nowait(_frame(message))
        except queue.Full:
            self._drop()

    def _write(self) -> None:
        while True:
            data = self.outbox.get()
            if data is None:
                return
            try:
                self.sock.sendall(data)
            except OSError:
                self._drop()
                return

    # -- object references -------------------------------------------------

    def encode_result(self, value: Any) -> Any:
        if hasattr(value, "add_done_callback") and hasattr(value, "done") and hasattr(value, "result"):
            future_id = next(self.ids)
            with self.lock:
                self.futures[future_id] = value
            value.add_done_callback(lambda done, future_id=future_id: self._future_done(future_id, done))
            return {"__ref__": {"kind": "future", "id": future_id}}
        if hasattr(value, "get_result_async") and hasattr(value, "accepted"):
            handle_id = next(self.ids)
            with self.lock:
                self.handles[handle_id] = value
            return {
                "__ref__": {
                    "kind": "handle",
                    "id": handle_id,
                    "accepted": bool(value.accepted),
                    "status": int(getattr(value, "status", 0) or 0),
                }
            }
        return _encode_plain(value)

    def _future_done(self, future_id: int, future) -> None:
        try:
            value = self.encode_result(future.result())
        except BaseException as exc:
            self.send({"event": "future", "future": future_id, "ok": False, "error": _encode_error(exc)})
        else:
            self.send({"event": "future", "future": future_id, "ok": True, "value": value})

    def decode_argument(self, value: Any) -> Any:
        if isinstance(value, dict) and set(value) == {"__ref__"}:
            ref = value["__ref__"]
            kind, ref_id = ref["kind"], int(ref["id"])
            if kind == "future":
                with self.lock:
                    return self.futures[ref_id]
            if kind == "handle":
                with self.lock:
                    return self.handles[ref_id]
            if kind == "callback":
                return lambda payload, ref_id=ref_id: self.send(
                    {"event": "callback", "callback": ref_id, "value": _encode_plain(payload)}
                )
            raise ValueError(f"unknown reference kind: {kind}")
        if isinstance(value, list):
            return [self.decode_argument(item) for item in value]
        if isinstance(value, dict):
            return {key: self.decode_argument(item) for key, item in value.items()}
        return value

    # -- requests ----------------------------------------------------------

    def handle(self, message: dict[str, Any]) -> Any:
        op = message.get("op")
        if op == "shutdown":
            self.server.request_stop(drain_sec=DRAIN_MAX_SEC)
            return None
        if op == "open":
            if int(message.get("version", -1)) != PROTOCOL_VERSION:
                raise ValueError("resident bridge protocol mismatch")
            parameters = message.get("bridge")
            if not isinstance(parameters, dict) or not set(parameters) <= BRIDGE_PARAMETERS:
                raise ValueError("unsupported resident bridge parameters")
            if not all(isinstance(item, str) for item in parameters.values()):
                raise ValueError("resident bridge parameters must be strings")
            shared, error = self.server.bridge_for(parameters)
            token = str(message.get("caller_token", "") or "")
            # The token travels per session; only the participant is shared.
            self.bridge = shared.for_caller(token) if shared is not None and hasattr(shared, "for_caller") else shared
            return {"started": shared is not None, "start_error": error}
        if self.bridge is None:
            raise RuntimeError("resident bridge is not open")
        if op == "call":
            method = str(message["method"])
            if method not in REMOTE_METHODS:
                raise ValueError(f"method is not available remotely: {method}")
            target = getattr(self.bridge, method)
            args = [self.decode_argument(item) for item in message.get("args", [])]
            kwargs = {key: self.decode_argument(item) for key, item in message.get("kwargs", {}).items()}
            return self.encode_result(target(*args, **kwargs))
        if op == "handle_call":
            method = str(message["method"])
            if method not in {"get_result_async", "cancel_goal_async"}:
                raise ValueError(f"goal handle method is not available: {method}")
            with self.lock:
                handle = self.handles[int(message["handle"])]
            return self.encode_result(getattr(handle, method)())
        if op == "future_cancel":
            with self.lock:
                future = self.futures.get(int(message["future"]))
            if future is not None:
                future.cancel()
            return None
        raise ValueError(f"unknown operation: {op}")

    def run_request(self, message: dict[str, Any]) -> None:
        try:
            value = self.handle(message)
        except BaseException as exc:
            self.send({"id": message.get("id"), "ok": False, "error": _encode_error(exc)})
        else:
            self.send({"id": message.get("id"), "ok": True, "value": value})

    def serve(self) -> None:
        workers = ThreadPoolExecutor(max_workers=REQUEST_WORKERS, thread_name_prefix="robot-skill-daemon-request")
        try:
            for message in _read_lines(self.sock):
                workers.submit(self.run_request, message)
        except (OSError, ValueError):
            pass
        finally:
            self.alive = False
            workers.shutdown(wait=False, cancel_futures=True)
            with self.lock:
                self.futures.clear()
                self.handles.clear()
            with contextlib.suppress(queue.Full):
                self.outbox.put_nowait(None)
            with contextlib.suppress(OSError):
                self.sock.close()


class _BridgeEntry:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.bridge = None
        self.warmed = threading.Event()
        self.warm_thread: threading.Thread | None = None


class BridgeDaemon:
    """Accept clients on one socket and share started bridges between them."""

    def __init__(
        self,
        path: Path,
        *,
        bridge_factory=None,
        idle_timeout_sec: float = DEFAULT_IDLE_TIMEOUT_SEC,
        warm_up_total_sec: float = WARM_UP_TOTAL_SEC,
        open_warm_wait_sec: float = OPEN_WARM_WAIT_SEC,
    ) -> None:
        self._warm_up_total_sec = warm_up_total_sec
        self._open_warm_wait_sec = open_warm_wait_sec
        self.path = Path(path)
        self._bridge_factory = bridge_factory
        self._idle_timeout_sec = idle_timeout_sec
        self._entries: dict[str, _BridgeEntry] = {}
        self._entries_lock = threading.Lock()
        self._active = 0
        self._active_lock = threading.Lock()
        self._last_activity = time.monotonic()
        self._stop = threading.Event()
        self._drain_deadline: float | None = None
        self._listener: socket.socket | None = None
        self._identity: tuple[int, int] | None = None

    # -- bridges -----------------------------------------------------------

    def _entry(self, key: str) -> _BridgeEntry:
        with self._entries_lock:
            return self._entries.setdefault(key, _BridgeEntry())

    def bridge_for(self, bridge_kwargs: dict[str, Any], *, wait_sec: float | None = None):
        """Return the started bridge for ``bridge_kwargs``, starting and warming it if needed.

        Each parameter set has its own lock, so one slow start never delays
        another config.  Warm-up runs in the background; the caller waits for
        it at most ``wait_sec`` and then uses the bridge anyway.
        """
        entry = self._entry(json.dumps(bridge_kwargs, sort_keys=True))
        with entry.lock:
            if entry.bridge is None:
                if self._bridge_factory is not None:
                    bridge = self._bridge_factory(**bridge_kwargs)
                else:
                    from robot_skill_cli.ros_bridge import RosBridge

                    bridge = RosBridge(**bridge_kwargs)
                if not bridge.start():
                    return None, str(getattr(bridge, "start_error", ""))
                entry.bridge = bridge
            if not entry.warmed.is_set() and (entry.warm_thread is None or not entry.warm_thread.is_alive()):
                entry.warm_thread = threading.Thread(
                    target=self._warm_entry, args=(entry,), name="robot-skill-daemon-warm-up", daemon=True
                )
                entry.warm_thread.start()
        entry.warmed.wait(self._open_warm_wait_sec if wait_sec is None else wait_sec)
        return entry.bridge, ""

    def _warm_entry(self, entry: _BridgeEntry) -> None:
        if self._warm_up(entry.bridge, self._warm_up_total_sec, self._stop):
            entry.warmed.set()

    def prewarm(self, bridge_kwargs: dict[str, Any]) -> threading.Thread:
        """Open and warm a bridge in the background while clients may already connect."""

        def run() -> None:
            parameters = _encode_plain(bridge_kwargs)
            bridge, error = self.bridge_for(parameters, wait_sec=0.0)
            if bridge is None:
                print(f"robot-skill-daemon: prewarmed bridge failed: {error}", file=sys.stderr, flush=True)
                return
            warmed = self._entry(json.dumps(parameters, sort_keys=True)).warmed.wait(self._warm_up_total_sec)
            state = "ready" if warmed else "not answered by the Gateway yet"
            print(f"robot-skill-daemon: prewarmed bridge {state}", file=sys.stderr, flush=True)

        thread = threading.Thread(target=run, name="robot-skill-prewarm", daemon=True)
        thread.start()
        return thread

    @staticmethod
    def _warm_up(bridge, total_sec: float = WARM_UP_TOTAL_SEC, stop: threading.Event | None = None) -> bool:
        """Wait until a read-only status request round-trips; never fails the open."""
        from robot_skill_cli.ros_bridge import BridgeError

        deadline = time.monotonic() + total_sec
        while time.monotonic() < deadline and not (stop is not None and stop.is_set()):
            try:
                bridge.get_status(timeout_sec=WARM_UP_ATTEMPT_SEC)
            except BridgeError:
                continue
            except Exception:
                return False
            return True
        return False

    # -- socket ------------------------------------------------------------

    def bind(self) -> None:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.path))
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                listener.close()
                raise
            try:
                _connect(self.path).close()
            except OSError:
                # Stale socket from a daemon that died: replace it.
                self.path.unlink()
                listener.bind(str(self.path))
            else:
                listener.close()
                raise
        os.chmod(self.path, 0o600)
        listener.listen(16)
        listener.settimeout(0.5)
        self._listener = listener
        info = os.stat(self.path)
        self._identity = (info.st_dev, info.st_ino)

    def _release_socket(self) -> None:
        """Stop accepting clients: unlink our socket path, then close the listener."""
        with contextlib.suppress(OSError):
            info = os.stat(self.path)
            if self._identity == (info.st_dev, info.st_ino):
                self.path.unlink()
        if self._listener is not None:
            with contextlib.suppress(OSError):
                self._listener.close()
            self._listener = None

    def _session(self, sock: socket.socket) -> None:
        with self._active_lock:
            self._active += 1
        try:
            _ClientSession(self, sock).serve()
        finally:
            with self._active_lock:
                self._active -= 1
                self._last_activity = time.monotonic()

    def active_sessions(self) -> int:
        with self._active_lock:
            return self._active

    def serve_forever(self) -> None:
        if self._listener is None:
            self.bind()
        try:
            while not self._stop.is_set():
                try:
                    sock, _address = self._listener.accept()
                except TimeoutError:
                    with self._active_lock:
                        idle = (
                            self._idle_timeout_sec > 0
                            and self._active == 0
                            and time.monotonic() - self._last_activity > self._idle_timeout_sec
                        )
                    if idle:
                        return
                    continue
                except OSError:
                    if self._stop.is_set():
                        break
                    raise
                if _peer_uid(sock) != os.getuid():
                    sock.close()
                    continue
                with self._active_lock:
                    self._last_activity = time.monotonic()
                threading.Thread(target=self._session, args=(sock,), daemon=True).start()
            self._drain()
        finally:
            self.shutdown()

    def request_stop(self, *, drain_sec: float = 0.0) -> None:
        """Stop accepting clients; keep serving connected ones up to ``drain_sec``."""
        deadline = time.monotonic() + max(drain_sec, 0.0)
        if self._drain_deadline is None or deadline < self._drain_deadline:
            self._drain_deadline = deadline
        self._stop.set()

    def _drain(self) -> None:
        self._release_socket()
        deadline = self._drain_deadline if self._drain_deadline is not None else time.monotonic()
        while self.active_sessions() > 0 and time.monotonic() < deadline:
            time.sleep(0.05)
            if self._drain_deadline is not None:
                deadline = self._drain_deadline

    def shutdown(self) -> None:
        self._stop.set()
        self._release_socket()
        with self._entries_lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for entry in entries:
            if entry.bridge is not None:
                with contextlib.suppress(Exception):
                    entry.bridge.close()


def stop_daemon() -> bool:
    """Ask the daemon that serves this environment to stop; False when none runs.

    The daemon stops accepting clients at once and exits after its connected
    clients finish (at most ``DRAIN_MAX_SEC``).
    """
    try:
        connection = _Connection(connect_or_spawn(spawn=False))
    except (DaemonUnavailable, OSError):
        return False
    try:
        connection.request({"op": "shutdown"}, timeout_sec=5.0)
    except (DaemonUnavailable, RuntimeError):
        return False
    finally:
        connection.close()
    return True


def _replace_running_daemon(path: Path, timeout_sec: float = 5.0) -> None:
    """Ask a daemon serving ``path`` to release it; it drains its clients in the background."""
    try:
        connection = _Connection(_connect(path))
    except OSError:
        return
    try:
        connection.request({"op": "shutdown"}, timeout_sec=timeout_sec)
    except (DaemonUnavailable, RuntimeError):
        pass
    finally:
        connection.close()
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline and path.exists():
        time.sleep(0.02)


def _prewarm_kwargs(config_name: str | None, config_path: str | None) -> dict[str, Any]:
    from robot_skill_cli.catalog import load_runtime_context
    from robot_skill_cli.cli import bridge_kwargs

    _context, transport = load_runtime_context(config_name=config_name, config_path=config_path)
    return bridge_kwargs(transport)


def main(argv: list[str] | None = None) -> int:
    import signal

    _scrub_private_environment()
    parser = argparse.ArgumentParser(
        prog="robot-skill-daemon",
        description="Resident ROS bridge for robot-skill (started by the embodied launch, or on demand when opted in)",
    )
    parser.add_argument("--socket", help="serve on this socket (used by robot-skill when it starts the daemon)")
    parser.add_argument("--stop", action="store_true", help="stop the daemon that serves the current environment")
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument("--config-path", help="pre-open and warm the Gateway bridge for this robot config")
    config_group.add_argument("--config-name", help="pre-open and warm the Gateway bridge for this robot config")
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=None,
        help=f"exit after this many idle seconds; 0 never exits (default {DEFAULT_IDLE_TIMEOUT_SEC:.0f}, "
        f"or {IDLE_TIMEOUT_ENV})",
    )
    parser.add_argument(
        "--replace", action="store_true", help="take the socket over from a running daemon, which drains its clients"
    )
    args = parser.parse_args(argv)
    if args.stop:
        print("stopped" if stop_daemon() else "not running")
        return 0
    if not daemon_enabled():
        print(f"robot-skill-daemon: disabled by {DISABLE_ENV}", file=sys.stderr)
        return 0
    prewarm_kwargs = None
    if args.config_path or args.config_name:
        # Resolve the bridge exactly as robot-skill would for this config, so
        # the warmed participant is the one a later call asks for.
        prewarm_kwargs = _prewarm_kwargs(args.config_name, args.config_path)
    path = Path(args.socket) if args.socket else socket_path()
    idle = args.idle_timeout
    if idle is None:
        idle = float(os.environ.get(IDLE_TIMEOUT_ENV, DEFAULT_IDLE_TIMEOUT_SEC))
    daemon = BridgeDaemon(
        path,
        idle_timeout_sec=idle,
        warm_up_total_sec=PREWARM_TOTAL_SEC if prewarm_kwargs is not None else WARM_UP_TOTAL_SEC,
    )
    try:
        if args.replace:
            # Same lock robot-skill holds while it spawns a daemon, so replacing
            # and binding never race another daemon's bind.
            with _socket_lock(path):
                _replace_running_daemon(path)
                daemon.bind()
        else:
            daemon.bind()
    except OSError:
        print(f"robot-skill-daemon: socket already served: {path}", file=sys.stderr)
        return 0  # another daemon owns the socket
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda _signum, _frame: daemon.request_stop(drain_sec=SIGNAL_DRAIN_SEC))
    print(f"robot-skill-daemon: serving {path}", file=sys.stderr, flush=True)
    if prewarm_kwargs is not None:
        daemon.prewarm(prewarm_kwargs)
    daemon.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
