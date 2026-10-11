"""Resident ROS bridge shared by successive ``robot-skill`` invocations.

Every ``robot-skill`` call used to create a fresh rclpy context: a new DDS
participant that the running Gateway had to discover before the first
service response, about 1 s (up to 3 s) with Fast DDS.  This module keeps one
:class:`~robot_skill_cli.ros_bridge.RosBridge` alive in a per-user background
process and lets each CLI process drive it over a private Unix socket, so the
discovery cost is paid once instead of on every command.

Only the ROS transport moves into the resident process.  Argument parsing,
catalog loading, signal handling, stop-state files and stdout stay in the CLI
process, exactly as before.  The client implements the same methods as
``RosBridge``; goal futures, goal handles and feedback callbacks are proxied
and ROS result messages are returned as attribute namespaces.

Safety properties:

* The socket lives in a ``0700`` directory owned by the user and every peer is
  checked with ``SO_PEERCRED``.
* The socket name is derived from the ROS transport environment
  (``ROS_DOMAIN_ID``, ``RMW_IMPLEMENTATION`` ...) and from the installed code,
  so a client never reaches a daemon on another domain, RMW or build.
* Any failure to reach or start the daemon falls back to an in-process
  ``RosBridge``; behavior then matches the previous per-process bridge.
* A client that disconnects does not cancel its goals, matching a CLI process
  that exits while a goal is running.
* ``IBROBOT_ROBOT_SKILL_DAEMON=0`` disables the daemon entirely.
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
import socket
import stat
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROTOCOL_VERSION = 1
DISABLE_ENV = "IBROBOT_ROBOT_SKILL_DAEMON"
IDLE_TIMEOUT_ENV = "IBROBOT_ROBOT_SKILL_DAEMON_IDLE_SEC"
DEFAULT_IDLE_TIMEOUT_SEC = 900.0
SPAWN_WAIT_SEC = 5.0
# A new DDS participant can see the Gateway before the Gateway has discovered
# its reply reader; a request sent in that window loses its response (seen as
# a 30 s RESULT_TIMEOUT with Fast DDS).  A fresh daemon bridge first sends
# short, read-only status probes until one is answered.
WARM_UP_ATTEMPT_SEC = 1.0
WARM_UP_TOTAL_SEC = 15.0

# Environment that selects the DDS graph.  A daemon only serves clients whose
# values are identical, because they are part of the socket name.
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
    "FASTDDS_BUILTIN_TRANSPORTS",
)
# Per-invocation state that must never leak into the long-lived daemon.
_PRIVATE_ENV_PREFIXES = ("IBROBOT_STOP_", "IBROBOT_HERMES_", "IBROBOT_EXECUTION_TOKEN_")

# Methods that are evaluated in the client instead of being forwarded.
_LOCAL_METHODS = frozenset({"start", "close", "wait_future"})


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


def _send(sock: socket.socket, lock: threading.Lock, message: dict[str, Any]) -> None:
    data = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
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
    # so every package it executes is part of its identity.
    return [
        sys.executable,
        sys.version,
        os.environ.get("AMENT_PREFIX_PATH", ""),
        os.environ.get("PYTHONPATH", ""),
        *(_package_stamp(package) for package in ("robot_skill_cli", "embodied_common", "ibrobot_msgs")),
    ]


def _runtime_directory() -> Path:
    base = os.environ.get("XDG_RUNTIME_DIR")
    root = Path(base) / "ibrobot" if base and os.path.isdir(base) else Path(f"/tmp/ibrobot-{os.getuid()}")
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
    return os.environ.get(DISABLE_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def _peer_uid(sock: socket.socket) -> int | None:
    so_peercred = getattr(socket, "SO_PEERCRED", None)
    if so_peercred is None:
        return None
    raw = sock.getsockopt(socket.SOL_SOCKET, so_peercred, struct.calcsize("3i"))
    return struct.unpack("3i", raw)[1]


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
        error = _disconnected()
        with self._state_lock:
            pending = list(self._pending.values())
            futures = list(self._futures.values())
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
                future = self._futures.get(int(message["future"]))
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
                    self._futures[future.future_id] = future
                    early = self._early_futures.pop(future.future_id, None)
                if early is not None:
                    self._settle(future, early)
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


def _daemon_environment() -> dict[str, str]:
    return {
        name: value
        for name, value in os.environ.items()
        if not any(name.startswith(prefix) for prefix in _PRIVATE_ENV_PREFIXES)
    }


def _spawn(path: Path) -> None:
    log_path = path.with_suffix(".log")
    with open(log_path, "ab") as log:
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


def connect_or_spawn(*, spawn: bool = True) -> socket.socket:
    path = socket_path()
    try:
        return _connect(path)
    except OSError as exc:
        if not spawn:
            raise DaemonUnavailable(str(exc)) from exc
    import fcntl

    with open(path.with_suffix(".lock"), "a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
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


class ResidentBridge:
    """Drop-in ``RosBridge`` that prefers the resident daemon.

    ``start()`` connects to (or starts) the daemon and opens a bridge with the
    same constructor arguments there.  If that is not possible, an in-process
    ``RosBridge`` is used instead, so callers see identical behavior apart from
    the startup cost.
    """

    def __init__(self, **bridge_kwargs: Any) -> None:
        self._bridge_kwargs = bridge_kwargs
        self._connection: _Connection | None = None
        self._local = None
        self.start_error = ""
        self.resident = False

    def start(self) -> bool:
        if daemon_enabled():
            try:
                connection = _Connection(connect_or_spawn())
                try:
                    started = connection.request(
                        {"op": "open", "version": PROTOCOL_VERSION, "bridge": _encode_plain(self._bridge_kwargs)},
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

    def close(self) -> None:
        if self._local is not None:
            self._local.close()
            self._local = None
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __getattr__(self, name: str):
        if name.startswith("_") or name in _LOCAL_METHODS:
            raise AttributeError(name)
        local = self.__dict__.get("_local")
        if local is not None:
            return getattr(local, name)
        from robot_skill_cli.ros_bridge import RosBridge

        if not callable(getattr(RosBridge, name, None)):
            raise AttributeError(name)
        connection = self.__dict__.get("_connection")

        def _forward(*args, **kwargs):
            if connection is None:
                from robot_skill_cli.output import EXIT_ROS_UNAVAILABLE
                from robot_skill_cli.ros_bridge import BridgeError

                raise BridgeError("ROS_UNAVAILABLE", "ROS bridge is not started", exit_code=EXIT_ROS_UNAVAILABLE)
            return connection.call(name, args, kwargs)

        return _forward


# --------------------------------------------------------------------------- #
# Daemon
# --------------------------------------------------------------------------- #


class _ClientSession:
    """Objects one client may reference; dropped (not cancelled) on disconnect."""

    def __init__(self, server: BridgeDaemon, sock: socket.socket) -> None:
        self.server = server
        self.sock = sock
        self.send_lock = threading.Lock()
        self.bridge = None
        self.ids = itertools.count(1)
        self.futures: dict[int, Any] = {}
        self.handles: dict[int, Any] = {}
        self.lock = threading.Lock()
        self.alive = True

    def send(self, message: dict[str, Any]) -> None:
        if not self.alive:
            return
        try:
            _send(self.sock, self.send_lock, message)
        except OSError:
            self.alive = False

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
            self.server.request_stop()
            return None
        if op == "open":
            if int(message.get("version", -1)) != PROTOCOL_VERSION:
                raise ValueError("resident bridge protocol mismatch")
            self.bridge, error = self.server.bridge_for(message["bridge"])
            return {"started": self.bridge is not None, "start_error": error}
        if self.bridge is None:
            raise RuntimeError("resident bridge is not open")
        if op == "call":
            method = str(message["method"])
            if method.startswith("_") or method in _LOCAL_METHODS:
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
        try:
            for message in _read_lines(self.sock):
                threading.Thread(target=self.run_request, args=(message,), daemon=True).start()
        except (OSError, ValueError):
            pass
        finally:
            self.alive = False
            with self.lock:
                self.futures.clear()
                self.handles.clear()
            with contextlib.suppress(OSError):
                self.sock.close()


class BridgeDaemon:
    """Accept clients on one socket and share started bridges between them."""

    def __init__(self, path: Path, *, bridge_factory=None, idle_timeout_sec: float = DEFAULT_IDLE_TIMEOUT_SEC) -> None:
        self.path = Path(path)
        self._bridge_factory = bridge_factory
        self._idle_timeout_sec = idle_timeout_sec
        self._bridges: dict[str, Any] = {}
        self._bridge_lock = threading.Lock()
        self._active = 0
        self._active_lock = threading.Lock()
        self._last_activity = time.monotonic()
        self._stop = threading.Event()
        self._listener: socket.socket | None = None
        self._inode: int | None = None

    def bridge_for(self, bridge_kwargs: dict[str, Any]):
        key = json.dumps(bridge_kwargs, sort_keys=True)
        with self._bridge_lock:
            bridge = self._bridges.get(key)
            if bridge is not None:
                return bridge, ""
            if self._bridge_factory is not None:
                bridge = self._bridge_factory(**bridge_kwargs)
            else:
                from robot_skill_cli.ros_bridge import RosBridge

                bridge = RosBridge(**bridge_kwargs)
            if not bridge.start():
                return None, str(getattr(bridge, "start_error", ""))
            self._warm_up(bridge)
            self._bridges[key] = bridge
            return bridge, ""

    @staticmethod
    def _warm_up(bridge) -> bool:
        """Wait until a read-only status request round-trips; never fails the open."""
        from robot_skill_cli.ros_bridge import BridgeError

        deadline = time.monotonic() + WARM_UP_TOTAL_SEC
        while time.monotonic() < deadline:
            try:
                bridge.get_status(timeout_sec=WARM_UP_ATTEMPT_SEC)
            except BridgeError:
                continue
            except Exception:
                return False
            return True
        return False

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
        listener.settimeout(1.0)
        self._listener = listener
        self._inode = os.stat(self.path).st_ino

    def _session(self, sock: socket.socket) -> None:
        with self._active_lock:
            self._active += 1
        try:
            _ClientSession(self, sock).serve()
        finally:
            with self._active_lock:
                self._active -= 1
                self._last_activity = time.monotonic()

    def serve_forever(self) -> None:
        if self._listener is None:
            self.bind()
        try:
            while not self._stop.is_set():
                try:
                    sock, _address = self._listener.accept()
                except TimeoutError:
                    with self._active_lock:
                        idle = self._active == 0 and time.monotonic() - self._last_activity > self._idle_timeout_sec
                    if idle:
                        return
                    continue
                except OSError:
                    if self._stop.is_set():
                        return
                    raise
                if _peer_uid(sock) not in {None, os.getuid()}:
                    sock.close()
                    continue
                with self._active_lock:
                    self._last_activity = time.monotonic()
                threading.Thread(target=self._session, args=(sock,), daemon=True).start()
        finally:
            self.shutdown()

    def request_stop(self) -> None:
        self._stop.set()

    def shutdown(self) -> None:
        self._stop.set()
        if self._listener is not None:
            with contextlib.suppress(OSError):
                self._listener.close()
            self._listener = None
        with contextlib.suppress(OSError):
            if self._inode is not None and os.stat(self.path).st_ino == self._inode:
                self.path.unlink()
        with self._bridge_lock:
            bridges = list(self._bridges.values())
            self._bridges.clear()
        for bridge in bridges:
            with contextlib.suppress(Exception):
                bridge.close()


def stop_daemon() -> bool:
    """Ask the daemon that serves this environment to exit; False when none runs."""
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


def main(argv: list[str] | None = None) -> int:
    import signal

    parser = argparse.ArgumentParser(
        prog="robot-skill-daemon", description="Resident ROS bridge for robot-skill (started on demand)"
    )
    parser.add_argument("--socket", help="serve on this socket (used by robot-skill when it starts the daemon)")
    parser.add_argument("--stop", action="store_true", help="stop the daemon that serves the current environment")
    args = parser.parse_args(argv)
    if args.stop:
        print("stopped" if stop_daemon() else "not running")
        return 0
    path = Path(args.socket) if args.socket else socket_path()
    idle = float(os.environ.get(IDLE_TIMEOUT_ENV, DEFAULT_IDLE_TIMEOUT_SEC))
    daemon = BridgeDaemon(path, idle_timeout_sec=idle)
    try:
        daemon.bind()
    except OSError:
        return 0  # another daemon owns the socket
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda _signum, _frame: daemon.request_stop())
    daemon.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
