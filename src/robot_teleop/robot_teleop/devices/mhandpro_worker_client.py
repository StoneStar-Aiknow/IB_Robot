"""Parent-side client for the isolated mHandPro SDK worker."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

from .mhandpro_sdk import CONNECTED_BOTH_GLOVES, CONNECTED_NONE, connection_includes_side


def connection_state_after_break(connection_state: int, break_mode: int) -> int:
    """Convert the SDK's disconnected-glove mode into the remaining connection state."""
    if break_mode in (1, 2) and connection_state == CONNECTED_BOTH_GLOVES:
        return CONNECTED_BOTH_GLOVES ^ break_mode
    if break_mode in (connection_state, CONNECTED_BOTH_GLOVES):
        return CONNECTED_NONE
    return connection_state


class MHandProWorkerClient:
    def __init__(
        self,
        lib_path: str,
        side: str,
        *,
        startup_timeout: float = 10.0,
        runtime_dir: str | Path | None = None,
        failure_policy: str = "require_all",
    ):
        if side not in ("left", "right", "both"):
            raise ValueError("mHandPro side must be left, right, or both")
        if failure_policy not in ("require_all", "allow_available"):
            raise ValueError("mHandPro failure_policy must be require_all or allow_available")
        self.lib_path = str(lib_path)
        self.side = side
        self.failure_policy = failure_policy
        self.startup_timeout = float(startup_timeout)
        self.runtime_dir = Path(runtime_dir or Path.home() / ".cache" / "ibrobot" / "mhandpro").expanduser()
        self.sdk_version = "unknown"
        self.connection_state = 0
        self._process: subprocess.Popen | None = None
        self._ready = threading.Event()
        self._connected = False
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._latest_frames: dict[str, dict] = {}
        self._error: str | None = None
        self._request_sequence = 0
        self._request_events: dict[int, threading.Event] = {}
        self._request_results: dict[int, dict] = {}
        self._output_lines: deque[str] = deque(maxlen=20)
        self._event_thread: threading.Thread | None = None
        self._output_thread: threading.Thread | None = None

    @property
    def is_connected(self) -> bool:
        process = self._process
        return self._connected and process is not None and process.poll() is None

    def is_side_connected(self, side: str) -> bool:
        """Return whether the worker currently reports the requested glove side."""
        if side not in ("left", "right"):
            raise ValueError("mHandPro side must be 'left' or 'right'")
        process = self._process
        return (
            self._connected
            and process is not None
            and process.poll() is None
            and connection_includes_side(self.connection_state, side)
        )

    def connect(self) -> int:
        if self.is_connected:
            return self.connection_state
        self._ready.clear()
        read_fd, write_fd = os.pipe()
        worker_path = Path(__file__).with_name("mhandpro_worker.py")
        interpreter = self._prepare_interpreter()
        command = [
            str(interpreter),
            "-I",
            "-S",
            str(worker_path),
            "--lib-path",
            self.lib_path,
            "--side",
            self.side,
            "--failure-policy",
            self.failure_policy,
            "--event-fd",
            str(write_fd),
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                pass_fds=(write_fd,),
            )
        finally:
            os.close(write_fd)
        self._event_thread = threading.Thread(
            target=self._read_events,
            args=(read_fd,),
            name=f"mhandpro-events-{self.side}",
            daemon=True,
        )
        self._event_thread.start()
        self._output_thread = threading.Thread(
            target=self._read_output,
            name=f"mhandpro-output-{self.side}",
            daemon=True,
        )
        self._output_thread.start()
        if not self._ready.wait(self.startup_timeout):
            details = self._failure_details("mHandPro worker startup timed out")
            self.disconnect()
            raise ConnectionError(details)
        if not self.is_connected:
            details = self._failure_details("mHandPro worker failed to start")
            self.disconnect()
            raise ConnectionError(details)
        return self.connection_state

    def _prepare_interpreter(self) -> Path:
        """Give the vendor SDK a writable directory beside /proc/self/exe."""
        source = Path(sys.executable).resolve()
        self.runtime_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.runtime_dir.chmod(0o700)
        target = self.runtime_dir / "python3"
        source_stat = source.stat()
        if target.is_file():
            target_stat = target.stat()
            if target_stat.st_size == source_stat.st_size and target_stat.st_mtime_ns == source_stat.st_mtime_ns:
                return target

        temporary = self.runtime_dir / f".python3.{os.getpid()}.tmp"
        try:
            shutil.copy2(source, temporary)
            temporary.chmod(0o700)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def latest_frame(self, side: str | None = None) -> dict | None:
        requested_side = side or self.side
        if requested_side == "both":
            raise ValueError("latest_frame requires an explicit side when the worker serves both gloves")
        with self._lock:
            stored = self._latest_frames.get(requested_side)
            if stored is None:
                return None
            frame = dict(stored)
            for key in ("positions", "quaternions", "virtual_positions", "gyroscope", "accelerations", "velocities"):
                if key in stored:
                    frame[key] = [list(value) for value in stored[key]]
            if "sensor_states" in stored:
                frame["sensor_states"] = list(stored["sensor_states"])
            return frame

    def start_calibration(self, mode: int, timeout: float) -> tuple[int, float]:
        if not self.is_connected:
            raise ConnectionError(self._failure_details("mHandPro worker is not connected"))
        with self._lock:
            self._request_sequence += 1
            request_id = self._request_sequence
            request_event = threading.Event()
            self._request_events[request_id] = request_event
        self._send({"command": "calibrate", "request_id": request_id, "mode": int(mode), "timeout": float(timeout)})
        if not request_event.wait(float(timeout) + 2.0):
            with self._lock:
                self._request_events.pop(request_id, None)
            raise TimeoutError(self._failure_details("mHandPro calibration worker timed out"))
        with self._lock:
            result = self._request_results.pop(request_id, None)
            self._request_events.pop(request_id, None)
        if result is None:
            raise RuntimeError(self._failure_details("mHandPro calibration worker exited"))
        return int(result["state"]), float(result["progress"])

    def disconnect(self) -> None:
        process = self._process
        self._connected = False
        if process is None:
            return
        if process.poll() is None:
            try:
                self._send({"command": "shutdown"})
                process.wait(timeout=3.0)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
        if process.stdin is not None:
            process.stdin.close()
        self._process = None

    def _send(self, payload: dict) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise BrokenPipeError("mHandPro worker stdin is unavailable")
        line = json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n"
        with self._write_lock:
            process.stdin.write(line)
            process.stdin.flush()

    def _read_events(self, read_fd: int) -> None:
        try:
            with os.fdopen(read_fd, "r", encoding="utf-8") as stream:
                for line in stream:
                    event = json.loads(line)
                    event_type = event.get("type")
                    if event_type == "ready":
                        self.connection_state = int(event["state"])
                        self.sdk_version = str(event.get("sdk_version", "unknown"))
                        self._connected = True
                        self._ready.set()
                    elif event_type == "frame":
                        with self._lock:
                            self._latest_frames[str(event["side"])] = event
                    elif event_type == "break":
                        break_mode = int(event.get("state", 0))
                        self.connection_state = connection_state_after_break(self.connection_state, break_mode)
                        self._connected = self.connection_state != 0
                    elif event_type == "calibration_result":
                        request_id = int(event["request_id"])
                        with self._lock:
                            self._request_results[request_id] = event
                            request_event = self._request_events.get(request_id)
                        if request_event is not None:
                            request_event.set()
                    elif event_type == "error":
                        self._error = str(event.get("message", "unknown worker error"))
                        self._connected = False
                        self._ready.set()
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._error = f"Invalid mHandPro worker event: {exc}"
            self._connected = False
            self._ready.set()
        finally:
            self._connected = False
            self._ready.set()
            with self._lock:
                request_events = list(self._request_events.values())
            for request_event in request_events:
                request_event.set()

    def _read_output(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            stripped = line.strip()
            if stripped:
                self._output_lines.append(stripped)

    def _failure_details(self, prefix: str) -> str:
        details = self._error
        if not details and self._output_lines:
            details = self._output_lines[-1]
        process = self._process
        if not details and process is not None and process.poll() is not None:
            details = f"worker exited with code {process.returncode}"
        return f"{prefix}: {details}" if details else prefix
