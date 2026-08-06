"""Browser transport for WebXR and optical-flow phone teleoperation."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import ssl
import struct
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import websockets
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation

from .config_phone import PhoneConfig
from .web_server import WebServer, get_access_urls

logger = logging.getLogger(__name__)

_BINARY_PROTOCOL_VERSION = 0x01
_BINARY_FRAME_SIZE = 99
_OPTICAL_FLOW_MIN_QUALITY = 0.20
_POSE_CENTER_JUMP_REJECT_M = 0.25
_POSE_ROTATION_JUMP_REJECT_RAD = np.deg2rad(45.0)


class WebPhone:
    """Receive one fail-closed WebPhone control session over WebSocket."""

    def __init__(self, phone_config: PhoneConfig, *, web_root: Path | None = None) -> None:
        self.phone_config = phone_config
        self.web_config = phone_config.web
        self._web_root_override = web_root
        self._lock = threading.RLock()

        self._latest_pose: np.ndarray | None = None
        self._latest_message: dict[str, Any] | None = None
        self._latest_tracking_mode = "disabled"
        self._latest_tracking_quality = 0.0
        self._last_rx_monotonic = 0.0
        self._tracking_lost_since = 0.0

        self._enabled = False
        self._reset_requested = False
        self._go_home_pending = False
        self._release_required = False
        self._stop_pending = False
        self._stop_reason = ""
        self._active_client: Any | None = None
        self._is_connected = False

        self._calib_pos: np.ndarray | None = None
        self._calib_rot_inv: Rotation | None = None
        self._world_to_control_linear: np.ndarray | None = None

        self._tls_active = False
        self._cert_file: Path | None = None
        self._key_file: Path | None = None
        self._http_server: WebServer | None = None
        self._ws_loop: asyncio.AbstractEventLoop | None = None
        self._ws_stop_event: asyncio.Event | None = None
        self._ws_thread: threading.Thread | None = None
        self._ws_started = threading.Event()
        self._ws_shutdown_requested = threading.Event()
        self._ws_start_error: BaseException | None = None

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def tls_active(self) -> bool:
        return self._tls_active

    @property
    def page_scheme(self) -> str:
        return "https" if self._tls_active else "http"

    @property
    def websocket_scheme(self) -> str:
        return "wss" if self._tls_active else "ws"

    @property
    def access_urls(self) -> list[str]:
        return get_access_urls(
            self.web_config.bind_address,
            self.web_config.http_port,
            scheme=self.page_scheme,
        )

    @staticmethod
    def _resolve_path(path: str) -> Path:
        env_pattern = re.compile(r"\$\(env\s+(\w+)\)")
        resolved = path
        for match in env_pattern.finditer(path):
            resolved = resolved.replace(match.group(0), os.environ.get(match.group(1), ""))
        return Path(resolved).expanduser()

    @staticmethod
    def parse_binary_message(data: bytes) -> dict[str, Any]:
        """Decode a versioned 99-byte WebPhone command frame."""
        if len(data) != _BINARY_FRAME_SIZE:
            raise ValueError(f"Binary frame must be {_BINARY_FRAME_SIZE} bytes, got {len(data)}")
        version = data[0]
        if version != _BINARY_PROTOCOL_VERSION:
            raise ValueError(f"Unsupported binary protocol version: {version}")
        flags = data[1]
        matrix_flat = np.frombuffer(data, dtype="<f4", count=16, offset=2)
        linear_vel = np.frombuffer(data, dtype="<f4", count=3, offset=66)
        angular_vel = np.frombuffer(data, dtype="<f4", count=3, offset=78)
        scale = float(np.frombuffer(data, dtype="<f4", count=1, offset=90)[0])
        tracking_quality = float(np.frombuffer(data, dtype="<f4", count=1, offset=94)[0])
        tracking_modes = {0: "disabled", 1: "optical_flow", 2: "ar_6dof"}
        platform_map = {0: "unknown", 1: "ios", 2: "android"}
        tracking_mode = tracking_modes.get((flags >> 4) & 0x03, "disabled")
        if tracking_mode == "ar_6dof":
            velocity_source = "ar-pose"
            angular_velocity_source = "webxr-orientation"
        elif tracking_mode == "optical_flow":
            velocity_source = "optical-pose"
            angular_velocity_source = "device-orientation"
        else:
            velocity_source = "none"
            angular_velocity_source = "none"
        return {
            "pose": matrix_flat.tolist(),
            "linearVelocity": linear_vel.tolist(),
            "angularVelocity": angular_vel.tolist(),
            "trackingMode": tracking_mode,
            "trackingQuality": tracking_quality,
            "velocitySource": velocity_source,
            "angularVelocitySource": angular_velocity_source,
            "move": bool(flags & 0x01),
            "buttonA": bool(flags & 0x02),
            "buttonB": bool(flags & 0x04),
            "reset": bool(flags & 0x08),
            "goHome": bool(flags & 0x40),
            "scale": scale,
            "platform": platform_map.get(data[98], "unknown"),
        }

    @staticmethod
    def _finite_float(value: Any, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if np.isfinite(parsed) else default

    @staticmethod
    def _strict_bool(value: Any, field_name: str) -> bool:
        if isinstance(value, bool):
            return value
        raise ValueError(f"WebPhone field {field_name!r} must be a boolean")

    @staticmethod
    def _request_headers(websocket: Any) -> Any | None:
        headers = getattr(websocket, "request_headers", None)
        if headers is not None:
            return headers
        request = getattr(websocket, "request", None)
        return getattr(request, "headers", None)

    def _origin_allowed(self, websocket: Any) -> bool:
        """Allow browser control only from the page host served by this WebPhone instance."""
        headers = self._request_headers(websocket)
        if headers is None:
            return False
        origin = headers.get("Origin")
        request_host = headers.get("Host")
        if not origin or not request_host:
            return False
        try:
            parsed_origin = urlsplit(origin)
            parsed_host = urlsplit(f"//{request_host}")
            origin_port = parsed_origin.port or (443 if parsed_origin.scheme == "https" else 80)
        except ValueError:
            return False
        return (
            parsed_origin.scheme == self.page_scheme
            and origin_port == self.web_config.http_port
            and parsed_origin.hostname is not None
            and parsed_origin.hostname.lower() == (parsed_host.hostname or "").lower()
        )

    def _web_root(self) -> Path:
        if self._web_root_override is not None:
            return self._web_root_override
        return Path(get_package_share_directory("robot_teleop")) / "web"

    def _configure_tls(self) -> None:
        tls = self.web_config.tls
        self._tls_active = False
        self._cert_file = self._resolve_path(tls.cert_file) if tls.cert_file else None
        self._key_file = self._resolve_path(tls.key_file) if tls.key_file else None
        if not tls.enabled:
            return
        cert_ok = self._cert_file is not None and self._cert_file.is_file()
        key_ok = self._key_file is not None and self._key_file.is_file()
        if cert_ok and key_ok:
            self._tls_active = True
            return
        if tls.allow_insecure_http:
            logger.warning(
                "WebPhone TLS files are unavailable; explicitly falling back to HTTP. "
                "WebXR sensor APIs may not work from a phone."
            )
            return
        raise FileNotFoundError(
            "WebPhone TLS is enabled but cert_file/key_file are missing; "
            "configure valid files or explicitly allow insecure HTTP"
        )

    def connect(self) -> bool:
        """Start the installed HTTP page and WebSocket command endpoint."""
        if self._is_connected:
            return True
        try:
            self._configure_tls()
            client_config = {
                "ar_enabled": self.web_config.ar_enabled,
                "binary_protocol_enabled": self.web_config.binary_protocol_enabled,
                "binary_protocol_version": _BINARY_PROTOCOL_VERSION,
                "websocket_port": self.web_config.websocket_port,
                "websocket_scheme": self.websocket_scheme,
                "optical_flow_fallback_enabled": self.phone_config.optical_flow_fallback_enabled,
            }
            self._http_server = WebServer(
                web_root=self._web_root(),
                bind_address=self.web_config.bind_address,
                port=self.web_config.http_port,
                use_https=self._tls_active,
                cert_file=self._cert_file if self._tls_active else None,
                key_file=self._key_file if self._tls_active else None,
                client_config=client_config,
            )
            self._http_server.start()
            self._start_websocket_server()
            self._is_connected = True
            return True
        except Exception:
            logger.exception("Failed to start WebPhone transport")
            self.disconnect()
            return False

    def _ssl_context(self) -> ssl.SSLContext | None:
        if not self._tls_active:
            return None
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(self._cert_file, self._key_file)
        return context

    def _start_websocket_server(self) -> None:
        if self._ws_thread is not None and self._ws_thread.is_alive():
            raise RuntimeError("Previous WebPhone WebSocket server is still shutting down")
        self._ws_thread = None
        self._ws_started.clear()
        self._ws_shutdown_requested.clear()
        self._ws_start_error = None

        async def serve() -> None:
            self._ws_stop_event = asyncio.Event()
            if self._ws_shutdown_requested.is_set():
                self._ws_stop_event.set()
            try:
                server = await websockets.serve(
                    self._handle_websocket,
                    self.web_config.bind_address,
                    self.web_config.websocket_port,
                    ssl=self._ssl_context(),
                    ping_interval=10,
                    ping_timeout=10,
                    close_timeout=1,
                    max_size=64 * 1024,
                )
            except BaseException as exc:
                self._ws_start_error = exc
                self._ws_started.set()
                return
            self._ws_started.set()
            await self._ws_stop_event.wait()
            server.close()
            await server.wait_closed()

        def run_server() -> None:
            loop = asyncio.new_event_loop()
            self._ws_loop = loop
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(serve())
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                self._ws_loop = None
                self._ws_stop_event = None
                loop.close()

        self._ws_thread = threading.Thread(target=run_server, name="webphone-websocket", daemon=True)
        self._ws_thread.start()
        if not self._ws_started.wait(timeout=3.0):
            raise TimeoutError("Timed out while starting WebPhone WebSocket server")
        if self._ws_start_error is not None:
            raise RuntimeError("Failed to start WebPhone WebSocket server") from self._ws_start_error

    async def _handle_websocket(self, websocket: Any, _path: str | None = None) -> None:
        if not self._origin_allowed(websocket):
            await websocket.close(code=1008, reason="WebPhone origin is not allowed")
            return
        with self._lock:
            if self._active_client is not None and self._active_client is not websocket:
                reject = True
            else:
                reject = False
                self._active_client = websocket
        if reject:
            await websocket.close(code=1008, reason="another WebPhone client owns control")
            return

        try:
            async for message in websocket:
                try:
                    if isinstance(message, bytes):
                        if not self.web_config.binary_protocol_enabled:
                            await websocket.close(code=1003, reason="WebPhone binary protocol is disabled")
                            return
                        try:
                            payload = self.parse_binary_message(message)
                        except (struct.error, TypeError, ValueError) as exc:
                            logger.warning(f"Rejected WebPhone binary protocol frame: {exc}")
                            await websocket.close(code=1003, reason="invalid WebPhone binary protocol frame")
                            return
                    else:
                        payload = json.loads(message)
                    if not isinstance(payload, dict):
                        raise ValueError("WebPhone message must be an object")
                    self._accept_message(payload, time.monotonic())
                except (json.JSONDecodeError, struct.error, TypeError, ValueError) as exc:
                    logger.warning(f"Rejected malformed WebPhone message: {exc}")
        except (ConnectionResetError, websockets.exceptions.ConnectionClosed):
            logger.debug("WebPhone client disconnected")
        finally:
            with self._lock:
                if self._active_client is websocket:
                    self._active_client = None
                    self._mark_safety_stop_locked("client disconnected")

    def _accept_message(self, payload: dict[str, Any], timestamp: float) -> None:
        with self._lock:
            self._check_stale_locked(timestamp)
            payload = dict(payload)
            requested_move = self._strict_bool(payload.get("move", False), "move")
            payload["move"] = requested_move
            payload["buttonA"] = self._strict_bool(
                payload.get("buttonA", payload.get("reservedButtonA", False)), "buttonA"
            )
            payload["buttonB"] = self._strict_bool(
                payload.get("buttonB", payload.get("reservedButtonB", False)), "buttonB"
            )
            payload["reset"] = self._strict_bool(payload.get("reset", False), "reset")
            payload["goHome"] = self._strict_bool(payload.get("goHome", False), "goHome")
            go_home_requested = payload["goHome"]
            tracking_mode = str(payload.get("trackingMode", self._latest_tracking_mode))
            if tracking_mode not in {"disabled", "optical_flow", "ar_6dof"}:
                raise ValueError(f"Unsupported WebPhone tracking mode: {tracking_mode}")
            pose = payload.get("pose")
            if pose is None:
                if requested_move:
                    raise ValueError("enabled WebPhone messages require a fresh pose")
                matrix = self._latest_pose.copy() if self._latest_pose is not None else np.eye(4)
                pose_updated = False
            else:
                try:
                    matrix = np.asarray(pose, dtype=float).reshape(4, 4)
                except (TypeError, ValueError) as exc:
                    raise ValueError("WebPhone pose must contain exactly 16 numeric values") from exc
                if not np.all(np.isfinite(matrix)):
                    raise ValueError("WebPhone pose contains non-finite values")
                if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
                    raise ValueError("WebPhone pose must be a homogeneous transform")
                rotation_matrix = matrix[:3, :3]
                if not np.allclose(rotation_matrix.T @ rotation_matrix, np.eye(3), atol=1e-4) or not np.isclose(
                    np.linalg.det(rotation_matrix), 1.0, atol=1e-4
                ):
                    raise ValueError("WebPhone pose rotation must be orthonormal")
                rotation = Rotation.from_matrix(rotation_matrix)
                matrix = matrix.copy()
                # Both tracking paths describe the viewer camera centre:
                # ARCore/WebXR reports it directly, while optical flow integrates
                # rotation-compensated camera translation into a virtual pose.
                # Move either pose to the same configured phone control point so
                # rotating the phone contributes the physical lever-arm motion
                # exactly once.
                if tracking_mode in {"ar_6dof", "optical_flow"}:
                    rotated_offset = rotation.apply(self.phone_config.camera_offset)
                    if tracking_mode == "optical_flow":
                        # The monocular fallback cannot reliably cancel the
                        # camera-centre arc during pitch/roll. Preserve the
                        # requested horizontal lever arm, but do not turn phone
                        # tilt into robot vertical motion. AR keeps the complete
                        # metric 3-D camera-to-control-point transform.
                        rotated_offset[1] = 0.0
                    matrix[:3, 3] -= rotated_offset
                pose_updated = True

            tracking_quality = self._finite_float(payload.get("trackingQuality", 0.0), 0.0)
            if not 0.0 <= tracking_quality <= 1.0:
                raise ValueError("WebPhone trackingQuality must be in range 0..1")
            previous_move = bool(self._latest_message and self._latest_message.get("move", False))
            if (
                pose_updated
                and requested_move
                and previous_move
                and tracking_mode == self._latest_tracking_mode
                and self._latest_pose is not None
                and not self._release_required
            ):
                position_jump = float(np.linalg.norm(matrix[:3, 3] - self._latest_pose[:3, 3]))
                previous_rotation = Rotation.from_matrix(self._latest_pose[:3, :3])
                rotation_jump = float((rotation * previous_rotation.inv()).magnitude())
                if position_jump >= _POSE_CENTER_JUMP_REJECT_M or rotation_jump >= _POSE_ROTATION_JUMP_REJECT_RAD:
                    self._mark_safety_stop_locked(
                        "tracking pose jumped "
                        f"(position={position_jump:.3f}m, rotation={np.rad2deg(rotation_jump):.1f}deg)"
                    )
            if (
                requested_move
                and previous_move
                and tracking_mode != self._latest_tracking_mode
                and not self._release_required
            ):
                self._mark_safety_stop_locked("tracking mode changed")
            reset_requested = payload["reset"]
            reset_edge = reset_requested and not self._reset_requested
            self._reset_requested = reset_requested

            if reset_edge:
                self._reset_origin_locked(matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]))
                self._mark_safety_stop_locked("origin reset")
            pose_tracking_allowed = tracking_mode == "ar_6dof" or (
                self.phone_config.optical_flow_fallback_enabled and tracking_mode == "optical_flow"
            )
            if requested_move and not pose_tracking_allowed and not self._release_required:
                self._mark_safety_stop_locked("WebPhone requires ar_6dof or enabled optical-flow fallback")

            if tracking_mode == "optical_flow" and requested_move and not self._release_required:
                if tracking_quality >= _OPTICAL_FLOW_MIN_QUALITY:
                    self._tracking_lost_since = 0.0
                elif self._tracking_lost_since <= 0.0:
                    self._tracking_lost_since = timestamp
                elif timestamp - self._tracking_lost_since >= self.web_config.command_stale_s:
                    self._mark_safety_stop_locked("optical-flow tracking lost")
            else:
                self._tracking_lost_since = 0.0

            if self._release_required:
                if requested_move:
                    payload = dict(payload)
                    payload["move"] = False
                else:
                    self._release_required = False

            # Home is an edge-triggered UI action, unlike the continuously
            # sampled deadman and gripper buttons. Latch it until get_action()
            # consumes it so a following AR/optical heartbeat cannot overwrite
            # the request before the 50 Hz PhoneDevice loop observes it.
            if go_home_requested and not self._release_required:
                self._go_home_pending = True

            self._latest_pose = matrix
            self._latest_tracking_mode = tracking_mode
            self._latest_tracking_quality = tracking_quality
            self._latest_message = dict(payload)
            self._last_rx_monotonic = timestamp

    def _mark_safety_stop_locked(self, reason: str) -> None:
        self._enabled = False
        self._release_required = True
        self._stop_pending = True
        self._stop_reason = reason
        self._latest_message = None
        self._go_home_pending = False
        self._last_rx_monotonic = 0.0
        self._tracking_lost_since = 0.0

    def consume_stop_request(self) -> str | None:
        """Return and clear the edge-triggered transport safety-stop reason."""
        with self._lock:
            self._check_stale_locked(time.monotonic())
            if not self._stop_pending:
                return None
            self._stop_pending = False
            return self._stop_reason

    def require_release(self, reason: str, *, request_stop: bool = True) -> None:
        """Latch recovery until release, optionally requesting an immediate stop."""
        with self._lock:
            if request_stop:
                self._mark_safety_stop_locked(reason)
                return
            self._enabled = False
            self._release_required = True
            self._go_home_pending = False
            if self._latest_message is not None:
                self._latest_message = dict(self._latest_message)
                self._latest_message["move"] = False

    def _check_stale_locked(self, now: float) -> None:
        if self._last_rx_monotonic <= 0.0:
            return
        command_age = now - self._last_rx_monotonic
        if command_age > self.web_config.command_stale_s:
            self._mark_safety_stop_locked(f"command stream stale ({command_age * 1000.0:.0f}ms without a frame)")

    @staticmethod
    def _yaw_alignment(rotation: Rotation) -> np.ndarray:
        """Build a roll-independent heading frame from an ARCore/WebXR viewer pose."""
        world_up = np.array([0.0, 1.0, 0.0])
        # ARCore/WebXR viewer coordinates are +X right, +Y up, +Z back.  The
        # viewer +Z axis is invariant when the phone rolls about the camera
        # axis, while +X becomes vertical near a 90-degree roll and cannot
        # define a horizontal heading.  Use +Z first, with +X only as the
        # fallback for the camera-pointing-straight-up/down singularity.
        viewer_back = rotation.apply(np.array([0.0, 0.0, 1.0]))
        back = viewer_back - np.dot(viewer_back, world_up) * world_up
        back_norm = float(np.linalg.norm(back))
        if back_norm > 1e-6:
            back /= back_norm
            right = np.cross(world_up, back)
            return np.vstack([right, world_up, back])

        viewer_right = rotation.apply(np.array([1.0, 0.0, 0.0]))
        right = viewer_right - np.dot(viewer_right, world_up) * world_up
        right_norm = float(np.linalg.norm(right))
        if right_norm <= 1e-6:
            return np.eye(3)
        right /= right_norm
        back = np.cross(right, world_up)
        return np.vstack([right, world_up, back])

    def _reset_origin_locked(self, position: np.ndarray, rotation: Rotation) -> None:
        self._calib_pos = position.copy()
        self._calib_rot_inv = rotation.inv()
        self._world_to_control_linear = self._yaw_alignment(rotation)

    def _world_to_control(self, vector: np.ndarray) -> np.ndarray:
        if self._world_to_control_linear is None:
            return np.asarray(vector, dtype=float)
        return self._world_to_control_linear @ np.asarray(vector, dtype=float)

    def _rotation_to_control(self, rotation: Rotation) -> Rotation:
        if self._world_to_control_linear is None:
            return rotation
        return Rotation.from_matrix(self._world_to_control_linear) * rotation

    def get_action(self) -> dict[str, Any]:
        """Return a normalized action, failing closed when input becomes stale."""
        with self._lock:
            self._check_stale_locked(time.monotonic())
            if self._latest_pose is None or self._latest_message is None:
                return {}
            pose = self._latest_pose.copy()
            raw_position = pose[:3, 3]
            raw_rotation = Rotation.from_matrix(pose[:3, :3])
            message = dict(self._latest_message)
            go_home_requested = self._go_home_pending
            self._go_home_pending = False
            enable = bool(message.get("move", False))

            if self._calib_pos is None or self._calib_rot_inv is None:
                self._reset_origin_locked(raw_position, raw_rotation)
            if enable and not self._enabled:
                self._reset_origin_locked(raw_position, raw_rotation)

            position = self._world_to_control(raw_position - self._calib_pos)
            rotation = self._rotation_to_control(raw_rotation)
            self._enabled = enable
            return {
                "phone.pos": position,
                "phone.rot": rotation,
                "phone.raw_inputs": {
                    "move": enable,
                    "scale": np.clip(self._finite_float(message.get("scale", 1.0), 1.0), 0.1, 4.0),
                    "reservedButtonA": bool(message.get("buttonA", message.get("reservedButtonA", False))),
                    "reservedButtonB": bool(message.get("buttonB", message.get("reservedButtonB", False))),
                    "reset": bool(message.get("reset", False)),
                    "goHome": go_home_requested,
                    "platform": str(message.get("platform", "unknown")),
                },
                "phone.enabled": enable,
                "phone.tracking_mode": self._latest_tracking_mode,
                "phone.tracking_quality": self._latest_tracking_quality,
            }

    def disconnect(self) -> None:
        """Fail closed, then stop both network servers idempotently."""
        with self._lock:
            self._mark_safety_stop_locked("WebPhone transport stopped")
            self._active_client = None
            self._latest_pose = None
        # This thread-safe latch also covers disconnect racing server startup,
        # before the event-loop-owned asyncio.Event has been created.
        self._ws_shutdown_requested.set()
        loop, stop_event = self._ws_loop, self._ws_stop_event
        if loop is not None and stop_event is not None and loop.is_running():
            loop.call_soon_threadsafe(stop_event.set)
        thread = self._ws_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        if thread is not None and thread.is_alive():
            logger.error("WebPhone WebSocket thread did not stop within 3 seconds")
        else:
            self._ws_thread = None
        if self._http_server is not None:
            self._http_server.stop()
            self._http_server = None
        self._is_connected = False
