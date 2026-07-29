"""Configuration models for built-in WebPhone teleoperation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _parse_bool(value: Any, *, field_name: str, default: bool) -> bool:
    """Parse configuration booleans without treating every non-empty string as true."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    raise ValueError(f"{field_name} must be a boolean")


@dataclass(frozen=True)
class WebTLSConfig:
    """TLS settings for the WebPhone HTTP and WebSocket servers."""

    enabled: bool = True
    cert_file: str = "$(env HOME)/.ssl/ib_robot/web_phone_cert.pem"
    key_file: str = "$(env HOME)/.ssl/ib_robot/web_phone_key.pem"
    allow_insecure_http: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WebTLSConfig:
        return cls(
            enabled=_parse_bool(data.get("enabled"), field_name="phone_config.web.tls.enabled", default=True),
            cert_file=str(data.get("cert_file", "$(env HOME)/.ssl/ib_robot/web_phone_cert.pem")),
            key_file=str(data.get("key_file", "$(env HOME)/.ssl/ib_robot/web_phone_key.pem")),
            allow_insecure_http=_parse_bool(
                data.get("allow_insecure_http"),
                field_name="phone_config.web.tls.allow_insecure_http",
                default=False,
            ),
        )


@dataclass(frozen=True)
class WebPhoneConfig:
    """Network, watchdog, and protocol settings for the browser backend."""

    bind_address: str = "0.0.0.0"
    http_port: int = 8765
    websocket_port: int = 8766
    command_stale_s: float = 0.18
    ar_enabled: bool = True
    binary_protocol_enabled: bool = True
    tls: WebTLSConfig = field(default_factory=WebTLSConfig)

    def __post_init__(self) -> None:
        if not self.bind_address:
            raise ValueError("phone_config.web.bind_address must not be empty")
        for name, port in (("http_port", self.http_port), ("websocket_port", self.websocket_port)):
            if not 1 <= port <= 65535:
                raise ValueError(f"phone_config.web.{name} must be in range 1..65535")
        if self.http_port == self.websocket_port:
            raise ValueError("phone_config.web HTTP and WebSocket ports must differ")
        if not np.isfinite(self.command_stale_s) or self.command_stale_s <= 0.0:
            raise ValueError("phone_config.web.command_stale_s must be finite and positive")
        if self.tls.enabled and bool(self.tls.cert_file) != bool(self.tls.key_file):
            raise ValueError("phone_config.web.tls cert_file and key_file must be configured together")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WebPhoneConfig:
        tls_data = data.get("tls", {})
        if not isinstance(tls_data, dict):
            raise ValueError("phone_config.web.tls must be a mapping")
        return cls(
            bind_address=str(data.get("bind_address", "0.0.0.0")),
            http_port=int(data.get("http_port", 8765)),
            websocket_port=int(data.get("websocket_port", 8766)),
            command_stale_s=float(data.get("command_stale_s", 0.18)),
            ar_enabled=_parse_bool(data.get("ar_enabled"), field_name="phone_config.web.ar_enabled", default=True),
            binary_protocol_enabled=_parse_bool(
                data.get("binary_protocol_enabled"),
                field_name="phone_config.web.binary_protocol_enabled",
                default=True,
            ),
            tls=WebTLSConfig.from_dict(tls_data),
        )


@dataclass
class PhoneConfig:
    """Motion and transport settings for built-in WebPhone teleoperation."""

    backend: str = "webphone"
    camera_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, -0.02, 0.04]))
    position_scale: float = 0.7
    angular_scale: float = 1.0
    optical_flow_fallback_enabled: bool = True
    orientation_axis_mask: np.ndarray = field(default_factory=lambda: np.ones(3))
    orientation_deadzone_rad: float = 0.0
    orientation_filter_alpha: float = 1.0
    end_effector_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {"min": [-0.5, -0.5, -0.5], "max": [0.5, 0.5, 0.5]}
    )
    max_ee_step_m: float = 0.05
    max_angular_step_rad: float = 0.1
    gripper_speed_factor: float = 20.0
    gripper_range: tuple[float, float] = (0.0, 1.0)
    web: WebPhoneConfig = field(default_factory=WebPhoneConfig)

    def __post_init__(self) -> None:
        if self.backend != "webphone":
            raise ValueError("phone_config.backend must be 'webphone'")
        if self.camera_offset.shape != (3,) or not np.all(np.isfinite(self.camera_offset)):
            raise ValueError("phone_config.camera_offset must contain three finite values")
        if not np.isfinite(self.position_scale) or self.position_scale < 0.0:
            raise ValueError("phone_config.position_scale must be finite and non-negative")
        if not np.isfinite(self.angular_scale) or self.angular_scale < 0.0:
            raise ValueError("phone_config.angular_scale must be finite and non-negative")
        if self.orientation_axis_mask.shape != (3,) or not np.all(np.isfinite(self.orientation_axis_mask)):
            raise ValueError("phone_config.orientation_axis_mask must contain three finite values")
        if np.any(self.orientation_axis_mask < 0.0) or np.any(self.orientation_axis_mask > 1.0):
            raise ValueError("phone_config.orientation_axis_mask values must be in range 0..1")
        if not np.isfinite(self.orientation_deadzone_rad) or self.orientation_deadzone_rad < 0.0:
            raise ValueError("phone_config.orientation_deadzone_rad must be finite and non-negative")
        if not np.isfinite(self.orientation_filter_alpha) or not 0.0 < self.orientation_filter_alpha <= 1.0:
            raise ValueError("phone_config.orientation_filter_alpha must be in range (0, 1]")
        if not isinstance(self.end_effector_bounds, dict):
            raise ValueError("phone_config.end_effector_bounds must be a mapping")
        try:
            bounds_min = np.asarray(self.end_effector_bounds["min"], dtype=float).reshape(3)
            bounds_max = np.asarray(self.end_effector_bounds["max"], dtype=float).reshape(3)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("phone_config.end_effector_bounds must contain three-value min/max vectors") from exc
        if not np.all(np.isfinite(bounds_min)) or not np.all(np.isfinite(bounds_max)):
            raise ValueError("phone_config.end_effector_bounds must contain finite values")
        if np.any(bounds_min >= bounds_max):
            raise ValueError("phone_config.end_effector_bounds min values must be less than max values")
        self.end_effector_bounds = {"min": bounds_min.tolist(), "max": bounds_max.tolist()}
        if np.any(bounds_min >= 0.0) or np.any(bounds_max <= 0.0):
            raise ValueError("phone_config.end_effector_bounds must contain zero strictly inside every axis")
        if (
            not np.isfinite(self.max_ee_step_m)
            or not np.isfinite(self.max_angular_step_rad)
            or self.max_ee_step_m <= 0.0
            or self.max_angular_step_rad <= 0.0
        ):
            raise ValueError("phone_config Cartesian step limits must be finite and positive")
        if not np.isfinite(self.gripper_speed_factor) or self.gripper_speed_factor < 0.0:
            raise ValueError("phone_config.gripper_speed_factor must be finite and non-negative")
        if (
            len(self.gripper_range) != 2
            or not np.all(np.isfinite(self.gripper_range))
            or self.gripper_range[0] > self.gripper_range[1]
        ):
            raise ValueError("phone_config.gripper_range must contain finite ordered min/max values")
        if not self.web.ar_enabled and not self.optical_flow_fallback_enabled:
            raise ValueError("WebPhone requires WebXR AR or optical-flow fallback")

    @property
    def cartesian_input_mode(self) -> str:
        """Return the fixed WebPhone-to-Placo solver contract."""
        return "pose"

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a JSON-serializable dictionary."""
        return {
            "backend": self.backend,
            "camera_offset": self.camera_offset.tolist(),
            "position_scale": self.position_scale,
            "angular_scale": self.angular_scale,
            "optical_flow_fallback_enabled": self.optical_flow_fallback_enabled,
            "orientation_axis_mask": self.orientation_axis_mask.tolist(),
            "orientation_deadzone_rad": self.orientation_deadzone_rad,
            "orientation_filter_alpha": self.orientation_filter_alpha,
            "end_effector_bounds": self.end_effector_bounds,
            "max_ee_step_m": self.max_ee_step_m,
            "max_angular_step_rad": self.max_angular_step_rad,
            "gripper_speed_factor": self.gripper_speed_factor,
            "gripper_range": list(self.gripper_range),
            "web": {
                "bind_address": self.web.bind_address,
                "http_port": self.web.http_port,
                "websocket_port": self.web.websocket_port,
                "command_stale_s": self.web.command_stale_s,
                "ar_enabled": self.web.ar_enabled,
                "binary_protocol_enabled": self.web.binary_protocol_enabled,
                "tls": {
                    "enabled": self.web.tls.enabled,
                    "cert_file": self.web.tls.cert_file,
                    "key_file": self.web.tls.key_file,
                    "allow_insecure_http": self.web.tls.allow_insecure_http,
                },
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhoneConfig:
        """Create a WebPhone configuration; legacy phone_os is ignored."""
        backend = str(data.get("backend", "webphone")).lower()
        if backend != "webphone":
            raise ValueError("phone_config.backend must be 'webphone'")
        legacy_input_mode = data.get("input_mode")
        if legacy_input_mode is not None and str(legacy_input_mode).lower() != "pose":
            raise ValueError(
                f"phone_config.input_mode={legacy_input_mode!r} is no longer supported for "
                "WebPhone; phone teleoperation uses the fixed 'pose' contract"
            )
        web_data = data.get("web", {})
        if not isinstance(web_data, dict):
            raise ValueError("phone_config.web must be a mapping")
        return cls(
            backend=backend,
            camera_offset=np.asarray(data.get("camera_offset", [0.0, -0.02, 0.04]), dtype=float),
            position_scale=float(data.get("position_scale", 0.7)),
            angular_scale=float(data.get("angular_scale", 1.0)),
            optical_flow_fallback_enabled=_parse_bool(
                data.get("optical_flow_fallback_enabled"),
                field_name="phone_config.optical_flow_fallback_enabled",
                default=True,
            ),
            orientation_axis_mask=np.asarray(data.get("orientation_axis_mask", [1.0, 1.0, 1.0]), dtype=float),
            orientation_deadzone_rad=float(data.get("orientation_deadzone_rad", 0.0)),
            orientation_filter_alpha=float(data.get("orientation_filter_alpha", 1.0)),
            end_effector_bounds=data.get("end_effector_bounds", {"min": [-0.5, -0.5, -0.5], "max": [0.5, 0.5, 0.5]}),
            max_ee_step_m=float(data.get("max_ee_step_m", 0.05)),
            max_angular_step_rad=float(data.get("max_angular_step_rad", 0.1)),
            gripper_speed_factor=float(data.get("gripper_speed_factor", 20.0)),
            gripper_range=tuple(float(v) for v in data.get("gripper_range", [0.0, 1.0])),
            web=WebPhoneConfig.from_dict(web_data),
        )
