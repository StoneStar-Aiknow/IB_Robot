"""The ``operator`` section of the business config. No ROS.

Every key has a default; a section may override any part of it. Every level
accepts only its known keys, so a misspelt key fails at startup instead of
being ignored. Times are seconds.

``frame_rate_max_hz``  camera frames decoded per second at most
``frame_gap_reset_s``  a longer gap between frames clears the gesture state
                       (a stalled stream is not a cover)
``startup_clear_s``    after startup, the camera must be clear this long
                       before a gesture counts
``double_cover``       two short covers: each lasts ``cover_min_s`` to
                       ``cover_max_s``, the gap between them ``gap_min_s`` to
                       ``gap_max_s``, both within ``total_max_s``
``long_cover_s``       a cover this long is a long cover, once
``rearm_clear_s``      after a gesture, clear this long before the next counts
``debounce_s``         a gesture within this of the previous one is dropped
``touch``              ``hold_s`` held, ``debounce_s``, and the vendor
                       ``event_type`` values that count as pressed
``led``                the light service, its type name, ``stable_s`` the
                       session must stay on/off before the light follows,
                       the vendor ``priority``, and the ``"on"``/``"off"``
                       colours (``mode`` and ``rgb``)
``session_silent_s``   no session state message for longer: no session
``max_session_s``      a session on (STANDBY or MIRRORING) this long is
                       stopped like a long cover; counted from the first
                       state that is not WAIT, cleared only by WAIT. 0: off
``leave_stop_min_round_s``
                       a round (MIRRORING) at least this long, cancelled
                       because the person left, stops the session like a
                       long cover once it is back in STANDBY. 0: off

``"on"`` and ``"off"`` must be quoted in YAML, which otherwise reads them as
booleans.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from manipulation_execution.mirror_operator.vendor import (
    LED_MODES,
    LED_SERVICE,
    LED_SERVICE_TYPE,
    TOUCH_PRESSED_EVENT_TYPES,
)

INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1


@dataclass(frozen=True)
class DoubleCover:
    cover_min_s: float = 0.2
    cover_max_s: float = 0.8
    gap_min_s: float = 0.2
    gap_max_s: float = 0.8
    total_max_s: float = 2.0


@dataclass(frozen=True)
class TouchConfig:
    hold_s: float = 1.0
    debounce_s: float = 3.0
    pressed_event_types: tuple[int, ...] = TOUCH_PRESSED_EVENT_TYPES


@dataclass(frozen=True)
class LedColor:
    mode: int
    rgb: tuple[int, int, int]


@dataclass(frozen=True)
class LedConfig:
    service: str = LED_SERVICE
    service_type: str = LED_SERVICE_TYPE
    stable_s: float = 3.0
    priority: int = 0
    on: LedColor = LedColor(0, (0, 255, 0))
    off: LedColor = LedColor(0, (0, 0, 255))


@dataclass(frozen=True)
class OperatorConfig:
    frame_rate_max_hz: float = 10.0
    frame_gap_reset_s: float = 0.5
    startup_clear_s: float = 1.0
    double_cover: DoubleCover = field(default_factory=DoubleCover)
    long_cover_s: float = 2.5
    rearm_clear_s: float = 1.0
    debounce_s: float = 3.0
    touch: TouchConfig = field(default_factory=TouchConfig)
    led: LedConfig = field(default_factory=LedConfig)
    session_silent_s: float = 3.0
    max_session_s: float = 0.0
    leave_stop_min_round_s: float = 0.0

    @classmethod
    def from_mapping(cls, section: Mapping[str, Any] | None) -> OperatorConfig:
        """The section over the defaults; ``ValueError`` names the first bad key."""
        section = _mapping(section, "operator", _TOP_KEYS)
        values = {}
        for name in _TOP_KEYS:
            if name in section and name not in _NESTED:
                values[name] = _seconds(section[name], f"operator.{name}", positive=name in _POSITIVE)
        values["double_cover"] = _double_cover(section.get("double_cover"))
        values["touch"] = _touch(section.get("touch"))
        values["led"] = _led(section.get("led"))
        config = cls(**values)
        cover = config.double_cover
        if cover.cover_min_s > cover.cover_max_s:
            raise ValueError("operator.double_cover: cover_min_s must not exceed cover_max_s")
        if cover.gap_min_s > cover.gap_max_s:
            raise ValueError("operator.double_cover: gap_min_s must not exceed gap_max_s")
        if cover.cover_max_s >= config.long_cover_s:
            raise ValueError("operator: double_cover.cover_max_s must be shorter than long_cover_s")
        return config


_NESTED = ("double_cover", "touch", "led")
_TOP_KEYS = tuple(name for name in OperatorConfig.__dataclass_fields__)
_POSITIVE = ("frame_rate_max_hz", "long_cover_s", "session_silent_s")


def _mapping(value: Any, path: str, known: tuple[str, ...]) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    unknown = sorted(str(key) for key in set(value) - set(known))
    if unknown:
        raise ValueError(f"{path} has unknown keys {unknown}; known: {list(known)}")
    return value


def _seconds(value: Any, path: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{path} must be a finite number, not negative")
    if positive and value == 0:
        raise ValueError(f"{path} must be greater than 0")
    return float(value)


def _integer(value: Any, path: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{path} must be an integer from {low} to {high}")
    return value


def _double_cover(value: Any) -> DoubleCover:
    known = tuple(DoubleCover.__dataclass_fields__)
    section = _mapping(value, "operator.double_cover", known)
    return DoubleCover(**{name: _seconds(section[name], f"operator.double_cover.{name}") for name in section})


def _touch(value: Any) -> TouchConfig:
    section = _mapping(value, "operator.touch", tuple(TouchConfig.__dataclass_fields__))
    values: dict[str, Any] = {
        name: _seconds(section[name], f"operator.touch.{name}") for name in ("hold_s", "debounce_s") if name in section
    }
    if "pressed_event_types" in section:
        types = section["pressed_event_types"]
        path = "operator.touch.pressed_event_types"
        if isinstance(types, str | bytes) or not isinstance(types, list | tuple) or not types:
            raise ValueError(f"{path} must be a non-empty list of event types")
        values["pressed_event_types"] = tuple(_integer(item, path, 0, 255) for item in types)
    return TouchConfig(**values)


def _led(value: Any) -> LedConfig:
    if isinstance(value, Mapping) and any(isinstance(key, bool) for key in value):
        raise ValueError('operator.led: quote the colour keys as "on" and "off"; YAML reads bare on/off as booleans')
    section = _mapping(value, "operator.led", tuple(LedConfig.__dataclass_fields__))
    values: dict[str, Any] = {}
    for name in ("service", "service_type"):
        if name in section:
            if not isinstance(section[name], str) or not section[name]:
                raise ValueError(f"operator.led.{name} must be a non-empty string")
            values[name] = section[name]
    if "service" in values and not values["service"].startswith("/"):
        raise ValueError("operator.led.service must be an absolute ROS name")
    if "stable_s" in section:
        values["stable_s"] = _seconds(section["stable_s"], "operator.led.stable_s")
    if "priority" in section:
        values["priority"] = _integer(section["priority"], "operator.led.priority", INT32_MIN, INT32_MAX)
    defaults = LedConfig()
    for name in ("on", "off"):
        if name in section:
            values[name] = _color(section[name], f"operator.led.{name}", getattr(defaults, name))
    return LedConfig(**values)


def _color(value: Any, path: str, default: LedColor) -> LedColor:
    section = _mapping(value, path, ("mode", "rgb"))
    mode = default.mode
    if "mode" in section:
        mode = _integer(section["mode"], f"{path}.mode", LED_MODES[0], LED_MODES[-1])
    rgb = default.rgb
    if "rgb" in section:
        rgb = section["rgb"]
        if isinstance(rgb, str | bytes) or not isinstance(rgb, list | tuple) or len(rgb) != 3:
            raise ValueError(f"{path}.rgb must be three integers from 0 to 255")
        rgb = tuple(_integer(item, f"{path}.rgb", 0, 255) for item in rgb)
    return LedColor(mode, rgb)
