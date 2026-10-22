"""The two vendor types the operator touches: the head touch and the light strip.

The only file in this package that names the vendor. ``manipulation_execution``
is a generic package and does not depend on the vendor's message package: both
types are loaded by name at runtime, so a machine without the vendor
environment still builds and runs it, with the touch and the light off.

* Head touch: the type name is the ``message_type`` of ``touch.head`` in the
  runtime description. Only ``event_type`` is read (``0`` unknown, ``1`` idle,
  ``2`` touch; the pats ``4``/``5``/``6`` are not a hold).
* Light strip: the type name is ``operator.led.service_type``. Only
  ``led_strip_mode`` (``0`` constant, ``1`` breath, ``2`` flash, ``3`` flow),
  ``r``, ``g``, ``b``, ``priority`` and ``reset_priority`` are written, and only
  ``status_code`` is read.
"""

from __future__ import annotations

from typing import Any

from rosidl_runtime_py.utilities import get_message, get_service

LED_SERVICE = "/aimdk_5Fmsgs/srv/SetPmuLed"
LED_SERVICE_TYPE = "aimdk_msgs/srv/SetPmuLed"
LED_MODES = (0, 1, 2, 3)
LED_STATUS_OK = 0
# Another holder with a higher priority has the light (the vendor's own interaction).
LED_PRIORITY_REJECTED = 0x1024
TOUCH_PRESSED_EVENT_TYPES = (2,)


def load_message(name: str) -> tuple[Any | None, str]:
    """The message class named ``pkg/msg/Name``, or ``(None, reason)``; never raises."""
    try:
        return get_message(name), ""
    except Exception as exc:  # noqa: BLE001 - any failure turns the touch off, never the node
        return None, f"message type {name!r} cannot be loaded: {exc}"


def load_service(name: str) -> tuple[Any | None, str]:
    """The service class named ``pkg/srv/Name``, or ``(None, reason)``; never raises."""
    try:
        return get_service(name), ""
    except Exception as exc:  # noqa: BLE001 - any failure turns the light off, never the node
        return None, f"service type {name!r} cannot be loaded: {exc}"


def touch_event_type(message: Any) -> int:
    return int(message.event_type)


def led_request(service_type: Any, mode: int, rgb: tuple[int, int, int], priority: int) -> Any:
    """A light request that never takes the light by force (``reset_priority`` False)."""
    request = service_type.Request()
    request.led_strip_mode = int(mode)
    request.r, request.g, request.b = (int(value) for value in rgb)
    request.priority = int(priority)
    request.reset_priority = False
    return request


def led_status_code(response: Any) -> int:
    return int(response.status_code)
