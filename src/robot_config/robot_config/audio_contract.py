"""Shared audio configuration predicates and peripheral lookup helpers."""

from collections.abc import Iterable, Mapping
from typing import Any


def is_audio_io_enabled(audio_io: Any) -> bool:
    """Return whether the shared audio_common device owners are enabled."""

    if isinstance(audio_io, Mapping):
        return bool(audio_io.get("enabled", False))
    return bool(audio_io) and bool(getattr(audio_io, "enabled", False))


def find_microphones(peripherals: Iterable[Any], name: str) -> list[Any]:
    """Return all microphones matching ``name`` for validation or launch use."""

    matches = []
    for peripheral in peripherals or ():
        if isinstance(peripheral, Mapping):
            peripheral_type = peripheral.get("type")
            peripheral_name = peripheral.get("name")
        else:
            peripheral_type = getattr(peripheral, "type", None)
            peripheral_name = getattr(peripheral, "name", None)
        if peripheral_type == "microphone" and peripheral_name == name:
            matches.append(peripheral)
    return matches


def find_microphone_params(peripherals: Iterable[Any], name: str) -> Mapping[str, Any]:
    """Return parameters for the first matching microphone, or an empty mapping."""

    matches = find_microphones(peripherals, name)
    peripheral = matches[0] if matches else None
    params = peripheral.get("params", {}) if isinstance(peripheral, Mapping) else getattr(peripheral, "params", {})
    return params if isinstance(params, Mapping) else {}
