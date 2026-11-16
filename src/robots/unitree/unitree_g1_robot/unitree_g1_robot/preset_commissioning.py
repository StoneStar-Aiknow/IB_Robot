"""Per-run commissioning gate for the preset-only G1 surface.

The physical arm-preset path needs three facts for exactly one launch
invocation: the operator's confirmation, the robot-facing NIC, and the vendor
FSM the session is allowed to run in. They are launch parameters, never a
persisted ``confirmed: true`` profile: the shipped profiles stay disabled and
the runtime re-checks the same facts itself, so a direct ``ros2 run`` cannot
skip the gate by reusing a stale profile.

Both the launch entry (before it builds the nodes) and the runtime node (before
it opens the vendor tier) call this module. Keeping the rule here means the two
layers cannot drift into one accepting what the other rejects.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Prefix shared with the launch entry and the node's error messages.
PRESET_HARDWARE_DISABLED = "PRESET_HARDWARE_DISABLED"

_MAX_FSM_ID = 2**31 - 1


def _parse_expected_fsm_id(value: Any, *, label: str) -> int | None:
    """Parse an explicit or audited FSM id; ``None`` means "not supplied"."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be an integer FSM id, got {value!r}")
    if isinstance(value, int):
        fsm_id = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            fsm_id = int(text, 10)
        except ValueError as exc:
            raise RuntimeError(f"{label} must be an integer FSM id, got {value!r}") from exc
    else:
        raise RuntimeError(f"{label} must be an integer FSM id, got {value!r}")
    if not 1 <= fsm_id <= _MAX_FSM_ID:
        raise RuntimeError(f"{label} must be a positive FSM id (1..{_MAX_FSM_ID}), got {fsm_id!r}")
    return fsm_id


def resolve_preset_commissioning(
    profile: Mapping[str, Any],
    *,
    confirmed: Any,
    runtime_network_interface: Any,
    expected_fsm_id: Any,
    vendor_network_interface: Any = "",
) -> tuple[str, int]:
    """Validate one physical preset run and return ``(nic, expected_fsm_id)``.

    ``confirmed`` must be the boolean ``True`` supplied for this run. A
    ``vendor.operator_validation.confirmed: true`` left in a profile is not a
    confirmation and fails with a message that says so; a surviving old
    run-scoped copy behaves the same way.

    ``expected_fsm_id`` may be the explicit per-run value, an audited
    ``vendor.operator_validation.expected_fsm_id``, or both if they agree. There
    is no automatic FSM selection: neither value present is a refusal.

    ``vendor_network_interface`` is the NIC the vendor tier will actually bind.
    When supplied it must be the same robot-facing NIC; the two cannot be
    allowed to name different networks.
    """
    vendor = profile.get("vendor") or {}
    if not isinstance(vendor, Mapping):
        raise RuntimeError("profile vendor block must be a mapping")
    validation = vendor.get("operator_validation") or {}
    if not isinstance(validation, Mapping):
        raise RuntimeError("vendor.operator_validation must be a mapping")

    if confirmed is not True:
        if validation.get("confirmed") is True:
            raise RuntimeError(
                f"{PRESET_HARDWARE_DISABLED}: the profile's persisted "
                "vendor.operator_validation.confirmed=true is not a per-run confirmation; "
                "pass preset_operator_confirmed:=true for this launch"
            )
        raise RuntimeError(
            f"{PRESET_HARDWARE_DISABLED}: on-site operator confirmation required (preset_operator_confirmed:=true)"
        )

    nic = str(runtime_network_interface or "").strip()
    if not nic:
        raise RuntimeError(
            f"{PRESET_HARDWARE_DISABLED}: runtime_network_interface must name the robot-facing NIC "
            "(non-empty, not loopback)"
        )
    if nic == "lo":
        raise RuntimeError(
            f"{PRESET_HARDWARE_DISABLED}: runtime_network_interface must name the robot-facing NIC "
            "(non-empty, not loopback)"
        )

    profile_nic = str(validation.get("network_interface", "") or "").strip()
    if profile_nic and profile_nic != nic:
        raise RuntimeError(
            "preset commissioning has conflicting network interface overrides: "
            f"runtime_network_interface={nic!r}, "
            f"vendor.operator_validation.network_interface={profile_nic!r}"
        )

    bound_nic = str(vendor_network_interface or "").strip()
    if bound_nic and bound_nic != nic:
        raise RuntimeError(
            "preset commissioning has conflicting network interface overrides: "
            f"runtime_network_interface={nic!r}, vendor_network_interface={bound_nic!r}"
        )

    explicit = _parse_expected_fsm_id(expected_fsm_id, label="preset_expected_fsm_id")
    audited = _parse_expected_fsm_id(
        validation.get("expected_fsm_id"),
        label="vendor.operator_validation.expected_fsm_id",
    )
    if explicit is not None and audited is not None and explicit != audited:
        raise RuntimeError(
            "preset commissioning has conflicting expected FSM ids: "
            f"preset_expected_fsm_id={explicit!r}, "
            f"vendor.operator_validation.expected_fsm_id={audited!r}"
        )
    expected = explicit if explicit is not None else audited
    if expected is None:
        raise RuntimeError(
            f"{PRESET_HARDWARE_DISABLED}: preset_expected_fsm_id is required; "
            "the runtime never selects an FSM automatically"
        )
    return nic, expected
