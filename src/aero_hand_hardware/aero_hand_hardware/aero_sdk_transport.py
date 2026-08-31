"""Compatibility transport for the non-blocking Aero Hand state protocol.

The released ``aero-open-sdk`` exposes command writes publicly, but its compact
state getter blocks until a complete reply arrives.  This adapter is the only
place where the driver relies on the SDK's private serial/protocol members.  It
keeps that compatibility code isolated so the rest of the driver depends on a
small, testable transport contract.
"""

from __future__ import annotations

import math
import struct

JOINT_COUNT = 7
STATE_FRAME_BYTES = 2 + JOINT_COUNT * 2
UINT16_MAX_FALLBACK = 65535.0


class AeroSdkTransport:
    """Issue non-blocking position requests and decode compact replies."""

    def __init__(self, hand):
        self.hand = hand
        for attribute in ("ser", "_send_data", "actuation_lower_limits", "actuation_upper_limits"):
            if not hasattr(hand, attribute):
                raise RuntimeError(f"Aero SDK object does not expose required transport member {attribute!r}")
        if not hasattr(hand, "actuations_to_joints_model"):
            raise RuntimeError("Aero SDK object does not expose its actuation-to-joint model")

    def request_position(self) -> None:
        from aero_open_sdk.aero_hand import GET_POS

        self.hand.ser.reset_input_buffer()
        self.hand._send_data(GET_POS)

    def poll_position(self) -> list[float] | None:
        """Return decoded degree positions, or ``None`` when a frame is incomplete."""
        from aero_open_sdk.aero_hand import GET_POS

        if self.hand.ser.in_waiting < STATE_FRAME_BYTES:
            return None
        frame = self.hand.ser.read(STATE_FRAME_BYTES)
        if len(frame) != STATE_FRAME_BYTES:
            raise RuntimeError(f"Aero Hand state reply has {len(frame)} bytes; expected {STATE_FRAME_BYTES}")

        data = struct.unpack("<2B7H", frame)
        if data[0] != GET_POS:
            raise RuntimeError(f"Aero Hand state reply opcode mismatch: {data[0]:#04x}")

        try:
            from aero_open_sdk.aero_hand import _UINT16_MAX

            uint16_max = float(_UINT16_MAX)
        except (ImportError, AttributeError):
            uint16_max = UINT16_MAX_FALLBACK
        if uint16_max <= 0.0:
            raise RuntimeError("Aero SDK exposes an invalid uint16 actuation scale")

        lower = self.hand.actuation_lower_limits
        upper = self.hand.actuation_upper_limits
        actuations_deg = [
            float(lower[index]) + (data[2 + index] / uint16_max) * (float(upper[index]) - float(lower[index]))
            for index in range(JOINT_COUNT)
        ]
        joints_rad = self.hand.actuations_to_joints_model.hand_joints(
            [value * math.radians(1.0) for value in actuations_deg]
        )
        return [math.degrees(float(value)) for value in joints_rad]

    def close(self) -> None:
        close = getattr(self.hand, "close", None)
        if callable(close):
            close()
        elif getattr(self.hand, "ser", None) is not None:
            self.hand.ser.close()
