"""The confirmed value mapping, read from JSON rather than hardcoded.

`x2_retarget_ledger.json` is the record of seven operator confirmations: for
each X2 arm channel, which SMPL-X quantity drives it, what its zero is, and
which way positive points. Every one of those was decided by a human watching a
rendered comparison, not by a fit, and the ledger is where that decision lives.

This module refuses to hand out a channel that is not marked `verified`. That
is the whole safety story of the conversion stage: an unconfirmed number cannot
leak into a trajectory by being merely present in the file.

The mapping itself is one line, the same for every channel:

    x2_arm_pos[index] = sign * (source_rad - offset_rad) * scale

with one documented exception. `shoulder_yaw`'s `source_rad` is not a function
of the source alone -- it depends on the commands already issued to pitch, roll
and elbow -- so that channel must be evaluated last. See `retarget.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import numpy as np

from .paths import LEDGER_PATH

SIDES: tuple[str, ...] = ("left", "right")

# Presentation order: proximal to distal, which is also the order the channels
# were confirmed in. NOT the evaluation order -- see `retarget.EVALUATION_ORDER`.
CHANNEL_ORDER: tuple[str, ...] = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_yaw",
    "wrist_pitch",
    "wrist_roll",
)

ARM_DOF = 14


class LedgerError(ValueError):
    """The ledger is missing, malformed, or holds an unconfirmed channel."""


@dataclass(frozen=True)
class Channel:
    """One confirmed channel: two arm_pos slots, two signs, two zeros, one scale."""

    name: str
    arm_pos: Mapping[str, int]
    sign: Mapping[str, float]
    offset_rad: Mapping[str, float]
    scale: float
    source_type: str
    euler_axis: int | None
    body_index: Mapping[str, int] | None
    readout: str
    verified: bool
    verified_by: str | None

    def index(self, side: str) -> int:
        return self.arm_pos[side]

    def command_rad(self, side: str, source_rad: np.ndarray | float) -> np.ndarray:
        """Apply the ledger formula. Radians in, radians out, no clipping.

        Clipping is deliberately not done here: a caller that wants the raw
        command in order to count how far it overshoots must be able to see it.
        `retarget` clips once, at the end, in radians.
        """
        source = np.asarray(source_rad, dtype=np.float64)
        return self.sign[side] * (source - self.offset_rad[side]) * self.scale


@dataclass(frozen=True)
class Ledger:
    path: Path
    formula: str
    status: str
    channels: Mapping[str, Channel]

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Ledger":
        path = Path(path) if path is not None else LEDGER_PATH
        if not path.is_file():
            raise LedgerError(f"ledger not found: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        try:
            mapping = raw["value_mapping"]
            entries = mapping["channels"]
        except KeyError as exc:  # pragma: no cover - malformed file
            raise LedgerError(f"{path} has no value_mapping.channels ({exc})") from exc

        channels: dict[str, Channel] = {}
        for name, entry in entries.items():
            source = entry.get("smplx_source") or {}
            body_index = source.get("body_index")
            # The three wrist channels were written into the ledger before the
            # `source_type` key existed; they carry a `euler_axis` instead. A
            # missing type with an axis present therefore means "euler", and a
            # missing type with no axis is a malformed entry, not a default.
            source_type = str(source.get("source_type") or "")
            if not source_type:
                if source.get("euler_axis") is None:
                    raise LedgerError(
                        f"channel {name!r} has neither smplx_source.source_type nor euler_axis"
                    )
                source_type = "euler"

            channels[name] = Channel(
                name=name,
                arm_pos={"left": int(entry["arm_pos"][0]), "right": int(entry["arm_pos"][1])},
                sign={side: float(entry["sign"][side]) for side in SIDES},
                offset_rad={side: float(entry["offset_rad"][side]) for side in SIDES},
                scale=float(entry["scale"]),
                source_type=source_type,
                euler_axis=None if source.get("euler_axis") is None else int(source["euler_axis"]),
                body_index=None if body_index is None
                else {side: int(body_index[side]) for side in SIDES},
                readout=str(source.get("readout", "")),
                verified=bool(entry.get("verified", False)),
                verified_by=entry.get("verified_by"),
            )

        ledger = cls(
            path=path.resolve(),
            formula=str(mapping.get("formula", "")),
            status=str(raw.get("status", "")),
            channels=channels,
        )
        ledger.validate()
        return ledger

    def validate(self) -> None:
        """Structural checks that must hold before any channel is used."""
        missing = [name for name in CHANNEL_ORDER if name not in self.channels]
        if missing:
            raise LedgerError(f"{self.path.name} is missing channels: {missing}")
        extra = sorted(set(self.channels) - set(CHANNEL_ORDER))
        if extra:
            raise LedgerError(f"{self.path.name} has unknown channels: {extra}")

        slots: dict[int, str] = {}
        for channel in self.channels.values():
            for side in SIDES:
                index = channel.index(side)
                if not 0 <= index < ARM_DOF:
                    raise LedgerError(
                        f"{channel.name}.{side} points at arm_pos[{index}], outside 0..{ARM_DOF - 1}"
                    )
                if index in slots:
                    raise LedgerError(
                        f"arm_pos[{index}] is claimed by both {slots[index]} and {channel.name}.{side}"
                    )
                slots[index] = f"{channel.name}.{side}"
        if len(slots) != ARM_DOF:
            unclaimed = sorted(set(range(ARM_DOF)) - set(slots))
            raise LedgerError(f"no channel drives arm_pos {unclaimed}")

    def require_confirmed(self, names: Iterable[str] | None = None) -> None:
        """Raise unless every named channel carries an operator confirmation.

        Both flags are checked. `verified` alone would pass a channel that was
        flipped true by hand without anyone having watched anything, and
        `verified_by` records who watched.
        """
        for name in (CHANNEL_ORDER if names is None else names):
            channel = self.channels[name]
            if not channel.verified or channel.verified_by is None:
                raise LedgerError(
                    f"channel {name!r} is not confirmed in {self.path.name} "
                    f"(verified={channel.verified}, verified_by={channel.verified_by!r}); "
                    "it must not drive a joint"
                )

    def __getitem__(self, name: str) -> Channel:
        return self.channels[name]

    def __iter__(self) -> Iterator[Channel]:
        return (self.channels[name] for name in CHANNEL_ORDER)

    def slot_names(self) -> list[str]:
        """`arm_pos` index -> "<side> <channel>", for reports and CSV headers."""
        names = [""] * ARM_DOF
        for channel in self.channels.values():
            for side in SIDES:
                names[channel.index(side)] = f"{side}_{channel.name}"
        return names
