"""The recording script for the key-action evaluation, and its labels.

One person, one long video: a cue track played next to them names each slot,
beeps once to start the move and again to put the hands down. Since every
slot has a fixed length, the labels of a whole video follow from one number,
the time in the video where the cue track started; key_eval finds it by
matching the video's motion against :func:`move_windows`.

Version 1 gave every slot 9.5 s, which makes that match nearly periodic: a
shift by one slot scores about 90 % of the true offset. Version 2 keeps the
same slots and beeps but varies the slot lengths (9–12.5 s, a fixed
non-repeating sequence), which brings the best wrong shift down to about 40 %.

This module is the single source for that script. The cue track builder (off
the repository, next to the TTS tool) and key_eval both read it; changing a
slot changes the cue track, so a video must be scored with the schedule
``version`` it was recorded with.

    python3 -m visual_interaction.session.key_eval_schedule --sheet   # printable list
    python3 -m visual_interaction.session.key_eval_schedule --json    # for the cue builder
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace

from visual_interaction.session.key_actions import KEY_ACTIONS

VERSION = 2
VERSIONS = (1, 2)

NONE = "none"

# Seconds of silence at the head of the cue track, before the first slot.
LEAD_IN_SEC = 3.0


@dataclass(frozen=True)
class Slot:
    """One cued move. ``go``/``down`` are seconds from the slot's start."""

    label: str  # a key action, or "none"
    variant: str  # e.g. "heart_finger", "none_ok"; the finer label
    cue: str  # what the cue track says at the slot's start
    length: float = 9.5
    go: float = 3.5
    down: float = 7.0
    beeps: bool = True


def _key(variant: str, cue: str) -> Slot:
    label = "heart" if variant.startswith("heart_") else variant.split("_")[0]
    if label not in KEY_ACTIONS:
        raise ValueError(f"unknown key action in {variant!r}")
    return Slot(label, variant, cue)


def _none(variant: str, cue: str) -> Slot:
    return Slot(NONE, f"none_{variant}", cue)


_SLOTS_V1: tuple[Slot, ...] = (
    _none("armsup", "准备，双手举过头顶"),
    _key("heart_finger", "单手比心，右手，胸前"),
    _key("heart_finger", "单手比心，左手，胸前"),
    _key("heart_finger", "单手比心，右手，脸旁"),
    _key("heart_finger", "单手比心，左手，脸旁"),
    _none("ok", "比个欧克手势"),
    _none("one", "比数字一"),
    _key("heart_twohand", "双手胸前比心"),
    _key("heart_twohand", "双手胸前比心"),
    _key("heart_twohand", "侧身，双手胸前比心"),
    _key("heart_twohand", "侧身，双手胸前比心"),
    _none("fist", "握拳举起来，加油"),
    _none("clap", "拍拍手"),
    _key("heart_overhead", "双手头顶比心"),
    _key("heart_overhead", "双手头顶比心"),
    _key("heart_overhead", "侧身，双手头顶比心"),
    _key("heart_overhead", "侧身，双手头顶比心"),
    _none("raise", "举手，不要挥"),
    _none("point", "用手指指一个方向"),
    _key("thumb_right", "点赞，右手"),
    _key("thumb_left", "点赞，左手"),
    _key("thumb_forward", "点赞，右手往前伸"),
    _key("thumb_both", "双手点赞"),
    _none("three", "比数字三"),
    _none("hips", "双手叉腰"),
    _key("yeah_right_face", "剪刀手，右手，脸旁"),
    _key("yeah_left_face", "剪刀手，左手，脸旁"),
    _key("yeah_right_chest", "剪刀手，右手，胸前"),
    _key("yeah_left_chest", "剪刀手，左手，胸前"),
    Slot(NONE, "none_rest", "休息一下，自然站着", length=15.0, go=15.0, down=15.0, beeps=False),
    _none("armsup", "双手举过头顶，不要比心"),
    _none("crossarms", "双手抱在胸前"),
    _key("wave_right_big", "右手大幅挥手"),
    _key("wave_left_big", "左手大幅挥手"),
    _key("wave_right_small", "右手胸前小幅挥手"),
    _key("wave_left_small", "左手胸前小幅挥手"),
    _none("phone", "拿手机看一眼"),
    _none("touchface", "摸摸脸或者头发"),
    _key("open_flat", "张开双臂，平举"),
    _key("open_up", "张开双臂，斜向上"),
    _key("open_flat_side", "侧身，张开双臂平举"),
    _key("open_half", "半张开双臂，手肘弯着"),
    Slot(NONE, "none_free", "自由活动，不做刚才五种动作", length=30.0, go=4.0, down=28.0),
)


# Version 2: the length of each standard slot, in order (rest and free keep theirs).
_V2_LENGTHS = (
    12.5,
    12.5,
    12.5,
    10.5,
    11.5,
    12.5,
    12.5,
    11.5,
    12.5,
    10.5,
    9.5,
    9.0,
    12.5,
    9.0,
    12.5,
    9.0,
    9.5,
    9.5,
    9.5,
    9.0,
    9.0,
    12.5,
    9.0,
    12.5,
    12.5,
    9.5,
    10.5,
    9.0,
    11.5,
    9.5,
    9.0,
    12.5,
    12.5,
    9.5,
    9.0,
    9.0,
    11.5,
    10.5,
    12.5,
    9.0,
    12.5,
)


def _with_v2_lengths(slots: tuple[Slot, ...]) -> tuple[Slot, ...]:
    """Version 1 slots with the standard ones (beep at 3.5 s, down at 7 s) re-timed."""
    standard = [i for i, s in enumerate(slots) if (s.go, s.down, s.length) == (Slot.go, Slot.down, Slot.length)]
    if len(standard) != len(_V2_LENGTHS):
        raise ValueError(f"{len(_V2_LENGTHS)} v2 lengths for {len(standard)} standard slots")
    out = list(slots)
    for i, length in zip(standard, _V2_LENGTHS, strict=True):
        out[i] = replace(out[i], length=length)
    return tuple(out)


_SLOTS = {1: _SLOTS_V1, 2: _with_v2_lengths(_SLOTS_V1)}


def slots(version: int = VERSION) -> tuple[Slot, ...]:
    if version not in _SLOTS:
        raise ValueError(f"unknown schedule version {version}; known: {VERSIONS}")
    return _SLOTS[version]


def starts(version: int = VERSION) -> list[float]:
    """Each slot's start, in seconds from the start of the cue track."""
    out, t = [], LEAD_IN_SEC
    for slot in slots(version):
        out.append(t)
        t += slot.length
    return out


def total_sec(version: int = VERSION) -> float:
    return LEAD_IN_SEC + sum(slot.length for slot in slots(version))


def move_windows(offset: float = 0.0, version: int = VERSION) -> list[tuple[float, float, Slot]]:
    """``(go, down, slot)`` in video seconds, for a cue track started at ``offset``."""
    pairs = zip(starts(version), slots(version), strict=True)
    return [(offset + t + s.go, offset + t + s.down, s) for t, s in pairs if s.go < s.down]


def label_at(video_sec: float, offset: float, version: int = VERSION) -> tuple[str, Slot | None]:
    """What the person should be doing at ``video_sec``: a key action, "none", or "" (unknown).

    "" covers the time before the cue track and after it, where nothing is
    known. Inside a slot, before ``go`` and after ``down`` is "none".
    """
    t = video_sec - offset
    if t < LEAD_IN_SEC:
        return "", None
    for start, slot in zip(starts(version), slots(version), strict=True):
        if start <= t < start + slot.length:
            inside = slot.go <= t - start < slot.down
            return (slot.label if inside else NONE), slot
    return "", None


def _sheet(version: int) -> str:
    lines = [f"# 关键动作录制顺序（schedule v{version}，全长 {total_sec(version):.0f} 秒）", ""]
    lines += ["| # | 开始（秒） | 听到什么 | 代号 |", "|---|---|---|---|"]
    for n, (start, slot) in enumerate(zip(starts(version), slots(version), strict=True), 1):
        lines.append(f"| {n} | {start:.0f} | {slot.cue} | {slot.variant} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sheet", action="store_true", help="print the slot list as a markdown table")
    group.add_argument("--json", action="store_true", help="print the schedule as JSON")
    parser.add_argument("--version", type=int, default=VERSION, choices=VERSIONS)
    args = parser.parse_args(argv)
    if args.sheet:
        sys.stdout.write(_sheet(args.version))
    else:
        payload = {
            "version": args.version,
            "lead_in_sec": LEAD_IN_SEC,
            "total_sec": total_sec(args.version),
            "slots": [
                dict(asdict(slot), start=start)
                for start, slot in zip(starts(args.version), slots(args.version), strict=True)
            ],
        }
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=1)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
