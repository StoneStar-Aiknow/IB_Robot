"""Canonical scheduled inference work-class identifiers."""

from __future__ import annotations

from enum import IntEnum


class WorkClass(IntEnum):
    """Wire-compatible work classes shared by Global and pipeline runtimes."""

    SESSION_CONTROL = 1
    ACTION_GENERATION = 2


def work_class_name(work_class: WorkClass | int) -> str:
    """Return the robot-config key for a work-class value."""

    return WorkClass(work_class).name.lower()


__all__ = ["WorkClass", "work_class_name"]
