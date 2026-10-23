"""Lightweight skill package compiler and immutable catalog registry.

This package owns manifest/profile loading, schema validation, canonical digest
computation, snapshot compilation and source discovery. It MUST NOT import
``robot_config`` or ``skill_library``; it only consumes plain data passed in by
the caller (see :class:`skill_catalog.models.SkillRobotContext`).
"""

from __future__ import annotations

__version__ = "0.1.0"
