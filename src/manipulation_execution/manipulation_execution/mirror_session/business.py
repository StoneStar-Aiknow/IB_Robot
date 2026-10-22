"""The business config file and the business hook classes, by name. No ROS.

The session node and the operator node both read the one business config file
(``business_config``). It has four sections, each a mapping, each optional:

``session``      given as is to the business hooks class; not checked here
``timing``       read by the session: ``start_confirm_s``, ``cancel_confirm_s``
``cover_judge``  given as is to the cover judge class; not checked here
``operator``     read by the operator node

The business hooks are chosen by name (``business_hooks``):

``""``             the mock playback when it is on, else the ``SessionHooks``
                   defaults
``"demo"``         the demo rules (:mod:`.demo`) around the hooks ``""`` would
                   choose, so the demo rules and the mock run together
``"pkg.mod:Class"``  ``Class(session section, HookContext)``; cannot be combined
                   with the mock playback

A file that cannot be read or a class that cannot be loaded raises
``ValueError``: the node fails at startup and never falls back to a default.
A business class is constructed in the node's ``__init__``; a slow constructor
delays the node's startup.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from manipulation_execution.mirror_session.demo import DemoHooks
from manipulation_execution.mirror_session.hooks import HookContext, SessionHooks

DEMO = "demo"
SECTIONS = ("session", "timing", "cover_judge", "operator")
TIMING_KEYS = ("start_confirm_s", "cancel_confirm_s")


@dataclass(frozen=True)
class BusinessConfig:
    session: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    cover_judge: dict = field(default_factory=dict)
    operator: dict = field(default_factory=dict)


def load_business_config(path: str) -> BusinessConfig:
    """The four sections of the file at ``path``; empty path: all four empty."""
    if not path:
        return BusinessConfig()
    try:
        content = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"business_config {path!r} cannot be read: {exc}") from exc
    if not isinstance(content, Mapping):
        raise ValueError(f"business_config {path!r} must be a mapping of the sections {list(SECTIONS)}")
    unknown = sorted(str(key) for key in set(content) - set(SECTIONS))
    if unknown:
        raise ValueError(f"business_config {path!r} has unknown sections {unknown}; known: {list(SECTIONS)}")
    sections = {}
    for name in SECTIONS:
        section = content.get(name)
        if section is None:  # missing, or written without a value
            section = {}
        if not isinstance(section, Mapping):
            raise ValueError(f"business_config {path!r}: {name} must be a mapping")
        sections[name] = dict(section)
    unknown = sorted(str(key) for key in set(sections["timing"]) - set(TIMING_KEYS))
    if unknown:
        raise ValueError(f"business_config {path!r}: timing has unknown keys {unknown}; known: {list(TIMING_KEYS)}")
    return BusinessConfig(**sections)


def load_class(name: str) -> type:
    """``"pkg.mod:Class"`` -> the class."""
    module_name, separator, class_name = name.partition(":")
    if not separator or not module_name or not class_name or ":" in class_name:
        raise ValueError(f"{name!r} is not 'package.module:Class'")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"{name!r}: module {module_name!r} cannot be imported: {exc}") from exc
    cls = getattr(module, class_name, None)
    if not isinstance(cls, type):
        raise ValueError(f"{name!r}: {module_name} has no class {class_name!r}")
    return cls


def build_hooks(
    name: str, business: BusinessConfig, context: HookContext, mock: SessionHooks | None
) -> SessionHooks:
    """The business hooks named ``name`` (see the module docstring)."""
    if not isinstance(name, str):
        raise ValueError("business_hooks must be a string")
    if not name:
        return mock if mock is not None else SessionHooks()
    if name == DEMO:
        return DemoHooks(business.session, context, inner=mock if mock is not None else SessionHooks())
    if mock is not None:
        raise ValueError(f"business_hooks {name!r} and the mock playback are exclusive; only {DEMO!r} wraps the mock")
    cls = load_class(name)
    if not issubclass(cls, SessionHooks):
        raise ValueError(f"business_hooks {name!r} is not a SessionHooks subclass")
    return cls(business.session, context)
