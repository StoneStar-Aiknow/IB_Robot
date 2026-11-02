"""Load a robot's grasp backends from configuration.

``grasp_execution.backend`` names a factory as ``"package.module:callable"``.
The factory receives the node and the robot's grasp configuration and returns a
``GraspBackends``. Loading by name keeps this package free of any dependency on
a robot package, the same way ``providers.py`` loads geometry providers.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping

from .contracts import GraspBackends


class BackendLoadError(ValueError):
    pass


def load_backends(spec: str, **context) -> GraspBackends:
    module_name, separator, attribute = str(spec).partition(":")
    if not separator or not module_name or not attribute:
        raise BackendLoadError(f"grasp backend must be 'package.module:callable', got {spec!r}")
    try:
        factory = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise BackendLoadError(f"cannot load grasp backend {spec!r}: {exc}") from exc
    backends = factory(**context)
    if not isinstance(backends, GraspBackends):
        raise BackendLoadError(f"{spec} returned {type(backends).__name__}, not GraspBackends")
    return backends


def backend_spec(grasp_execution: Mapping) -> str | None:
    """The configured factory, or None for the legacy primitive executor."""
    value = grasp_execution.get("backend")
    return None if value in (None, "") else str(value)
