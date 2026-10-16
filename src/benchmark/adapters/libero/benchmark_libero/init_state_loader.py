"""Explicit trusted init-state loader for the verified hf-libero provider.
LIBERO runtime trusted initial-state loading: ``benchmark.get_task_init_states(i)``
calls ``torch.load(init_states_path)`` without ``weights_only=False``. Under
torch 2.6+ the default changed to ``weights_only=True``, which rejects the
legacy numpy pickle used by LIBERO's init-state files.
This module resolves the trusted fixed init-state asset from the pinned
LIBERO task metadata + ``get_libero_path("init_states")`` and loads it
locally with an explicit ``torch.load(path, weights_only=False)``.
Security comment: provider identity verification pins ``hf-libero`` to the
accepted version range and proves that init-state paths resolve inside that
installed distribution. The adapter never accepts arbitrary user paths.
``weights_only=False`` is therefore acceptable
here; it must NOT be used for arbitrary external checkpoints.
"""

from __future__ import annotations

import os
import re
from typing import Any


class InitStateLoadError(RuntimeError):
    """Raised when the trusted LIBERO init-state file cannot be loaded."""


def resolve_init_states_path(task: Any, get_libero_path_fn: Any, *, provider: str = "hf-libero") -> str:
    """Resolve the trusted init-state file path from pinned LIBERO task metadata.
    ``task`` is a ``libero.libero.benchmark.Task`` NamedTuple carrying
    ``problem_folder`` and ``init_states_file``. ``get_libero_path_fn`` is the
    pinned ``libero.libero.get_libero_path`` callable. The resolved path is
    ``<get_libero_path("init_states")>/<task.problem_folder>/<task.init_states_file>``.

    ``provider`` selects the resolution mode:

    - ``"hf-libero"`` (default) is fail-closed: only the exact path is
      accepted. A missing file returns the exact path so the downstream
      loader raises, mirroring the pre-LIBERO-plus behavior. This guards
      against a mismatched ``libero`` package silently serving a different
      task table.
    - ``"libero-plus"`` mirrors the provider's own ``get_task_init_states``
      path resolution:
      - ``_add_``/``_level`` tasks (new objects) resolve under
        ``init_files/libero_newobj/<problem_folder>/`` and have per-task files.
      - Other perturbation suffixes (``_table_``, ``_tb_``, ``_view_``,
        ``_language_``, ``_initstate_``, ``_noise_``, ``_light_``) strip the
        suffix to the base task's ``.pruned_init`` file. Missing files
        fail closed rather than falling back to a different state set.
    """
    if task is None:
        raise InitStateLoadError("task metadata is None; cannot resolve init-state path")
    problem_folder = getattr(task, "problem_folder", None)
    init_states_file = getattr(task, "init_states_file", None)
    if not problem_folder or not init_states_file:
        raise InitStateLoadError(
            f"task metadata is missing problem_folder/init_states_file; "
            f"got problem_folder={problem_folder!r}, init_states_file={init_states_file!r}"
        )
    base = get_libero_path_fn("init_states")
    if not base:
        raise InitStateLoadError("get_libero_path('init_states') returned an empty path")
    name = str(init_states_file)
    folder = str(problem_folder)
    exact = os.path.join(str(base), folder, name)

    if provider != "libero-plus":
        return exact

    # LIBERO-plus perturbation resolution below.
    # _add_ and _level tasks have their own init files under libero_newobj
    # (upstream benchmark/__init__.py lines 226-232); the state dimension
    # differs because these tasks add new objects to the scene.
    if "_add_" in name or "_level" in name:
        return os.path.join(str(base), "libero_newobj", folder, name)
    if os.path.isfile(exact):
        return exact
    # Perturbation variants share the base task's init file. Strip the
    # suffix at the earliest known marker, retaining the upstream state set.
    stem = name.rsplit(".", 1)[0]
    # Base spatial task names also contain e.g. _table_center_; only numeric
    # table/background suffixes are perturbations, as in the pinned provider.
    stem = re.sub(r"_(?:table|tb)_\d+", "", stem)
    cut = len(stem)
    for marker in ("_view_", "_language_", "_initstate_", "_noise_", "_light_"):
        index = stem.find(marker)
        if 0 < index < cut:
            cut = index
    if cut < len(stem) or stem != name.rsplit(".", 1)[0]:
        return os.path.join(str(base), folder, stem[:cut] + ".pruned_init")
    return exact


def load_task_init_states(task: Any, get_libero_path_fn: Any, *, provider: str = "hf-libero") -> Any:
    """Load and normalize native states identically for planning and reset."""
    path = resolve_init_states_path(task, get_libero_path_fn, provider=provider)
    states = load_trusted_init_states(path)
    name = str(task.init_states_file)
    if provider == "libero-plus" and ("_add_" in name or "_level" in name):
        import numpy as np

        states = np.asarray(states).reshape(1, -1)
    return states


def load_trusted_init_states(init_states_path: str) -> Any:
    """Load a trusted LIBERO init-state file with explicit ``weights_only=False``.
    The file is a provider-owned asset inside the identity-verified
    ``hf-libero`` distribution. ``weights_only=False`` is required because
    LIBERO pickles init states with the legacy numpy format. This
    function must NOT be used for arbitrary external checkpoints.
    """
    import torch  # noqa: PLC0415 -- heavy import deferred to adapter boundary

    if not os.path.isfile(init_states_path):
        raise InitStateLoadError(f"trusted init-state file does not exist: {init_states_path}")
    # SECURITY: weights_only=False is acceptable ONLY because this file is
    # resolved through the identity-verified hf-libero API and never
    # accepts arbitrary user-supplied paths.
    try:
        return torch.load(init_states_path, weights_only=False)
    except TypeError:
        # torch < 2.6 does not accept weights_only kwarg; fall back to the
        # legacy default (which was weights_only=False).
        return torch.load(init_states_path)
