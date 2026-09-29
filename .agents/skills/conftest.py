"""Collaboration-test import blocker for the retired ``atomgit_sdk`` (task 6.3).

Every test under ``.agents/skills`` runs with ``atomgit_sdk`` and all its
submodules unimportable: a meta-path finder (from
``_shared/tests/sdk_blocker/sitecustomize.py``, the single blocker
implementation) refuses those imports in this pytest process, and the blocker
directory is prepended to ``PYTHONPATH`` so the ``site`` module loads the same
``sitecustomize`` in every subprocess the entrypoint fake-CLI tests spawn.

The block is genuine, not observational: a ``sys.modules`` snapshot cannot
detect optional-import fallbacks or preloaded modules, but a raising
meta-path finder makes every import attempt of the retired SDK fail
deterministically. A self-check at conftest import time proves the finder is
actually intercepting (``importlib.util.find_spec`` only raises ``ImportError``
with the blocker marker when our finder fired); collection fails loudly if
the blocker is not in effect. The SDK being still installed (user
``~/.local``) is irrelevant — nothing is uninstalled and no shared venv is
modified.
"""

import importlib.util
import os
import sys
from pathlib import Path

BLOCKER_DIR = Path(__file__).resolve().parent / "_shared" / "tests" / "sdk_blocker"


def _load_blocker_module():
    spec = importlib.util.spec_from_file_location("_collab_sdk_blocker_sitecustomize", BLOCKER_DIR / "sitecustomize.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # installs the meta-path finder here
    return module


def _propagate_to_subprocesses():
    """Put the blocker directory first on PYTHONPATH for spawned children."""
    blocker = str(BLOCKER_DIR)
    existing = os.environ.get("PYTHONPATH")
    if existing is None:
        os.environ["PYTHONPATH"] = blocker
    elif blocker not in existing.split(os.pathsep):
        os.environ["PYTHONPATH"] = blocker + os.pathsep + existing


def _self_check(blocker_module):
    """Prove the finder is intercepting; fail collection loudly otherwise.

    ``find_spec`` raises only through a meta-path finder, and only OUR finder
    raises ``ImportError`` carrying the blocker marker — a SDK-less
    environment would surface ``ModuleNotFoundError`` without it, so this
    check is deterministic whether or not the SDK is installed.
    """
    try:
        importlib.util.find_spec("atomgit_sdk")
    except ImportError as error:
        if blocker_module.BLOCKED_MESSAGE not in str(error):
            raise RuntimeError(
                f"collaboration sdk blocker self-check failed: unexpected import error: {error}"
            ) from error
        return
    raise RuntimeError(
        "collaboration sdk blocker self-check failed: importlib.util.find_spec('atomgit_sdk') "
        "did not raise — atomgit_sdk imports are NOT blocked in this process"
    )


blocker_module = _load_blocker_module()
_propagate_to_subprocesses()
_self_check(blocker_module)
