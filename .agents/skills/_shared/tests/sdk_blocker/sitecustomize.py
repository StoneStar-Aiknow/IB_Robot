"""sitecustomize: block ``atomgit_sdk`` imports in collaboration-test subprocesses.

This file is the single blocker implementation for task 6.3 (OpenSpec change
``migrate-atomgit-skills-to-cli``). The collaboration conftest
(``.agents/skills/conftest.py``) loads it into the pytest process, and puts
this directory first on ``PYTHONPATH`` so every subprocess spawned by the
entrypoint fake-CLI tests imports it through ``site`` at interpreter startup
and refuses the retired SDK exactly like the parent does.

The finder *raises* instead of returning ``None``: returning None would let
the next meta-path finder import the package normally (the SDK is still
installed in the user's ``~/.local``), while the raised ImportError fails the
import deterministically with a recognizable marker. The marker is what the
self-checks assert on — an accidental ``ModuleNotFoundError`` from a
SDK-less environment would NOT contain it.
"""

import sys

BLOCKED_MESSAGE = (
    "atomgit_sdk is retired (OpenSpec migrate-atomgit-skills-to-cli): "
    "imports are blocked by the collaboration test import blocker"
)


class AtomgitSdkBlocker:
    """Meta-path finder refusing ``atomgit_sdk`` and every submodule."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "atomgit_sdk" or fullname.startswith("atomgit_sdk."):
            raise ImportError(f"{BLOCKED_MESSAGE}: {fullname}")
        return None


if not any(isinstance(finder, AtomgitSdkBlocker) for finder in sys.meta_path):
    sys.meta_path.insert(0, AtomgitSdkBlocker())
