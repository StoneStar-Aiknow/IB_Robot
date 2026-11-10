"""Compatibility exports for the neutral catalog digest contract.

The canonical implementation moved to :mod:`embodied_common.catalog_digest` so
the public client can verify snapshots without importing the skill compiler.
The preimages, field sets and digest helpers are the same objects, and the
presentation chain (``display_name``/``tags`` validated through
``embodied_common.capability_presentation``) lives there too, so the compiler
and the client cannot drift apart.
"""

from __future__ import annotations

from embodied_common.catalog_digest import *  # noqa: F401,F403
