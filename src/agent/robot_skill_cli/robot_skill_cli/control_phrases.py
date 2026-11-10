"""Workspace re-export of the single STOP vocabulary.

Installed Hermes plugins do not have the IB-Robot packages on their path.
``hermes_configure`` therefore writes the same ``embodied_common/control_phrases.py``
next to ``stop_contract.py`` in the plugin directory. In the source workspace
this module provides that sibling, so ``stop_contract`` always imports one
module and never falls back to a second source after an ``ImportError``.
"""

from embodied_common.control_phrases import CONTINUE_PHRASES as CONTINUE_PHRASES
from embodied_common.control_phrases import STOP_PHRASES as STOP_PHRASES
