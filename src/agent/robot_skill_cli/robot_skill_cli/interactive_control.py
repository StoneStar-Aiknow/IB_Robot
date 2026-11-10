"""Compatibility alias for the shared controller, including private test hooks."""

import sys

from robot_client import workflow as _implementation

sys.modules[__name__] = _implementation
