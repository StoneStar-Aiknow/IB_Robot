"""Compatibility alias; old and new imports share the same transport module."""

import sys

from robot_client.transports import ros_bridge as _implementation

sys.modules[__name__] = _implementation
