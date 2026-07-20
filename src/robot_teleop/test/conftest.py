"""Make the ``robot_teleop`` source tree importable when tests run via bare
``pytest`` from the workspace root (dev/CI without a colcon-installed package).

Under ``colcon test`` the package is on the ament Python path and this is a
no-op. Here we register the package directory as ``robot_teleop`` WITHOUT
executing its ``__init__`` (which eager-imports the ROS device factory and its
heavy deps). Tests that need only leaf modules (``vr_rotation``) then import
cleanly; tests that stub their own infrastructure (``test_vr_teleop_deadman``)
are unaffected because they install a compatible package shim before import.
"""

import os
import sys
import types

_PKG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "robot_teleop")

if "robot_teleop" not in sys.modules:
    pkg = types.ModuleType("robot_teleop")
    pkg.__path__ = [_PKG_DIR]
    sys.modules["robot_teleop"] = pkg
