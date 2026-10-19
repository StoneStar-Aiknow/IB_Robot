"""Make the ``robot_teleop`` source tree importable when tests run via bare
``pytest`` from the workspace root (dev/CI without a colcon-installed package).

Under ``colcon test`` the package is on the ament Python path and this is a
no-op. This used to register a bare package module WITHOUT executing the
``__init__`` (which eager-imported the ROS device factory and its heavy
deps); the ``__init__`` is PEP 562-lazy now (openspec change
``migrate-business-launch-builders``, design decision 6), so the real
package import is cheap and the shim is gone — registering the source
parent on ``sys.path`` is enough. Tests that stub their own infrastructure
(``test_vr_teleop_deadman``, ``test_leader_arm``) still install their own
package shims before import and are unaffected.
"""

import os
import sys

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if "robot_teleop" not in sys.modules:
    try:
        import robot_teleop  # noqa: F401  (already resolvable? keep it)
    except ModuleNotFoundError:
        sys.path.insert(0, _PKG_PARENT)
