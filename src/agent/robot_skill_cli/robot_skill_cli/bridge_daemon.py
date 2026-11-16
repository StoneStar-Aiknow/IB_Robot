"""Compatibility import for the neutral resident transport implementation."""

import sys

from robot_client.transports import resident as _implementation

if __name__ == "__main__":
    from robot_skill_cli.daemon_cli import main

    raise SystemExit(main())
else:
    sys.modules[__name__] = _implementation
