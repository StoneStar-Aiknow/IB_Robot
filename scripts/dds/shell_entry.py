"""Load the private bridge from the same checkout without altering sys.path."""

import importlib
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("ibrobot_dds_shell", Path(__file__).with_name("__init__.py"))
assert spec is not None and spec.loader is not None
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
raise SystemExit(importlib.import_module("ibrobot_dds_shell.shell_bridge").main())
