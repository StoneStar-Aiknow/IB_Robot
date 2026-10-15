"""Pytest path bootstrap for the benchmark_robotwin package tests.

Keeps the tests runnable both from the repository root and from anywhere
inside the workspace, with or without the colcon install overlay sourced.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))
