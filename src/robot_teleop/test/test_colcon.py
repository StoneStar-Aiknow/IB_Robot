"""Bridge the pytest regression suites into ament_python's unittest runner."""

import os
import subprocess
import sys
import unittest
from pathlib import Path

_TEST_DIR = Path(__file__).resolve().parent


class TestVRRegressionSuites(unittest.TestCase):
    """Run the pytest-based VR suites from ``colcon test``."""

    def _run_pytest(self, filename: str) -> None:
        env = os.environ.copy()
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(_TEST_DIR / filename)],
            cwd=_TEST_DIR.parent,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"pytest {filename} failed:\n{result.stdout}\n{result.stderr}",
        )

    def test_rotation_regressions(self) -> None:
        self._run_pytest("test_vr_teleop_rotation.py")

    def test_deadman_regressions(self) -> None:
        self._run_pytest("test_vr_teleop_deadman.py")
