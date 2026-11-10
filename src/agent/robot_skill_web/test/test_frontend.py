"""Run browser-script behavior tests without a browser or a ROS connection."""

import shutil
import subprocess
from pathlib import Path

import pytest

#: ``frontend.test.cjs`` covers the UI state machine with a stubbed transport;
#: ``frontend_recovery.test.cjs`` drives the same script against a real
#: loopback HTTP server to prove a page refresh resumes a saved reference.
NODE_TEST_FILES = ("frontend.test.cjs", "frontend_recovery.test.cjs")


@pytest.mark.parametrize("name", NODE_TEST_FILES)
def test_frontend_behaviors(name):
    node = shutil.which("node")
    assert node, "Install the declared nodejs test dependency"
    result = subprocess.run(
        [node, "--test", str(Path(__file__).with_name(name))],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
