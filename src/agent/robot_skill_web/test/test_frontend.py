"""Run browser-script behavior tests without a browser or a ROS connection."""

import shutil
import subprocess
from pathlib import Path


def test_frontend_behaviors():
    node = shutil.which("node")
    assert node, "Install the declared nodejs test dependency"
    result = subprocess.run(
        [node, "--test", str(Path(__file__).with_name("frontend.test.cjs"))],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
