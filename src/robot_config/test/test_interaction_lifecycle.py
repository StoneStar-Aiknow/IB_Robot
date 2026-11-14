"""Lifecycle cleanup owns only this checkout, including orphaned node processes."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def deployment(tmp_path):
    root = tmp_path / "checkout"
    scripts = root / "commissioning"
    scripts.mkdir(parents=True)
    for name in ("interaction_processes.py", "start_interaction.sh", "stop_interaction.sh"):
        shutil.copyfile(ROOT / "commissioning" / name, scripts / name)
    envfile = scripts / "env_orin.sh"
    envfile.write_text(f'export WORKSPACE="{root}"\n')
    return root


def node(root):
    script = root / "install/lib/mock/node.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("import time\ntime.sleep(60)\n")
    return subprocess.Popen([sys.executable, str(script)], start_new_session=True)


def stop(root, env=None):
    return subprocess.run(
        ["bash", str(root / "commissioning/stop_interaction.sh"), "--head"],
        env=env, capture_output=True, text=True, timeout=30,
    )


def test_stop_finds_orphan_without_pidfile_and_preserves_other_checkout(tmp_path):
    root = deployment(tmp_path)
    other = deployment(tmp_path / "other")
    owned, foreign = node(root), node(other)
    try:
        # Wait until Python has installed its normal handlers.
        result = stop(root)
        owned.wait(timeout=3)
        assert result.returncode == 0, result.stderr
        assert foreign.poll() is None
        assert "no checkout-owned nodes remain" in result.stdout
        assert stop(root).returncode == 0  # idempotent
    finally:
        for process in (owned, foreign):
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)


def test_stop_rejects_reused_pid_but_still_attempts_remote_head(tmp_path):
    root = deployment(tmp_path)
    foreign = subprocess.Popen(["sleep", "60"], start_new_session=True)
    state = root / "log/agent_interaction"
    state.mkdir(parents=True)
    (state / "pipeline.pgid").write_text(str(foreign.pid))
    binary = tmp_path / "bin"
    binary.mkdir()
    called = tmp_path / "remote_called"
    ssh = binary / "ssh"
    ssh.write_text(f'#!/bin/bash\ntouch "{called}"\nexit 0\n')
    ssh.chmod(0o755)
    try:
        result = subprocess.run(
            ["bash", str(root / "commissioning/stop_interaction.sh")],
            env=dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}"),
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0 and "ownership mismatch" in result.stderr
        assert foreign.poll() is None and called.exists()
        assert (state / "pipeline.pgid").exists()
    finally:
        foreign.terminate()
        foreign.wait(timeout=5)


def test_start_refuses_orphan_even_without_pidfile(tmp_path):
    root = deployment(tmp_path)
    owned = node(root)
    try:
        result = subprocess.run(
            ["bash", str(root / "commissioning/start_interaction.sh"), "--head"],
            capture_output=True, text=True, timeout=5,
        )
        assert result.returncode != 0
        assert "Checkout-owned nodes remain" in result.stderr
        assert owned.poll() is None
        assert not (root / "log/grasp_head/head.pgid").exists()
    finally:
        owned.terminate()
        owned.wait(timeout=5)
