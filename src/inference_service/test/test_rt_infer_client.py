"""Behavioral tests for the RT-Infer client wrapper.

Review #1: these tests execute REAL subprocesses with observable stub mains
(never string assertions on the constructed argv). The contract under test:

* the child resolves ``module:function`` with importlib and CALLS it —
  a pasted ``-c "import m; m:main()"`` is an annotation and never runs;
* ``sys.argv`` inside the node is the node's own arguments (ROS remappings
  pass through), prefixed with the entry name;
* the wrapper's exit code is the node's exit code, so the vendor client
  bookkeeping records the real outcome;
* the argv contract for the vendor CLI still carries the role priority/CPU
  and the PEAR-only periodic environment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from inference_service.rt_infer_client import RtInferClientError, _config_from_snapshot, build_client_argv, main

# A stub node whose main() records its invocation where the test can see it.
_STUB_MAIN = textwrap.dedent(
    """
    import json
    import sys
    from pathlib import Path


    def main():
        payload = {
            "argv": sys.argv[1:],
            "called": True,
            "node_name": sys.argv[0],
        }
        marker = Path(sys.argv[-1])
        marker.write_text(json.dumps(payload), encoding="utf-8")
    """
)

_STUB_EXITING_MAIN = textwrap.dedent(
    """
    import sys


    def main():
        sys.exit(int(sys.argv[-1]))
    """
)


def _snapshot(tmp_path: Path, *, periodic: bool = False) -> Path:
    payload = {
        "bundle_version": "2026.11.05-test",
        "bundle_dir": "",
        "launch_token": "test-launch",
        "role_priorities": {"pear": 0, "yolox": 3, "tts": 3, "vl": 3},
        "role_cpus": {"pear": 13, "yolox": 10, "tts": 12, "vl": 11},
        "pear_periodic": (
            {"period_hz": 18, "budget_ns": 38_000_000, "control_path": "/run/p.bin"} if periodic else None
        ),
    }
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_stub(tmp_path: Path, name: str, body: str) -> str:
    """A stub module on disk, importable by the child process."""
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir(exist_ok=True)
    (stub_dir / f"{name}.py").write_text(body, encoding="utf-8")
    return name


def _run_child(tmp_path: Path, node_main: str, node_args: list[str]) -> subprocess.CompletedProcess:
    """Run the wrapper's child entry directly (no vendor CLI on this host).

    This is the exact code path priorityctl spawns: this module executed with
    ``python -m``, resolving ``--rt-infer-node-main`` with importlib.
    """
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir(exist_ok=True)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "inference_service.rt_infer_client",
            "--rt-infer-node-main",
            node_main,
            "--",
            *node_args,
        ],
        capture_output=True,
        text=True,
        cwd=str(stub_dir),
        env={
            **os.environ,
            "PYTHONPATH": f"{stub_dir}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
            "RTINFER_STATE_DIR": str(tmp_path),
            "RTINFER_LAUNCH_TOKEN": "test-launch",
        },
        timeout=60,
    )


def test_child_actually_calls_the_node_main_and_passes_arguments(tmp_path):
    """The importlib path calls the entry with the node's own argv (review #1).

    The stub records its argv to a file. The old pasted-``-c`` implementation
    raised NameError ('main' annotation) and never invoked the node, so this
    behavior test fails against it.
    """
    stub = _write_stub(tmp_path, "observable_stub", _STUB_MAIN)
    marker = tmp_path / "called.json"
    result = _run_child(tmp_path, f"{stub}:main", ["--ros-args", "-r", "__node:=hri_pear", "--", str(marker)])
    assert result.returncode == 0, result.stderr
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["called"] is True
    assert payload["node_name"] == "main"
    assert payload["argv"] == ["--ros-args", "-r", "__node:=hri_pear", "--", str(marker)]
    records = list(tmp_path.glob("client-*.token"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["token"] == "test-launch"


def test_child_propagates_the_node_exit_code(tmp_path):
    """A node exiting non-zero exits the child non-zero (vendor bookkeeping)."""
    stub = _write_stub(tmp_path, "exiting_stub", _STUB_EXITING_MAIN)
    result = _run_child(tmp_path, f"{stub}:main", ["7"])
    assert result.returncode == 7


def test_child_preserves_real_ros_parameters_and_remapping(tmp_path):
    stub = _write_stub(
        tmp_path,
        "ros_stub",
        textwrap.dedent("""
        import json,sys
        from pathlib import Path
        import rclpy
        def main():
            rclpy.init(args=sys.argv)
            node = rclpy.create_node('original_name')
            try:
                value = node.declare_parameter('sample_count',0).value
                Path(sys.argv[-1]).write_text(json.dumps({'name':node.get_name(),'sample_count':value}))
            finally:
                node.destroy_node()
                rclpy.shutdown()
    """),
    )
    marker = tmp_path / "ros-result.json"
    result = _run_child(
        tmp_path,
        f"{stub}:main",
        ["--ros-args", "-r", "__node:=mapped_name", "-p", "sample_count:=7", "--", str(marker)],
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(marker.read_text()) == {"name": "mapped_name", "sample_count": 7}


def test_child_reports_unresolvable_module_cleanly(tmp_path):
    result = _run_child(tmp_path, "no_such_module_xyz:main", [])
    assert result.returncode != 0
    assert "ModuleNotFoundError" in result.stderr or "No module named" in result.stderr


def test_child_rejects_malformed_entry(tmp_path):
    result = _run_child(tmp_path, "json", [])
    assert result.returncode != 0
    assert "--rt-infer-node-main must be 'module:function'" in result.stderr


def test_argv_uses_role_priority_and_cpu(tmp_path):
    config = _config_from_snapshot(json.loads(_snapshot(tmp_path).read_text()))
    argv = build_client_argv(config, "yolox", ["node.py"])
    assert "run" in argv
    assert argv[argv.index("--priority") + 1] == "3"
    assert argv[argv.index("--cpu") + 1] == "10"
    assert argv[-1] == "node.py"


def test_argv_pins_manifest_toolkit_home(tmp_path, monkeypatch):
    config = _config_from_snapshot(json.loads(_snapshot(tmp_path).read_text()))
    monkeypatch.setattr(
        "robot_config.rt_infer_bundle.expected_toolkit_home",
        lambda _prefix: "/usr/local/Ascend/ascend-toolkit/8.1.RC1",
    )

    joined = " ".join(build_client_argv(config, "yolox", ["node.py"]))

    assert "ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/8.1.RC1" in joined


def test_control_tool_and_prefix_are_canonical(tmp_path):
    config = _config_from_snapshot(json.loads(_snapshot(tmp_path).read_text()))
    assert config.prefix == "/opt/rt-infer/2026.11.05-test"
    assert config.control_tool == "/opt/rt-infer/2026.11.05-test/scripts/priorityctl.py"


def test_pear_role_carries_periodic_environment(tmp_path):
    config = _config_from_snapshot(json.loads(_snapshot(tmp_path, periodic=True).read_text()))
    argv = build_client_argv(config, "pear", ["node"])
    joined = " ".join(argv)
    assert "RT_INFER_PERIODIC_PATH=/run/p.bin" in joined
    assert "RT_INFER_PERIODIC_SESSION=/run/periodic-session.json" in joined
    assert "RT_INFER_PERIODIC_PERIOD_HZ=18" in joined
    assert "RT_INFER_PERIODIC_BUDGET_NS=38000000" in joined
    assert "RT_INFER_SAFE_ARGS_BATCH=1" in joined


def test_background_roles_do_not_get_pear_periodic_env(tmp_path):
    config = _config_from_snapshot(json.loads(_snapshot(tmp_path, periodic=True).read_text()))
    joined = " ".join(build_client_argv(config, "tts", ["node"]))
    assert "RT_INFER_PERIODIC_PATH" not in joined
    assert "RT_INFER_PERIODIC_SESSION" not in joined


def test_list_args_prints_argv_without_executing(tmp_path, capsys, monkeypatch):
    def _refuse_execvp(*_args, **_kwargs):
        raise AssertionError("execvp must not run with --list-args")

    monkeypatch.setattr("os.execvp", _refuse_execvp)
    snapshot = _snapshot(tmp_path)
    code = main(
        [
            "--role",
            "vl",
            "--rt-infer-config",
            str(snapshot),
            "--node-main",
            "inference_service.model_service_node:main",
            "--list-args",
        ]
    )
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed[0] == sys.executable


def test_unknown_role_rejected(tmp_path):
    config = _config_from_snapshot(json.loads(_snapshot(tmp_path).read_text()))
    with pytest.raises(Exception, match="supported roles"):
        build_client_argv(config, "chassis", ["node"])


def test_corrupt_snapshot_raises_client_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RtInferClientError, match="snapshot"):
        main(
            [
                "--role",
                "vl",
                "--rt-infer-config",
                str(path),
                "--node-main",
                "m:f",
                "--list-args",
            ]
        )


def test_wrapper_argv_spawns_importable_child_not_pasted_source(tmp_path):
    """The vendor command must import the child module, never paste source.

    Guards against regressing to ``-c "import m; m:main()"`` — an annotation
    that raises NameError and never runs the node (review #1).
    """
    config = _config_from_snapshot(json.loads(_snapshot(tmp_path).read_text()))
    argv = build_client_argv(
        config,
        "pear",
        [sys.executable, "-m", "inference_service.rt_infer_client", "--rt-infer-node-main", "m:main"],
    )
    joined = " ".join(argv)
    assert "-m inference_service.rt_infer_client --rt-infer-node-main" in joined
    assert "-c" not in argv
    assert "m:main()" not in joined


def test_real_wrapper_waits_through_parent_and_late_child_sigint(tmp_path):
    import signal
    import time

    stub = _write_stub(
        tmp_path,
        "closing_stub",
        textwrap.dedent("""
        import atexit, json, os, signal, sys, time
        from pathlib import Path
        def main():
            assert signal.getsignal(signal.SIGINT) != signal.SIG_IGN
            root = Path(sys.argv[-1])
            (root/'model-ready').touch()
            def finalize():
                (root/'in-finalizer').touch()
                deadline=time.monotonic()+5
                while not (root/'release-finalizer').exists() and time.monotonic()<deadline:
                    time.sleep(0.01)
                (root/'finalizer-complete').touch()
            atexit.register(finalize)
            return 0
    """),
    )
    controller = tmp_path / "controller.py"
    controller.write_text(
        textwrap.dedent("""
        import json, os, signal, subprocess, sys
        from pathlib import Path
        assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
        child = subprocess.Popen(sys.argv[sys.argv.index('--')+1:])
        Path(os.environ['TEST_CONTROLLER_INFO']).write_text(json.dumps({'pid':os.getpid(),'child':child.pid}))
        raise SystemExit(child.wait())
    """)
    )
    runner = tmp_path / "run_wrapper.py"
    runner.write_text(
        textwrap.dedent("""
        import sys
        from robot_config.rt_infer_config import RtInferConfig
        from inference_service.rt_infer_client import main
        control = sys.argv[1]
        RtInferConfig.control_tool = property(lambda self: control)
        raise SystemExit(main(['--role','yolox','--rt-infer-config',sys.argv[2],
                              '--node-main',sys.argv[3],'--',sys.argv[4]]))
    """)
    )
    info = tmp_path / "controller.json"
    process = subprocess.Popen(
        [sys.executable, str(runner), str(controller), str(_snapshot(tmp_path)), f"{stub}:main", str(tmp_path)],
        env={
            **os.environ,
            "RTINFER_STATE_DIR": str(tmp_path),
            "TEST_CONTROLLER_INFO": str(info),
            "PYTHONPATH": f"{tmp_path / 'stubs'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "in-finalizer").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (tmp_path / "in-finalizer").exists()
        child_pid = json.loads(info.read_text())["child"]
        record = json.loads((tmp_path / f"client-{child_pid}.token").read_text())
        assert record["closing"] is True
        process.send_signal(signal.SIGINT)
        os.kill(child_pid, signal.SIGINT)
        (tmp_path / "release-finalizer").touch()
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, (stdout, stderr)
        assert (tmp_path / "finalizer-complete").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
