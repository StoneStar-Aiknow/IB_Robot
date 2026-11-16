"""Public-entry regressions; fake accelerator dependencies, real process cleanup."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).parents[3]
SCRIPTS = ROOT / "scripts/rt_infer/periodic"
VENDOR = ROOT / "third_party/vendor/rt-infer/2026.11.05/scripts"
sys.path[:0] = [str(SCRIPTS), str(VENDOR)]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PeriodicTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        model = self.folder / "model.om"
        model.touch()
        mapping = self.folder / "models.json"
        mapping.write_text(
            json.dumps({name: str(model) for name in ("pear", "yolox", "te", "fm", "vision", "prefill", "decode")})
        )
        self.env = {
            "RTINFER_PREFIX": str(self.folder),
            "RTINFER_MODELS": str(mapping),
            "RTINFER_REFERENCES": str(self.folder),
            "RT_INFER_PERIODIC_PATH": str(self.folder / "control"),
            "RT_INFER_SAFE_ARGS_BATCH": "1",
            "RT_INFER_PERSISTENT_IO": "1",
        }
        self.patch = mock.patch.dict(os.environ, self.env)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.case = load("periodic_case", SCRIPTS / "latency_periodic10.py")
        self.runner = load("periodic_runner", SCRIPTS / "run.py")

    def test_worker_imports_dependencies_from_their_owners(self):
        (self.folder / "config.json").write_text("{}")
        fake_np = types.ModuleType("numpy")
        fake_np.load = mock.Mock(side_effect=RuntimeError("reached model input loading"))
        interface = types.ModuleType("ais_bench.infer.interface")
        interface.InferSession = object
        sys.modules.pop("baseline", None)
        with (
            mock.patch.dict(sys.modules, {"numpy": fake_np, "ais_bench.infer.interface": interface}),
            self.assertRaisesRegex(RuntimeError, "reached model input loading"),
        ):
            self.case.worker(argparse.Namespace(folder=self.folder, role="tts"))

    @unittest.skipUnless(os.name == "posix", "actual env command requires Linux")
    def test_worker_receives_explicit_config_after_environment_scrubbing(self):
        base = load("periodic_base", VENDOR / "base_control.py")
        probe = [sys.executable, "-c", "import os,json;print(json.dumps(dict(os.environ)))"]
        command = self.case.managed_command([], 0, 13, probe)
        command = command[command.index("/usr/bin/env") :]
        received = json.loads(subprocess.check_output(command, env=base.clean_environment(), text=True))
        for key, value in self.env.items():
            self.assertEqual(received[key], value)
        self.assertNotIn("RT_INFER_SKIP_MIXED", received)

    def test_idle_check_uses_nested_controller_state(self):
        for state in ("running", "starting", "start-failed", "recovery-required"):
            with self.assertRaises(RuntimeError):
                self.runner.require_idle({"state": {"status": state}, "device_users": []})
        self.runner.require_idle({"state": {"status": "stopped"}, "device_users": []})

    @unittest.skipUnless(os.name == "posix", "SIGTERM cleanup requires Linux")
    def test_timeout_allows_child_finally_to_reap_descendant(self):
        marker = self.folder / "cleaned"
        script = self.folder / "child.py"
        script.write_text(
            "import subprocess,signal,time,pathlib,sys\n"
            "def stop(*args): raise SystemExit(143)\n"
            "signal.signal(signal.SIGTERM,stop)\n"
            "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
            "try: time.sleep(60)\n"
            f"finally: child.terminate();child.wait();pathlib.Path({str(marker)!r}).write_text('reaped')\n"
        )
        with self.assertRaises(subprocess.TimeoutExpired):
            self.runner.run_workload([sys.executable, str(script)], timeout=1)
        self.assertEqual(marker.read_text(), "reaped")

    def test_failed_scheduler_start_reaps_owned_child(self):
        ctl = load("priority_test", VENDOR / "priorityctl.py")
        args = argparse.Namespace(
            scheduler="pfp",
            prefetch="on",
            cpu=15,
            pfp_periodic=None,
            pfp_exclusive="on",
            local_pool="on",
            pfp_batch_select="on",
            pfp_min_cache="on",
            pfp_hi_priority=0,
            trace=None,
            dispatch_audit=None,
            dispatch_timing=None,
        )
        child = mock.Mock()
        child.poll.return_value = None
        b = ctl.base
        with (
            mock.patch.multiple(
                b,
                state_read=mock.Mock(return_value={}),
                module_loaded=mock.Mock(return_value=False),
                ipc_segment=mock.Mock(return_value=None),
                preflight=mock.Mock(),
                device_users=mock.Mock(return_value=[]),
                boot_id=mock.Mock(return_value="boot"),
                state_write=mock.Mock(),
                command=mock.Mock(),
                artifact_path=mock.Mock(return_value=self.folder / "scheduler"),
                STATE=self.folder,
                IPC_PATH=self.folder / "ipc",
            ),
            mock.patch.object(ctl.os, "sched_getaffinity", return_value={15}, create=True),
            mock.patch.object(ctl.subprocess, "Popen", return_value=child),
            mock.patch.object(ctl.time, "monotonic", side_effect=[0, 16]),
            self.assertRaisesRegex(b.DeploymentError, "timed out"),
        ):
            ctl.start(self.folder, {"version": "test", "artifacts": {"module": {"sha256": "abc"}}}, args)
        child.terminate.assert_called_once()
        child.wait.assert_called_once_with(timeout=15)


if __name__ == "__main__":
    unittest.main()
