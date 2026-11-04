"""Offline deployment boundary tests; never load drivers or use an accelerator."""

import argparse
import contextlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SPEC = importlib.util.spec_from_file_location("rt_infer_control", Path(__file__).parents[3] / "third_party/vendor/rt-infer/2026.11.04/scripts/base_control.py")
ctl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ctl)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = {
            "schema_version": 1,
            "version": "test.1",
            "native_recovery": "reboot-required",
            "compatibility": {"architecture": "aarch64"},
            "artifacts": {},
        }
        elf = b"\x7fELF\x02\x01" + b"\x00" * 12 + b"\xb7\x00"
        for role in ctl.ROLES:
            path = self.root / role
            path.write_bytes(elf + role.encode())
            self.manifest["artifacts"][role] = {"path": role, "sha256": ctl.digest(path)}
        self.write_manifest()

    def write_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def test_valid_bundle_and_tamper(self):
        self.assertEqual(ctl.verify_bundle(self.root)["version"], "test.1")
        (self.root / "runtime").write_bytes(b"modified")
        with self.assertRaisesRegex(ctl.DeploymentError, "Checksum mismatch"):
            ctl.verify_bundle(self.root)

    def test_traversal_and_absolute_paths(self):
        for name in ("../outside", "/etc/passwd", "a/../../outside", "a\\file"):
            with self.subTest(name=name), self.assertRaises(ctl.DeploymentError):
                ctl.member(self.root, name)

    def test_reject_x86_binary_even_with_matching_hash(self):
        path = self.root / "runtime"
        path.write_bytes(b"\x7fELF\x02\x01" + b"\x00" * 12 + b"\x3e\x00")
        self.manifest["artifacts"]["runtime"]["sha256"] = ctl.digest(path)
        self.write_manifest()
        with self.assertRaisesRegex(ctl.DeploymentError, "aarch64"):
            ctl.verify_bundle(self.root)

    def test_cannot_claim_hot_recovery(self):
        self.manifest["native_recovery"] = "safe"
        self.write_manifest()
        with self.assertRaisesRegex(ctl.DeploymentError, "recovery"):
            ctl.verify_bundle(self.root)

    def test_operator_payload_tamper_is_rejected(self):
        path = self.root / "operator.json"
        path.write_text("original")
        self.manifest["payload_files"] = [{"path": path.name, "sha256": ctl.digest(path)}]
        self.write_manifest()
        ctl.verify_bundle(self.root)
        path.write_text("changed")
        with self.assertRaisesRegex(ctl.DeploymentError, "Checksum mismatch"):
            ctl.verify_bundle(self.root)

    def test_precompiled_graph_is_seeded_and_tampering_is_rejected(self):
        graph = self.root / "graphs" / "denoise" / "compiled_module"
        graph.parent.mkdir(parents=True)
        graph.write_bytes(b"pinned graph")
        self.manifest["graph_files"] = [{"path": "graphs/denoise/compiled_module", "sha256": ctl.digest(graph)}]
        cache = self.root / "cache"
        ctl.prepare_application_cache(self.root, self.manifest, cache, "pi05-765")
        self.assertEqual((cache / "denoise/compiled_module").read_bytes(), graph.read_bytes())
        ctl.prepare_application_cache(self.root, self.manifest, cache, "pi05-765")
        (cache / "denoise/compiled_module").write_bytes(b"different graph")
        with self.assertRaisesRegex(ctl.DeploymentError, "Precompiled graph changed"):
            ctl.prepare_application_cache(self.root, self.manifest, cache, "pi05-765")

    def test_binary_profile_requires_graph_assets(self):
        with self.assertRaisesRegex(ctl.DeploymentError, "requires its pinned precompiled graph"):
            ctl.prepare_application_cache(self.root, self.manifest, self.root / "cache", "pi05-765")

    def test_cache_cannot_cross_application_profiles(self):
        cache = self.root / "cache"
        ctl.prepare_application_cache(self.root, self.manifest, cache, "standard")
        with self.assertRaisesRegex(ctl.DeploymentError, "different runtime or application profile"):
            ctl.prepare_application_cache(self.root, self.manifest, cache, "pi05-765")

    def test_install_is_atomic_and_repeatable(self):
        destination = self.root / "install"
        args = argparse.Namespace(bundle=self.root, prefix=destination)
        with (
            mock.patch.object(ctl, "root_linux"),
            mock.patch.object(ctl, "preflight"),
            mock.patch.object(ctl, "locked", contextlib.nullcontext),
            mock.patch.object(ctl, "state_read", return_value={}),
            mock.patch.object(ctl, "trusted_tree"),
        ):
            ctl.install(args)
            ctl.install(args)
            self.assertEqual(ctl.verify_bundle(destination), self.manifest)
            self.assertFalse(list(self.root.glob(".rtinfer-*")))

    def test_incompatible_install_leaves_destination_absent(self):
        destination = self.root / "install"
        with (
            mock.patch.object(ctl, "root_linux"),
            mock.patch.object(ctl, "preflight", side_effect=ctl.DeploymentError("Incompatible kernel")),
            mock.patch.object(ctl, "locked", contextlib.nullcontext),
            mock.patch.object(ctl, "state_read", return_value={}),
            self.assertRaisesRegex(ctl.DeploymentError, "Incompatible kernel"),
        ):
            ctl.install(argparse.Namespace(bundle=self.root, prefix=destination))
        self.assertFalse(destination.exists())
        self.assertFalse(list(self.root.glob(".rtinfer-*")))

    def test_model_environment_mismatch_has_actionable_error(self):
        compatibility = {"python": f"{sys.version_info.major}.{sys.version_info.minor}", "torch": "2.5.1"}
        with (
            mock.patch.object(ctl.importlib.metadata, "version", return_value="2.6.0"),
            self.assertRaisesRegex(ctl.DeploymentError, "expected '2.5.1', found '2.6.0'"),
        ):
            ctl.model_environment({"compatibility": compatibility})

    def test_policy_rejects_missing_declared_assets(self):
        manifest = {
            "model": {"model_type": "pi05"},
            "deployments": {
                "torch-npu-fast": {
                    "extra_parameters": {
                        "denoising": {"solver": "ab2", "schedule": "uniform", "steps": 3},
                        "language_capacity": 48,
                        "vision_ffn": True,
                        "image_preprocessing": "layout_aware",
                        "action_output_device": "cpu",
                    }
                }
            },
            "bundle": {"files": [{"path": "tokenizer/tokenizer.json"}]},
        }
        for name in ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
            (self.root / name).write_text("fixture")
        (self.root / "inference_manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ctl.DeploymentError, "Missing model bundle asset: tokenizer/tokenizer.json"):
            ctl.check_pi05_policy(self.root)
        (self.root / "tokenizer").mkdir()
        (self.root / "tokenizer/tokenizer.json").write_text("fixture")
        self.assertEqual(ctl.check_pi05_policy(self.root)["declared_files_checked"], 5)

    def test_install_refuses_overwrite(self):
        destination = self.root / "install"
        destination.mkdir()
        (destination / "keep").write_text("user data")
        args = argparse.Namespace(bundle=self.root, prefix=destination)
        with (
            mock.patch.object(ctl, "root_linux"),
            mock.patch.object(ctl, "preflight"),
            mock.patch.object(ctl, "locked", contextlib.nullcontext),
            mock.patch.object(ctl, "state_read", return_value={}),
            self.assertRaises(ctl.DeploymentError),
        ):
            ctl.install(args)
        self.assertEqual((destination / "keep").read_text(), "user data")


class LifecycleTests(unittest.TestCase):
    def test_management_services_are_not_compute_clients(self):
        self.assertFalse(ctl.compute_device("/dev/davinci_manager"))
        self.assertFalse(ctl.compute_device("/dev/davinci_mdio"))
        self.assertTrue(ctl.compute_device("/dev/davinci0"))
        self.assertTrue(ctl.compute_device("/dev/aicore_device"))
        self.assertTrue(ctl.compute_device("/dev/devmm_svm"))

    @unittest.skipUnless(sys.platform == "linux" and hasattr(os, "pidfd_open"), "Linux pidfd test")
    def test_stop_signals_only_owned_process_and_preserves_recovery(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            identity = ctl.process_identity(child.pid)
            state = {"scheduler": identity, "shmid": "42", "status": "running"}
            with (
                mock.patch.object(ctl, "state_read", return_value=state),
                mock.patch.object(
                    ctl, "ipc_segment", return_value={"cpid": str(child.pid), "shmid": "42", "nattch": "1"}
                ),
                mock.patch.object(ctl, "device_users", return_value=[]),
                mock.patch.object(ctl, "state_write") as write,
            ):
                ctl.stop()
            self.assertEqual(child.wait(timeout=2), -15)
            self.assertEqual(write.call_args.args[0]["status"], "stopped")
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)

    def test_failed_driver_load_records_recovery_before_mutation(self):
        events = []
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(ctl, "IPC_PATH", Path(directory) / "ipc"),
            mock.patch.object(ctl, "state_read", return_value={}),
            mock.patch.object(ctl, "preflight"),
            mock.patch.object(ctl, "module_loaded", return_value=False),
            mock.patch.object(ctl, "ipc_segment", return_value=None),
            mock.patch.object(ctl, "device_users", return_value=[]),
            mock.patch.object(ctl.os, "sched_getaffinity", return_value={14}, create=True),
            mock.patch.object(ctl, "boot_id", return_value="test-boot"),
            mock.patch.object(ctl, "artifact_path", return_value=Path("/module")),
            mock.patch.object(ctl, "state_write", side_effect=lambda state: events.append(dict(state))),
            mock.patch.object(ctl, "command", side_effect=OSError("insmod failed")),
            self.assertRaisesRegex(OSError, "insmod failed"),
        ):
            ctl.start(
                Path("/bundle"),
                {"version": "test.1", "artifacts": {"module": {"sha256": "abc"}}},
                argparse.Namespace(experimental=True, cpu=14),
            )
        self.assertEqual([event["status"] for event in events], ["starting", "start-failed"])
        self.assertTrue(all(event["native_recovery"] == "reboot-required" for event in events))

    def test_pid_reuse_does_not_match(self):
        old = {"pid": 123, "start_ticks": "10", "executable": "/scheduler"}
        with mock.patch.object(ctl, "process_identity", return_value={**old, "start_ticks": "20"}):
            self.assertFalse(ctl.same_process(old))

    def test_client_exec_preserves_identity(self):
        client = {"pid": 123, "start_ticks": "10"}
        with mock.patch.object(ctl, "process_identity", return_value={**client, "executable": "/new-program"}):
            self.assertTrue(ctl.same_process(client))

    def test_restart_refused_after_failed_start(self):
        with (
            mock.patch.object(ctl, "state_read", return_value={"status": "start-failed"}),
            mock.patch.object(ctl, "command") as command,
            self.assertRaisesRegex(ctl.DeploymentError, "stopped cleanly"),
        ):
            ctl.start(Path("/bundle"), {}, argparse.Namespace(experimental=True))
        command.assert_not_called()

    def test_clean_restart_reuses_module_and_removes_only_detached_owned_ipc(self):
        row = {"key": "55", "shmid": "42", "cpid": "123", "ctime": "100", "uid": "0", "size": "4096", "nattch": "0"}
        current = {
            "status": "stopped",
            "prefix": str(Path("/bundle")),
            "version": "test.1",
            "module_owned": True,
            "module_sha256": "abc",
            "ipc_identity": ctl.shm_identity(row),
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(ctl, "IPC_PATH", Path(directory) / "ipc"),
            mock.patch.object(ctl, "STATE", Path(directory)),
            mock.patch.object(ctl, "state_read", return_value=current),
            mock.patch.object(ctl, "preflight"),
            mock.patch.object(ctl, "module_loaded", return_value=True),
            mock.patch.object(ctl, "ipc_segment", side_effect=[row, None]),
            mock.patch.object(ctl, "device_users", return_value=[]),
            mock.patch.object(ctl.os, "sched_getaffinity", return_value={14}, create=True),
            mock.patch.object(ctl, "boot_id", return_value="boot"),
            mock.patch.object(ctl, "artifact_path", return_value=Path("/bundle/scheduler")),
            mock.patch.object(ctl, "state_write"),
            mock.patch.object(ctl, "command") as command,
            mock.patch.object(ctl.subprocess, "Popen", side_effect=OSError("launch boundary")),
            self.assertRaisesRegex(OSError, "launch boundary"),
        ):
            ctl.start(
                Path("/bundle"),
                {"version": "test.1", "artifacts": {"module": {"sha256": "abc"}}},
                argparse.Namespace(experimental=True, cpu=14),
            )
        command.assert_called_once_with(["ipcrm", "-m", "42"])

    def test_ipc_cleanup_refuses_new_segment_and_attached_clients(self):
        original = {
            "key": "55",
            "shmid": "42",
            "cpid": "123",
            "ctime": "100",
            "uid": "0",
            "size": "4096",
            "nattch": "0",
        }
        for changed in ({**original, "ctime": "101"}, {**original, "nattch": "1"}):
            with (
                mock.patch.object(ctl, "ipc_segment", return_value=changed),
                mock.patch.object(ctl, "command") as command,
                self.assertRaises(ctl.DeploymentError),
            ):
                ctl.cleanup_stopped_ipc({"ipc_identity": ctl.shm_identity(original)})
            command.assert_not_called()

    def test_native_refused_after_clean_soft_stop(self):
        with (
            mock.patch.object(ctl, "state_read", return_value={"status": "stopped"}),
            self.assertRaisesRegex(ctl.DeploymentError, "reboot"),
        ):
            ctl.native_check()

    def test_native_refused_after_failed_start(self):
        with (
            mock.patch.object(ctl, "state_read", return_value={"status": "start-failed"}),
            self.assertRaisesRegex(ctl.DeploymentError, "reboot"),
        ):
            ctl.native_check()

    def test_stop_refuses_attached_unmanaged_client(self):
        state = {"scheduler": {"pid": 123}, "shmid": "42", "status": "running"}
        with (
            mock.patch.object(ctl, "state_read", return_value=state),
            mock.patch.object(ctl, "same_process", side_effect=lambda identity: bool(identity)),
            mock.patch.object(ctl, "ipc_segment", return_value={"cpid": "123", "shmid": "42", "nattch": "2"}),
            self.assertRaisesRegex(ctl.DeploymentError, "clients"),
        ):
            ctl.stop()

    def test_stop_refuses_foreign_segment(self):
        state = {"scheduler": {"pid": 123}, "shmid": "42", "status": "running"}
        with (
            mock.patch.object(ctl, "state_read", return_value=state),
            mock.patch.object(ctl, "same_process", side_effect=lambda identity: bool(identity)),
            mock.patch.object(ctl, "ipc_segment", return_value={"cpid": "999", "shmid": "42", "nattch": "1"}),
            self.assertRaisesRegex(ctl.DeploymentError, "creator"),
        ):
            ctl.stop()

    def test_stop_absent_process_preserves_reboot_requirement(self):
        state = {"scheduler": {"pid": 123}, "status": "running"}
        with (
            mock.patch.object(ctl, "state_read", return_value=state),
            mock.patch.object(ctl, "same_process", return_value=False),
            mock.patch.object(ctl, "state_write") as write,
        ):
            ctl.stop()
        self.assertEqual(write.call_args.args[0]["status"], "recovery-required")

    def test_environment_does_not_inherit_diagnostic_switches(self):
        environment = {
            "PATH": "/bin",
            "LD_PRELOAD": "old.so",
            "RT_INFER_SKIP_MIXED": "1",
            "AICORE_SCHED_SERIAL": "1",
            "IBROBOT_PI05_TORCHAIR_CACHE_HOME": "/native",
        }
        with mock.patch.dict(ctl.os.environ, environment, clear=True):
            self.assertEqual(ctl.clean_environment(), {"PATH": "/bin"})


if __name__ == "__main__":
    unittest.main()
