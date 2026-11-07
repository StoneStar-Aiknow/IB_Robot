"""Exercise the public controller's mutable-state boundary without an NPU."""

import importlib.util
import json
import multiprocessing
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


@pytest.fixture
def controller(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[3] / "third_party/vendor/rt-infer/2026.11.06/scripts/base_control.py"
    spec = importlib.util.spec_from_file_location("state_security_controller", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.STATE = tmp_path / "state"
    module.STATE.mkdir(mode=0o700)
    monkeypatch.setattr(module, "root_linux", lambda: None)
    original = module._trusted_state_info

    def fixture_owner(info, path, *, directory=False):
        # Nonroot CI owns its fixture; only /tmp's ancestor mode is adapted.
        # The actual checks still validate links, types and fixture permissions.
        adapted = SimpleNamespace(
            st_uid=0 if info.st_uid == os.geteuid() else info.st_uid,
            st_mode=(info.st_mode & ~0o022) if directory and path == "tmp" else info.st_mode,
            st_nlink=info.st_nlink,
        )
        original(adapted, path, directory=directory)

    monkeypatch.setattr(module, "_trusted_state_info", fixture_owner)
    return module


def _rename_when_requested(source, destination, ready, done):
    assert ready.wait(5)
    Path(source).replace(destination)
    done.set()


@pytest.mark.parametrize("temporary", ["state.tmp", "periodic-session.tmp", "client-101.tmp"])
def test_rt_infer_state_scan_accepts_real_process_atomic_publish(controller, monkeypatch, temporary):
    source = controller.STATE / temporary
    source.write_text("{}")
    destination = source.with_suffix(".token" if temporary.startswith("client-") else ".json")
    context = multiprocessing.get_context("fork")
    ready, done = context.Event(), context.Event()
    child = context.Process(target=_rename_when_requested, args=(source, destination, ready, done))
    child.start()
    listdir = os.listdir

    def enumerated_then_renamed(fd):
        names = listdir(fd)
        ready.set()
        assert done.wait(5)
        return names

    monkeypatch.setattr(controller.os, "listdir", enumerated_then_renamed)
    try:
        with controller.state_locked(blocking=False):
            assert destination.read_text() == "{}"
    finally:
        child.join(5)
        if child.is_alive():
            child.kill()
            child.join()
    assert child.exitcode == 0


def test_rt_infer_missing_state_json_is_not_a_transient_exception(controller, monkeypatch):
    path = controller.STATE / "state.json"
    path.write_text("{}")
    listdir = os.listdir

    def disappear(fd):
        names = listdir(fd)
        path.rename(controller.STATE.parent / "saved-state.json")
        return names

    monkeypatch.setattr(controller.os, "listdir", disappear)
    with pytest.raises(FileNotFoundError), controller.state_locked(blocking=False):
        pytest.fail("missing stable state accepted")


@pytest.mark.parametrize("name", ["unrecognized.tmp", "client-0.tmp", "state.tmp.extra", "unknown"])
def test_rt_infer_rejects_unknown_state_entries(controller, name):
    (controller.STATE / name).write_text("{}")
    with (
        pytest.raises(controller.DeploymentError, match="Unknown state entry"),
        controller.state_locked(blocking=False),
    ):
        pytest.fail("unknown entry accepted")


@pytest.mark.parametrize("name", ["state.json", "state.tmp", "lock", "client-12.token"])
def test_rt_infer_rejects_state_and_lock_symlinks(controller, name):
    target = controller.STATE.parent / "target"
    target.write_text("{}")
    (controller.STATE / name).symlink_to(target)
    with pytest.raises((controller.DeploymentError, OSError)), controller.state_locked(blocking=False):
        pytest.fail("symlink accepted")


@pytest.mark.parametrize("name", ["state.json", "state.tmp", "lock"])
def test_rt_infer_rejects_world_writable_state(controller, name):
    path = controller.STATE / name
    path.write_text("{}")
    path.chmod(0o666)
    with pytest.raises(controller.DeploymentError, match="ownership/mode"), controller.state_locked(blocking=False):
        pytest.fail("writable entry accepted")


def test_rt_infer_rejects_foreign_file_owner(controller):
    info = SimpleNamespace(st_uid=999999, st_mode=stat.S_IFREG | 0o600, st_nlink=1)
    with pytest.raises(controller.DeploymentError, match="ownership/mode"):
        controller._trusted_state_info(info, "state.json")


@pytest.mark.parametrize("kind", ["directory", "fifo", "hardlink"])
def test_rt_infer_rejects_nonregular_and_multilink_state(controller, kind):
    path = controller.STATE / "state.json"
    if kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        target = controller.STATE.parent / "original"
        target.write_text("{}")
        os.link(target, path)
    with pytest.raises(controller.DeploymentError), controller.state_locked(blocking=False):
        pytest.fail("nonregular or linked entry accepted")


@pytest.mark.parametrize("location", ["root", "ancestor"])
def test_rt_infer_rejects_state_directory_symlinks(controller, location):
    real = controller.STATE
    link = real.parent / "redirect"
    link.symlink_to(real if location == "root" else real.parent, target_is_directory=True)
    controller.STATE = link if location == "root" else link / real.name
    with pytest.raises(OSError), controller.state_locked(blocking=False):
        pytest.fail("directory symlink accepted")


def test_rt_infer_rejects_world_writable_state_directory(controller):
    controller.STATE.chmod(0o777)
    with pytest.raises(controller.DeploymentError, match="ownership/mode"), controller.state_locked(blocking=False):
        pytest.fail("writable state directory accepted")


def test_rt_infer_detects_lock_inode_replacement(controller):
    with (
        pytest.raises(controller.DeploymentError, match="lock inode changed"),
        controller.state_locked(blocking=False),
    ):
        (controller.STATE / "lock").rename(controller.STATE / "old-lock")
        (controller.STATE / "lock").write_text("replacement")
        (controller.STATE / "lock").chmod(0o600)


def test_rt_infer_detects_state_directory_replacement(controller):
    with (
        pytest.raises(controller.DeploymentError, match="directory identity changed"),
        controller.state_locked(blocking=False),
    ):
        controller.STATE.rename(controller.STATE.with_name("old-state"))
        controller.STATE.mkdir(mode=0o700)


def test_rt_infer_second_process_cannot_acquire_live_state_lock(controller):
    context = multiprocessing.get_context("fork")
    result = context.Queue()

    def compete():
        try:
            with controller.state_locked(blocking=False):
                result.put("acquired")
        except BlockingIOError:
            result.put("refused")

    with controller.state_locked(blocking=False):
        child = context.Process(target=compete)
        child.start()
        child.join(5)
        assert child.exitcode == 0
        assert result.get(timeout=1) == "refused"


def test_rt_infer_default_control_revision_verifies_reused_binaries(controller):
    root = Path(__file__).resolve().parents[3]
    profile = yaml.safe_load((root / "src/robot_config/config/robots/aimdk_x2_magicmirror_board.yaml").read_text())
    version = profile["robot"]["rt_infer"]["bundle_version"]
    assert version == "2026.11.06"
    bundle = root / "third_party/vendor/rt-infer" / version
    manifest = controller.verify_bundle(bundle)
    old = json.loads((bundle.with_name("2026.11.05") / "manifest.json").read_text())
    assert manifest["artifacts"] == old["artifacts"]
    old_payload = {entry["path"]: entry for entry in old["payload_files"]}
    for entry in manifest["payload_files"]:
        if entry["path"].endswith((".so", ".ko")) or entry["path"].startswith("bin/"):
            assert entry == old_payload[entry["path"]]
