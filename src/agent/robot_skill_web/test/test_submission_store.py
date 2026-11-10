from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from robot_client.errors import ClientError
from robot_skill_web.serialization import SubmissionStore


def test_reserve_is_atomic_once_marker(tmp_path: Path):
    store = SubmissionStore(tmp_path)
    first = store.reserve("preview", "session", {"request_id": "request", "target_key": "target"})
    second = store.reserve("preview", "other", {"request_id": "other", "target_key": "other"})
    assert first == second
    assert first["reference"]["request_id"] == "request"
    assert store.unresolved()
    store.resolve("preview", "SUCCEEDED")
    assert not store.unresolved()


def test_directory_permissions_are_validated_and_not_silently_repaired(tmp_path: Path):
    tmp_path.chmod(0o755)

    with pytest.raises(ClientError) as error:
        SubmissionStore(tmp_path).reserve("preview", "session", {"request_id": "request", "target_key": "target"})
    assert error.value.code == "UNSAFE_LOCAL_STATE"


def test_bad_record_fails_closed(tmp_path: Path):
    (tmp_path / "preview.json").write_text("{broken", encoding="utf-8")
    (tmp_path / "preview.json").chmod(0o600)

    with pytest.raises(ClientError) as error:
        SubmissionStore(tmp_path).get("preview")
    assert error.value.code == "INVALID_LOCAL_STATE"


def test_records_are_private_and_directory_is_fsynced(tmp_path: Path, monkeypatch):
    synced = []
    original_fsync = os.fsync

    def fsync(fd):
        synced.append(stat.S_ISDIR(os.fstat(fd).st_mode))
        original_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    store = SubmissionStore(tmp_path)
    store.reserve("preview", "session", {"request_id": "request", "target_key": "target"})

    record_path = tmp_path / "preview.json"
    assert stat.S_IMODE(record_path.stat().st_mode) == 0o600
    assert json.loads(record_path.read_text(encoding="utf-8"))["preview_id"] == "preview"
    assert synced == [False, True]


def test_preview_id_cannot_escape_directory(tmp_path: Path):
    store = SubmissionStore(tmp_path)

    with pytest.raises(Exception, match="preview_id is not a valid local record identifier"):
        store.reserve("../escape", "session", {"request_id": "request", "target_key": "target"})


def test_reconciled_unknown_is_not_unresolved_after_restart(tmp_path: Path):
    store = SubmissionStore(tmp_path)
    store.reserve("preview", "session", {"request_id": "request", "target_key": "target"})

    store.resolve("preview", "UNKNOWN", reconciled=True)

    assert not store.unresolved()
    reloaded = SubmissionStore(tmp_path)
    assert reloaded.get("preview")["reconciled"] is True
    assert not reloaded.unresolved()


def test_unreconciled_unknown_is_not_locked_out_after_load(tmp_path: Path):
    store = SubmissionStore(tmp_path)
    store.reserve("preview", "session", {"request_id": "request", "target_key": "target"})

    assert SubmissionStore(tmp_path).unresolved()
    store.resolve("preview", "UNKNOWN", reconciled=True)
    assert not store.unresolved()


def test_rejected_submission_is_terminal_but_not_unknown(tmp_path: Path):
    store = SubmissionStore(tmp_path)
    store.reserve("preview", "session", {"request_id": "request", "target_key": "target"})

    store.resolve("preview", "REJECTED")

    assert store.get("preview")["state"] == "REJECTED"
    assert not store.unresolved()
