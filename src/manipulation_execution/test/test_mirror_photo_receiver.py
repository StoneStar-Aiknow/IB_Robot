"""The shipped receiver (``scripts/x2_photo_capture/receiver.py``) against the photo node's own client.

Every photo goes through ``uploader.post_photo`` or a whole ``Uploader`` pass,
so the two ends are checked against each other, not against a copy of the
protocol. The receiver runs on localhost in a thread; photos are a synthetic
JPEG with no one in it. ``--discard`` (the development mode) must leave no
photo bytes anywhere.
"""

import hashlib
import importlib.util
import json
import shlex
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

from manipulation_execution.mirror_photo.outbox import Outbox
from manipulation_execution.mirror_photo.selector import SCHEMA, Photo
from manipulation_execution.mirror_photo.uploader import ENDPOINT, META_HEADER, Refused, Uploader, post_photo

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "x2_photo_capture"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"x2_photo_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


receiver = _load("receiver")

JPEG = cv2.imencode(".jpg", np.random.default_rng(1).integers(0, 256, (48, 64, 3), dtype=np.uint8))[1].tobytes()
OTHER = cv2.imencode(".jpg", np.zeros((48, 64, 3), dtype=np.uint8))[1].tobytes()


def meta_for(round_id: str, jpeg: bytes = JPEG, **extra) -> dict:
    meta = {
        "schema": SCHEMA,
        "round_id": round_id,
        "quality": "normal",
        "source": "gesture",
        "label": "Victory",
        "bbox": [16.0, 8.0, 48.0, 40.0],
        "width": 64,
        "height": 48,
        "bytes": len(jpeg),
        "sha256": hashlib.sha256(jpeg).hexdigest(),
    }
    meta.update(extra)
    return meta


class Running:
    def __init__(self, inbox: Path, *, discard: bool = False, allow=(), max_mb: float = 5.0, **options):
        self.inbox = receiver.Inbox(inbox, discard=discard, **options)
        self.server = receiver.Receiver(("127.0.0.1", 0), self.inbox, allow=allow, max_mb=max_mb)
        self.target = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()

    def post(self, round_id: str, jpeg: bytes = JPEG, **extra) -> None:
        post_photo(self.target, jpeg, meta_for(round_id, jpeg, **extra), 3.0)

    def raw(self, method: str, path: str, body: bytes | None = None, headers=None) -> tuple[int, dict]:
        request = urllib.request.Request(self.target + path, data=body, headers=headers or {}, method=method)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=3.0) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def ledger(self) -> list[dict]:
        path = self.inbox.path / receiver.LEDGER
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def files(self) -> list[str]:
        """Everything in the inbox but the ledger (checked by ``ledger()``)."""
        return sorted(path.name for path in self.inbox.path.iterdir() if path.name != receiver.LEDGER)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def run(tmp_path):
    started = []

    def start(**options) -> Running:
        one = Running(tmp_path / "inbox", **options)
        started.append(one)
        return one

    yield start
    for one in started:
        one.close()


@pytest.fixture
def said(monkeypatch):
    lines = []
    monkeypatch.setattr(receiver, "log", lines.append)
    return lines


# -- receiving ---------------------------------------------------------------------------------


def test_a_photo_is_stored_with_its_metadata_and_confirmed(run, said):
    one = run()
    one.post("20261109-101500-abcd")
    assert one.files() == ["20261109-101500-abcd.jpg", "20261109-101500-abcd.json"]
    assert (one.inbox.path / "20261109-101500-abcd.jpg").read_bytes() == JPEG
    stored = json.loads((one.inbox.path / "20261109-101500-abcd.json").read_text(encoding="utf-8"))
    assert stored["bbox"] == [16.0, 8.0, 48.0, 40.0]
    assert stored["sha256"] == hashlib.sha256(JPEG).hexdigest()
    assert stored["peer"] == "127.0.0.1"
    (entry,) = one.ledger()
    assert (entry["round_id"], entry["duplicate"], entry["stored"]) == (
        "20261109-101500-abcd",
        False,
        "20261109-101500-abcd.jpg",
    )
    assert any("20261109-101500-abcd" in line and "Victory" in line for line in said)


def test_discard_keeps_no_photo_bytes_only_the_ledger(run, said):
    one = run(discard=True)
    one.post("r1")
    one.post("r2", OTHER)
    assert one.files() == []
    ledger = one.ledger()
    assert [entry["round_id"] for entry in ledger] == ["r1", "r2"]
    assert [entry["sha256"] for entry in ledger] == [
        hashlib.sha256(JPEG).hexdigest(),
        hashlib.sha256(OTHER).hexdigest(),
    ]
    assert all(entry["stored"] is None and entry["bytes"] > 0 for entry in ledger)
    assert JPEG[:64] not in (one.inbox.path / receiver.LEDGER).read_bytes()


def test_a_resend_is_confirmed_and_not_stored_twice(run, said):
    one = run()
    one.post("r1")
    one.post("r1")
    assert one.files() == ["r1.jpg", "r1.json"]
    assert [entry["duplicate"] for entry in one.ledger()] == [False, True]
    assert (one.inbox.received, one.inbox.duplicates) == (1, 1)


def test_resends_are_recognized_after_a_restart(run, tmp_path):
    first = run(discard=True)
    first.post("r1")
    first.close()
    second = run(discard=True)
    second.post("r1")
    assert [entry["duplicate"] for entry in second.ledger()] == [False, True]


def test_a_resend_racing_its_first_copy_is_still_one_photo(run):
    one = run()
    threads = [threading.Thread(target=one.post, args=("r1",)) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5.0)
    assert one.files() == ["r1.jpg", "r1.json"]
    assert sum(not entry["duplicate"] for entry in one.ledger()) == 1


def test_other_bytes_under_a_known_round_id_are_kept_beside_it(run):
    one = run()
    one.post("r1")
    one.post("r1", OTHER)
    suffix = hashlib.sha256(OTHER).hexdigest()[:8]
    assert one.files() == ["r1.jpg", "r1.json", f"r1_{suffix}.jpg", f"r1_{suffix}.json"]
    assert (one.inbox.path / f"r1_{suffix}.jpg").read_bytes() == OTHER


def test_the_metadata_lands_before_the_photo(run, monkeypatch):
    one = run()
    order = []
    write = receiver._write_atomic
    monkeypatch.setattr(receiver, "_write_atomic", lambda path, data: (order.append(path.suffix), write(path, data)))
    one.post("r1")
    assert order == [".json", ".jpg"]


# -- refusing (each one is retried by the board, never confirmed) --------------------------------


@pytest.mark.parametrize(
    ("jpeg", "extra", "why"),
    [
        (b"GIF89a" + JPEG, {}, "HTTP 415"),
        (JPEG, {"round_id": "../escape"}, "HTTP 400"),
        (JPEG, {"round_id": ".hidden"}, "HTTP 400"),
        (JPEG, {"round_id": ""}, "HTTP 400"),
        (JPEG, {"sha256": "0" * 64}, "HTTP 400"),
    ],
)
def test_a_bad_photo_is_refused_and_nothing_is_written(run, said, jpeg, extra, why):
    one = run()
    meta = meta_for("r1", jpeg)
    meta.update(extra)
    with pytest.raises(Refused, match=why):
        post_photo(one.target, jpeg, meta, 3.0)
    assert one.files() == []
    assert not (one.inbox.path / receiver.LEDGER).exists()


def test_metadata_that_is_not_json_is_refused(run):
    one = run()
    status, answer = one.raw("POST", ENDPOINT, JPEG, {META_HEADER: "{not json"})
    assert (status, answer["ok"]) == (400, False)


def test_a_photo_over_the_size_limit_is_refused(run):
    one = run(max_mb=0.001)
    with pytest.raises((Refused, OSError)):
        one.post("r1", JPEG + b"\0" * 2048)
    assert one.files() == []


def test_a_sender_not_allowed_gets_403_and_health_too(run, said):
    one = run(allow=["10.9.9.9"])
    with pytest.raises(Refused, match="HTTP 403"):
        one.post("r1")
    assert one.raw("GET", receiver.HEALTH)[0] == 403
    assert one.files() == []
    assert len([line for line in said if "not in --allow" in line]) == 1


def test_the_allowed_sender_gets_through(run):
    one = run(allow=["127.0.0.1"])
    one.post("r1")
    assert "r1.jpg" in one.files()


def test_only_two_paths_answer_and_photos_cannot_be_fetched(run):
    one = run()
    one.post("r1")
    assert one.raw("GET", "/r1.jpg")[0] == 404
    assert one.raw("GET", "/")[0] == 404
    assert one.raw("POST", "/upload", JPEG)[0] == 404


def test_health_says_what_it_is(run):
    one = run(discard=True)
    one.post("r1")
    status, answer = one.raw("GET", receiver.HEALTH)
    assert status == 200
    assert answer == {"ok": True, "service": receiver.SERVICE, "discard": True, "received": 1}


def test_a_disk_that_refuses_is_not_a_confirmation(run, monkeypatch):
    one = run()

    def full(path, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(receiver, "_write_atomic", full)
    with pytest.raises(Refused, match="HTTP 500"):
        one.post("r1")
    monkeypatch.undo()
    one.post("r1")  # the resend once there is room: stored, not taken for a duplicate
    assert "r1.jpg" in one.files()


# -- after the receipt -------------------------------------------------------------------------


def _wait_for(path: Path, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.02)
    return False


def test_crop_saves_the_person_box_beside_the_original(run):
    pytest.importorskip("PIL")
    from PIL import Image

    one = run(crop=True)
    one.post("r1")
    crop = one.inbox.path / "r1_crop.jpg"
    assert _wait_for(crop)
    with Image.open(crop) as picture:
        assert picture.size == (42, 42)  # [16, 8, 48, 40] widened by 15 % each side: (11, 3, 53, 45)
    assert (one.inbox.path / "r1.jpg").read_bytes() == JPEG


def test_crop_without_pillow_keeps_the_original_and_says_so_once(run, said, monkeypatch):
    monkeypatch.setitem(sys.modules, "PIL", None)
    one = run(crop=True)
    one.post("r1")
    one.post("r2", OTHER)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not any("pip install pillow" in line for line in said):
        time.sleep(0.02)
    time.sleep(0.1)
    assert len([line for line in said if "pip install pillow" in line]) == 1
    assert {"r1.jpg", "r2.jpg"} <= set(one.files())
    assert not any(name.endswith("_crop.jpg") for name in one.files())


@pytest.mark.parametrize(
    ("bbox", "box"),
    [
        ([16, 8, 48, 40], (11, 3, 53, 45)),
        ([0, 0, 64, 48], (0, 0, 64, 48)),
        ([60, 40, 70, 60], (58, 37, 64, 48)),
        ([10, 10, 10, 20], None),
        (None, None),
        ([1, 2, 3], None),
        ([float("nan"), 0, 10, 10], None),
    ],
)
def test_crop_box(bbox, box):
    assert receiver.crop_box(bbox, 64, 48) == box


def test_on_photo_runs_a_command_with_the_path(run, tmp_path):
    marker = tmp_path / "marker with space.txt"
    script = "import os, sys; open(sys.argv[2] + '.part', 'w').write(sys.argv[1]); os.replace(sys.argv[2] + '.part', sys.argv[2])"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)} {{path}} {shlex.quote(str(marker))}"
    one = run(on_photo=command)
    one.post("r1")
    assert _wait_for(marker)
    assert marker.read_text() == str(one.inbox.path / "r1.jpg")


def test_discard_runs_no_command(run, tmp_path):
    marker = tmp_path / "marker"
    one = run(discard=True, on_photo=f"touch {marker}")
    one.post("r1")
    time.sleep(0.3)
    assert not marker.exists()


# -- the node's whole upload path --------------------------------------------------------------


def test_the_node_s_uploader_empties_its_outbox_into_the_receiver(run, tmp_path):
    one = run(discard=True)
    lines = []
    outbox = Outbox(
        str(tmp_path / "photos"), max_files=200, max_mb=100.0, min_free_gb=0.0, log=lambda *line: lines.append(line)
    )
    outbox.prepare()
    names = [f"20261109-10150{index}-abcd" for index in range(3)]
    for name in names:
        jpeg = cv2.imencode(".jpg", np.full((48, 64, 3), 40 * names.index(name), dtype=np.uint8))[1].tobytes()
        outbox.put(Photo(name, jpeg, meta_for(name, jpeg)))
    target_file = tmp_path / "photo_target"
    target_file.write_text(one.target.removeprefix("http://") + "\n")
    uploader = Uploader(
        outbox, target_file=str(target_file), default_target="", timeout_sec=3.0, log=lambda *line: None
    )
    uploader.step()
    assert outbox.names() == []
    assert sorted(entry["round_id"] for entry in one.ledger()) == sorted(names)
    assert uploader.status()["delivered"] == 3
