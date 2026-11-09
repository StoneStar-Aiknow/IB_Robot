"""No photo is lost on the way: the outbox on disk and the upload over real HTTP on localhost.

The receiver here is a minimal stand-in written for the test (the shipped one,
``scripts/x2_photo_capture/receiver.py``, has its own tests): it confirms a
photo by repeating its sha256, or misbehaves on purpose. Photos are synthetic
bytes in a JPEG wrapper; nothing here is a picture of anyone.
"""

import hashlib
import http.server
import json
import threading
import time
from pathlib import Path

import pytest

from manipulation_execution.mirror_photo import uploader as uploader_module
from manipulation_execution.mirror_photo.outbox import Outbox
from manipulation_execution.mirror_photo.selector import SCHEMA, Photo
from manipulation_execution.mirror_photo.uploader import (
    BACKOFF_SEC,
    ENDPOINT,
    META_HEADER,
    POLL_SEC,
    Uploader,
    read_target,
)


def photo(name: str, size: int = 1000) -> Photo:
    jpeg = b"\xff\xd8" + hashlib.sha256(name.encode()).digest() * (size // 32) + b"\xff\xd9"
    meta = {"schema": SCHEMA, "round_id": name, "quality": "normal", "sha256": hashlib.sha256(jpeg).hexdigest()}
    return Photo(name, jpeg, meta)


def names(count: int, start: int = 0) -> list[str]:
    return [f"20261109-1200{index:02d}-abcd" for index in range(start, start + count)]


class Log:
    def __init__(self):
        self.lines: list[tuple[str, str]] = []

    def __call__(self, level: str, text: str) -> None:
        self.lines.append((level, text))

    def at(self, level: str) -> list[str]:
        return [text for kind, text in self.lines if kind == level]


class Receiver:
    """``mode``: ok | badsha | error | notjson. ``received``: (meta, body) of every request, confirmed or not."""

    def __init__(self, port: int = 0):
        self.mode = "ok"
        self.received: list[tuple[dict, bytes]] = []
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - the http.server name
                body = self.rfile.read(int(self.headers["Content-Length"]))
                owner.received.append((json.loads(self.headers[META_HEADER]), body))
                assert self.path == ENDPOINT
                if owner.mode == "error":
                    self.send_error(500)
                    return
                digest = hashlib.sha256(body).hexdigest() if owner.mode == "ok" else "0" * 64
                answer = b"not json" if owner.mode == "notjson" else json.dumps({"ok": True, "sha256": digest}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(answer)))
                self.end_headers()
                self.wfile.write(answer)

            def log_message(self, *_args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()

    @property
    def address(self) -> str:
        return f"127.0.0.1:{self.port}"

    @property
    def delivered(self) -> list[str]:
        return [meta["round_id"] for meta, _ in self.received] if self.mode == "ok" else []

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def receiver():
    one = Receiver()
    yield one
    one.close()


def _dead_port() -> int:
    """A localhost port nobody listens on (it was bound a moment ago)."""
    one = Receiver()
    one.close()
    return one.port


class Rig:
    def __init__(self, root: Path, target: str = "", *, default: str = "", free=None, **limits):
        self.log = Log()
        self.root = root
        self.target_file = root / "photo_target"
        if target:
            self.target_file.write_text(target + "\n")
        self.outbox = self._outbox(free, **limits)
        self.uploader = Uploader(
            self.outbox,
            target_file=str(self.target_file),
            default_target=default,
            timeout_sec=2.0,
            log=self.log,
        )

    def _outbox(self, free=None, max_files: int = 200, max_mb: float = 100.0) -> Outbox:
        outbox = Outbox(
            str(self.root / "photos"),
            max_files=max_files,
            max_mb=max_mb,
            min_free_gb=2.0,
            log=self.log,
            free_bytes=free,
        )
        outbox.prepare()
        return outbox

    def files(self) -> list[str]:
        return sorted(path.name for path in (self.root / "photos" / "outbox").iterdir())


# -- the outbox --------------------------------------------------------------------------


def test_a_photo_is_two_files_and_reads_back(tmp_path):
    rig = Rig(tmp_path)
    one = photo(names(1)[0])
    assert rig.outbox.put(one) == "disk"
    assert rig.files() == [f"{one.name}.jpg", f"{one.name}.json"]
    jpeg, meta = rig.outbox.load(one.name)
    assert jpeg == one.jpeg
    assert meta["round_id"] == one.name
    assert meta["sha256"] == hashlib.sha256(jpeg).hexdigest()
    rig.outbox.remove(one.name)
    assert rig.files() == []


def test_the_newest_photo_is_first(tmp_path):
    rig = Rig(tmp_path)
    for name in names(3):
        rig.outbox.put(photo(name))
    assert rig.outbox.names() == names(3)[::-1]


def test_past_the_file_limit_the_oldest_go_with_an_error_each(tmp_path):
    rig = Rig(tmp_path, max_files=3)
    for name in names(5):
        rig.outbox.put(photo(name))
    assert rig.outbox.names() == names(5)[2:][::-1]
    errors = rig.log.at("error")
    assert len(errors) == 2
    assert names(5)[0] in errors[0] and "never delivered" in errors[0]
    assert rig.outbox.stats()["dropped"] == 2


def test_past_the_size_limit_the_oldest_go(tmp_path):
    rig = Rig(tmp_path, max_mb=0.25)  # about 2 photos of 100 kB
    for name in names(4):
        rig.outbox.put(photo(name, size=100_000))
    assert rig.outbox.names() == names(4)[2:][::-1]
    assert rig.outbox.stats()["bytes"] <= 0.25 * 1024 * 1024


def test_a_restart_with_a_smaller_limit_trims_at_once(tmp_path):
    rig = Rig(tmp_path)
    for name in names(5):
        rig.outbox.put(photo(name))
    smaller = rig._outbox(max_files=2)
    assert smaller.names() == names(5)[3:][::-1]


def test_a_write_cut_short_is_tidied_and_kept(tmp_path):
    rig = Rig(tmp_path)
    one, two = (photo(name) for name in names(2))
    rig.outbox.put(one)
    folder = tmp_path / "photos" / "outbox"
    (folder / f"{two.name}.jpg").write_bytes(two.jpeg)  # the JSON never came
    (folder / f"{two.name}.json.tmp").write_bytes(b"{")
    (folder / "20261109-120100-abcd.json").write_text("{}")  # metadata without its JPEG
    restarted = rig._outbox()
    assert restarted.names() == [two.name, one.name]
    assert not list(folder.glob("*.tmp"))
    assert not (folder / "20261109-120100-abcd.json").exists()
    jpeg, meta = restarted.load(two.name)
    assert jpeg == two.jpeg
    assert meta["quality"] == "recovered"
    assert any(two.name in line for line in rig.log.at("warning"))


def test_a_round_the_node_never_finished_is_sent_from_its_provisional_copy(tmp_path):
    rig = Rig(tmp_path)
    one = photo(names(1)[0])
    rig.outbox.write_current(Photo(one.name, one.jpeg, {**one.meta, "quality": "provisional"}))
    restarted = rig._outbox()
    assert restarted.names() == [one.name]
    jpeg, meta = restarted.load(one.name)
    assert jpeg == one.jpeg
    assert (meta["quality"], meta["end_reason"]) == ("recovered", "restart")
    assert "current.jpg" not in rig.files()


def test_a_provisional_copy_of_a_round_already_in_is_dropped(tmp_path):
    rig = Rig(tmp_path)
    one = photo(names(1)[0])
    rig.outbox.write_current(Photo(one.name, b"\xff\xd8older\xff\xd9", one.meta))
    rig.outbox.put(one)  # the node died between this and clear_current
    restarted = rig._outbox()
    assert restarted.names() == [one.name]
    assert restarted.load(one.name)[0] == one.jpeg
    assert rig.files() == [f"{one.name}.jpg", f"{one.name}.json"]


def test_a_provisional_copy_without_readable_metadata_is_still_sent(tmp_path):
    rig = Rig(tmp_path)
    folder = tmp_path / "photos" / "outbox"
    (folder / "current.jpg").write_bytes(photo("x").jpeg)
    (folder / "current.json").write_text("{broken")
    restarted = rig._outbox()
    assert len(restarted.names()) == 1
    assert restarted.names()[0].endswith("-rec")


def test_a_normal_round_end_clears_the_provisional_copy(tmp_path):
    rig = Rig(tmp_path)
    one = photo(names(1)[0])
    rig.outbox.write_current(one)
    rig.outbox.put(one)
    rig.outbox.clear_current()
    assert rig.files() == [f"{one.name}.jpg", f"{one.name}.json"]
    assert rig.outbox.names() == [one.name]


def test_with_the_disk_nearly_full_a_photo_is_kept_in_memory_and_still_sent(tmp_path, receiver):
    rig = Rig(tmp_path, receiver.address, free=lambda _path: 10 * 1024**2)
    one = photo(names(1)[0])
    rig.outbox.write_current(one)
    assert rig.outbox.put(one) == "memory"
    assert rig.files() == []
    assert "kept in memory only" in rig.log.at("error")[0]
    assert rig.outbox.stats()["in_memory"] == 1
    rig.uploader.step()
    assert receiver.delivered == [one.name]
    assert rig.outbox.names() == []


# -- the upload --------------------------------------------------------------------------


def test_a_photo_is_delivered_byte_for_byte_and_then_removed(tmp_path, receiver):
    rig = Rig(tmp_path, receiver.address)
    one = photo(names(1)[0])
    rig.outbox.put(one)
    assert rig.uploader.step() == POLL_SEC
    meta, body = receiver.received[0]
    assert body == one.jpeg
    assert meta["round_id"] == one.name
    assert rig.files() == []
    status = rig.uploader.status()
    assert (status["delivered"], status["last_delivered"], status["failures"]) == (1, one.name, 0)
    assert status["target"] == f"http://{receiver.address}"


def test_the_newest_is_sent_first(tmp_path, receiver):
    rig = Rig(tmp_path, receiver.address)
    for name in names(3):
        rig.outbox.put(photo(name))
    rig.uploader.step()
    assert receiver.delivered == names(3)[::-1]


def test_with_the_receiver_down_the_photos_wait_and_go_once_it_is_up(tmp_path):
    port = _dead_port()
    rig = Rig(tmp_path, f"127.0.0.1:{port}")
    for name in names(2):
        rig.outbox.put(photo(name))
    delays = [rig.uploader.step() for _ in range(7)]
    assert delays == [*BACKOFF_SEC, BACKOFF_SEC[-1], BACKOFF_SEC[-1]]
    assert rig.outbox.names() == names(2)[::-1]
    assert len(rig.log.at("warning")) == 1  # once per kind of failure, not per attempt
    assert "did not answer" in rig.uploader.status()["last_error"]
    up = Receiver(port)
    try:
        rig.uploader.step()
        assert up.delivered == names(2)[::-1]
    finally:
        up.close()
    assert rig.outbox.names() == []
    assert rig.uploader.status()["failures"] == 0
    assert any("back after" in line for line in rig.log.at("info"))


@pytest.mark.parametrize("mode", ["badsha", "error", "notjson"])
def test_an_answer_that_does_not_confirm_the_photo_keeps_it(tmp_path, receiver, mode):
    rig = Rig(tmp_path, receiver.address)
    one = photo(names(1)[0])
    rig.outbox.put(one)
    receiver.mode = mode
    assert rig.uploader.step() == BACKOFF_SEC[0]
    assert rig.outbox.names() == [one.name]
    assert "refused" in rig.uploader.status()["last_error"]
    receiver.mode = "ok"
    rig.uploader.step()
    assert receiver.delivered[-1] == one.name
    assert rig.outbox.names() == []


def test_one_refused_photo_does_not_hold_up_the_others(tmp_path, receiver):
    rig = Rig(tmp_path, receiver.address)
    for name in names(3):
        rig.outbox.put(photo(name))
    sent = []

    def picky(target, jpeg, meta, timeout):
        if meta["round_id"] == names(3)[1]:
            raise uploader_module.Refused("no")
        sent.append(meta["round_id"])

    rig.uploader._post = picky
    rig.uploader.step()
    assert sent == [names(3)[2], names(3)[0]]
    assert rig.outbox.names() == [names(3)[1]]


def test_a_node_killed_with_photos_waiting_sends_them_after_the_restart(tmp_path, receiver):
    first = Rig(tmp_path, f"127.0.0.1:{_dead_port()}")
    for name in names(3):
        first.outbox.put(photo(name))
    first.uploader.step()
    # The process is gone; a new one starts on the same directory.
    second = Rig(tmp_path, receiver.address)
    second.uploader.step()
    assert receiver.delivered == names(3)[::-1]
    assert second.files() == []


def test_the_upper_pc_is_whatever_the_target_file_says_now(tmp_path, receiver):
    other = Receiver()
    try:
        rig = Rig(tmp_path, receiver.address)
        rig.outbox.put(photo(names(1)[0]))
        rig.uploader.step()
        rig.target_file.write_text(f"# moved at the venue\n{other.address}\n")
        rig.outbox.put(photo(names(1, 1)[0]))
        rig.uploader.step()
        assert receiver.delivered == names(1)
        assert other.delivered == names(1, 1)
    finally:
        other.close()


def test_a_waiting_uploader_retries_at_once_when_the_target_changes(tmp_path, receiver):
    rig = Rig(tmp_path, f"127.0.0.1:{_dead_port()}")
    rig.outbox.put(photo(names(1)[0]))
    waited = []
    thread = threading.Thread(target=lambda: waited.append(_timed(rig.uploader._wait, 30.0)))
    rig.uploader.step()
    thread.start()
    time.sleep(0.2)
    rig.target_file.write_text(receiver.address)
    thread.join(5.0)
    assert waited and waited[0] < 3 * POLL_SEC


def test_a_new_photo_wakes_a_waiting_uploader(tmp_path):
    rig = Rig(tmp_path, f"127.0.0.1:{_dead_port()}")
    waited = []
    thread = threading.Thread(target=lambda: waited.append(_timed(rig.uploader._wait, 30.0)))
    thread.start()
    time.sleep(0.1)
    rig.uploader.wake()
    thread.join(5.0)
    assert waited and waited[0] < 1.0


def _timed(call, *args) -> float:
    started = time.monotonic()
    call(*args)
    return time.monotonic() - started


def test_the_running_uploader_rides_out_a_receiver_outage(tmp_path):
    port = _dead_port()
    rig = Rig(tmp_path, f"127.0.0.1:{port}")
    thread = threading.Thread(target=rig.uploader.run, daemon=True)
    thread.start()
    try:
        rig.outbox.put(photo(names(1)[0]))
        rig.uploader.wake()
        time.sleep(0.5)
        assert rig.outbox.names() == names(1)
        up = Receiver(port)
        try:
            deadline = time.monotonic() + 2 * BACKOFF_SEC[1] + 1.0
            while rig.outbox.names() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert up.delivered == names(1)
        finally:
            up.close()
    finally:
        rig.uploader.stop()
        thread.join(5.0)
    assert not thread.is_alive()


def test_a_bug_in_one_pass_does_not_stop_the_uploader(tmp_path):
    rig = Rig(tmp_path, "127.0.0.1:9")
    calls = []

    def broken():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        rig.uploader.stop()
        return 0.0

    rig.uploader.step = broken
    rig.uploader._wait = lambda delay: None
    rig.uploader.run()
    assert len(calls) == 2
    assert "RuntimeError: boom" in rig.uploader.status()["last_error"]


def test_a_proxy_in_the_environment_is_not_used(tmp_path, receiver, monkeypatch):
    for name in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.setenv(name, f"http://127.0.0.1:{_dead_port()}")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    rig = Rig(tmp_path, receiver.address)
    rig.outbox.put(photo(names(1)[0]))
    rig.uploader.step()
    assert receiver.delivered == names(1)


# -- the target --------------------------------------------------------------------------


def test_the_target_is_the_first_line_that_is_not_a_comment(tmp_path):
    target = tmp_path / "photo_target"
    target.write_text("\n# the venue router\n  192.168.3.20:9000  \n10.0.0.1\n")
    assert read_target(str(target)) == ("http://192.168.3.20:9000", "")


def test_a_missing_or_empty_target_file_falls_back_to_the_default(tmp_path):
    target = tmp_path / "photo_target"
    assert read_target(str(target), "192.168.3.10") == ("http://192.168.3.10:8765", "")
    target.write_text("# nothing yet\n")
    assert read_target(str(target), "192.168.3.10") == ("http://192.168.3.10:8765", "")


def test_a_bad_address_in_the_file_does_not_fall_back(tmp_path):
    target = tmp_path / "photo_target"
    target.write_text("192.168.3.10/x2\n")
    url, problem = read_target(str(target), "192.168.3.99")
    assert url == ""
    assert str(target) in problem


def test_without_any_address_the_photos_wait_and_the_reason_is_told(tmp_path):
    rig = Rig(tmp_path)
    rig.outbox.put(photo(names(1)[0]))
    assert rig.uploader.step() == POLL_SEC
    assert rig.outbox.names() == names(1)
    assert "no upper PC address" in rig.uploader.status()["last_error"]
    assert len(rig.log.at("warning")) == 1


def test_the_default_target_is_used_when_there_is_no_file(tmp_path, receiver):
    rig = Rig(tmp_path, default=receiver.address)
    rig.outbox.put(photo(names(1)[0]))
    rig.uploader.step()
    assert receiver.delivered == names(1)
