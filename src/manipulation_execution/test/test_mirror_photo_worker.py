"""The round worker: every finished round with a frame lands in the outbox, whatever goes wrong while choosing.

Real ``PhotoCore`` rounds (synthetic frames, as in the selector tests) and a
real outbox in a temp directory; the worker thread runs as the node runs it.
"""

import json
import threading

import cv2
import numpy as np

from embodied_common.mirror_session_contract import STATE_MIRRORING, STATE_STANDBY
from manipulation_execution.mirror_photo.config import PhotoSection
from manipulation_execution.mirror_photo.outbox import Outbox
from manipulation_execution.mirror_photo.selector import PhotoCore
from manipulation_execution.mirror_photo.worker import RoundWorker

FPS = 15
FRAME = cv2.imencode(".jpg", np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8))[1].tobytes()


def state(name: str) -> str:
    return json.dumps({"state": name})


class Log:
    def __init__(self):
        self.lines: list[tuple[str, str]] = []

    def __call__(self, level: str, text: str) -> None:
        self.lines.append((level, text))

    def at(self, level: str) -> list[str]:
        return [text for kind, text in self.lines if kind == level]


class Rig:
    def __init__(self, tmp_path, *, choose=None):
        self.log = Log()
        section = PhotoSection(enabled=True, photos_dir=str(tmp_path / "photos"), target_file=str(tmp_path / "t"))
        self.outbox = Outbox(section.photos_dir, max_files=200, max_mb=100.0, min_free_gb=0.0, log=self.log)
        self.outbox.prepare()
        self.woken = 0
        extra = {} if choose is None else {"choose": choose}
        self.worker = RoundWorker(section, self.outbox, wake_uploader=self._wake, log=self.log, **extra)
        self.core = PhotoCore(
            section, on_round_end=self.worker.finished, on_provisional=self.worker.provisional, wall=lambda: 1.0e9
        )
        self.thread = threading.Thread(target=self.worker.run, daemon=True)
        self.thread.start()

    def _wake(self):
        self.woken += 1

    def round(self, seconds: float, start: float = 0.0) -> None:
        import numpy as np
        STILL_POSE = np.tile(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), 52)
        self.core.on_state(state(STATE_MIRRORING), 0.0)
        for index in range(round(seconds * FPS)):
            stamp = start + index / FPS
            self.core.on_image(stamp, FRAME)
            # Add PEAR frames to create anchors (needed for normal photos)
            self.core.on_pear(stamp, STILL_POSE, (10.0, 10.0, 50.0, 40.0))
        self.core.on_state(state(STATE_STANDBY), 0.0)

    def drain(self) -> None:
        self.worker.stop()
        self.thread.join(5.0)
        assert not self.thread.is_alive()

    def files(self) -> list[str]:
        return sorted(path.name for path in self.outbox.path.iterdir())


def test_a_round_becomes_one_photo_and_the_crash_copy_goes(tmp_path):
    rig = Rig(tmp_path)
    rig.round(8.0)
    rig.drain()
    (name,) = rig.outbox.names()
    assert rig.files() == [f"{name}.jpg", f"{name}.json"]
    assert rig.woken == 1
    status = rig.worker.status()
    assert status["rounds_sent"] == 1 and status["backlog"] == 0
    assert status["last_photo"]["quality"] == "normal"
    assert status["last_photo"]["stored"] == "disk"
    assert any(name in line and "queued on disk" in line for line in rig.log.at("info"))


def test_the_crash_copy_is_kept_while_the_round_runs(tmp_path):
    import numpy as np
    STILL_POSE = np.tile(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), 52)
    rig = Rig(tmp_path)
    rig.core.on_state(state(STATE_MIRRORING), 0.0)
    for index in range(8 * FPS):
        stamp = index / FPS
        rig.core.on_image(stamp, FRAME)
        # Add PEAR to create anchors and trigger provisional
        rig.core.on_pear(stamp, STILL_POSE, (10.0, 10.0, 50.0, 40.0))
    rig.drain()
    assert rig.files() == ["current.jpg", "current.json"]


def test_a_failure_while_choosing_still_sends_the_round(tmp_path):
    def broken(_finished, _section):
        raise MemoryError("decoding")

    rig = Rig(tmp_path, choose=broken)
    rig.round(8.0)
    rig.drain()
    (name,) = rig.outbox.names()
    _, meta = rig.outbox.load(name)
    assert (meta["quality"], meta["scoring"]) == ("fallback", "error")
    assert "MemoryError: decoding" in meta["error"]
    assert "current.jpg" not in rig.files()
    assert any("choosing failed" in line for line in rig.log.at("error"))


def test_a_round_without_a_frame_sends_nothing_and_says_so(tmp_path):
    rig = Rig(tmp_path)
    rig.round(0.0)
    rig.drain()
    assert rig.outbox.names() == []
    assert rig.woken == 0
    assert any("no camera frame" in line for line in rig.log.at("warning"))


def test_one_bad_round_does_not_stop_the_next(tmp_path):
    calls = []

    def flaky(finished, section):
        from manipulation_execution.mirror_photo.selector import choose_photo

        calls.append(finished.round_id)
        if len(calls) == 1:
            raise ValueError("first round")  # sent unscored, then the outbox refuses it
        return choose_photo(finished, section)

    rig = Rig(tmp_path, choose=flaky)
    put = rig.outbox.put
    first = [True]

    def put_once_broken(photo):
        if first:
            first.pop()
            raise RuntimeError("disk gone")
        return put(photo)

    rig.outbox.put = put_once_broken
    rig.round(8.0)
    rig.round(8.0, start=100.0)
    rig.round(8.0, start=200.0)
    rig.drain()
    assert len(calls) == 3
    errors = rig.log.at("error")
    assert any("ValueError: first round" in line for line in errors)
    assert any("RuntimeError: disk gone" in line for line in errors)
    assert len(rig.outbox.names()) == 2  # rounds two and three


def test_a_disk_that_refuses_the_crash_copy_is_told_once(tmp_path):
    rig = Rig(tmp_path)
    refused = []

    def refuse(photo):
        refused.append(photo)
        raise OSError(28, "No space left on device")

    rig.outbox.write_current = refuse
    rig.core.on_state(state(STATE_MIRRORING), 0.0)
    for index in range(20 * FPS):
        rig.core.on_image(index / FPS, FRAME)
    rig.core.on_state(state(STATE_STANDBY), 0.0)
    rig.drain()
    assert len(refused) > 1  # refused again and again, told once
    assert len([line for line in rig.log.at("warning") if "no crash copy" in line]) == 1
    assert len(rig.outbox.names()) == 1
