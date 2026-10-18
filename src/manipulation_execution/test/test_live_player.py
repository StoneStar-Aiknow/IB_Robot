import threading
import time

import numpy as np

from manipulation_execution.imitation_retargeting.live_player import LivePlayer

LOW = np.full(14, -1.0)
HIGH = np.full(14, 1.0)
RATED = np.full(14, 2.0)
VEL_SCALE = 0.5
RATE_HZ = 50.0
BASE = np.zeros(14)


class FakeStream:
    joint_names = tuple(f"j{i}" for i in range(14))

    def __init__(self):
        self.calls = []
        self.polled = []        # what poll returns
        self.fail = False

    def retarget(self, t_camera, frame_id, raw):
        self.calls.append(("retarget", frame_id))
        if self.fail:
            raise RuntimeError("boom")
        return np.full(14, 0.01)

    def push_pose(self, t_camera, pose):
        self.calls.append(("push_pose", t_camera))

    def reset_input(self):
        self.calls.append(("reset_input",))

    def start_round(self):
        self.calls.append(("start_round",))

    def poll(self, now):
        self.calls.append(("poll", now))
        return list(self.polled)

    def status(self):
        return {"stream": "fake"}

    def names(self):
        return [c[0] for c in self.calls]


def _player(stream, queue_size=4):
    return LivePlayer(stream, low_rad=LOW, high_rad=HIGH, rated_rad_s=RATED, vel_scale=VEL_SCALE,
                      rate_hz=RATE_HZ, staleness_s=0.5, base_pose=BASE, queue_size=queue_size)


def _frame(player, now, frame_id, source="yolox"):
    player.on_frame(now, now, frame_id, source, np.zeros(312))


def test_paused_tick_does_not_poll_and_play_starts_round():
    stream = FakeStream()
    player = _player(stream)
    _frame(player, 1.0, 0)
    assert player.tick(1.0) is None
    assert "poll" not in stream.names()
    player.play(True)
    assert stream.names()[-1] == "start_round"
    stream.polled = [np.full(14, 0.001), np.full(14, 0.002)]
    assert player.tick(1.02) == np.full(14, 0.002).tolist()


def test_non_yolox_frames_are_not_queued_and_not_alive():
    stream = FakeStream()
    player = _player(stream)
    player.play(True)
    stream.polled = [np.full(14, 0.001)]
    for i in range(3):
        _frame(player, 1.0 + 0.02 * i, i, source="center_fallback")
    assert player.process_one(timeout=0) is False
    assert player.tick(1.1) is None
    status = player.status(1.1)
    assert (status["input_frames"], status["yolox_frames"], status["held_stale"]) == (3, 0, 1)
    assert status["input_age_sec"] is None


def test_reset_input_precedes_first_frame_after_gap():
    stream = FakeStream()
    player = _player(stream)
    for frame_id, now in enumerate((1.0, 1.02, 1.04, 2.0, 2.02)):
        _frame(player, now, frame_id)
        assert player.process_one(timeout=0)
    names = [c for c in stream.calls if c[0] in ("reset_input", "retarget")]
    assert names == [("reset_input",), ("retarget", 0), ("retarget", 1), ("retarget", 2),
                     ("reset_input",), ("retarget", 3), ("retarget", 4)]


def test_retarget_error_is_counted_and_not_pushed():
    stream = FakeStream()
    stream.fail = True
    player = _player(stream)
    _frame(player, 1.0, 0)
    assert player.process_one(timeout=0) is True
    status = player.status(1.0)
    assert status["retarget_errors"] == 1 and "boom" in status["last_error"]
    assert "push_pose" not in stream.names()


def test_big_jump_is_rate_capped_and_clamped():
    stream = FakeStream()
    player = _player(stream)
    player.play(True)
    stream.polled = [np.full(14, 5.0)]
    prev = BASE
    for i in range(60):
        now = 1.0 + i / RATE_HZ
        _frame(player, now, i)
        out = np.array(player.tick(now))
        assert np.all(np.abs(out - prev) <= RATED * VEL_SCALE / RATE_HZ + 1e-12)
        assert np.all((out >= LOW) & (out <= HIGH))
        prev = out
    assert np.allclose(prev, HIGH)


def test_full_queue_drops_oldest():
    stream = FakeStream()
    player = _player(stream, queue_size=4)
    for i in range(6):
        _frame(player, 1.0 + 0.02 * i, i)
    while player.process_one(timeout=0):
        pass
    assert [c[1] for c in stream.calls if c[0] == "retarget"] == [2, 3, 4, 5]
    assert player.status(1.2)["dropped"] == 2


def test_worker_thread_accounts_for_every_frame():
    stream = FakeStream()
    player = _player(stream)
    player.start()
    feeder = threading.Thread(target=lambda: [_frame(player, 1.0 + 0.001 * i, i) for i in range(50)])
    feeder.start()
    feeder.join()
    started = time.monotonic()
    player.close()
    assert time.monotonic() - started < 1.0
    assert not player._thread.is_alive()
    status = player.status(2.0)
    assert status["yolox_frames"] == 50
    assert status["processed"] + status["retarget_errors"] + status["dropped"] == 50
