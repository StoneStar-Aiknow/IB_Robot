"""Deterministic concurrency contracts for the ROS-free speech scheduler."""

import threading
from dataclasses import dataclass

import pytest

from ibrobot_agent.speech_pipeline import SpeechPipeline


@dataclass
class Speech:
    text: str
    then: tuple = ()
    terminal: bool = False
    kind: str = "reply"
    execution_id: str = "execution"


class Harness:
    def __init__(self, capacity=32):
        self.changed = threading.Condition()
        self.synthesized = []
        self.played = []
        self.completed = []
        self.logs = []
        self.synth_blocks = {}
        self.play_blocks = {}
        self.expired = set()
        self.fail_synth = set()
        self.fail_play = set()
        self.active = {"synth": 0, "play": 0}
        self.maximum = {"synth": 0, "play": 0}
        self.pipeline = SpeechPipeline(
            synthesize=self.synthesize,
            play=self.play,
            is_current=lambda action: action.text not in self.expired,
            completed=lambda action: self.record(self.completed, action.text),
            log=lambda line: self.record(self.logs, line),
            capacity=capacity,
        )

    def record(self, target, value):
        with self.changed:
            target.append(value)
            self.changed.notify_all()

    def wait(self, predicate):
        with self.changed:
            assert self.changed.wait_for(predicate, timeout=3), self.logs

    def synthesize(self, action):
        self.active["synth"] += 1
        self.maximum["synth"] = max(self.maximum["synth"], self.active["synth"])
        self.record(self.synthesized, action.text)
        try:
            if action.text in self.synth_blocks:
                assert self.synth_blocks[action.text].wait(5)
            if action.text in self.fail_synth:
                raise ValueError("synthesis test failure")
            return [action.text + "-1", action.text + "-2"]
        finally:
            self.active["synth"] -= 1

    def play(self, action, audio):
        self.active["play"] += 1
        self.maximum["play"] = max(self.maximum["play"], self.active["play"])
        self.record(self.played, action.text)
        try:
            assert audio == [action.text + "-1", action.text + "-2"]
            if action.text in self.play_blocks:
                assert self.play_blocks[action.text].wait(5)
            if action.text in self.fail_play:
                raise ValueError("play test failure")
            return True
        finally:
            self.active["play"] -= 1

    def submit(self, name, **kwargs):
        assert self.pipeline.submit(Speech(name, **kwargs))

    def stop(self):
        for event in [*self.synth_blocks.values(), *self.play_blocks.values()]:
            event.set()

    def close(self):
        self.pipeline.close(stop=self.stop)


@pytest.fixture
def harness():
    value = Harness()
    yield value
    value.close()


def test_overlap_latest_ready_and_grouped_audio(harness):
    h = harness
    h.play_blocks["A"] = threading.Event()
    h.submit("A")
    h.wait(lambda: h.played == ["A"])
    h.submit("B")
    h.wait(lambda: any("seq=2" in line and "synthesis returned" in line for line in h.logs))
    h.submit("C")
    h.wait(lambda: any("reason=replaced" in line and "seq=2" in line for line in h.logs))
    h.play_blocks["A"].set()
    h.wait(lambda: h.completed == ["A", "C"])
    assert h.played == ["A", "C"]
    assert h.maximum == {"synth": 1, "play": 1}


def test_ready_does_not_wait_for_new_synthesis(harness):
    h = harness
    h.play_blocks["A"] = threading.Event()
    h.synth_blocks["C"] = threading.Event()
    h.submit("A")
    h.wait(lambda: h.played == ["A"])
    h.submit("B")
    h.submit("C")
    h.wait(lambda: "C" in h.synthesized)
    h.play_blocks["A"].set()
    h.wait(lambda: "B" in h.completed)
    assert h.played == ["A", "B"]


def test_then_barrier_partitions_ready_and_confirms_once(harness):
    h = harness
    h.play_blocks["A"] = threading.Event()
    h.synth_blocks["barrier"] = threading.Event()
    h.submit("A")
    h.wait(lambda: h.played == ["A"])
    h.submit("B")
    h.submit("barrier", then=("control",))
    h.submit("C")
    h.wait(lambda: "barrier" in h.synthesized)
    h.play_blocks["A"].set()
    h.wait(lambda: "B" in h.completed)
    assert "barrier" not in h.played
    h.synth_blocks["barrier"].set()
    h.wait(lambda: "C" in h.completed)
    assert h.played == ["A", "B", "barrier", "C"]
    assert h.completed.count("barrier") == 1


def test_terminal_invalidates_inflight_and_ready_but_not_claimed(harness):
    h = harness
    h.play_blocks["A"] = threading.Event()
    h.synth_blocks["C"] = threading.Event()
    h.submit("A")
    h.wait(lambda: h.played == ["A"])
    h.submit("B")
    h.submit("C")
    h.wait(lambda: "C" in h.synthesized)
    h.submit("pending")
    h.submit("terminal", kind="terminal", terminal=True)
    h.submit("new")
    h.synth_blocks["C"].set()
    h.wait(lambda: "new" in h.synthesized)
    h.play_blocks["A"].set()
    h.wait(lambda: "new" in h.completed)
    assert h.played == ["A", "terminal", "new"]
    assert "pending" not in h.synthesized


def test_execution_terminal_keeps_then_and_other_execution(harness):
    h = harness
    h.synth_blocks["blocked"] = threading.Event()
    h.submit("blocked")
    h.wait(lambda: "blocked" in h.synthesized)
    h.submit("old")
    h.submit("barrier", then=("control",))
    h.submit("other", execution_id="other")
    h.submit("final", terminal=True)
    h.synth_blocks["blocked"].set()
    h.wait(lambda: "final" in h.completed)
    assert h.played == ["barrier", "other", "final"]


def test_failures_do_not_confirm_or_kill_workers(harness):
    h = harness
    h.fail_synth.add("synth-error")
    h.fail_play.add("play-error")
    h.submit("synth-error", then=("control",))
    h.submit("play-error", then=("control",))
    h.submit("good", then=("control",))
    h.wait(lambda: "good" in h.completed)
    assert h.completed == ["good"]
    assert any("synthesis failed" in line for line in h.logs)
    assert any("playback/completion failed" in line for line in h.logs)


def test_receipt_busy_replacement_is_pending_only(harness):
    h = harness
    h.play_blocks["A"] = threading.Event()
    h.synth_blocks["blocked"] = threading.Event()
    h.submit("A")
    h.wait(lambda: "A" in h.played)
    h.submit("ready", kind="busy")
    h.wait(lambda: any("seq=2" in line and "synthesis returned" in line for line in h.logs))
    h.submit("blocked")
    h.wait(lambda: "blocked" in h.synthesized)
    h.submit("pending-busy", kind="busy")
    h.submit("receipt", kind="receipt")
    h.play_blocks["A"].set()
    h.wait(lambda: "ready" in h.completed)
    assert "pending-busy" not in h.synthesized


def test_validity_before_after_synthesis_and_before_play(harness):
    h = harness
    h.expired.add("before")
    h.submit("before", kind="receipt")
    h.synth_blocks["after"] = threading.Event()
    h.submit("after")
    h.wait(lambda: "after" in h.synthesized)
    h.expired.add("after")
    h.synth_blocks["after"].set()
    h.play_blocks["A"] = threading.Event()
    h.submit("A")
    h.wait(lambda: "A" in h.played)
    h.submit("ready", then=("control",))
    h.submit("sentinel", then=("control",))
    h.wait(lambda: "sentinel" in h.synthesized)
    h.expired.add("ready")
    h.play_blocks["A"].set()
    h.wait(lambda: "sentinel" in h.completed)
    assert h.played == ["A", "sentinel"]
    assert "before" not in h.synthesized
    assert sum("reason=expired" in line for line in h.logs) == 3


def test_capacity_rejects_protected_and_evicts_ordinary():
    h = Harness(capacity=2)
    try:
        h.synth_blocks["barrier"] = threading.Event()
        h.submit("barrier", then=("control",))
        h.wait(lambda: "barrier" in h.synthesized)
        h.submit("old")
        h.submit("new", then=("control",))
        assert not h.pipeline.submit(Speech("rejected", then=("control",)))
        assert any("reason=capacity" in line for line in h.logs)
        h.synth_blocks["barrier"].set()
        h.wait(lambda: "new" in h.completed)
        assert h.played == ["barrier", "new"]
    finally:
        h.close()


@pytest.mark.parametrize("stage", ["synth", "play", "idle"])
def test_close_unblocks_workers_suppresses_late_play_and_completion(stage):
    h = Harness()
    if stage != "idle":
        blocks = h.synth_blocks if stage == "synth" else h.play_blocks
        blocks["A"] = threading.Event()
        h.submit("A", then=("control",))
        observed = h.synthesized if stage == "synth" else h.played
        h.wait(lambda: "A" in observed)
    h.close()
    assert h.completed == []
    assert not h.pipeline.submit(Speech("late"))
    assert not any(worker.is_alive() for worker in h.pipeline._workers)
    if stage != "play":
        assert h.played == []


def test_close_timeout_is_visible_and_can_be_retried():
    h = Harness()
    h.synth_blocks["A"] = threading.Event()
    h.submit("A")
    h.wait(lambda: "A" in h.synthesized)
    try:
        with pytest.raises(TimeoutError, match="speech-synthesis"):
            h.pipeline.close(timeout=0)
    finally:
        h.close()
    assert h.played == []


def test_completion_admitted_before_close_finishes_before_close_returns():
    h = Harness()
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    def completed(action):
        entered.set()
        assert release.wait(3)
        h.record(h.completed, action.text)

    h.pipeline._completed = completed
    h.submit("A")
    assert entered.wait(3)
    closer = threading.Thread(target=lambda: (h.close(), closed.set()))
    closer.start()
    assert not closed.is_set()
    release.set()
    closer.join(3)
    assert closed.is_set()
    assert h.completed == ["A"]


def test_external_callbacks_do_not_hold_scheduler_lock():
    h = Harness()
    checked = threading.Event()

    def current(action):
        thread = threading.Thread(target=lambda: (h.pipeline.submit(Speech("nested", then=(1,))), checked.set()))
        if action.text == "A" and not checked.is_set():
            thread.start()
            assert checked.wait(3)
            thread.join(3)
        return True

    h.pipeline._is_current = current
    try:
        h.submit("A", then=(1,))
        h.wait(lambda: "nested" in h.completed)
    finally:
        h.close()
