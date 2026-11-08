"""``AsrMute``: the ASR held quiet while a prerecorded clip plays, on a fake clock.

The commands are the self introduction's: ``stop`` / ``start`` on ``/voice_control``.
"""

from __future__ import annotations

import pytest

from manipulation_execution.mirror_voice.asr_mute import START, STOP, AsrMute


class Clock:
    def __init__(self):
        self.t = 50.0

    def __call__(self):
        return self.t


def mute(mode="mute", delay=0.8):
    clock = Clock()
    sent: list[tuple[float, str]] = []
    return AsrMute(mode, delay, lambda command: sent.append((clock.t, command)), clock=clock), clock, sent


def commands(sent):
    return [command for _, command in sent]


class TestMute:
    def test_the_commands_are_the_self_introductions(self):
        assert (STOP, START) == ("stop", "start")

    def test_stop_before_the_clip_and_start_0_8_s_after_it_ends(self):
        asr, clock, sent = mute()
        asr.on_asr_status("listening")

        asr.playback_started()
        assert sent == [(50.0, "stop")]
        clock.t = 52.0
        asr.playback_ended()
        clock.t = 52.79
        asr.tick()
        assert commands(sent) == ["stop"]
        clock.t = 52.8
        asr.tick()

        assert sent == [(50.0, "stop"), (52.8, "start")]
        assert asr.status()["muted"] is False

    def test_a_clip_starting_inside_the_window_keeps_it_stopped(self):
        asr, clock, sent = mute()
        asr.playback_started()
        clock.t = 51.0
        asr.playback_ended()
        clock.t = 51.5
        asr.playback_started()  # the next clip, 0.5 s later
        clock.t = 53.0
        asr.tick()  # the first window is long over, but a clip is playing
        assert commands(sent) == ["stop"]

        asr.playback_ended()
        clock.t = 53.8
        asr.tick()
        assert commands(sent) == ["stop", "start"]

    def test_unknown_asr_state_counts_as_running(self):
        asr, _, sent = mute()
        asr.playback_started()
        assert commands(sent) == ["stop"]

    def test_an_idle_asr_is_left_alone_and_not_started_after(self):
        asr, clock, sent = mute()
        asr.on_asr_status("IDLE")
        asr.playback_started()
        asr.playback_ended()
        clock.t += 5
        asr.tick()
        asr.shutdown()

        assert sent == []

    def test_shutdown_starts_what_it_stopped(self):
        asr, _, sent = mute()
        asr.playback_started()
        asr.shutdown()
        assert commands(sent) == ["stop", "start"]

    def test_shutdown_after_the_start_sends_nothing_more(self):
        asr, clock, sent = mute()
        asr.playback_started()
        asr.playback_ended()
        clock.t += 1
        asr.tick()
        asr.shutdown()
        assert commands(sent) == ["stop", "start"]

    def test_a_zero_delay_starts_at_the_end(self):
        asr, _, sent = mute(delay=0.0)
        asr.playback_started()
        asr.playback_ended()
        assert commands(sent) == ["stop", "start"]

    def test_a_publish_error_is_counted_not_raised(self):
        def broken(command):
            raise RuntimeError("context down")

        asr = AsrMute("mute", 0.8, broken)
        asr.playback_started()
        asr.shutdown()
        assert asr.status()["publish_errors"] == 1


class Flaky:
    """A publisher that raises while ``broken``."""

    def __init__(self):
        self.broken = False
        self.sent: list[str] = []

    def __call__(self, command):
        if self.broken:
            raise RuntimeError("publisher down")
        self.sent.append(command)


def flaky(mode="mute"):
    clock = Clock()
    publish = Flaky()
    log: list[tuple[str, str]] = []
    asr = AsrMute(mode, 0.8, publish, clock=clock, log=lambda level, message: log.append((level, message)))
    return asr, clock, publish, log


def stopped_and_due(asr, clock, publish):
    """A clip played; its ``start`` is due and the publisher has just broken."""
    asr.playback_started()
    asr.playback_ended()
    clock.t += 0.8
    publish.broken = True
    asr.tick()


class TestStartRetry:
    def test_a_failed_start_is_tried_again_on_the_next_tick(self):
        asr, clock, publish, _ = flaky()
        stopped_and_due(asr, clock, publish)
        assert asr.status()["muted"] is True
        assert asr.status()["publish_errors"] == 1

        asr.tick()
        assert asr.status()["publish_errors"] == 2

        publish.broken = False
        asr.tick()
        assert publish.sent == ["stop", "start"]
        assert asr.status()["muted"] is False

        asr.tick()  # done: no further start
        assert publish.sent == ["stop", "start"]

    def test_it_warns_once_for_a_run_of_failures(self):
        asr, clock, publish, log = flaky()
        stopped_and_due(asr, clock, publish)
        for _ in range(20):
            asr.tick()

        assert len(log) == 1
        assert asr.status()["publish_errors"] == 21

    def test_no_retry_while_a_clip_plays_and_the_end_of_it_asks_again(self):
        asr, clock, publish, _ = flaky()
        stopped_and_due(asr, clock, publish)
        publish.broken = False
        asr.playback_started()
        clock.t += 5
        asr.tick()
        assert publish.sent == ["stop"]  # still stopped by us: no second stop, no start

        asr.playback_ended()
        clock.t += 0.79
        asr.tick()
        assert publish.sent == ["stop"]
        clock.t += 0.01
        asr.tick()
        assert publish.sent == ["stop", "start"]

    def test_off_does_not_retry_on_a_tick(self):
        asr, clock, publish, _ = flaky("off")
        asr.activate()
        publish.broken = True
        asr.shutdown()
        errors = asr.status()["publish_errors"]
        clock.t += 5
        asr.tick()

        assert asr.status()["publish_errors"] == errors


class TestShutdownIsIdempotent:
    def test_a_second_shutdown_sends_nothing(self):
        asr, _, publish, _ = flaky()
        asr.playback_started()
        asr.shutdown()
        asr.shutdown()

        assert publish.sent == ["stop", "start"]

    def test_a_shutdown_that_failed_is_finished_by_the_late_one(self):
        asr, _, publish, _ = flaky()
        asr.playback_started()
        publish.broken = True
        asr.shutdown()
        publish.broken = False
        asr.shutdown()
        asr.shutdown()

        assert publish.sent == ["stop", "start"]

    def test_a_failed_shutdown_is_not_retried_by_ticks(self):
        asr, clock, publish, _ = flaky()
        asr.playback_started()
        publish.broken = True
        asr.shutdown()
        publish.broken = False
        clock.t += 5
        asr.tick()

        assert publish.sent == ["stop"]

    @pytest.mark.parametrize("mode", ["mute", "off"])
    def test_nothing_is_stopped_after_shutdown(self, mode):
        asr, clock, publish, _ = flaky(mode)
        asr.playback_started()
        asr.shutdown()
        sent = list(publish.sent)
        asr.playback_started()
        asr.on_asr_status("listening")
        asr.playback_ended()
        clock.t += 5
        asr.tick()

        assert publish.sent == sent


class TestOtherModes:
    def test_off_stops_at_activation_before_each_clip_and_when_it_comes_back(self):
        asr, clock, sent = mute("off")
        asr.activate()
        asr.playback_started()
        asr.playback_ended()
        clock.t += 5
        asr.tick()
        asr.on_asr_status("idle")
        asr.on_asr_status("listening")  # someone else started it

        assert commands(sent) == ["stop", "stop", "stop"]
        asr.shutdown()
        assert commands(sent)[-1] == "start"

    def test_keep_sends_nothing(self):
        asr, clock, sent = mute("keep")
        asr.activate()
        asr.on_asr_status("listening")
        asr.playback_started()
        asr.playback_ended()
        clock.t += 5
        asr.tick()
        asr.shutdown()
        assert sent == []

    @pytest.mark.parametrize("mode,delay", [("loud", 0.8), ("mute", -1), ("mute", float("nan")), ("mute", True)])
    def test_bad_arguments_are_refused(self, mode, delay):
        with pytest.raises(ValueError):
            AsrMute(mode, delay, lambda command: None)
