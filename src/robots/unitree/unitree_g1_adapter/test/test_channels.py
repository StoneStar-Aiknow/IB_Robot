"""Channel wrapper tests against the injectable fake vendor surface."""

from __future__ import annotations

import pytest
from fake_vendor import make_fake_vendor
from unitree_g1_adapter.channels import CommandWriter, StateFeed


def test_state_feed_caches_the_newest_sample_and_stamp():
    vendor_surface = make_fake_vendor()
    stamps = iter([100.0, 200.0])
    seen = []
    feed = StateFeed(
        "rt/lowstate",
        "low_state",
        vendor=vendor_surface,
        handler=seen.append,
        now=lambda: next(stamps),
    )
    subscriber = vendor_surface["_state"]["subscribers"][0]
    assert subscriber.init_calls == 1
    assert subscriber.queue_len == 0

    assert feed.last_sample is None
    assert feed.last_stamp == 0.0

    subscriber.deliver({"tick": 1})
    assert feed.last_sample == {"tick": 1}
    assert feed.last_stamp == 100.0
    assert seen == [{"tick": 1}]

    subscriber.deliver({"tick": 2})
    assert feed.last_sample == {"tick": 2}
    assert feed.last_stamp == 200.0
    assert seen == [{"tick": 1}, {"tick": 2}]


def test_state_feed_close_is_idempotent_and_stops_delivery():
    vendor_surface = make_fake_vendor()
    feed = StateFeed("rt/lowstate", "low_state", vendor=vendor_surface)
    subscriber = vendor_surface["_state"]["subscribers"][0]

    feed.close()
    feed.close()
    assert subscriber.closed == 1


def test_command_writer_builds_dict_low_cmd_frames_and_passes_idl_through():
    vendor_surface = make_fake_vendor()
    writer = CommandWriter("rt/arm_sdk", "low_cmd", vendor=vendor_surface)
    publisher = vendor_surface["_state"]["publishers"][0]
    assert publisher.init_calls == 1

    slots = [{"q": 0.0}] * 35
    assert writer.write(slots) is True
    kind, _timeout = publisher.written[0]
    assert kind == {"kind": "low_cmd", "slots": slots}

    idl_sample = object()
    assert writer.write(idl_sample) is True
    assert publisher.written[1][0] is idl_sample
    assert writer.write(idl_sample, timeout=0.2) is True
    assert publisher.written[2][1] == 0.2


def test_command_writer_supports_hand_cmd_dicts_and_refuses_others():
    vendor_surface = make_fake_vendor()
    writer = CommandWriter("rt/dex3/left/cmd", "hand_cmd", vendor=vendor_surface)
    assert writer.write([{"q": 0.5}]) is True

    state_reader = CommandWriter("rt/lowstate-out", "low_state", vendor=vendor_surface)
    with pytest.raises(TypeError, match="low_state"):
        state_reader.write({"any": "dict"})
    with pytest.raises(TypeError, match="low_state"):
        state_reader.write([{"any": "dict"}])

    # A bare mapping is refused for command types: the converter takes the
    # slot list, and building from a dict's keys would silently corrupt the
    # frame. A LowCmdFrame mapping (motor_cmd list + frame-level metadata) is
    # the accepted form for low_cmd.
    command_writer = CommandWriter("rt/arm_sdk", "low_cmd", vendor=vendor_surface)
    with pytest.raises(TypeError, match="slot list"):
        command_writer.write({"any": "dict"})

    # The low_cmd frame mapping carries the vendor header fields through to
    # the converter; the hand_cmd frame mapping stays refused.
    assert command_writer.write({"motor_cmd": [{"q": 0.1}], "mode_pr": 1, "mode_machine": 5, "crc": 0}) is True
    built = command_writer._publisher.written[-1][0]
    assert built == {"kind": "low_cmd", "slots": [{"q": 0.1}], "mode_pr": 1, "mode_machine": 5, "crc": 0}
    hand_writer = CommandWriter("rt/dex3/left/cmd", "hand_cmd", vendor=vendor_surface)
    with pytest.raises(TypeError, match="slot list"):
        hand_writer.write({"motor_cmd": [{"q": 0.1}]})


def test_command_writer_write_failure_passthrough_and_closed_guard():
    vendor_surface = make_fake_vendor(publisher_write_ok=False)
    writer = CommandWriter("rt/arm_sdk", "low_cmd", vendor=vendor_surface)
    assert writer.write([{"q": 0.0}] * 35) is False

    writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        writer.write([{"q": 0.0}] * 35)
