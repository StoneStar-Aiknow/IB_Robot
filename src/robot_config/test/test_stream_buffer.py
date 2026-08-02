from __future__ import annotations

import pytest

from robot_config.contract_utils import StreamBuffer


def test_stream_buffer_selects_ordered_asof_sample_after_out_of_order_pushes():
    buffer = StreamBuffer("hold", 50, retention_ns=1_000)

    assert buffer.push(200, "second", receive_time_ns=210)
    assert buffer.push(100, "first", receive_time_ns=220)
    assert buffer.push(150, "middle", receive_time_ns=230)

    assert [item[0] for item in buffer.history] == [100, 150, 200]
    assert buffer.sample(175, now_ns=230) == "middle"


def test_stream_buffer_replaces_duplicate_capture_timestamp():
    buffer = StreamBuffer("hold", 50)

    buffer.push(100, "old", receive_time_ns=100)
    buffer.push(100, "new", receive_time_ns=110)

    assert buffer.history == [(100, 110, "new")]


@pytest.mark.parametrize(
    ("policy", "tolerance_ns", "tick_ns", "constraint"),
    [("asof", 20, 121, "asof"), ("drop", 0, 150, "drop")],
)
def test_stream_buffer_reports_alignment_tolerance_failures(policy, tolerance_ns, tick_ns, constraint):
    buffer = StreamBuffer(policy, 50, tolerance_ns)
    buffer.push(100, "value")

    value, issue = buffer.select(tick_ns)

    assert value is None
    assert issue["reason"] == "stale"
    assert issue["constraint"] == constraint


def test_stream_buffer_uses_receive_timestamp_for_live_age():
    buffer = StreamBuffer("hold", 50, max_age_ns=100)
    buffer.push(10, "value", receive_time_ns=1_000)

    assert buffer.sample(1_050, now_ns=1_100) == "value"
    value, issue = buffer.select(1_050, now_ns=1_101)
    assert value is None
    assert issue["constraint"] == "max_age"


def test_stream_buffer_prunes_retention_without_future_sample_poisoning_history():
    buffer = StreamBuffer("hold", 50, retention_ns=100)
    buffer.push(100, "old", receive_time_ns=100)
    buffer.push(150, "middle", receive_time_ns=150)
    buffer.push(250, "latest", receive_time_ns=250)

    assert [item[0] for item in buffer.history] == [150, 250]
    assert not buffer.push(1_000, "future", receive_time_ns=260)
    assert [item[0] for item in buffer.history] == [150, 250]


def test_stream_buffer_reports_missing_and_newer_than_request():
    buffer = StreamBuffer("hold", 50)

    _, missing = buffer.select(100)
    assert missing["reason"] == "missing"

    buffer.push(200, "future")
    _, future = buffer.select(199)
    assert future["reason"] == "newer_than_request"


def test_stream_buffer_reset_removes_all_samples():
    buffer = StreamBuffer("hold", 50)
    buffer.push(100, "value")

    buffer.reset()

    assert len(buffer) == 0
    assert buffer.history == []
