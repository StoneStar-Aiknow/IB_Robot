import pytest

from object_tracker.bounded_queue import LatestQueue


def test_latest_queue_bounds_memory_and_reports_drops():
    queue = LatestQueue[int](2)
    queue.push(1)
    queue.push(2)
    queue.push(3)

    assert len(queue) == 2
    assert queue.dropped == 1
    assert queue.pop_latest() == 3
    assert len(queue) == 0


def test_latest_queue_rejects_invalid_capacity():
    with pytest.raises(ValueError, match="positive"):
        LatestQueue(0)
