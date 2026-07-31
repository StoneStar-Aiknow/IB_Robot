import threading
import time

from semantic_mapping.pipeline import BoundedFrameQueue, SerializedCommitter


def test_drop_oldest_queue_never_returns_dropped_frame():
    queue = BoundedFrameQueue(capacity=2, policy="drop_oldest")
    queue.put("first")
    queue.put("second")
    queue.put("third")

    assert queue.get().payload == "second"
    assert queue.get().payload == "third"
    assert queue.diagnostics.dropped_oldest == 1


def test_drop_newest_queue_reports_rejection():
    queue = BoundedFrameQueue(capacity=1, policy="drop_newest")
    assert queue.put("first")
    assert not queue.put("second")

    assert queue.get().payload == "first"
    assert queue.diagnostics.dropped_newest == 1


def test_backpressure_waits_until_consumer_makes_capacity():
    queue = BoundedFrameQueue(capacity=1, policy="backpressure")
    queue.put("first")
    result = []
    producer = threading.Thread(target=lambda: result.append(queue.put("second", timeout=1.0)))
    producer.start()
    time.sleep(0.02)

    assert result == []
    assert queue.get().payload == "first"
    producer.join(timeout=1.0)
    assert result == [True]
    assert queue.get().payload == "second"
    assert queue.diagnostics.backpressure_waits == 1


def test_serialized_committer_prevents_overlapping_mutation():
    committer = SerializedCommitter()
    active = 0
    maximum_active = 0
    state_lock = threading.Lock()

    def mutate():
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.01)
        with state_lock:
            active -= 1

    threads = [threading.Thread(target=lambda: committer.commit(mutate)) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert maximum_active == 1
    assert committer.diagnostics.committed == 4
    assert len(committer.diagnostics.commit_time_ns) == 4
