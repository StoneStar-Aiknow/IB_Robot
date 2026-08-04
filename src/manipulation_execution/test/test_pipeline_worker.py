import time

import pytest

from manipulation_execution.pipeline_worker import dynamic_worker_map


def test_dynamic_worker_map_balances_work_and_restores_input_order():
    assignments = []

    def process(worker_index: int, item: int) -> tuple[int, int]:
        if item == 0:
            time.sleep(0.05)
        assignments.append((item, worker_index))
        return item, worker_index

    results = dynamic_worker_map(
        list(range(6)),
        2,
        process,
        thread_name_prefix="pipeline-worker-test",
    )

    assert [item for item, _ in results] == list(range(6))
    assert any(item % 2 != worker_index for item, worker_index in assignments)


def test_dynamic_worker_map_rejects_non_positive_worker_count():
    with pytest.raises(ValueError, match="worker_count must be positive"):
        dynamic_worker_map([1], 0, lambda _worker_index, item: item, thread_name_prefix="invalid")
