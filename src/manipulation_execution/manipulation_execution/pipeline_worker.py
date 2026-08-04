"""Robot-independent worker scheduling for grasp candidate preparation."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, SimpleQueue
from typing import TypeVar

__all__ = ["dynamic_worker_map"]

_ItemT = TypeVar("_ItemT")
_ResultT = TypeVar("_ResultT")


def dynamic_worker_map(
    items: list[_ItemT],
    worker_count: int,
    worker: Callable[[int, _ItemT], _ResultT],
    *,
    thread_name_prefix: str,
) -> list[_ResultT]:
    """Process one shared queue concurrently while preserving input order."""

    if not items:
        return []
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")

    active_workers = min(int(worker_count), len(items))
    work_queue: SimpleQueue[tuple[int, _ItemT]] = SimpleQueue()
    for position, item in enumerate(items):
        work_queue.put((position, item))
    ordered_results: list[_ResultT | None] = [None] * len(items)

    def consume(worker_index: int) -> list[tuple[int, _ResultT]]:
        completed = []
        while True:
            try:
                position, item = work_queue.get_nowait()
            except Empty:
                break
            completed.append((position, worker(worker_index, item)))
        return completed

    with ThreadPoolExecutor(max_workers=active_workers, thread_name_prefix=thread_name_prefix) as pool:
        futures = [pool.submit(consume, worker_index) for worker_index in range(active_workers)]
        for future in futures:
            for position, result in future.result():
                ordered_results[position] = result

    if any(result is None for result in ordered_results):
        raise RuntimeError("dynamic worker queue returned incomplete results")
    return [result for result in ordered_results if result is not None]
