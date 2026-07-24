import time
from types import SimpleNamespace
from typing import Any, cast

from manipulation_execution.pick_executor_node import PickExecutorNode


class _Logger:
    def info(self, _message: str) -> None:
        pass

    def warning(self, _message: str) -> None:
        pass


def test_parallel_candidate_preparation_preserves_ranked_order_and_worker_affinity():
    harness = SimpleNamespace(
        _ik_worker_clients=["ik_0", "ik_1", "ik_2"],
        _fk_worker_clients=["fk_0", "fk_1", "fk_2"],
        _ik_worker_count=3,
        get_logger=lambda: _Logger(),
    )
    harness._verify_ik_worker_pool = lambda *_args: None
    calls: list[tuple[int, str, str]] = []

    def prepare(candidate, _scene, _goal, _deadline, *, initial_seed, ik_client, fk_client):
        del initial_seed
        time.sleep(0.002 * (3 - candidate.index % 3))
        calls.append((candidate.index, ik_client, fk_client))
        return candidate.index

    harness._prepare_candidate = prepare
    ranked = [SimpleNamespace(index=index) for index in range(8)]

    prepared, error = PickExecutorNode._prepare_ranked_candidates(
        cast(Any, harness),
        cast(Any, ranked),
        cast(Any, None),
        SimpleNamespace(),
        None,
        time.monotonic() + 10.0,
    )

    assert error is None
    assert prepared == list(range(8))
    assert sorted(calls) == [(index, f"ik_{index % 3}", f"fk_{index % 3}") for index in range(8)]
