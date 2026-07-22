import numpy as np
import torch

from action_dispatch.action_dispatcher_node import _normalize_action_chunk


def test_normalize_action_chunk_removes_singleton_batch_dimension():
    tensor, array = _normalize_action_chunk(np.zeros((1, 100, 6), dtype=np.float32))

    assert tuple(tensor.shape) == (100, 6)
    assert array.shape == (100, 6)


def test_normalize_action_chunk_preserves_unbatched_chunks():
    tensor, array = _normalize_action_chunk(torch.zeros((50, 6)))

    assert tuple(tensor.shape) == (50, 6)
    assert array.shape == (50, 6)
