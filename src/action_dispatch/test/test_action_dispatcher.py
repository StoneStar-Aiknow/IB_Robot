import numpy as np
import pytest
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


@pytest.mark.parametrize(
    "action_chunk",
    [
        np.arange(6, dtype=np.float32),
        np.arange(12, dtype=np.float32).reshape(2, 6),
        np.arange(12, dtype=np.float32).reshape(1, 2, 6),
        torch.arange(6, dtype=torch.float32),
        torch.arange(12, dtype=torch.float32).reshape(2, 6),
        torch.arange(12, dtype=torch.float32).reshape(1, 2, 6),
    ],
)
def test_normalize_action_chunk_matches_legacy_contract(action_chunk):
    expected_tensor = action_chunk if hasattr(action_chunk, "detach") else torch.from_numpy(action_chunk)
    expected_array = expected_tensor.detach().cpu().numpy() if hasattr(action_chunk, "detach") else action_chunk
    if expected_array.ndim == 3 and expected_array.shape[0] == 1:
        expected_array = expected_array[0]
        expected_tensor = expected_tensor[0]
    if expected_array.ndim == 1:
        expected_array = expected_array.reshape(1, -1)
        expected_tensor = expected_tensor.reshape(1, -1)

    tensor, array = _normalize_action_chunk(action_chunk)

    assert type(tensor) is type(expected_tensor)
    assert tensor.dtype == expected_tensor.dtype
    assert tensor.device == expected_tensor.device
    assert tuple(tensor.shape) == tuple(expected_tensor.shape)
    assert torch.equal(tensor, expected_tensor)
    assert type(array) is type(expected_array)
    assert array.dtype == expected_array.dtype
    assert array.shape == expected_array.shape
    assert np.array_equal(array, expected_array)
    if isinstance(action_chunk, np.ndarray) and action_chunk.ndim == 2:
        assert array is action_chunk


def test_normalize_action_chunk_preserves_legacy_rejection_of_non_ndarray_input():
    with pytest.raises(TypeError, match="expected np.ndarray"):
        _normalize_action_chunk([[1.0, 2.0]])
