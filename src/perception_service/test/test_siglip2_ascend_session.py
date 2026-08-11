from types import MethodType

import numpy as np
import pytest

from inference_service.backends.errors import BackendInferenceError
from inference_service.generic_runtime import NamedTensorRequest
from perception_service.siglip2_ascend_session import SigLIP2AscendSession


def _session(calls):
    session = SigLIP2AscendSession()
    session._text_batch_size = MethodType(lambda _self: 4, session)
    session._embedding_dimension = MethodType(lambda _self, _role: 3, session)

    def run_role(_self, _index, role, values):
        calls.append((role, values))
        if role == "vision":
            value = np.asarray(values["host.siglip2.image"], dtype=np.float32)
            return {"host.siglip2.image_embedding": np.repeat(value.mean()[None, None], 3, axis=1)}
        value = np.asarray(values["host.siglip2.input_ids"], dtype=np.float32)
        return {"host.siglip2.text_embeddings": np.repeat(value[:, :1], 3, axis=1)}

    session._run_role = MethodType(run_role, session)
    return session


def test_siglip2_ascend_splits_images_and_pads_text_to_compiled_batches():
    calls = []
    session = _session(calls)
    request = NamedTensorRequest(
        request_id="siglip2",
        inputs={
            "masked_images": np.stack(
                [np.ones((3, 2, 2), dtype=np.float32), np.full((3, 2, 2), 2.0, dtype=np.float32)]
            ),
            "text_tokens": np.arange(6 * 5, dtype=np.int64).reshape(6, 5),
            "text_attention_mask": np.ones((6, 5), dtype=np.int64),
        },
    )

    result = session._execute(request)

    assert [role for role, _values in calls] == ["vision", "vision", "text", "text"]
    assert calls[-1][1]["host.siglip2.input_ids"].shape == (4, 5)
    np.testing.assert_array_equal(calls[-1][1]["host.siglip2.input_ids"][2:], 0)
    assert result["image_embeddings"].shape == (2, 3)
    assert result["text_embeddings"].shape == (6, 3)
    assert result["image_embeddings"].flags.c_contiguous
    assert result["text_embeddings"].flags.c_contiguous


def test_siglip2_ascend_rejects_mismatched_token_and_attention_shapes():
    session = _session([])
    request = NamedTensorRequest(
        request_id="siglip2",
        inputs={
            "masked_images": np.empty((0, 3, 384, 384), dtype=np.float32),
            "text_tokens": np.empty((2, 64), dtype=np.int64),
            "text_attention_mask": np.empty((1, 64), dtype=np.int64),
        },
    )

    with pytest.raises(BackendInferenceError, match="token and attention shapes differ"):
        session._execute(request)
