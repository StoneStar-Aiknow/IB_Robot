from __future__ import annotations

import numpy as np
import pytest

from inference_service.backends.ascend.backend import AscendBackend
from inference_service.codecs import BoundInputs, BoundTensor


def test_ascend_explicit_diagnostic_capture_records_named_values_without_acl():
    captured = {}
    backend = AscendBackend(
        0,
        diagnostic_capture=lambda name, value: captured.setdefault(name, np.asarray(value).copy()),
    )
    inputs = BoundInputs(
        (
            BoundTensor("observation.images.top", "image", 0, np.array([1.0], dtype=np.float32)),
            BoundTensor("observation.language.tokens", "tokens", 1, np.array([2], dtype=np.int64)),
            BoundTensor("observation.language.attention_mask", "masks", 2, np.array([True], dtype=np.bool_)),
            BoundTensor("prefix_att_2d_masks_4d", "prefix", 3, np.array([3.0], dtype=np.float32)),
        )
    )

    backend._capture_bound_inputs("vlm", inputs)
    backend._capture_vlm_outputs(
        {
            "internal.past_kv": np.array([4.0], dtype=np.float16),
            "internal.prefix_pad_masks": np.array([False], dtype=np.bool_),
        }
    )

    assert set(captured) == {
        "vlm_in_image_0",
        "vlm_in_lang_tokens",
        "vlm_in_lang_masks",
        "vlm_in_prefix_mask_4d",
        "past_kv_tensor",
        "prefix_pad_masks",
    }
    np.testing.assert_array_equal(captured["past_kv_tensor"], np.array([4.0], dtype=np.float16))


def test_ascend_diagnostic_capture_is_not_a_runtime_option():
    try:
        AscendBackend._validate_runtime_options({"diagnostic_capture": True})
    except Exception as exc:
        assert getattr(exc, "code", None) == "invalid_runtime_options"
    else:  # pragma: no cover
        raise AssertionError("diagnostic_capture must not be accepted as a runtime option")


def test_ascend_diagnostic_capture_constructor_rejects_non_callable():
    with pytest.raises(TypeError, match="callable"):
        AscendBackend(0, diagnostic_capture=True)
