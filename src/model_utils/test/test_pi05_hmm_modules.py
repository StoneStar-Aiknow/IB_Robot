from __future__ import annotations

import torch

from model_utils.pi05_export.build_hmm_modules import MODULES, _hmonnx_path
from model_utils.pi05_export.export_hmm_modules import _cache_names, _flatten_cache, _unflatten_cache


def test_pi05_hmm_module_table_matches_the_manifest_execution_roles(tmp_path):
    assert tuple(MODULES) == (
        "vision",
        "prefill",
        "action_in_proj",
        "time_mlp",
        "decode",
        "action_out_proj",
    )
    assert MODULES["vision"] == ("siglip", False, -27, 2)
    assert MODULES["prefill"] == ("gemma_2b_prefill", True, -18, 0)
    assert MODULES["decode"] == ("gemma_expert_300m_decode", True, -18, 0)
    assert MODULES["action_in_proj"] == ("action_in_proj", False, None, None)
    assert MODULES["time_mlp"] == ("time_mlp", False, None, None)
    assert MODULES["action_out_proj"] == ("action_out_proj", False, None, None)

    output_dir = tmp_path / "xh2"
    assert _hmonnx_path(output_dir, "vision") == output_dir / "hmonnx/vision.onnx"
    assert _hmonnx_path(output_dir, "action_in_proj") == output_dir / "hmonnx/action_in_proj.onnx"
    assert _hmonnx_path(output_dir, "time_mlp") == output_dir / "hmonnx/time_mlp.onnx"
    assert _hmonnx_path(output_dir, "action_out_proj") == output_dir / "hmonnx/action_out_proj.onnx"
    assert _hmonnx_path(output_dir, "prefill") == output_dir / "hmonnx/prefill.onnx"
    assert _hmonnx_path(output_dir, "decode") == output_dir / "hmonnx/decode.onnx"


def test_pi05_cache_helpers_round_trip_transformers_dynamic_cache():
    cache = _unflatten_cache(
        (
            torch.ones(1, 1, 3, 2),
            torch.full((1, 1, 3, 2), 2.0),
            torch.full((1, 1, 3, 2), 3.0),
            torch.full((1, 1, 3, 2), 4.0),
        )
    )

    flat = _flatten_cache(cache)

    assert len(flat) == 4
    assert torch.equal(flat[0], torch.ones(1, 1, 3, 2))
    assert torch.equal(flat[3], torch.full((1, 1, 3, 2), 4.0))


def test_pi05_cache_names_are_interleaved_by_layer():
    assert _cache_names(2) == ["past_key_0", "past_value_0", "past_key_1", "past_value_1"]
