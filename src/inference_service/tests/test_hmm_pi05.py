from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from inference_service.core.compiled_policy import (
    HMM_MANIFEST_BASENAME,
    PI05CompiledAdapter,
    create_runtime_session,
    load_compiled_manifest,
    resolve_pi05_hmm_paths,
)


def _pi05_config(**updates):
    config = {
        "type": "pi05",
        "chunk_size": 30,
        "max_action_dim": 7,
        "num_inference_steps": 5,
        "input_features": {
            "observation.images.front": {"type": "VISUAL", "shape": [3, 224, 224]},
            "observation.language.tokens": {"shape": [48]},
            "observation.language.attention_mask": {"shape": [48]},
        },
        "output_features": {"action": {"shape": [7]}},
    }
    config.update(updates)
    return config


def _write_pi05_policy(tmp_path: Path, config: dict) -> Path:
    """Lay out a fake pi05 HMM policy dir: config.json + config.hmm.json + 6 .hmm + embedding.pt."""
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "policy_type": "pi05",
        "backend": "hmm",
        "artifacts": {
            "vision": "model/siglip.hmm",
            "prefill": "model/gemma_2b_prefill.hmm",
            "decode": "model/gemma_expert_300m_decode.hmm",
            "time_mlp": "model/time_mlp.hmm",
            "action_in_proj": "model/action_in_proj.hmm",
            "action_out_proj": "model/action_out_proj.hmm",
            "embedding": "model/embedding.pt",
        },
        "execution": ["vision", "prefill", "decode", "time_mlp", "action_in_proj", "action_out_proj"],
    }
    (tmp_path / HMM_MANIFEST_BASENAME).write_text(json.dumps(manifest), encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in (
        "siglip",
        "gemma_2b_prefill",
        "gemma_expert_300m_decode",
        "time_mlp",
        "action_in_proj",
        "action_out_proj",
    ):
        (model_dir / f"{name}.hmm").write_bytes(b"hmm")
    (model_dir / "embedding.pt").write_bytes(b"pt")
    return tmp_path


def test_pi05_adapter_accepts_hmm_backend():
    adapter = PI05CompiledAdapter.from_config(_pi05_config(), "hmm")
    assert adapter.policy_type == "pi05"
    assert adapter.uses_action_chunking is True
    assert adapter.get_chunk_size() == 30


def test_pi05_adapter_still_accepts_ascend_om():
    adapter = PI05CompiledAdapter.from_config(_pi05_config(), "ascend_om")
    assert adapter.policy_type == "pi05"


def test_pi05_adapter_rejects_unsupported_backend():
    with pytest.raises(ValueError, match="does not support PI05"):
        PI05CompiledAdapter.from_config(_pi05_config(), "rknn")


def test_create_runtime_session_hmm_pi05_dispatch():
    session = create_runtime_session("hmm", {"type": "pi05"})
    assert session.__class__.__name__ == "PI05HMMRuntimeSession"


def test_create_runtime_session_hmm_non_pi05_stays_single_module():
    # ACT (and anything non-pi05) must keep using the single-module HMMRuntimeSession.
    session = create_runtime_session("hmm", {"type": "act"})
    assert session.__class__.__name__ == "HMMRuntimeSession"


def test_resolve_pi05_hmm_paths(tmp_path):
    policy = _write_pi05_policy(tmp_path, _pi05_config())
    manifest = load_compiled_manifest(str(policy), "hmm", "pi05")
    vision, prefill, decode, time_mlp, action_in_proj, action_out_proj, embedding = resolve_pi05_hmm_paths(
        str(policy), manifest=manifest
    )
    assert vision.name == "siglip.hmm"
    assert prefill.name == "gemma_2b_prefill.hmm"
    assert decode.name == "gemma_expert_300m_decode.hmm"
    assert time_mlp.name == "time_mlp.hmm"
    assert action_in_proj.name == "action_in_proj.hmm"
    assert action_out_proj.name == "action_out_proj.hmm"
    assert embedding.name == "embedding.pt"


def test_resolve_pi05_hmm_paths_requires_full_execution(tmp_path):
    # An incomplete execution list must be rejected by require_execution.
    policy = _write_pi05_policy(tmp_path, _pi05_config())
    bad_manifest_path = tmp_path / HMM_MANIFEST_BASENAME
    data = json.loads(bad_manifest_path.read_text(encoding="utf-8"))
    data["execution"] = ["vision", "prefill"]  # incomplete
    bad_manifest_path.write_text(json.dumps(data), encoding="utf-8")
    manifest = load_compiled_manifest(str(policy), "hmm", "pi05")
    with pytest.raises(ValueError, match="execution must be"):
        resolve_pi05_hmm_paths(str(policy), manifest=manifest)


# ---------------------------------------------------------------------------
# Orchestration test: mock tcim_lite with 6 modules and confirm forward()
# wires KV-cache handoff + runs the denoise loop without hitting real hardware.
# ---------------------------------------------------------------------------
class _FakeOutput:
    def __init__(self, data):
        self._data = np.asarray(data, dtype=np.float32)

    def numpy(self):
        return self._data


@dataclass
class _FakeTensorInfo:
    shape: list


class _FakeModule:
    def __init__(self, output_name, output_shape, input_specs=None):
        self._output_name = output_name
        self._output_shape = output_shape
        # input_specs: dict name -> shape
        self._input_specs = input_specs or {}
        self.set_inputs: dict[str, np.ndarray] = {}
        self.dev_inputs_handled: list[str] = []
        self.runs = 0

    def get_num_inputs(self):
        return len(self._input_specs)

    def get_input_name(self, idx):
        return list(self._input_specs)[idx]

    def get_input_info(self, name):
        return _FakeTensorInfo(list(self._input_specs[name]))

    def set_input(self, name, data):
        self.set_inputs[name] = np.asarray(data)

    def get_dev_input(self, name):
        self.dev_inputs_handled.append(name)
        return f"dev_ptr:{name}"

    def run(self):
        self.runs += 1

    def sync(self):
        pass

    def get_num_outputs(self):
        return 1

    def get_output_name(self, idx):
        del idx
        return self._output_name

    def get_output_info(self, name):
        del name
        return _FakeTensorInfo(list(self._output_shape))

    def get_output(self, name):
        assert name == self._output_name
        return _FakeOutput(np.zeros(self._output_shape, dtype=np.float16))


def _install_fake_tcim_lite(monkeypatch, modules: dict[str, _FakeModule]):
    runtime = types.ModuleType("tcim_lite.runtime")
    load_calls: list[str] = []

    def _load(path, *args, **kwargs):
        load_calls.append(str(path))
        for key, mod in modules.items():
            if str(path).endswith(key):
                return mod
        raise KeyError(f"no fake module for {path}")

    runtime.load = _load
    api = types.ModuleType("tcim_lite")
    api.runtime = runtime
    monkeypatch.setitem(sys.modules, "tcim_lite", api)
    monkeypatch.setitem(sys.modules, "tcim_lite.runtime", runtime)
    return load_calls


def test_pi05_hmm_model_forward_orchestration(tmp_path, monkeypatch):
    # Build the 6 fake modules with plausible shapes (matches pi05 demo).
    vision = _FakeModule("image_features", [1, 256, 2048], {"pixel_values": [1, 3, 224, 224]})
    # prefill has many inputs incl. model_layer...cache slots (for KV handoff) + attention_mask
    prefill_inputs = {
        "input_1": [1, 968, 2048],
        "valid_length": [1],
        "current_length": [1],
        "attention_mask": [1, 1, 968, 2048],
    }
    for i in range(4):
        prefill_inputs[f"model_layer_{i}_cache"] = [1, 1, 1024]
    prefill = _FakeModule("past_kv", [1, 968, 2048], prefill_inputs)
    decode = _FakeModule(
        "suffix_out",
        [1, 50, 1024],
        {
            "input_1": [1, 50, 1024],
            "valid_length": [1],
            "current_length": [1],
            "cond": [1, 1024],
            "attention_mask": [1, 1, 50, 2048],
        },
    )
    time_mlp = _FakeModule("time_mlp_out", [1, 1024], {"time_emb": [1, 1024]})
    action_in_proj = _FakeModule("action_in_proj_out", [1, 50, 1024], {"action_in": [1, 50, 32]})
    action_out_proj = _FakeModule("action_out_proj_out", [1, 50, 32], {"action_out": [1, 50, 1024]})

    modules = {
        "siglip.hmm": vision,
        "gemma_2b_prefill.hmm": prefill,
        "gemma_expert_300m_decode.hmm": decode,
        "time_mlp.hmm": time_mlp,
        "action_in_proj.hmm": action_in_proj,
        "action_out_proj.hmm": action_out_proj,
    }
    _install_fake_tcim_lite(monkeypatch, modules)

    # embedding.pt: a small embedding weight to satisfy torch.load.
    import torch as _torch

    embedding_path = tmp_path / "embedding.pt"
    _torch.save({"weight": _torch.zeros(4, 2048)}, embedding_path)

    @dataclass
    class Cfg:
        chunk_size: int = 30
        max_action_dim: int = 7
        num_inference_steps: int = 2

    from inference_service.core.hmm.pi05.PI05HMMModel import PI05HMMModel

    model = PI05HMMModel(
        vision_path=str(tmp_path / "siglip.hmm"),
        prefill_path=str(tmp_path / "gemma_2b_prefill.hmm"),
        decode_path=str(tmp_path / "gemma_expert_300m_decode.hmm"),
        time_mlp_path=str(tmp_path / "time_mlp.hmm"),
        action_in_proj_path=str(tmp_path / "action_in_proj.hmm"),
        action_out_proj_path=str(tmp_path / "action_out_proj.hmm"),
        embedding_path=str(embedding_path),
        config=Cfg(),
    )

    # KV-cache handoff happens at __init__: the prefill module is the *source*
    # (its get_dev_input is read for every model_layer_*cache slot) and decode
    # is the *sink* (set_input binds those dev pointers).
    assert prefill.dev_inputs_handled, "prefill did not expose KV-cache dev inputs"
    assert all("cache" in n for n in prefill.dev_inputs_handled)
    handoff_cache_names = [n for n in decode.set_inputs if "cache" in n]
    assert handoff_cache_names, "decode did not receive KV-cache handoff from prefill"
    assert all(str(decode.set_inputs[n]).startswith("dev_ptr:") for n in handoff_cache_names)

    images = [np.zeros((1, 3, 224, 224), dtype=np.float32)]
    tokens = np.zeros((1, 8), dtype=np.int64)
    masks = np.ones((1, 8), dtype=bool)
    prefix_mask = np.zeros((1, 1, 264, 2048), dtype=np.float32)  # 256 img + 8 lang

    action = model.forward(images=images, tokens=tokens, masks=masks, prefix_att_2d_masks_4d=prefix_mask)

    # forward returns a batched float32 tensor (B, chunk_size, action_dim); the
    # PI05 adapter squeezes the leading dim downstream (decode_outputs).
    assert isinstance(action, torch.Tensor)
    assert action.dtype == torch.float32
    assert action.shape == (1, 30, 7)
    assert torch.isfinite(action).all()

    # prefill ran once; decode ran once per denoise step.
    assert prefill.runs == 1
    assert decode.runs == Cfg.num_inference_steps
    assert action_in_proj.runs == Cfg.num_inference_steps
    assert time_mlp.runs == Cfg.num_inference_steps
    assert action_out_proj.runs == Cfg.num_inference_steps
