from __future__ import annotations

import json
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import model_utils.loss_compare as loss_compare


class _FakeEngine:
    policy_type = "pi05"
    backend_type = "ascend"
    nominal_chunk_size = 2
    max_action_dimension = 8

    def __init__(self, *, stateful: bool = True) -> None:
        self.capabilities = SimpleNamespace(stateful=stateful, resettable=stateful)
        self.calls = []
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def __call__(
        self,
        batch,
        *,
        request_id=None,
        control_inputs=None,
        capture_raw_action=False,
    ):
        noise = None if control_inputs is None else control_inputs.get("noise")
        value = float(torch.as_tensor(noise).mean()) if noise is not None else float(len(self.calls) + 1)
        raw = torch.full((2, 6), value)
        final = raw + 10.0
        self.calls.append(
            {
                "batch": batch,
                "request_id": request_id,
                "control_inputs": control_inputs,
                "capture_raw_action": capture_raw_action,
            }
        )
        return SimpleNamespace(raw_action=raw, action=final)


def _args(tmp_path, *, generate_target: bool = False) -> Namespace:
    return Namespace(
        policy_path=str(tmp_path / "bundle"),
        deployment="ascend",
        model_dtype="native",
        batch_path=str(tmp_path / "batches.json"),
        target_path=str(tmp_path / "target.json"),
        raw_target_path=str(tmp_path / "target_raw.json"),
        noise_dir=str(tmp_path / "noise"),
        generate_target=generate_target,
        task="pick",
        seed=42,
        policy_type="pi05",
    )


def _utils(tmp_path, engine: _FakeEngine, *, generate_target: bool = False):
    utils = object.__new__(loss_compare.LossUtils)
    utils.args = _args(tmp_path, generate_target=generate_target)
    utils.engine = engine
    return utils


def test_forward_resets_stateful_pipeline_and_separates_raw_from_final_actions(monkeypatch, tmp_path):
    engine = _FakeEngine()
    utils = _utils(tmp_path, engine)
    utils.args.noise_dir = ""
    monkeypatch.setattr(loss_compare, "np", np)
    monkeypatch.setattr(loss_compare, "torch", torch)
    monkeypatch.setattr(loss_compare, "tqdm", lambda values, **_kwargs: values)

    outputs = utils.forward([{"observation.state": np.zeros(6)}, {"observation.state": np.ones(6)}])

    assert engine.reset_calls == 2
    assert [call["request_id"] for call in engine.calls] == ["loss-compare-0", "loss-compare-1"]
    assert all(call["capture_raw_action"] is True for call in engine.calls)
    assert torch.equal(utils._raw_preds[0], torch.full((2, 6), 1.0))
    assert torch.equal(outputs[0], torch.full((2, 6), 11.0))


def test_pi05_noise_self_check_uses_metadata_shape_and_resets_each_run(monkeypatch, tmp_path):
    engine = _FakeEngine()
    utils = _utils(tmp_path, engine)
    monkeypatch.setattr(loss_compare, "torch", torch)

    utils._assert_noise_effective({"observation.state": np.zeros(6)})

    assert engine.reset_calls == 3
    assert [tuple(call["control_inputs"]["noise"].shape) for call in engine.calls] == [(1, 2, 8)] * 3
    assert torch.equal(engine.calls[0]["control_inputs"]["noise"], engine.calls[2]["control_inputs"]["noise"])
    assert not torch.equal(engine.calls[0]["control_inputs"]["noise"], engine.calls[1]["control_inputs"]["noise"])


def test_generate_target_writes_final_and_raw_action_documents(monkeypatch, tmp_path):
    engine = _FakeEngine(stateful=False)
    utils = _utils(tmp_path, engine, generate_target=True)
    monkeypatch.setattr(loss_compare, "torch", torch)
    monkeypatch.setattr(loss_compare, "tqdm", lambda values, **_kwargs: values)
    monkeypatch.setattr(utils, "load_batches_as_tensors", lambda: [{"observation.state": np.zeros(6)}])
    monkeypatch.setattr(utils, "_assert_noise_effective", lambda _batch: None)
    monkeypatch.setattr(utils, "_resolve_noise", lambda _index: None)

    utils.generate_target()

    assert json.loads((tmp_path / "target.json").read_text(encoding="utf-8")) == [[[11.0] * 6, [11.0] * 6]]
    assert json.loads((tmp_path / "target_raw.json").read_text(encoding="utf-8")) == [[[1.0] * 6, [1.0] * 6]]


def test_pi05_noise_shape_requires_bundle_metadata(tmp_path):
    engine = _FakeEngine()
    engine.max_action_dimension = None
    utils = _utils(tmp_path, engine)

    with pytest.raises(RuntimeError, match="max_action_dim"):
        utils._pi05_noise_shape()
