from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from model_utils.pi05_om_dump import dump_pi05_om


class _Engine:
    policy_type = "pi05"
    backend_type = "ascend"
    nominal_chunk_size = 2
    max_action_dimension = 4

    def __init__(self, created, **kwargs):
        self.created = created
        self.created.append(kwargs)
        self.calls = []
        self.closed = False

    def __call__(self, batch, **kwargs):
        self.calls.append((batch, kwargs))
        noise = kwargs["control_inputs"]["noise"]
        capture = self.created[-1]["runtime_options"]["diagnostic_capture"]
        capture("past_kv_tensor", np.array([[3.0, 4.0]], dtype=np.float16))
        capture("prefix_pad_masks", np.array([[True, False]]))
        capture("x_t_step00", noise - 1.0)
        return SimpleNamespace(raw_action=noise + 1.0, action=noise + 2.0)

    def close(self):
        self.closed = True


def _batch_file(tmp_path):
    path = tmp_path / "batches.json"
    path.write_text(
        json.dumps([{"observation.state": [1, 2], "observation.images.top_view": [[[255, 0, 0]]]}]),
        encoding="utf-8",
    )
    return path


def test_dump_forwards_exact_deployment_saves_deterministic_values_and_closes(tmp_path):
    created = []
    engines = []

    def factory(**kwargs):
        capture = kwargs.pop("diagnostic_capture")
        kwargs["runtime_options"] = {"diagnostic_capture": capture}
        engine = _Engine(created, **kwargs)
        engines.append(engine)
        return engine

    output = dump_pi05_om(
        policy_path=str(tmp_path / "bundle"),
        deployment="site.ascend-310p3",
        batch_path=str(_batch_file(tmp_path)),
        batch_index=0,
        output_dir=str(tmp_path / "dump"),
        task="pick",
        seed=17,
        engine_factory=factory,
    )

    assert created[0]["deployment"] == "site.ascend-310p3"
    assert set(created[0]["runtime_options"]) == {"diagnostic_capture"}
    assert engines[0].closed is True
    expected = torch.Generator(device="cpu").manual_seed(17)
    expected_noise = torch.randn((1, 2, 4), generator=expected, dtype=torch.float32).numpy()
    np.testing.assert_array_equal(np.load(output / "ae_in_noise.npy"), expected_noise)
    np.testing.assert_array_equal(np.load(output / "raw_action.npy"), expected_noise + 1.0)
    np.testing.assert_array_equal(np.load(output / "action.npy"), expected_noise + 2.0)
    np.testing.assert_array_equal(np.load(output / "past_kv_tensor.npy"), np.array([[3.0, 4.0]], dtype=np.float16))
    np.testing.assert_array_equal(np.load(output / "x_t_step00.npy"), expected_noise - 1.0)
    np.testing.assert_array_equal(
        np.load(output / "input_observation.images.top.npy"),
        np.array([[[[1]], [[0]], [[0]]]]),
    )
    index = json.loads((output / "diagnostic_capture.json").read_text(encoding="utf-8"))
    assert index["deployment"] == "site.ascend-310p3"
    assert index["values"]["raw_action"]["shape"] == [1, 2, 4]


def test_dump_closes_engine_when_inference_fails(tmp_path):
    engines = []

    class FailingEngine(_Engine):
        def __call__(self, batch, **kwargs):
            raise RuntimeError("inference failed")

    def factory(**kwargs):
        capture = kwargs.pop("diagnostic_capture")
        kwargs["runtime_options"] = {"diagnostic_capture": capture}
        engine = FailingEngine([], **kwargs)
        engines.append(engine)
        return engine

    with pytest.raises(RuntimeError, match="inference failed"):
        dump_pi05_om(
            policy_path=str(tmp_path / "bundle"),
            deployment="ascend",
            batch_path=str(_batch_file(tmp_path)),
            batch_index=0,
            output_dir=str(tmp_path / "dump"),
            engine_factory=factory,
        )

    assert engines[0].closed is True
