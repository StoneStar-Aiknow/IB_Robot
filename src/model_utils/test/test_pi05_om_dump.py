from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import model_utils.pi05_om_dump as pi05_om_dump_module
from model_utils.observation_batch import FieldSpec, save_observation_batch
from model_utils.pi05_om_dump import dump_pi05_om, dump_pi05_om_batches


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


def _batch_file(tmp_path, count=1, task=None):
    path = tmp_path / "batches.json"
    samples = [{"observation.state": [1, 2], "observation.images.top_view": [[[255, 0, 0]]]} for _ in range(count)]
    if task is not None:
        for sample in samples:
            sample["task"] = task
    path.write_text(
        json.dumps(samples),
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


@pytest.mark.parametrize("failure", ["inference", "index", "close"])
def test_dump_failure_leaves_previous_root_unchanged_and_removes_staging(tmp_path, monkeypatch, failure):
    engines = []
    output_root = tmp_path / "dump"
    output_root.mkdir()
    (output_root / "completed.txt").write_text("previous", encoding="utf-8")

    class FailingEngine(_Engine):
        def __call__(self, batch, **kwargs):
            if failure == "inference":
                raise RuntimeError("inference failed")
            return super().__call__(batch, **kwargs)

        def close(self):
            self.closed = True
            if failure == "close":
                raise RuntimeError("close failed")

    if failure == "index":

        def fail_index(self, metadata):
            raise RuntimeError("index failed")

        monkeypatch.setattr(DiagnosticCapture, "write_index", fail_index)

    def factory(**kwargs):
        capture = kwargs.pop("diagnostic_capture")
        kwargs["runtime_options"] = {"diagnostic_capture": capture}
        engine = FailingEngine([], **kwargs)
        engines.append(engine)
        return engine

    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        dump_pi05_om(
            policy_path=str(tmp_path / "bundle"),
            deployment="ascend",
            batch_path=str(_batch_file(tmp_path)),
            batch_index=0,
            output_dir=str(output_root),
            engine_factory=factory,
        )

    assert engines[0].closed is True
    assert [path.name for path in output_root.iterdir()] == ["completed.txt"]
    assert (output_root / "completed.txt").read_text(encoding="utf-8") == "previous"
    assert not list(tmp_path.glob(".dump.staging-*"))


def test_dump_atomic_exchange_failure_keeps_previous_root(tmp_path, monkeypatch):
    output_root = tmp_path / "dump"
    output_root.mkdir()
    (output_root / "completed.txt").write_text("previous", encoding="utf-8")

    def fail_exchange(_left, _right):
        raise RuntimeError("exchange failed")

    monkeypatch.setattr(pi05_om_dump_module, "_rename_exchange", fail_exchange)

    with pytest.raises(RuntimeError, match="exchange failed"):
        dump_pi05_om(
            policy_path=str(tmp_path / "bundle"),
            deployment="ascend",
            batch_path=str(_batch_file(tmp_path)),
            batch_index=0,
            output_dir=str(output_root),
            engine_factory=lambda **kwargs: _Engine(
                [], runtime_options={"diagnostic_capture": kwargs["diagnostic_capture"]}
            ),
        )

    assert (output_root / "completed.txt").read_text(encoding="utf-8") == "previous"
    assert not list(tmp_path.glob(".dump.staging-*"))


def test_dump_accepts_safetensors_batch(tmp_path):
    pytest.importorskip("safetensors")
    batch_path = tmp_path / "batches.safetensors"
    save_observation_batch(
        batch_path,
        [
            {
                "observation.state": np.array([1, 2], dtype=np.float32),
                "observation.images.top_view": np.array([[[255, 0, 0]]], dtype=np.uint8),
            }
        ],
        field_specs=[
            FieldSpec("observation.state", (2,), "float32"),
            FieldSpec("observation.images.top_view", (1, 1, 3), "uint8", semantic="image", layout="HWC"),
        ],
    )
    engines = []

    def factory(**kwargs):
        capture = kwargs.pop("diagnostic_capture")
        kwargs["runtime_options"] = {"diagnostic_capture": capture}
        engine = _Engine([], **kwargs)
        engines.append(engine)
        return engine

    output = dump_pi05_om(
        policy_path=str(tmp_path / "bundle"),
        deployment="ascend",
        batch_path=str(batch_path),
        batch_index=0,
        output_dir=str(tmp_path / "dump-safetensors"),
        engine_factory=factory,
    )

    np.testing.assert_array_equal(
        np.load(output / "input_observation.images.top.npy"),
        np.array([[[[1]], [[0]], [[0]]]]),
    )
    assert engines[0].closed is True


def test_dump_batches_loads_once_reuses_one_engine_and_writes_sample_directories(tmp_path, monkeypatch):
    created = []
    engines = []
    load_calls = []
    real_load = pi05_om_dump_module.load_observation_batch

    def tracked_load(path):
        load_calls.append(path)
        return real_load(path)

    monkeypatch.setattr(pi05_om_dump_module, "load_observation_batch", tracked_load)

    def factory(**kwargs):
        capture = kwargs.pop("diagnostic_capture")
        kwargs["runtime_options"] = {"diagnostic_capture": capture}
        engine = _Engine(created, **kwargs)
        engines.append(engine)
        return engine

    outputs = dump_pi05_om_batches(
        policy_path=str(tmp_path / "bundle"),
        deployment="ascend",
        batch_path=str(_batch_file(tmp_path, count=2)),
        batch_indices=[0, 1],
        output_dir=str(tmp_path / "batch-dump"),
        engine_factory=factory,
    )

    assert len(engines) == 1
    assert len(load_calls) == 1
    assert len(engines[0].calls) == 2
    assert engines[0].closed is True
    assert [path.name for path in outputs] == ["sample_0000", "sample_0001"]
    assert all((path / "diagnostic_capture.json").is_file() for path in outputs)


def test_dump_batches_shorter_rerun_removes_old_samples_and_files(tmp_path):
    engines = []

    def factory(**kwargs):
        capture = kwargs.pop("diagnostic_capture")
        kwargs["runtime_options"] = {"diagnostic_capture": capture}
        engine = _Engine([], **kwargs)
        engines.append(engine)
        return engine

    output_root = tmp_path / "batch-dump"
    batch_path = _batch_file(tmp_path, count=2)
    dump_pi05_om_batches(
        policy_path=str(tmp_path / "bundle"),
        deployment="ascend",
        batch_path=str(batch_path),
        batch_indices=[0, 1],
        output_dir=str(output_root),
        engine_factory=factory,
    )
    (output_root / "sample_0000" / "velocity_step99.npy").write_bytes(b"stale")

    outputs = dump_pi05_om_batches(
        policy_path=str(tmp_path / "bundle"),
        deployment="ascend",
        batch_path=str(batch_path),
        batch_indices=[0],
        output_dir=str(output_root),
        engine_factory=factory,
    )

    assert outputs == [output_root / "sample_0000"]
    assert not (output_root / "sample_0001").exists()
    assert not (output_root / "sample_0000" / "velocity_step99.npy").exists()
    assert all(engine.closed for engine in engines)


def test_dump_task_override_replaces_stored_task_and_empty_override_preserves_it(tmp_path):
    engines = []

    def factory(**kwargs):
        capture = kwargs.pop("diagnostic_capture")
        kwargs["runtime_options"] = {"diagnostic_capture": capture}
        engine = _Engine([], **kwargs)
        engines.append(engine)
        return engine

    batch_path = _batch_file(tmp_path, task="stored task")
    dump_pi05_om(
        policy_path=str(tmp_path / "bundle"),
        deployment="ascend",
        batch_path=str(batch_path),
        batch_index=0,
        output_dir=str(tmp_path / "preserved"),
        engine_factory=factory,
    )
    dump_pi05_om(
        policy_path=str(tmp_path / "bundle"),
        deployment="ascend",
        batch_path=str(batch_path),
        batch_index=0,
        output_dir=str(tmp_path / "overridden"),
        task="explicit task",
        engine_factory=factory,
    )

    assert engines[0].calls[0][0]["task"] == "stored task"
    assert engines[1].calls[0][0]["task"] == "explicit task"


def test_diagnostic_capture_numbers_repeated_action_expert_trajectory_values(tmp_path):
    capture = DiagnosticCapture(tmp_path)
    initial = np.zeros((1, 2, 4), dtype=np.float16)
    next_state = np.ones((1, 2, 4), dtype=np.float16)

    capture("action_expert_in_noise", initial)
    capture("action_expert_in_time", np.array([1.0], dtype=np.float16))
    capture("action_expert_out_action", np.full_like(initial, 2.0))
    capture("action_expert_in_noise", next_state)
    capture("action_expert_in_time", np.array([0.5], dtype=np.float16))
    capture("action_expert_out_action", np.full_like(initial, 3.0))

    np.testing.assert_array_equal(np.load(tmp_path / "ae_in_noise.npy"), initial)
    np.testing.assert_array_equal(np.load(tmp_path / "x_t_step00.npy"), next_state)
    assert float(np.load(tmp_path / "ae_in_time_step00.npy")[0]) == 1.0
    assert float(np.load(tmp_path / "ae_in_time_step01.npy")[0]) == 0.5
    np.testing.assert_array_equal(np.load(tmp_path / "velocity_step00.npy"), np.full_like(initial, 2.0))
    np.testing.assert_array_equal(np.load(tmp_path / "velocity_step01.npy"), np.full_like(initial, 3.0))
