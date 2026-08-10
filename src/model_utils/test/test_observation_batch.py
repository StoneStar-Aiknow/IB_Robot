from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from model_utils.observation_batch import (
    FieldSpec,
    extract_lerobot_observations,
    generate_random_observations,
    load_observation_batch,
    save_observation_batch,
    select_dataset_indices,
)


def test_random_generation_is_deterministic_and_inclusive():
    fields = [FieldSpec("state", (3,), "float32", -1, 1), FieldSpec("flag", (1,), "uint8", 2, 2)]
    first = generate_random_observations(fields, 4, seed=7)
    second = generate_random_observations(fields, 4, seed=7)
    assert all(np.array_equal(a["state"], b["state"]) for a, b in zip(first, second, strict=True))
    assert all(np.array_equal(sample["flag"], [2]) for sample in first)


def test_legacy_json_preserves_strings(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"samples": [{"task": "pick", "state": [1, 2]}]}), encoding="utf-8")
    batch = load_observation_batch(path)
    assert batch[0] == {"task": "pick", "state": [1, 2]}
    assert batch.sample_provenance == {}


def test_safetensors_round_trip_canonicalizes_image_and_task(tmp_path):
    pytest.importorskip("safetensors")
    path = tmp_path / "batch.safetensors"
    samples = [
        {"observation.image": np.ones((3, 2, 4), dtype=np.float32), "state": np.array([1.0]), "task": "pick"},
        {"observation.image": np.zeros((3, 2, 4), dtype=np.float32), "state": np.array([2.0]), "task": "place"},
    ]
    specs = [
        FieldSpec("observation.image", (3, 2, 4), "float32", semantic="image", layout="CHW"),
        FieldSpec("state", (1,), "float64"),
    ]
    sample_provenance = {"dataset_index": [4, 8], "episode_index": [1, 2], "frame_index": [10, 20]}
    save_observation_batch(
        path,
        samples,
        field_specs=specs,
        provenance={"seed": 3},
        sample_provenance=sample_provenance,
    )
    batch = load_observation_batch(path)
    assert batch[0]["observation.image"].shape == (2, 4, 3)
    assert batch[0]["observation.image"].dtype == np.uint8
    assert batch[0]["task"] == "pick"
    assert batch.provenance == {"seed": 3}
    for name, values in sample_provenance.items():
        np.testing.assert_array_equal(batch.sample_provenance[name], values)
    with pytest.raises(FileExistsError):
        save_observation_batch(path, samples, field_specs=specs)


def test_chw_and_hwc_images_are_canonicalized_equivalently(tmp_path):
    pytest.importorskip("safetensors")
    path = tmp_path / "equivalent.safetensors"
    hwc = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    chw = np.moveaxis(hwc, -1, 0)
    save_observation_batch(
        path,
        [{"from_chw": chw, "from_hwc": hwc}],
        field_specs=[
            FieldSpec("from_chw", chw.shape, "uint8", semantic="image", layout="CHW"),
            FieldSpec("from_hwc", hwc.shape, "uint8", semantic="image", layout="HWC"),
        ],
    )

    batch = load_observation_batch(path)
    np.testing.assert_array_equal(batch[0]["from_chw"], hwc)
    np.testing.assert_array_equal(batch[0]["from_hwc"], hwc)


def test_one_channel_chw_and_hwc_images_are_supported(tmp_path):
    pytest.importorskip("safetensors")
    path = tmp_path / "one-channel.safetensors"
    hwc = np.arange(8, dtype=np.uint8).reshape(2, 4, 1)
    chw = np.moveaxis(hwc, -1, 0)
    save_observation_batch(
        path,
        [{"chw": chw, "hwc": hwc}],
        field_specs=[
            FieldSpec("chw", chw.shape, "uint8", semantic="image", layout="CHW"),
            FieldSpec("hwc", hwc.shape, "uint8", semantic="image", layout="HWC"),
        ],
    )

    batch = load_observation_batch(path)
    assert batch[0]["chw"].shape == (2, 4, 1)
    np.testing.assert_array_equal(batch[0]["chw"], batch[0]["hwc"])


@pytest.mark.parametrize("shape", [(3, 224, 3), (3, 3, 3)])
def test_ambiguous_image_shapes_follow_declared_layout(tmp_path, shape):
    pytest.importorskip("safetensors")
    path = tmp_path / f"ambiguous-{shape[1]}.safetensors"
    values = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    save_observation_batch(
        path,
        [{"chw": values, "hwc": values}],
        field_specs=[
            FieldSpec("chw", shape, "uint8", semantic="image", layout="CHW"),
            FieldSpec("hwc", shape, "uint8", semantic="image", layout="HWC"),
        ],
    )

    batch = load_observation_batch(path)
    np.testing.assert_array_equal(batch[0]["chw"], np.moveaxis(values, 0, -1))
    np.testing.assert_array_equal(batch[0]["hwc"], values)


def test_image_spec_requires_layout(tmp_path):
    pytest.importorskip("safetensors")
    with pytest.raises(ValueError, match="must declare layout"):
        save_observation_batch(
            tmp_path / "missing-layout.safetensors",
            [{"pixels": np.zeros((3, 2, 4), dtype=np.float32)}],
            field_specs=[FieldSpec("pixels", (3, 2, 4), "float32", semantic="image")],
        )


def test_save_rejects_declared_shape_mismatch(tmp_path):
    pytest.importorskip("safetensors")
    with pytest.raises(ValueError, match="does not match declared shape/dtype"):
        save_observation_batch(
            tmp_path / "shape-mismatch.safetensors",
            [{"state": np.zeros(3, dtype=np.float32)}],
            field_specs=[FieldSpec("state", (4,), "float32")],
        )


def test_save_requires_a_spec_for_every_numeric_field(tmp_path):
    pytest.importorskip("safetensors")
    with pytest.raises(ValueError, match="requires an explicit field spec"):
        save_observation_batch(
            tmp_path / "missing-spec.safetensors",
            [{"state": np.zeros(3, dtype=np.float32)}],
            field_specs=[],
        )


def test_stratified_selection_covers_episodes_and_has_unique_frames():
    groups = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
    selected = select_dataset_indices(groups, 5, seed=4)
    assert len(selected) == len(set(selected)) == 5
    assert {next(ep for ep, frames in groups.items() if index in frames) for index in selected} == set(groups)


def _install_fake_lerobot(monkeypatch, items):
    class FakeDataset:
        def __init__(self, **_kwargs):
            self.meta = SimpleNamespace(episodes=[{"dataset_from_index": 0, "dataset_to_index": len(items)}])

        def __len__(self):
            return len(items)

        def __getitem__(self, index):
            return items[index]

    lerobot = ModuleType("lerobot")
    lerobot.__path__ = []
    datasets = ModuleType("lerobot.datasets")
    datasets.__path__ = []
    dataset_module = ModuleType("lerobot.datasets.lerobot_dataset")
    dataset_module.LeRobotDataset = FakeDataset
    monkeypatch.setitem(sys.modules, "lerobot", lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", dataset_module)


def _write_policy_config(tmp_path):
    policy_path = tmp_path / "policy"
    policy_path.mkdir()
    (policy_path / "config.json").write_text(
        json.dumps(
            {
                "input_features": {
                    "observation.state": {"type": "STATE", "shape": [2]},
                    "camera": {"type": "VISUAL", "shape": [3, 2, 4]},
                }
            }
        ),
        encoding="utf-8",
    )
    return policy_path


def test_dataset_explicit_fields_are_a_strict_whitelist(tmp_path, monkeypatch):
    item = {
        "observation.state": np.array([1, 2], dtype=np.float32),
        "camera": np.zeros((3, 2, 4), dtype=np.float32),
        "task": "pick",
        "episode_index": 3,
        "frame_index": 4,
        "index": 5,
    }
    _install_fake_lerobot(monkeypatch, [item])
    policy_path = _write_policy_config(tmp_path)

    samples, specs, _, _ = extract_lerobot_observations(
        tmp_path / "dataset",
        1,
        policy_path=policy_path,
        fields=["observation.state"],
    )
    assert set(samples[0]) == {"observation.state"}
    assert [spec.name for spec in specs] == ["observation.state"]

    samples, specs, _, _ = extract_lerobot_observations(
        tmp_path / "dataset",
        1,
        policy_path=policy_path,
        fields=["task"],
    )
    assert samples == [{"task": "pick"}]
    assert specs == []


def test_dataset_default_fields_come_from_policy_and_include_task(tmp_path, monkeypatch):
    item = {
        "observation.state": np.array([1, 2], dtype=np.float32),
        "camera": np.zeros((3, 2, 4), dtype=np.float32),
        "task": "pick",
        "episode_index": 3,
        "frame_index": 4,
        "index": 5,
    }
    _install_fake_lerobot(monkeypatch, [item])
    policy_path = _write_policy_config(tmp_path)

    samples, specs, _, _ = extract_lerobot_observations(
        tmp_path / "dataset",
        1,
        policy_path=policy_path,
    )

    assert set(samples[0]) == {"observation.state", "camera", "task"}
    assert [(spec.name, spec.semantic, spec.layout) for spec in specs] == [
        ("observation.state", "tensor", ""),
        ("camera", "image", "CHW"),
    ]


def test_dataset_arrays_must_match_policy_shape(tmp_path, monkeypatch):
    item = {
        "observation.state": np.array([1, 2], dtype=np.float32),
        "camera": np.zeros((2, 4, 3), dtype=np.float32),
        "episode_index": 0,
        "frame_index": 0,
        "index": 0,
    }
    _install_fake_lerobot(monkeypatch, [item])

    with pytest.raises(ValueError, match="does not match declared shape/dtype"):
        extract_lerobot_observations(
            tmp_path / "dataset",
            1,
            policy_path=_write_policy_config(tmp_path),
        )
