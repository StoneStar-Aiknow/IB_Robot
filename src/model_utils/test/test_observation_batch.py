from __future__ import annotations

import json

import numpy as np
import pytest

from model_utils.observation_batch import (
    FieldSpec,
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
    assert load_observation_batch(path)[0] == {"task": "pick", "state": [1, 2]}


def test_safetensors_round_trip_canonicalizes_image_and_task(tmp_path):
    pytest.importorskip("safetensors")
    path = tmp_path / "batch.safetensors"
    samples = [
        {"observation.image": np.ones((3, 2, 4), dtype=np.float32), "state": np.array([1.0]), "task": "pick"},
        {"observation.image": np.zeros((3, 2, 4), dtype=np.float32), "state": np.array([2.0]), "task": "place"},
    ]
    save_observation_batch(path, samples, provenance={"seed": 3})
    batch = load_observation_batch(path)
    assert batch[0]["observation.image"].shape == (2, 4, 3)
    assert batch[0]["observation.image"].dtype == np.uint8
    assert batch[0]["task"] == "pick"
    assert batch.provenance == {"seed": 3}
    with pytest.raises(FileExistsError):
        save_observation_batch(path, samples)


def test_stratified_selection_covers_episodes_and_has_unique_frames():
    groups = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
    selected = select_dataset_indices(groups, 5, seed=4)
    assert len(selected) == len(set(selected)) == 5
    assert {next(ep for ep, frames in groups.items() if index in frames) for index in selected} == set(groups)
