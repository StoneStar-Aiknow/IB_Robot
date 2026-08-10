from __future__ import annotations

import json

import numpy as np
import pytest

from model_utils import observation_batch_cli
from model_utils.observation_batch import FieldSpec, save_observation_batch


def test_parse_field():
    field = observation_batch_cli.parse_field("state=2x3,float32,-1,1")
    assert field.shape == (2, 3)
    assert field.minimum == -1

    image = observation_batch_cli.parse_field("camera=3x2x4,float32,0,1,image,CHW")
    assert image.semantic == "image"
    assert image.layout == "CHW"


def test_parse_field_rejects_invalid_syntax():
    with pytest.raises(Exception, match="NAME=SHAPE"):
        observation_batch_cli.parse_field("state:3")


def test_random_and_inspect_cli(tmp_path, capsys):
    pytest.importorskip("safetensors")
    output = tmp_path / "batch.safetensors"
    assert (
        observation_batch_cli.main(
            [
                "random",
                "--output",
                str(output),
                "--samples",
                "2",
                "--field",
                "state=3,float32,-1,1",
                "--field",
                "pixels=3x2x4,float32,0,1,image,CHW",
            ]
        )
        == 0
    )
    assert observation_batch_cli.main(["inspect", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["samples"] == 2
    assert report["fields"]["state"]["shape"] == [3]
    assert report["fields"]["pixels"]["shape"] == [2, 4, 3]
    assert report["fields"]["pixels"]["semantic"] == "image"
    assert report["sample_provenance"] == {}


def test_dataset_cli_passes_policy_specs_to_save(tmp_path, monkeypatch):
    captured = {}
    spec = FieldSpec("state", (2,), "float32")

    def extract(root, samples, **kwargs):
        assert root == "dataset"
        assert samples == 1
        assert kwargs["policy_path"] == "policy"
        return [{"state": np.zeros(2, dtype=np.float32)}], [spec], {"source": "test"}, {"dataset_index": [2]}

    def save(path, samples, **kwargs):
        captured.update(path=path, samples=samples, kwargs=kwargs)

    monkeypatch.setattr(observation_batch_cli, "extract_lerobot_observations", extract)
    monkeypatch.setattr(observation_batch_cli, "save_observation_batch", save)

    assert (
        observation_batch_cli.main(
            [
                "dataset",
                "--dataset-root",
                "dataset",
                "--policy-path",
                "policy",
                "--output",
                str(tmp_path / "batch.safetensors"),
                "--samples",
                "1",
            ]
        )
        == 0
    )
    assert captured["kwargs"]["field_specs"] == [spec]
    assert captured["kwargs"]["sample_provenance"] == {"dataset_index": [2]}


def test_inspect_summarizes_sample_provenance(tmp_path, capsys):
    pytest.importorskip("safetensors")
    output = tmp_path / "provenance.safetensors"
    save_observation_batch(
        output,
        [{"state": np.zeros(2, dtype=np.float32)}, {"state": np.ones(2, dtype=np.float32)}],
        field_specs=[FieldSpec("state", (2,), "float32")],
        sample_provenance={"dataset_index": [7, 11], "frame_index": [2, 5]},
    )

    assert observation_batch_cli.main(["inspect", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["sample_provenance"] == {
        "dataset_index": {"dtype": "int64", "shape": [2], "min": 7, "max": 11},
        "frame_index": {"dtype": "int64", "shape": [2], "min": 2, "max": 5},
    }
