from __future__ import annotations

import json

import pytest

from model_utils import observation_batch_cli


def test_parse_field():
    field = observation_batch_cli.parse_field("state=2x3,float32,-1,1")
    assert field.shape == (2, 3)
    assert field.minimum == -1


def test_parse_field_rejects_invalid_syntax():
    with pytest.raises(Exception, match="NAME=SHAPE"):
        observation_batch_cli.parse_field("state:3")


def test_random_and_inspect_cli(tmp_path, capsys):
    pytest.importorskip("safetensors")
    output = tmp_path / "batch.safetensors"
    assert (
        observation_batch_cli.main(
            ["random", "--output", str(output), "--samples", "2", "--field", "state=3,float32,-1,1"]
        )
        == 0
    )
    assert observation_batch_cli.main(["inspect", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["samples"] == 2
    assert report["fields"]["state"]["shape"] == [3]
