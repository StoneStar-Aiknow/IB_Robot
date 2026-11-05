"""Fixed-library contents are bound to the canonical startup identity."""

import copy
import hashlib

import pytest

from robot_config.fixed_poses import resolve_fixed_poses
from robot_config.loader import robot_config_digest


def test_relative_artifact_is_bound_and_modified_snapshot_is_rejected(tmp_path):
    artifact = tmp_path / "poses.json"
    artifact.write_text('{"revision": 1}')
    original = {"pose_composer": {"mode": "fixed", "fixed_library": "poses.json"}}
    first = copy.deepcopy(original)
    resolve_fixed_poses(first, tmp_path / "robot.yaml")
    config = first["pose_composer"]
    assert config["fixed_library"] == str(artifact)
    assert config["fixed_library_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    artifact.write_text('{"revision": 2}')
    with pytest.raises(ValueError, match="changed"):
        resolve_fixed_poses(first, tmp_path / "robot.yaml")
    second = copy.deepcopy(original)
    resolve_fixed_poses(second, tmp_path / "robot.yaml")
    assert robot_config_digest(first) != robot_config_digest(second)


def test_fixed_mode_requires_an_existing_artifact(tmp_path):
    with pytest.raises(ValueError, match="requires fixed_library"):
        resolve_fixed_poses({"pose_composer": {"mode": "fixed"}}, tmp_path / "robot.yaml")
    with pytest.raises(FileNotFoundError):
        resolve_fixed_poses(
            {"pose_composer": {"mode": "fixed", "fixed_library": "missing.json"}}, tmp_path / "robot.yaml"
        )


def test_generated_mode_does_not_load_fixed_artifacts(tmp_path):
    config = {"pose_composer": {"mode": "generated"}}
    resolve_fixed_poses(config, tmp_path / "robot.yaml")
    assert config == {"pose_composer": {"mode": "generated"}}
