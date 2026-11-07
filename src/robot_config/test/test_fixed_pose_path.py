"""Portable fixed artifacts resolve identically for compilation and live loading."""

import hashlib

import pytest

from robot_config.fixed_poses import fixed_library_path, resolve_fixed_poses


def test_workspace_artifact_is_hashed_and_frozen_after_source_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE", str(tmp_path))
    artifact = tmp_path / "commissioning/fixed_poses.json"
    artifact.parent.mkdir()
    artifact.write_text('{"artifact": "test"}')
    config = {"pose_composer": {"mode": "fixed", "fixed_library": "$(env WORKSPACE)/commissioning/fixed_poses.json"}}
    assert fixed_library_path(config, tmp_path / "robot.yaml") == artifact
    resolve_fixed_poses(config, tmp_path / "robot.yaml")
    assert config["pose_composer"]["fixed_library"] == str(artifact)
    assert config["pose_composer"]["fixed_library_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    monkeypatch.setenv("WORKSPACE", str(tmp_path / "different"))
    resolve_fixed_poses(config, tmp_path / "snapshot.yaml")
    artifact.write_text("modified")
    with pytest.raises(ValueError, match="changed"):
        resolve_fixed_poses(config, tmp_path / "snapshot.yaml")


def test_unset_workspace_fails_before_selecting_an_unintended_output(tmp_path, monkeypatch):
    monkeypatch.delenv("WORKSPACE", raising=False)
    config = {"pose_composer": {"fixed_library": "$(env WORKSPACE)/commissioning/fixed_poses.json"}}
    with pytest.raises(ValueError, match="WORKSPACE"):
        fixed_library_path(config, tmp_path / "robot.yaml")
