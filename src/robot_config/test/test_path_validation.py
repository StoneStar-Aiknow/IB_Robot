"""Startup policy changes invalidate plans; frozen files cannot silently change."""

import copy
import hashlib

import pytest

from robot_config.loader import robot_config_digest
from robot_config.path_validation import resolve_path_validation
from robot_runtime.path_validation import normalize_path_validation, path_validation_identity
from robot_runtime.task_poses import task_pose_identity


@pytest.mark.parametrize(
    "value",
    [False, "off", {"hand": False}, {"hand": "false"}, {"body": "on"}, {"hands": "off"}, {"body": "prevalidated"}],
)
def test_policy_rejects_ambiguous_or_incomplete_configuration(value):
    with pytest.raises(ValueError, match="path_validation"):
        normalize_path_validation(value)


def test_default_and_explicit_online_have_same_identity_but_off_does_not():
    base = {"pose_composer": {"mode": "fixed"}}
    online = copy.deepcopy(base)
    online["pose_composer"]["path_validation"] = {"hand": "online", "body": "online"}
    off = copy.deepcopy(base)
    off["pose_composer"]["path_validation"] = {"hand": "off"}
    assert robot_config_digest(base) == robot_config_digest(online)
    assert robot_config_digest(base) != robot_config_digest(off)
    assert task_pose_identity(None, base["pose_composer"]) != task_pose_identity(None, off["pose_composer"])
    assert path_validation_identity({}) == {}
    with pytest.raises(ValueError, match="fixed"):
        path_validation_identity({"path_validation": {"hand": "off"}})


def test_certificate_resolution_freezes_bytes_and_excludes_deployment_path_from_identity(tmp_path):
    artifact = tmp_path / "certificate.json"
    artifact.write_text("{}")
    robot = {
        "pose_composer": {
            "mode": "fixed",
            "path_validation": {"hand": "prevalidated", "certificate_file": "certificate.json"},
        }
    }
    resolve_path_validation(robot, tmp_path / "robot.yaml")
    policy = robot["pose_composer"]["path_validation"]
    assert policy["certificate_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    changed = copy.deepcopy(robot)
    changed["pose_composer"]["path_validation"]["certificate_file"] = "/another/host/certificate.json"
    assert robot_config_digest(robot) == robot_config_digest(changed)
    artifact.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="changed"):
        resolve_path_validation(robot, tmp_path / "robot.yaml")
