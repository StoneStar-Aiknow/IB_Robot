"""Startup-only validation shared by configuration, catalog and the X2 grasp backends."""

import pytest

from robot_runtime.hand_mounts import installed_hands
from robot_runtime.path_validation import normalize_path_validation, path_validation_identity, policy_digest
from robot_runtime.pose_backend import intersect_joint_limits
from robot_runtime.task_poses import pose_catalog


def test_installed_hands_are_explicit_and_ordered():
    assert installed_hands(["right", "left"]) == ("left", "right")
    assert installed_hands(()) == ()
    for value in ("left", ["left", "left"], ["middle"], [1], None):
        with pytest.raises(ValueError):
            installed_hands(value)


def test_path_validation_defaults_online_and_binds_policy_not_deployment_path():
    assert normalize_path_validation() == {"hand": "online", "body": "online"}
    sha = "a" * 64
    policy = {"hand": "prevalidated", "body": "off", "certificate_file": "/x/cert.json", "certificate_sha256": sha}
    identity = path_validation_identity({"mode": "fixed", "path_validation": policy})
    moved = path_validation_identity(
        {"mode": "fixed", "path_validation": {**policy, "certificate_file": "/elsewhere/cert.json"}}
    )
    assert identity == moved
    assert identity["path_validation_digest"] == policy_digest(
        {"hand": "prevalidated", "body": "off", "certificate_sha256": sha}
    )
    assert path_validation_identity(None) == {}


@pytest.mark.parametrize(
    "value",
    [
        {"hand": "maybe"},
        {"extra": 1},
        {"hand": "prevalidated"},
        {"certificate_sha256": "a" * 64},
        {"certificate_file": "/c", "certificate_sha256": "A" * 64},
        # Without its digest a certificate binds nothing: the identity drops the path.
        {"hand": "prevalidated", "certificate_file": "/a/cert.json"},
        {"certificate_file": "/a/cert.json"},
    ],
)
def test_path_validation_refuses_unsafe_or_incomplete_policies(value):
    with pytest.raises(ValueError):
        normalize_path_validation(value)


def test_a_different_certificate_content_gives_a_different_identity():
    def ident(path, sha):
        policy = {"hand": "prevalidated", "certificate_file": path, "certificate_sha256": sha}
        return path_validation_identity({"mode": "fixed", "path_validation": policy})

    assert ident("/a/cert.json", "a" * 64) != ident("/b/cert.json", "b" * 64)
    assert ident("/a/cert.json", "a" * 64) == ident("/b/cert.json", "a" * 64)


def test_path_validation_only_applies_to_fixed_poses():
    with pytest.raises(ValueError):
        path_validation_identity({"mode": "generated", "path_validation": {}})


def test_joint_limits_never_widen_runtime_travel():
    vendor = {"a": (-1.0, 1.0), "b": (0.0, 2.0)}
    assert intersect_joint_limits(("a", "b"), vendor) == {"a": (-1.0, 1.0), "b": (0.0, 2.0)}
    assert intersect_joint_limits(("a",), vendor, {"a": (-2.0, 0.5)}) == {"a": (-1.0, 0.5)}
    with pytest.raises(ValueError):
        intersect_joint_limits(("a",), vendor, {"a": (1.5, 2.0)})
    with pytest.raises(ValueError):
        intersect_joint_limits(("a", "b"), vendor, {"a": (-1.0, 1.0)})
    with pytest.raises(ValueError):
        intersect_joint_limits(("c",), vendor)


@pytest.mark.parametrize("record", [{"phrases": []}, {"user_message": "wave"}, "wave"])
def test_a_fixed_pose_without_its_message_or_phrases_is_a_configuration_error(tmp_path, record):
    import hashlib
    import json

    library = tmp_path / "poses.json"
    library.write_text(json.dumps({"poses": {"wave": record}}))
    composer = {
        "mode": "fixed",
        "fixed_library": str(library),
        "fixed_library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }
    with pytest.raises(ValueError, match="user_message and phrases"):
        pose_catalog(None, composer)
