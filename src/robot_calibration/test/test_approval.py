import tarfile

import pytest
import yaml

from robot_calibration.approval import approve

CANDIDATE_FILES = {
    "base_to_front_camera.candidate.yaml",
    "base_to_mid360.yaml",
    "front_camera_intrinsics.yaml",
}
CURRENT_FILES = {
    "base_to_front_camera.yaml",
    "base_to_mid360.yaml",
    "front_camera_intrinsics.yaml",
}


def _archive(
    tmp_path,
    serial: str,
    *,
    missing: str | None = None,
    malformed: str | None = None,
    intrinsics_override: tuple[str, object] | None = None,
):
    candidate = tmp_path / "base_to_front_camera.candidate.yaml"
    candidate_value = {
        "schema_version": "1.0",
        "calibration_version": "test-1",
        "status": "candidate",
        "device": {"name": "front_camera", "serial": serial},
        "transform": {
            "parent_frame": "base_link",
            "child_frame": "camera_front_optical_frame",
            "translation": [0.0, 0.0, 0.0],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }
    mid360 = tmp_path / "base_to_mid360.yaml"
    mid360_value = {
        "schema_version": "1.0",
        "calibration_version": "mount-1",
        "status": "candidate",
        "device": {"name": "MID-360", "serial": "unavailable"},
        "transform": {
            "parent_frame": "base_link",
            "child_frame": "body",
            "translation": [-0.08, 0.0, 0.2],
            "rotation_xyzw": [0.0, 0.0, 0.7071067811865475, 0.7071067811865476],
        },
    }
    intrinsics = tmp_path / "front_camera_intrinsics.yaml"
    intrinsics_value = {
        "schema_version": "1.0",
        "calibration_version": "camera-info-1",
        "status": "candidate",
        "device": {"name": "front_camera", "serial": serial},
        "camera_info": {
            "frame_id": "camera_front_optical_frame",
            "width": 848,
            "height": 480,
            "distortion_model": "plumb_bob",
            "d": [0.1, -0.2, 0.003, 0.004, 0.0],
            "k": [606.0, 0.0, 321.0, 0.0, 605.0, 254.0, 0.0, 0.0, 1.0],
            "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "p": [606.0, 0.0, 321.0, 0.0, 0.0, 605.0, 254.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        },
    }
    if malformed == candidate.name:
        candidate_value["transform"]["child_frame"] = "camera_front_link"
    elif malformed == mid360.name:
        mid360_value["transform"]["translation"] = [0.0, 0.0]
    elif malformed == intrinsics.name:
        intrinsics_value["camera_info"]["distortion_model"] = "equidistant"
    if intrinsics_override is not None:
        key, value = intrinsics_override
        intrinsics_value["camera_info"][key] = value
    for path, value in (
        (candidate, candidate_value),
        (mid360, mid360_value),
        (intrinsics, intrinsics_value),
    ):
        path.write_text(
            yaml.safe_dump(value, sort_keys=False),
            encoding="utf-8",
        )
    files = {
        candidate.name: candidate,
        mid360.name: mid360,
        intrinsics.name: intrinsics,
    }
    archive = tmp_path / "calib-1.candidate.tar"
    with tarfile.open(archive, "w") as output:
        for name, path in files.items():
            if name != missing:
                output.add(path, arcname=name)
    return archive


def test_candidate_archive_contains_exactly_three_calibration_artifacts(tmp_path):
    with tarfile.open(_archive(tmp_path, "123456789")) as source:
        assert set(source.getnames()) == CANDIDATE_FILES


def test_approve_installs_three_approved_current_files(tmp_path):
    current = approve(_archive(tmp_path, "123456789"), tmp_path / "production")

    assert current == tmp_path / "production/current/base_to_front_camera.yaml"
    assert {path.name for path in current.parent.iterdir()} == CURRENT_FILES
    assert not any(path.is_symlink() for path in current.parent.iterdir())
    values = {path.name: yaml.safe_load(path.read_text(encoding="utf-8")) for path in current.parent.iterdir()}
    assert all(value["status"] == "approved" for value in values.values())
    assert values["base_to_front_camera.yaml"]["device"]["serial"] == "123456789"


def test_approve_allows_unavailable_serial(tmp_path):
    current = approve(_archive(tmp_path, "unavailable"), tmp_path / "production")

    value = yaml.safe_load(current.read_text(encoding="utf-8"))
    assert value["status"] == "approved"
    assert value["device"]["serial"] == "unavailable"


def test_approve_replaces_legacy_symlink_without_modifying_target(tmp_path):
    production = tmp_path / "production"
    legacy_artifact = production / "artifacts/base_to_front_camera-old/base_to_front_camera.yaml"
    legacy_artifact.parent.mkdir(parents=True)
    legacy_bytes = b"legacy approved artifact\n"
    legacy_artifact.write_bytes(legacy_bytes)
    current = production / "current"
    current.mkdir()
    (current / "base_to_front_camera.yaml").symlink_to(legacy_artifact)

    approve(_archive(tmp_path, "123456789"), production)

    assert legacy_artifact.read_bytes() == legacy_bytes
    assert {path.name for path in current.iterdir()} == CURRENT_FILES
    assert all(path.is_file() and not path.is_symlink() for path in current.iterdir())


@pytest.mark.parametrize("missing", sorted(CANDIDATE_FILES))
def test_approve_rejects_missing_candidate(tmp_path, missing):
    with pytest.raises(ValueError, match=missing):
        approve(_archive(tmp_path, "123456789", missing=missing), tmp_path / "production")


@pytest.mark.parametrize("malformed", sorted(CANDIDATE_FILES))
def test_approve_rejects_semantically_malformed_candidate(tmp_path, malformed):
    with pytest.raises(ValueError):
        approve(_archive(tmp_path, "123456789", malformed=malformed), tmp_path / "production")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("frame_id", "camera_front_link"),
        ("width", 0),
        ("height", 0),
        ("distortion_model", "equidistant"),
        ("d", [0.0] * 4),
        ("k", [0.0] * 8),
        ("r", [0.0] * 8),
        ("p", [0.0] * 11),
        ("d", [0.0] * 4 + [float("inf")]),
        ("k", [0.0] * 8 + [float("inf")]),
        ("r", [0.0] * 8 + [float("inf")]),
        ("p", [0.0] * 11 + [float("inf")]),
    ],
)
def test_approve_intrinsics_match_consumer_contract(tmp_path, field, value):
    with pytest.raises(ValueError):
        approve(
            _archive(tmp_path, "123456789", intrinsics_override=(field, value)),
            tmp_path / "production",
        )
