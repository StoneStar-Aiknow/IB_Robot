import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_calibration.detector import prepare_detector_parameters
from robot_calibration.offline import (
    PATCH_DIFF_SHA256,
    _matrix_from_quaternion,
    _quaternion_from_matrix,
    create_candidate_artifact,
    create_supporting_artifacts,
    solve_joint_calibration,
)

OBSERVATIONS = {
    "scene-01": {
        "camera_centers": [
            [0.744663, -1.16879, 4.90816],
            [1.21756, -1.17236, 4.74581],
            [1.21181, -0.773213, 4.7203],
            [0.738911, -0.769651, 4.88265],
        ],
        "lidar_centers": [
            [-0.884256, -5.08407, 1.02062],
            [-1.37722, -4.90892, 0.626971],
            [-0.90046, -5.07899, 0.621736],
            [-1.35244, -4.91701, 1.04503],
        ],
    },
    "scene-02": {
        "camera_centers": [
            [-0.324412, -1.17197, 5.05813],
            [0.175416, -1.1753, 5.04545],
            [0.178244, -0.775364, 5.05174],
            [-0.321584, -0.772029, 5.06441],
        ],
        "lidar_centers": [
            [0.161266, -5.20748, 1.01822],
            [0.165756, -5.20665, 0.617123],
            [-0.343916, -5.16278, 0.613254],
            [-0.324446, -5.16571, 1.02646],
        ],
    },
    "scene-03": {
        "camera_centers": [
            [-0.28754, -1.18516, 4.9481],
            [0.212437, -1.18829, 4.95162],
            [0.214732, -0.789357, 4.98071],
            [-0.285246, -0.786225, 4.9772],
        ],
        "lidar_centers": [
            [0.121815, -5.10919, 0.609867],
            [-0.38331, -5.10367, 0.619287],
            [0.112424, -5.11159, 1.00479],
            [-0.372759, -5.10632, 1.01943],
        ],
    },
    "scene-04-test": {
        "camera_centers": [
            [0.661194, -1.19241, 4.56736],
            [1.13123, -1.20015, 4.39704],
            [1.13547, -0.800227, 4.39059],
            [0.665441, -0.792489, 4.56091],
        ],
        "lidar_centers": [
            [-0.812836, -4.73171, 0.64973],
            [-0.8072, -4.73189, 1.0313],
            [-1.27521, -4.55481, 1.03981],
            [-1.28028, -4.55491, 0.642438],
        ],
    },
}


def test_setup_and_offline_share_fast_calib_patch_hash():
    repository = Path(__file__).parents[3]
    setup_script = (repository / "scripts/setup/ros_third_party.sh").read_text(encoding="utf-8")
    match = re.search(r'local expected_diff_sha256="([0-9a-f]{64})"', setup_script)

    assert match is not None
    assert match.group(1) == PATCH_DIFF_SHA256


def test_offline_module_exposes_cli_help():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[1])
    completed = subprocess.run(
        [sys.executable, "-m", "robot_calibration.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0
    assert "legacy-import" in completed.stdout


def test_detector_parameters_bind_export_and_output_paths(tmp_path):
    template = tmp_path / "template.yaml"
    template.write_text(
        "fast_calib:\n  ros__parameters:\n    scene_id: scene-01\n"
        "    bag_path: /REPLACE/bag\n    image_path: /REPLACE/image\n    output_path: /REPLACE/output\n",
        encoding="utf-8",
    )
    exported = tmp_path / "export" / "scene-01"
    (exported / "dense_bag_v1").mkdir(parents=True)
    (exported / "image.png").write_bytes(b"png")
    (exported / "camera_info.yaml").write_text(
        yaml.safe_dump(
            {
                "width": 640,
                "height": 480,
                "distortion_model": "plumb_bob",
                "D": [0.1, -0.2, 0.003, 0.004, 0.0],
                "K": [606.0, 0.0, 321.0, 0.0, 605.0, 254.0, 0.0, 0.0, 1.0],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "observations" / "scene-01"
    parameters = tmp_path / "parameters.yaml"

    prepare_detector_parameters(template, exported, output, parameters, "scene-01")

    value = yaml.safe_load(parameters.read_text())["fast_calib"]["ros__parameters"]
    assert output.is_dir()
    assert value["bag_path"] == str(exported / "dense_bag_v1")
    assert value["image_path"] == str(exported / "image.png")
    assert value["output_path"] == str(output)


def test_detector_parameters_bind_exported_camera_info(tmp_path):
    template = tmp_path / "template.yaml"
    template.write_text(
        "fast_calib:\n  ros__parameters:\n    scene_id: scene-01\n"
        "    fx: 1.0\n    fy: 2.0\n    cx: 3.0\n    cy: 4.0\n"
        "    k1: 5.0\n    k2: 6.0\n    p1: 7.0\n    p2: 8.0\n",
        encoding="utf-8",
    )
    exported = tmp_path / "export" / "scene-01"
    (exported / "dense_bag_v1").mkdir(parents=True)
    (exported / "image.png").write_bytes(b"png")
    (exported / "camera_info.yaml").write_text(
        yaml.safe_dump(
            {
                "D": [0.1, -0.2, 0.003, 0.004],
                "K": [606.0, 0.0, 321.0, 0.0, 605.0, 254.0, 0.0, 0.0, 1.0],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "observations" / "scene-01"
    parameters = tmp_path / "parameters.yaml"

    prepare_detector_parameters(template, exported, output, parameters, "scene-01")

    value = yaml.safe_load(parameters.read_text())["fast_calib"]["ros__parameters"]
    assert (value["fx"], value["fy"], value["cx"], value["cy"]) == (606.0, 605.0, 321.0, 254.0)
    assert (value["k1"], value["k2"], value["p1"], value["p2"]) == (0.1, -0.2, 0.003, 0.004)


def test_detector_parameters_reject_incomplete_camera_info(tmp_path):
    template = tmp_path / "template.yaml"
    template.write_text("fast_calib:\n  ros__parameters:\n    scene_id: scene-01\n", encoding="utf-8")
    exported = tmp_path / "export" / "scene-01"
    (exported / "dense_bag_v1").mkdir(parents=True)
    (exported / "image.png").write_bytes(b"png")
    (exported / "camera_info.yaml").write_text("K: [1.0]\nD: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="camera_info K"):
        prepare_detector_parameters(
            template,
            exported,
            tmp_path / "observations" / "scene-01",
            tmp_path / "parameters.yaml",
            "scene-01",
        )


def _write_observations(root):
    paths = {}
    for scene, centers in OBSERVATIONS.items():
        value = {
            "schema_version": "1.0",
            "scene_id": scene,
            "source_frame": "body",
            "target_frame": "camera_front_optical_frame",
            **centers,
        }
        path = root / scene / "observation.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
        paths[scene] = path
    return paths


def test_joint_solver_reproduces_historical_candidate_metrics(tmp_path):
    observations = _write_observations(tmp_path / "observations")

    result, report = solve_joint_calibration(
        observations=observations,
        output=tmp_path / "extrinsic.yaml",
        report=tmp_path / "report.json",
        max_training_rmse_m=0.04,
        max_test_rmse_m=0.04,
        max_baseline_m=0.5,
        min_correspondence_margin_m=0.05,
    )

    assert result["source_frame"] == "body"
    assert report["status"] == "candidate"
    assert report["training_joint_rmse_m"] == pytest.approx(0.025902042465795867)
    assert report["test_rmse_m"] == pytest.approx(0.03356981902924989)
    assert report["correspondence_margin_m"] == pytest.approx(0.073147374, abs=1e-9)
    assert report["correspondence_indices"]["scene-04-test"] == [1, 2, 3, 0]

    with pytest.raises(ValueError, match="test RMSE"):
        solve_joint_calibration(
            observations=observations,
            output=tmp_path / "rejected.yaml",
            report=tmp_path / "rejected.json",
            max_training_rmse_m=0.04,
            max_test_rmse_m=0.03,
            max_baseline_m=0.5,
            min_correspondence_margin_m=0.05,
        )


def test_candidate_artifact_composes_mount_and_binds_evidence(tmp_path):
    observations = _write_observations(tmp_path / "observations")
    result_path = tmp_path / "extrinsic.yaml"
    report_path = tmp_path / "report.json"
    solve_joint_calibration(
        observations=observations,
        output=result_path,
        report=report_path,
        max_training_rmse_m=0.04,
        max_test_rmse_m=0.04,
        max_baseline_m=0.5,
        min_correspondence_margin_m=0.05,
    )
    mount_path = tmp_path / "base_to_mid360.yaml"
    mount_path.write_text(
        "schema_version: '1.0'\ncalibration_version: mount-1\nstatus: candidate\n"
        "device: {name: MID-360, serial: L1}\n"
        "transform:\n  parent_frame: base_link\n  child_frame: body\n"
        "  translation: [-0.08, 0.0, 0.2]\n"
        "  rotation_xyzw: [0.0, 0.0, 0.7071067811865475, 0.7071067811865476]\n",
        encoding="utf-8",
    )
    capture_manifest = tmp_path / "manifest.json"
    capture_manifest.write_text('{"capture_id":"calib-19700101-000154"}\n', encoding="ascii")
    output = tmp_path / "base_to_front_camera.yaml"

    artifact = create_candidate_artifact(
        result=result_path,
        report=report_path,
        mount=mount_path,
        capture_manifest=capture_manifest,
        camera_serial="C1",
        producer_commit="1" * 40,
        parameters_sha256="2" * 64,
        output=output,
    )

    assert artifact["status"] == "candidate"
    assert artifact["transform"]["parent_frame"] == "base_link"
    assert artifact["transform"]["child_frame"] == "camera_front_optical_frame"
    solver = yaml.safe_load(result_path.read_text())
    rotation = np.asarray(solver["rotation_matrix"])
    body_from_camera_translation = -rotation.T @ np.asarray(solver["translation"])
    expected_translation = (
        np.asarray([-0.08, 0.0, 0.2])
        + np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) @ body_from_camera_translation
    )
    assert artifact["transform"]["translation"] == pytest.approx(expected_translation)
    assert artifact["metrics"]["test_rmse_m"] == pytest.approx(0.03356981902924989)
    assert artifact["provenance"]["solver_commit"] == "7747dfc6109c04b4bf81d2e3661e41626c8392e1"
    assert (
        artifact["provenance"]["capture_manifest_sha256"] == hashlib.sha256(capture_manifest.read_bytes()).hexdigest()
    )
    assert output.is_file()
    assert json.loads(json.dumps(artifact))["calibration_version"].startswith("fast-calib-")

    mount_source = tmp_path / "lekiwi_mid360_mount.yaml"
    mount_source.write_text(
        "schema_version: '1.0'\nstatus: provisional\nparent_frame: base_link\n"
        "lidar_frame: livox_frame\nbody_frame: body\n"
        "translation_m: [-0.08, 0.0, 0.2]\nrpy_deg: [0.0, 0.0, 90.0]\n",
        encoding="utf-8",
    )
    exported = tmp_path / "exported"
    camera_info = {
        "schema_version": "1.0",
        "frame_id": "camera_front_optical_frame",
        "width": 848,
        "height": 480,
        "distortion_model": "plumb_bob",
        "D": [0.1, -0.2, 0.003, 0.004, 0.0],
        "K": [606.0, 0.0, 321.0, 0.0, 605.0, 254.0, 0.0, 0.0, 1.0],
        "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "P": [606.0, 0.0, 321.0, 0.0, 0.0, 605.0, 254.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    }
    scene_path = exported / "scene-04-test"
    scene_path.mkdir(parents=True)
    (scene_path / "camera_info.yaml").write_text(yaml.safe_dump(camera_info, sort_keys=False), encoding="utf-8")
    base_to_mid360, intrinsics = create_supporting_artifacts(
        mount=mount_source,
        exported=exported,
        camera_serial="C1",
        lidar_serial="unavailable",
        output=tmp_path / "supporting",
    )
    derived_output = tmp_path / "base_to_front_camera-derived.yaml"
    derived = create_candidate_artifact(
        result=result_path,
        report=report_path,
        mount=base_to_mid360,
        capture_manifest=capture_manifest,
        camera_serial="C1",
        producer_commit="1" * 40,
        parameters_sha256="2" * 64,
        output=derived_output,
    )

    assert derived["transform"] == artifact["transform"]
    mount_artifact = yaml.safe_load(base_to_mid360.read_text(encoding="utf-8"))
    assert mount_artifact["transform"] == {
        "parent_frame": "base_link",
        "child_frame": "body",
        "translation": [-0.08, 0.0, 0.2],
        "rotation_xyzw": pytest.approx([0.0, 0.0, 0.7071067811865475, 0.7071067811865476]),
    }
    intrinsics_artifact = yaml.safe_load(intrinsics.read_text(encoding="utf-8"))
    assert intrinsics_artifact["camera_info"] == {
        "frame_id": camera_info["frame_id"],
        "width": camera_info["width"],
        "height": camera_info["height"],
        "distortion_model": camera_info["distortion_model"],
        "d": camera_info["D"],
        "k": camera_info["K"],
        "r": camera_info["R"],
        "p": camera_info["P"],
    }


def test_supporting_artifacts_use_only_test_scene_camera_info(tmp_path):
    mount = tmp_path / "mount.yaml"
    mount.write_text(
        "schema_version: '1.0'\nstatus: provisional\nparent_frame: base_link\n"
        "lidar_frame: livox_frame\nbody_frame: body\n"
        "translation_m: [0.0, 0.0, 0.0]\nrpy_deg: [0.0, 0.0, 0.0]\n",
        encoding="utf-8",
    )
    exported = tmp_path / "exported"
    test_camera_info = {
        "schema_version": "1.0",
        "frame_id": "camera_front_optical_frame",
        "width": 848,
        "height": 480,
        "distortion_model": "plumb_bob",
        "D": [0.1, -0.2, 0.003, 0.004, 0.0],
        "K": [606.0, 0.0, 321.0, 0.0, 605.0, 254.0, 0.0, 0.0, 1.0],
        "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "P": [606.0, 0.0, 321.0, 0.0, 0.0, 605.0, 254.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    }
    for index, scene in enumerate(("scene-01", "scene-02", "scene-03")):
        scene_path = exported / scene
        scene_path.mkdir(parents=True)
        (scene_path / "camera_info.yaml").write_text(
            yaml.safe_dump({"training_scene": index, "width": 640 + index}), encoding="utf-8"
        )
    test_scene = exported / "scene-04-test"
    test_scene.mkdir(parents=True)
    (test_scene / "camera_info.yaml").write_text(yaml.safe_dump(test_camera_info), encoding="utf-8")

    _, intrinsics = create_supporting_artifacts(
        mount=mount,
        exported=exported,
        camera_serial="C1",
        lidar_serial="L1",
        output=tmp_path / "supporting",
    )

    assert yaml.safe_load(intrinsics.read_text(encoding="utf-8"))["camera_info"]["k"] == test_camera_info["K"]


def test_supporting_artifacts_reject_test_scene_camera_info_with_wrong_frame(tmp_path):
    mount = tmp_path / "mount.yaml"
    mount.write_text(
        "schema_version: '1.0'\nstatus: provisional\nparent_frame: base_link\n"
        "lidar_frame: livox_frame\nbody_frame: body\n"
        "translation_m: [0.0, 0.0, 0.0]\nrpy_deg: [0.0, 0.0, 0.0]\n",
        encoding="utf-8",
    )
    test_scene = tmp_path / "exported/scene-04-test"
    test_scene.mkdir(parents=True)
    (test_scene / "camera_info.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "frame_id": "camera_front_link",
                "width": 848,
                "height": 480,
                "distortion_model": "plumb_bob",
                "D": [0.0] * 5,
                "K": [1.0] * 9,
                "R": [1.0] * 9,
                "P": [1.0] * 12,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frame_id must equal camera_front_optical_frame"):
        create_supporting_artifacts(
            mount=mount,
            exported=tmp_path / "exported",
            camera_serial="C1",
            lidar_serial="L1",
            output=tmp_path / "supporting",
        )


def test_quaternion_serialization_preserves_a_non_symmetric_rotation_matrix():
    rotation = np.array(
        [
            [-0.0015035342, -0.0054320073, 0.9999841162],
            [-0.9999599610, -0.0088130433, -0.0015513711],
            [0.0088213303, -0.9999464103, -0.0054185391],
        ]
    )

    restored = _matrix_from_quaternion(_quaternion_from_matrix(rotation))

    np.testing.assert_allclose(restored, rotation, atol=1e-8)
