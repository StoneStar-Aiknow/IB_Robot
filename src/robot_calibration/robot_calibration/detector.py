"""Run the pinned FAST-Calib detector on exported static scenes."""

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import yaml

from robot_calibration.offline import PATCH_DIFF_SHA256, REQUIRED_SCENES, SOLVER_COMMIT, _load_observation


def _command_output(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def validate_solver_workspace(workspace: Path) -> Path:
    """Return the detector executable only for the exact managed source identity."""
    source = workspace / "src" / "fast_calib"
    if _command_output(["git", "-C", str(source), "rev-parse", "HEAD"]) != SOLVER_COMMIT:
        raise ValueError("FAST-Calib commit does not match the pinned contract")
    diff = subprocess.run(
        ["git", "-C", str(source), "diff", "--binary", "HEAD"], check=True, capture_output=True
    ).stdout
    if hashlib.sha256(diff).hexdigest() != PATCH_DIFF_SHA256:
        raise ValueError("FAST-Calib managed diff does not match the pinned contract")
    candidates = (
        workspace / "install" / "lib" / "fast_calib" / "fast_calib",
        workspace / "install" / "fast_calib" / "lib" / "fast_calib" / "fast_calib",
    )
    for executable in candidates:
        if os.access(executable, os.X_OK):
            return executable
    raise ValueError("FAST-Calib executable is missing")


def _camera_parameters(camera_info_path: Path) -> dict[str, float]:
    value = yaml.safe_load(camera_info_path.read_bytes()) or {}
    matrix = value.get("K")
    distortion = value.get("D")
    if not isinstance(matrix, list) or len(matrix) != 9:
        raise ValueError("camera_info K must contain nine values")
    if not isinstance(distortion, list) or len(distortion) < 4:
        raise ValueError("camera_info D must contain at least four values")
    return {
        "fx": float(matrix[0]),
        "fy": float(matrix[4]),
        "cx": float(matrix[2]),
        "cy": float(matrix[5]),
        "k1": float(distortion[0]),
        "k2": float(distortion[1]),
        "p1": float(distortion[2]),
        "p2": float(distortion[3]),
    }


def prepare_detector_parameters(
    template: Path, exported_scene: Path, output: Path, parameters: Path, scene_id: str
) -> None:
    """Bind one checked template to immutable exported inputs and a fresh output."""
    value = yaml.safe_load(template.read_bytes())
    params = value["fast_calib"]["ros__parameters"]
    if params.get("scene_id") != scene_id:
        raise ValueError("detector template scene_id mismatch")
    bag = exported_scene / "dense_bag_v1"
    image = exported_scene / "image.png"
    camera_info = exported_scene / "camera_info.yaml"
    if not bag.is_dir() or not image.is_file() or not camera_info.is_file():
        raise ValueError(f"exported scene is incomplete: {scene_id}")
    output.mkdir(parents=True, exist_ok=False)
    params.update(_camera_parameters(camera_info))
    params["bag_path"] = str(bag.absolute())
    params["image_path"] = str(image.absolute())
    params["output_path"] = str(output.absolute())
    parameters.parent.mkdir(parents=True, exist_ok=True)
    parameters.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def run_detector(workspace: Path, templates: Path, exported: Path, output: Path) -> dict[str, str]:
    """Generate and validate one observation for every required scene."""
    executable = validate_solver_workspace(workspace)
    if os.path.lexists(output):
        raise FileExistsError(output)
    output.mkdir(parents=True)
    hashes = {}
    try:
        for scene in REQUIRED_SCENES:
            suffix = "04_test" if scene == "scene-04-test" else scene.removeprefix("scene-")
            template = templates / f"current_installation_scene{suffix}.yaml"
            scene_output = output / scene
            parameters = output / f"{scene}.yaml"
            prepare_detector_parameters(template, exported / scene, scene_output, parameters, scene)
            subprocess.run(
                [
                    str(executable),
                    "--ros-args",
                    "-r",
                    "__node:=fast_calib",
                    "--params-file",
                    str(parameters),
                ],
                check=True,
            )
            observation = scene_output / "observation.yaml"
            _, _, digest = _load_observation(observation, scene)
            hashes[scene] = digest
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return hashes
