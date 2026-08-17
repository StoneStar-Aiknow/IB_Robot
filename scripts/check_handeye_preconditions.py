#!/usr/bin/env python3
"""Static precheck for IB_Robot eye-in-hand calibration configuration."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("src/robot_config/config/robots/lekiwi_handeye_realsense_grasp.yaml")


@dataclass(frozen=True)
class CameraSummary:
    name: str
    driver: str
    parent_frame: str
    frame_id: str
    optical_frame_id: str
    image_topic: str
    camera_info_topic: str
    camera_info_url: str


@dataclass(frozen=True)
class PrecheckResult:
    robot_name: str
    config_path: Path
    base_frame: str
    ee_frame: str
    cameras: list[CameraSummary]
    mounted_cameras: list[CameraSummary]
    errors: list[str]
    warnings: list[str]

    @property
    def passed(self) -> bool:
        return not self.errors


def _load_robot_section(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict) or not isinstance(data.get("robot"), dict):
        raise ValueError(f"{config_path} must contain a top-level 'robot' mapping")
    return data["robot"]


def _resolve_env_path(value: str) -> Path:
    def replace_env(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), "")

    expanded = re.sub(r"\$\(env\s+([A-Za-z_][A-Za-z0-9_]*)\)", replace_env, value)
    return Path(os.path.expanduser(expanded))


def _camera_topics(camera: dict[str, Any]) -> tuple[str, str]:
    name = str(camera.get("name", ""))
    return f"/camera/{name}/image_raw", f"/camera/{name}/camera_info"


def _summarize_camera(camera: dict[str, Any]) -> CameraSummary:
    image_topic, camera_info_topic = _camera_topics(camera)
    transform = camera.get("transform") or {}
    return CameraSummary(
        name=str(camera.get("name", "")),
        driver=str(camera.get("driver", "")),
        parent_frame=str(transform.get("parent_frame", "")),
        frame_id=str(camera.get("frame_id", "")),
        optical_frame_id=str(camera.get("optical_frame_id", "")),
        image_topic=image_topic,
        camera_info_topic=camera_info_topic,
        camera_info_url=str(camera.get("camera_info_url", "")),
    )


def run_precheck(
    config_path: Path,
    *,
    camera_name: str = "",
    mount_parent: str = "",
    check_files: bool = False,
) -> PrecheckResult:
    config_path = config_path.expanduser().resolve()
    robot = _load_robot_section(config_path)
    moveit = robot.get("moveit") or {}
    base_frame = str(moveit.get("base_link", "base"))
    ee_frame = str(mount_parent or moveit.get("ee_link", "gripper"))

    raw_cameras = [
        item for item in robot.get("peripherals", []) or [] if isinstance(item, dict) and item.get("type") == "camera"
    ]
    cameras = [_summarize_camera(camera) for camera in raw_cameras]
    mounted = [
        camera
        for camera in cameras
        if camera.parent_frame == ee_frame and (not camera_name or camera.name == camera_name)
    ]

    errors: list[str] = []
    warnings: list[str] = []

    if not robot.get("ros2_control"):
        errors.append("robot.ros2_control is missing; joint states and robot TF cannot be configured.")

    calib_file = str((robot.get("ros2_control") or {}).get("calib_file", ""))
    if not calib_file:
        warnings.append(
            "robot.ros2_control.calib_file is not configured; zero calibration cannot be proven statically."
        )
    elif check_files and not _resolve_env_path(calib_file).exists():
        errors.append(f"configured joint calibration file does not exist: {calib_file}")

    if not raw_cameras:
        errors.append("no camera peripherals are configured.")

    if camera_name and not any(camera.name == camera_name for camera in cameras):
        errors.append(f"requested camera '{camera_name}' is not configured.")

    if not mounted:
        if camera_name:
            errors.append(
                f"camera '{camera_name}' is not mounted under '{ee_frame}', so it cannot satisfy eye-in-hand calibration."
            )
        else:
            errors.append(
                f"no camera peripheral is mounted under '{ee_frame}', so eye-in-hand calibration is not possible."
            )

    for camera in mounted:
        if not camera.frame_id:
            errors.append(f"camera '{camera.name}' is missing frame_id.")
        if not camera.optical_frame_id:
            errors.append(f"camera '{camera.name}' is missing optical_frame_id.")
        if camera.driver != "realsense" and not camera.camera_info_url:
            warnings.append(
                f"camera '{camera.name}' has no camera_info_url; runtime CameraInfo must still provide valid K/D."
            )
        if camera.camera_info_url.startswith("file://") and check_files:
            info_path = _resolve_env_path(camera.camera_info_url.replace("file://", ""))
            if not info_path.exists():
                errors.append(f"camera '{camera.name}' calibration file does not exist: {camera.camera_info_url}")

    if cameras and not mounted:
        external = ", ".join(f"{camera.name}(parent={camera.parent_frame or 'unset'})" for camera in cameras)
        warnings.append(
            "configured camera(s) are external/base-mounted for this check: "
            f"{external}. Use an eye-to-hand workflow or add a wrist/gripper-mounted camera."
        )

    return PrecheckResult(
        robot_name=str(robot.get("name", "")),
        config_path=config_path,
        base_frame=base_frame,
        ee_frame=ee_frame,
        cameras=cameras,
        mounted_cameras=mounted,
        errors=errors,
        warnings=warnings,
    )


def print_result(result: PrecheckResult) -> None:
    print(f"Hand-eye precheck for: {result.robot_name}")
    print(f"Config: {result.config_path}")
    print(f"Base frame: {result.base_frame}")
    print(f"End-effector frame: {result.ee_frame}")
    print("Cameras:")
    if not result.cameras:
        print("  - none")
    for camera in result.cameras:
        print(
            "  - "
            f"{camera.name}: driver={camera.driver}, parent={camera.parent_frame or 'unset'}, "
            f"image={camera.image_topic}, camera_info={camera.camera_info_topic}"
        )

    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")

    print(f"Result: {'PASS' if result.passed else 'FAIL'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check whether a robot_config can satisfy eye-in-hand hand-eye calibration prerequisites.",
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Robot config YAML path. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--camera-name",
        default="",
        help="Optional camera name that must be mounted on the end-effector.",
    )
    parser.add_argument(
        "--mount-parent",
        default="",
        help="Override the required camera parent frame. Default: robot.moveit.ee_link or gripper.",
    )
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Also require configured joint/camera calibration files to exist on disk.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        result = run_precheck(
            args.robot_config,
            camera_name=args.camera_name,
            mount_parent=args.mount_parent,
            check_files=args.check_files,
        )
    except Exception as exc:
        print(f"hand-eye precheck failed to run: {exc}", file=sys.stderr)
        return 2

    print_result(result)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
