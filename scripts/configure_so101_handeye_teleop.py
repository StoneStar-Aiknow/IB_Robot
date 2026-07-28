#!/usr/bin/env python3
"""Enable SO101 leader-arm teleoperation in a robot_config YAML."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DEVICE_NAME = "so101_leader"
DEFAULT_CALIB_FILE = "$(env HOME)/.calibrate/so101_leader_calibrate.json"
DEFAULT_ROBOT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "robot_config"
    / "config"
    / "robots"
    / "so101_handeye_realsense_grasp.yaml"
)


def _load_robot_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("robot"), dict):
        raise ValueError(f"{path} must contain a top-level 'robot' mapping")
    return data


def configure_teleop(
    data: dict[str, Any],
    *,
    leader_port: str,
    device_name: str,
    calib_file: str,
) -> tuple[str, str]:
    robot = data["robot"]
    follower_port = str((robot.get("ros2_control") or {}).get("port", ""))
    if not follower_port:
        raise ValueError("robot.ros2_control.port is missing; cannot verify leader/follower port separation")
    if leader_port == follower_port:
        raise ValueError(f"leader port must differ from follower port {follower_port}")

    teleop = robot.setdefault("teleoperation", {})
    teleop["enabled"] = True
    teleop["active_device"] = device_name
    teleop["devices"] = [
        {
            "name": device_name,
            "type": "leader_arm",
            "port": leader_port,
            "calib_file": calib_file,
        }
    ]
    return leader_port, follower_port


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enable SO101 leader-arm teleop in a hand-eye robot_config YAML.",
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=DEFAULT_ROBOT_CONFIG,
        help="Robot config YAML to update in place.",
    )
    parser.add_argument("--leader-port", required=True, help="Serial port for the SO101 leader arm.")
    parser.add_argument("--device-name", default=DEFAULT_DEVICE_NAME, help="Teleop device name to write.")
    parser.add_argument("--calib-file", default=DEFAULT_CALIB_FILE, help="Leader arm calibration file path.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the result without writing YAML.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        data = _load_robot_config(args.robot_config.expanduser())
        leader_port, follower_port = configure_teleop(
            data,
            leader_port=args.leader_port,
            device_name=args.device_name,
            calib_file=args.calib_file,
        )
    except Exception as exc:
        parser.error(str(exc))

    if not args.dry_run:
        args.robot_config.expanduser().write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    action = "Would enable" if args.dry_run else "Enabled"
    print(f"{action} teleop leader on {leader_port}; follower is {follower_port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
