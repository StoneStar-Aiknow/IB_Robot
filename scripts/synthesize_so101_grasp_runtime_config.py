#!/usr/bin/env python3
"""Refresh the SO101 grasp runtime YAML without losing host calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from robot_config.runtime_config import RUNTIME_OWNED_ROBOT_KEYS, synthesize_runtime_config

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = WORKSPACE / "src/robot_config/config/robots/so101_handeye_realsense_only.yaml"
DEFAULT_RUNTIME_CONFIG = Path("/tmp/so101_handeye_realsense_grasp.yaml")


def _load_mapping(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create a missing runtime file from the base config. Verify serial ports before use.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without writing.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    base_path = args.base_config.expanduser().resolve()
    runtime_path = args.runtime_config.expanduser().resolve()

    try:
        base = _load_mapping(base_path)
        if runtime_path.exists():
            runtime = _load_mapping(runtime_path)
        elif args.create:
            runtime = base
        else:
            raise FileNotFoundError(
                f"runtime config not found: {runtime_path}; create/calibrate it first or pass --create"
            )
        merged = synthesize_runtime_config(base, runtime)
    except Exception as exc:
        parser.error(str(exc))

    robot = merged["robot"]
    skill_templates = (robot.get("embodied") or {}).get("skill_templates") or {}
    grasp_enabled = bool((robot.get("grasp_execution") or {}).get("enabled", False))
    summary = (
        f"robot={robot.get('name', '')} grasp_enabled={grasp_enabled} "
        f"pick_object={'pick_object' in skill_templates} "
        f"preserved={','.join(RUNTIME_OWNED_ROBOT_KEYS)}"
    )
    if args.dry_run:
        print(f"Would synthesize {runtime_path}: {summary}")
        return 0

    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = runtime_path.with_name(f".{runtime_path.name}.tmp")
    temporary_path.write_text(
        yaml.safe_dump(merged, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary_path.replace(runtime_path)
    print(f"Synthesized {runtime_path}: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
