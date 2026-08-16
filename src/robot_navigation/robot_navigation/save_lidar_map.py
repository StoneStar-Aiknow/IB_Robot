"""Save the MID-360/SLAM Toolbox OccupancyGrid map."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def _default_output_prefix() -> str:
    return str(Path.home() / ".ros" / "ibrobot" / "maps" / "map")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save the LiDAR SLAM OccupancyGrid through nav2_map_server.")
    parser.add_argument("-f", "--file", default=_default_output_prefix(), help="Output map prefix without suffix")
    parser.add_argument("-t", "--topic", default="/map", help="OccupancyGrid topic to save")
    parser.add_argument("--timeout", type=float, default=10.0, help="Seconds to wait for the map topic")
    return parser


def build_save_command(args: argparse.Namespace) -> list[str]:
    output_prefix = Path(args.file).expanduser()
    return [
        "ros2",
        "run",
        "nav2_map_server",
        "map_saver_cli",
        "-t",
        args.topic,
        "-f",
        str(output_prefix),
        "--ros-args",
        "-p",
        f"save_map_timeout:={args.timeout}",
    ]


def _validate_saved_map(prefix: Path) -> tuple[Path, Path]:
    yaml_path = prefix.with_suffix(".yaml")
    image_path = prefix.with_suffix(".pgm")
    if not yaml_path.is_file() or yaml_path.stat().st_size == 0:
        raise RuntimeError(f"map saver did not produce a valid YAML file: {yaml_path}")
    if not image_path.is_file() or image_path.stat().st_size == 0:
        raise RuntimeError(f"map saver did not produce a valid PGM file: {image_path}")
    with yaml_path.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    if not isinstance(metadata, dict) or not isinstance(metadata.get("image"), str):
        raise RuntimeError(f"map YAML has no image reference: {yaml_path}")
    referenced_image = (yaml_path.parent / metadata["image"]).resolve()
    if referenced_image != image_path.resolve() or not referenced_image.is_file():
        raise RuntimeError(f"map YAML image reference is invalid: {metadata['image']}")
    return yaml_path, image_path


def _publish_map_atomically(staged_prefix: Path, output_prefix: Path) -> None:
    staged_yaml, staged_image = _validate_saved_map(staged_prefix)
    output_yaml = output_prefix.with_suffix(".yaml")
    output_image = output_prefix.with_suffix(".pgm")
    with staged_yaml.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    if metadata["image"] != output_image.name:
        metadata["image"] = output_image.name
        staged_yaml.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    old_yaml = output_yaml.with_suffix(".yaml.previous")
    old_image = output_image.with_suffix(".pgm.previous")
    backups = []
    published = []
    try:
        for source, backup in ((output_yaml, old_yaml), (output_image, old_image)):
            if source.exists():
                os.replace(source, backup)
                backups.append((backup, source))
        os.replace(staged_yaml, output_yaml)
        published.append(output_yaml)
        os.replace(staged_image, output_image)
    except Exception:
        for target in published:
            target.unlink(missing_ok=True)
        for backup, target in reversed(backups):
            if backup.exists():
                os.replace(backup, target)
        raise
    for backup, _target in backups:
        backup.unlink(missing_ok=True)


def promote_saved_map(source_prefix: Path, output_prefix: Path | None = None) -> None:
    """Publish an existing YAML/PGM map pair with rollback on partial failure."""
    source_prefix = source_prefix.expanduser()
    output_prefix = (output_prefix or Path(_default_output_prefix())).expanduser()
    _validate_saved_map(source_prefix)
    if source_prefix.resolve() == output_prefix.resolve():
        raise ValueError("source and output map prefixes must be different")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_prefix.name}.promote-", dir=output_prefix.parent))
    staged_prefix = staging_dir / output_prefix.name
    try:
        shutil.copy2(source_prefix.with_suffix(".yaml"), staged_prefix.with_suffix(".yaml"))
        shutil.copy2(source_prefix.with_suffix(".pgm"), staged_prefix.with_suffix(".pgm"))
        _publish_map_atomically(staged_prefix, output_prefix)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_prefix = Path(args.file).expanduser()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    staged_prefix = output_prefix.parent / f".{output_prefix.name}"
    for suffix in (".yaml", ".pgm"):
        staged_prefix.with_suffix(suffix).unlink(missing_ok=True)
    print(f"Saving LiDAR map from {args.topic} to {output_prefix}.yaml/.pgm")
    try:
        staged_values = vars(args).copy()
        staged_values["file"] = str(staged_prefix)
        staged_args = argparse.Namespace(**staged_values)
        completed = subprocess.run(build_save_command(staged_args), check=False)
    except FileNotFoundError:
        print("ros2 executable not found. Source the ROS workspace environment first.", file=sys.stderr)
        return 127
    if completed.returncode != 0:
        return completed.returncode
    try:
        _publish_map_atomically(staged_prefix, output_prefix)
    except (OSError, RuntimeError, yaml.YAMLError) as exc:
        print(f"LiDAR map was not published: {exc}", file=sys.stderr)
        return 1
    print(f"LiDAR map saved: {output_prefix}.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
