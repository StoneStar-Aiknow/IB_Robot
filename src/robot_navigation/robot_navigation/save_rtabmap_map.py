import argparse
import subprocess
import sys
from pathlib import Path


def _default_output_prefix() -> str:
    return str(Path.home() / ".ros" / "ibrobot" / "maps" / "rtabmap")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save the current RTAB-Map occupancy grid through nav2_map_server.",
    )
    parser.add_argument(
        "-f",
        "--file",
        default=_default_output_prefix(),
        help="Output map prefix, without .yaml/.pgm suffix. Default: ~/.ros/ibrobot/maps/rtabmap",
    )
    parser.add_argument(
        "-t",
        "--topic",
        default="/rtabmap/map",
        help="OccupancyGrid topic to save. Default: /rtabmap/map",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Seconds map_saver_cli should wait for the map topic. Default: 10.0",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_prefix = Path(args.file).expanduser()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    command = [
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
    print(f"Saving map from {args.topic} to {output_prefix}.yaml/.pgm")
    try:
        completed = subprocess.run(command, check=False)
    except FileNotFoundError:
        print(
            "ros2 executable not found. Source the ROS workspace environment before running this CLI.", file=sys.stderr
        )
        return 127

    if completed.returncode == 0:
        print(f"Map saved: {output_prefix}.yaml")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
