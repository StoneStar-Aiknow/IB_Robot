"""Launch the packaged RViz view for calibration capture or validation."""

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


def rviz_command(mode: str, share: Path) -> list[str]:
    if mode not in {"capture", "validate"}:
        raise ValueError(f"unknown viewer mode: {mode}")
    return ["rviz2", "-d", str(share / "rviz" / f"calib_{mode}.rviz")]


def preview_decoder_command() -> list[str]:
    return ["ros2", "run", "robot_calibration", "calib_preview_decode"]


def overlay_decoder_command() -> list[str]:
    return [
        "ros2",
        "run",
        "robot_calibration",
        "calib_preview_decode",
        "--input-topic",
        "/calib/overlay/compressed",
        "--output-topic",
        "/calib/overlay",
        "--node-name",
        "robot_calibration_overlay_decoder",
    ]


@dataclass
class ViewerSession:
    rviz: subprocess.Popen[str]
    decoder: subprocess.Popen[str] | None = None
    overlay_decoder: subprocess.Popen[str] | None = None


def package_share() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("robot_calibration"))
    except (ImportError, LookupError):
        return Path(__file__).parents[1]


def display_environment(
    environment: Mapping[str, str] | None = None,
    x11_dir: Path = Path("/tmp/.X11-unix"),
) -> dict[str, str]:
    """Return a child environment connected to the current local desktop."""
    resolved = dict(os.environ if environment is None else environment)
    if resolved.get("DISPLAY") or resolved.get("WAYLAND_DISPLAY"):
        return resolved

    sockets = [path for path in x11_dir.glob("X*") if path.name[1:].isdigit()]
    if not sockets:
        raise RuntimeError("当前主机没有可用的图形桌面")
    socket = max(sockets, key=lambda path: path.stat().st_mtime)
    resolved["DISPLAY"] = f":{socket.name[1:]}"

    if not resolved.get("XAUTHORITY"):
        authority = Path(f"/run/user/{os.getuid()}/gdm/Xauthority")
        if authority.is_file():
            resolved["XAUTHORITY"] = str(authority)
    return resolved


def start_viewer(mode: str, log_path: Path | None = None) -> ViewerSession | None:
    try:
        environment = display_environment()
    except RuntimeError:
        print(
            f"无图形环境，未自动打开 RViz。可在同一 ROS 环境的 PC 端运行: "
            f"ros2 run robot_calibration calib_view --mode {mode}"
        )
        return None
    decoder = None
    overlay_decoder = None
    if log_path is None:
        decoder = subprocess.Popen(preview_decoder_command(), env=environment, start_new_session=True, text=True)
        if mode == "validate":
            overlay_decoder = subprocess.Popen(
                overlay_decoder_command(), env=environment, start_new_session=True, text=True
            )
        rviz = subprocess.Popen(rviz_command(mode, package_share()), env=environment, start_new_session=True, text=True)
        return ViewerSession(rviz=rviz, decoder=decoder, overlay_decoder=overlay_decoder)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        decoder = subprocess.Popen(
            preview_decoder_command(),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        if mode == "validate":
            overlay_decoder = subprocess.Popen(
                overlay_decoder_command(),
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
        rviz = subprocess.Popen(
            rviz_command(mode, package_share()),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
    return ViewerSession(rviz=rviz, decoder=decoder, overlay_decoder=overlay_decoder)


def stop_viewer(session: ViewerSession | None) -> None:
    if session is None:
        return
    for process in (session.rviz, session.decoder, session.overlay_decoder):
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("capture", "validate"), required=True)
    args = parser.parse_args(argv)
    try:
        environment = display_environment()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    decoder = None
    overlay_decoder = None
    try:
        decoder = subprocess.Popen(preview_decoder_command(), env=environment, start_new_session=True, text=True)
        if args.mode == "validate":
            overlay_decoder = subprocess.Popen(
                overlay_decoder_command(), env=environment, start_new_session=True, text=True
            )
        return subprocess.run(rviz_command(args.mode, package_share()), env=environment, check=False).returncode
    finally:
        for process in (decoder, overlay_decoder):
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
