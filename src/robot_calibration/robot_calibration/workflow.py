"""User-facing calibration workflows.

The public commands intentionally hide observation files, serial numbers, and
solver parameters. Lower-level modules remain available for diagnostics.
"""

import argparse
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rosidl_runtime_py.utilities import get_message

from robot_calibration.bag import REQUIRED_TOPIC_TYPES, validate_fast_calib_bag
from robot_calibration.capture import REQUIRED_SCENES, CaptureError, finalize_capture
from robot_calibration.detector import run_detector
from robot_calibration.export import export_capture
from robot_calibration.offline import create_candidate_artifact, create_supporting_artifacts, solve_joint_calibration
from robot_calibration.store import ArtifactStore
from robot_calibration.viewer import start_viewer, stop_viewer

LOGICAL_SENSORS = {"camera_front": "front", "front_camera": "front", "wrist_camera": "wrist"}
REQUIRED_TOPICS = tuple(REQUIRED_TOPIC_TYPES)
CAPTURE_PREVIEW_COMMAND = [
    "ros2",
    "run",
    "robot_calibration",
    "calib_capture_preview",
    "--max-fps",
    "8.0",
    "--jpeg-quality",
    "70",
    "--max-points",
    "6000",
]


def sensor_calibration_launch_command() -> list[str]:
    return [
        "ros2",
        "launch",
        "robot_config",
        "sensor_calibration.launch.py",
        "robot_config:=lekiwi_sensor_calib",
    ]


def default_calib_root() -> Path:
    return Path.home() / ".ros/ibrobot/calib"


def default_raw_dir() -> Path:
    return default_calib_root() / "raw"


def default_process_dir(capture_id: str) -> Path:
    return default_calib_root() / "process" / capture_id


def default_candidate_dir() -> Path:
    return default_calib_root() / "candidates"


def default_log_dir() -> Path:
    return default_calib_root() / "logs"


def capture_initialization_message(log_path: Path) -> str:
    return f"初始化中... 详细日志请见 {log_path.absolute()}"


def logical_sensor_name(value: str) -> str:
    """Return the stable user-facing name for a sensor position."""
    normalized = value.strip().lower().replace("-", "_")
    return LOGICAL_SENSORS.get(normalized, normalized.removesuffix("_camera"))


def resolve_capture_input(value: Path) -> tuple[str, Path]:
    """Classify a sealed capture directory or a single capture archive."""
    path = Path(value).expanduser().absolute()
    if path.is_dir() and (path / "manifest.json").is_file():
        return "directory", path
    if path.is_file() and path.name.endswith(".raw.tar"):
        return "archive", path
    raise ValueError(f"capture input must be a sealed directory or .raw.tar archive: {path}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(_repo_root()), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _default_workspace() -> Path:
    configured = os.environ.get("FAST_CALIB_WORKSPACE")
    if configured:
        return Path(configured).expanduser()
    return _repo_root()


def _default_templates() -> Path:
    return _repo_root() / "src/robot_calibration/config/fast_calib/scenes"


def _default_mount() -> Path:
    return _repo_root() / "src/robot_config/config/hardware/lekiwi_mid360_mount.yaml"


def _capture_summary(capture_root: Path, capture_id: str) -> list[dict[str, Any]]:
    scenes = []
    for scene_id in REQUIRED_SCENES:
        bag = validate_fast_calib_bag(capture_root / scene_id)
        scenes.append(
            {
                "scene_id": scene_id,
                "role": "test" if scene_id == "scene-04-test" else "fit",
                "duration_s": bag.duration_s,
                "stationary_windows": 1,
                "topics": {
                    topic: {"type": REQUIRED_TOPIC_TYPES[topic], "count": count}
                    for topic, count in bag.topic_counts.items()
                },
                "tf_edges": ["base_link -> body", "body -> camera_front_link"],
                "files": [
                    {"path": f"{scene_id}/metadata.yaml"},
                    *({"path": f"{scene_id}/{path.name}"} for path in bag.storage_files),
                ],
            }
        )
    return scenes


def _copy_or_import_capture(input_path: Path, work: Path) -> Path:
    kind, path = resolve_capture_input(input_path)
    if kind == "directory":
        return path
    imported_root = work / "capture"
    result = ArtifactStore.import_capture(path, imported_root)
    return Path(result["path"])


def _parameters_sha256(observations: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    for scene in REQUIRED_SCENES:
        digest.update((observations / f"{scene}.yaml").read_bytes())
    return digest.hexdigest()


def _discover_realsense_serial() -> str:
    """Read the serial from the installed RealSense utility, if available."""
    try:
        completed = subprocess.run(
            ["rs-enumerate-devices", "-S"], capture_output=True, text=True, check=False, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    for line in completed.stdout.splitlines():
        match = re.search(r"(?:serial(?: number)?\s*:\s*)?(\d{6,})\s*$", line, re.IGNORECASE)
        if match:
            return match.group(1)
    return "unavailable"


def _record_command(scene_dir: Path) -> list[str]:
    return [
        "ros2",
        "bag",
        "record",
        "--storage",
        "mcap",
        "--output",
        str(scene_dir),
        *REQUIRED_TOPICS,
    ]


def _wait_for_required_topics(
    log_path: Path,
    *,
    sensor_process: subprocess.Popen[str] | None = None,
) -> None:
    """Wait until every capture topic has delivered a message."""
    node = None
    owns_context = False
    received: set[str] = set()
    subscriptions = []
    try:
        if not rclpy.ok():
            rclpy.init()
            owns_context = True
        node = rclpy.create_node("calib_capture_readiness")
        for topic, message_type in REQUIRED_TOPIC_TYPES.items():
            qos = (
                QoSProfile(
                    depth=1,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                )
                if topic == "/tf_static"
                else qos_profile_sensor_data
            )
            subscriptions.append(
                node.create_subscription(
                    get_message(message_type),
                    topic,
                    lambda _message, ready_topic=topic: received.add(ready_topic),
                    qos,
                )
            )
        next_report = time.monotonic()
        while True:
            missing = [topic for topic in REQUIRED_TOPICS if topic not in received]
            if not missing:
                print("所有采集 topic 均已收到消息，可以开始标定采集", flush=True)
                return
            if sensor_process is not None and sensor_process.poll() is not None:
                raise RuntimeError(
                    f"sensor calibration launch exited before readiness; topics without messages: {', '.join(missing)}; "
                    f"see {log_path.absolute()}"
                )
            now = time.monotonic()
            if now >= next_report:
                print(f"等待传感器初始化，尚未收到消息: {', '.join(missing)}", flush=True)
                next_report = now + 15.0
            rclpy.spin_once(node, timeout_sec=1.0)
    finally:
        subscriptions.clear()
        if node is not None:
            node.destroy_node()
        if owns_context and rclpy.ok():
            rclpy.shutdown()


def _start_logged_process(command: list[str], log_path: Path, **kwargs: Any) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, **kwargs)


def _wait_for_recording_window(duration_s: float, *, sleep_fn=time.sleep) -> None:
    """Allow rosbag startup to settle before measuring the capture window."""
    sleep_fn(2.0)
    sleep_fn(duration_s)


def _stop_owned_process(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    process_group = process.pid

    def group_exists() -> bool:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    with contextlib.suppress(ProcessLookupError):
        os.killpg(process_group, signal.SIGINT)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process_group, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process_group, signal.SIGKILL)
            process.wait(timeout=5)
    if not group_exists():
        return
    for termination_signal in (signal.SIGTERM, signal.SIGKILL):
        if not group_exists():
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process_group, termination_signal)


def _retain_failed_recording(recording: Path) -> Path:
    """Keep an incomplete capture available for diagnosis instead of deleting it."""
    failed = recording.with_name(recording.name.removesuffix(".recording") + ".failed")
    suffix = 1
    while failed.exists():
        failed = recording.with_name(recording.name.removesuffix(".recording") + f".failed-{suffix}")
        suffix += 1
    if recording.exists():
        os.rename(recording, failed)
    return failed


def _start_capture_preview(log_path: Path | None = None) -> subprocess.Popen[str]:
    if log_path is None:
        return subprocess.Popen(CAPTURE_PREVIEW_COMMAND, stdin=subprocess.DEVNULL, start_new_session=True, text=True)
    return _start_logged_process(
        CAPTURE_PREVIEW_COMMAND,
        log_path,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        text=True,
    )


def _start_sensor_calibration(log_path: Path) -> subprocess.Popen[str]:
    return _start_logged_process(
        sensor_calibration_launch_command(),
        log_path,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        text=True,
    )


def _write_artifact_archive(output: Path, capture_id: str, files: list[Path]) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{capture_id}.candidate.tar"
    with tarfile.open(archive, "x", format=tarfile.PAX_FORMAT) as target:
        for path in sorted(files, key=lambda item: item.name):
            info = target.gettarinfo(str(path), arcname=path.name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o644
            with path.open("rb") as stream:
                target.addfile(info, stream)
    return archive


def solve_user_workflow(
    input_path: Path,
    output: Path,
    *,
    workspace: Path | None = None,
    candidate_dir: Path | None = None,
) -> Path:
    """Run validation, export, detector, solve, test evaluation, and artifact generation."""
    output = output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    try:
        capture = _copy_or_import_capture(input_path, output / "staging")
        manifest = capture / "manifest.json"
        manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
        capture_id = manifest_value["capture_id"]
        exported = output / "exported"
        export_capture(capture, exported)
        camera_identity = str(manifest_value.get("devices", {}).get("camera") or "front")
        lidar_identity = str(manifest_value.get("devices", {}).get("lidar") or "unavailable")
        mount_artifact, intrinsics_artifact = create_supporting_artifacts(
            mount=_default_mount(),
            exported=exported,
            camera_serial=camera_identity,
            lidar_serial=lidar_identity,
            output=output,
        )
        observations = output / "observations"
        run_detector(workspace or _default_workspace(), _default_templates(), exported, observations)
        _, report = solve_joint_calibration(
            observations={scene: observations / scene / "observation.yaml" for scene in REQUIRED_SCENES},
            output=output / "extrinsic.yaml",
            report=output / "report.json",
            max_training_rmse_m=0.04,
            max_test_rmse_m=0.04,
            max_baseline_m=0.5,
            min_correspondence_margin_m=0.05,
        )
        artifact_path = output / "base_to_front_camera.candidate.yaml"
        create_candidate_artifact(
            result=output / "extrinsic.yaml",
            report=output / "report.json",
            mount=mount_artifact,
            capture_manifest=manifest,
            camera_serial=camera_identity,
            producer_commit=_git_commit(),
            parameters_sha256=_parameters_sha256(observations),
            output=artifact_path,
        )
        from robot_calibration.overlay import render_test_overlay

        overlay = output / "test-overlay.png"
        projected = render_test_overlay(output / "extrinsic.yaml", exported / "scene-04-test", overlay)
        summary = {
            "capture_id": capture_id,
            "artifact": str(artifact_path),
            "test_preview": str(overlay),
            "projected_point_count": projected,
            "status": "candidate",
            "training_rmse_m": report["training_joint_rmse_m"],
            "test_rmse_m": report["test_rmse_m"],
            "correspondence_margin_m": report["correspondence_margin_m"],
        }
        (output / "calibration_summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        archive = _write_artifact_archive(
            candidate_dir or default_candidate_dir(),
            capture_id,
            [
                artifact_path,
                mount_artifact,
                intrinsics_artifact,
                output / "extrinsic.yaml",
                output / "report.json",
                output / "calibration_summary.json",
                overlay,
            ],
        )
        return archive
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def capture_user_workflow(
    output: Path,
    *,
    duration_s: float = 10.0,
    input_fn=input,
    sleep_fn=time.sleep,
    record_command: list[str] | None = None,
) -> tuple[Path, Path]:
    """Interactively record four scenes from an already-running sensor graph."""
    if duration_s <= 0:
        raise ValueError("duration_s must be greater than zero")
    capture_id = f"calib-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    recording = output / f".{capture_id}.recording"
    log_path = default_log_dir() / f"{capture_id}.log"
    recording.mkdir(parents=True, exist_ok=False)
    preview_process = None
    viewer_process = None
    sensor_process = None
    try:
        print(capture_initialization_message(log_path), flush=True)
        sensor_process = _start_sensor_calibration(log_path)
        preview_process = _start_capture_preview(log_path)
        viewer_process = start_viewer("capture", log_path)
        _wait_for_required_topics(log_path, sensor_process=sensor_process)
        for index, scene_id in enumerate(REQUIRED_SCENES):
            input_fn(f"{scene_id}: 确认所有传感器均可见标定板，保持静止后按 Enter 开始录制（~10s）...")
            scene_dir = recording / scene_id
            command = record_command or _record_command(scene_dir)
            scene_dir.parent.mkdir(parents=True, exist_ok=True)
            recorder = _start_logged_process(command, log_path, start_new_session=True, stdin=subprocess.DEVNULL)
            try:
                _wait_for_recording_window(duration_s, sleep_fn=sleep_fn)
            finally:
                if recorder.poll() is None:
                    recorder.send_signal(signal.SIGINT)
                return_code = recorder.wait()
            if return_code != 0:
                raise RuntimeError(f"scene recorder exited with code {return_code}: {scene_id}")
            if not (scene_dir / "metadata.yaml").is_file():
                raise RuntimeError(f"scene recorder did not produce metadata.yaml: {scene_id}")
            validate_fast_calib_bag(scene_dir)
            if index + 1 < len(REQUIRED_SCENES):
                print(
                    f"录制完成，请移动到另一个位置，准备开始 {REQUIRED_SCENES[index + 1]}",
                    flush=True,
                )
            else:
                print("录制完成", flush=True)
        print("四个 scene 录制完成，正在保存并打包数据，请耐心等待...", flush=True)
        scenes = _capture_summary(recording, capture_id)
        camera_serial = _discover_realsense_serial()
        result = finalize_capture(
            output,
            capture_id,
            scenes,
            {
                "camera": camera_serial,
                "camera_position": logical_sensor_name("front"),
                "lidar": "unavailable",
                "lidar_position": "mid360",
            },
            source=recording,
        )
        shutil.rmtree(recording)
        archive = output / f"{capture_id}.raw.tar"
        ArtifactStore.export_capture(result, archive)
        return result, archive
    except Exception:
        failed_recording = _retain_failed_recording(recording)
        print(f"采集失败，未完成的数据已保留: {failed_recording.absolute()}", file=sys.stderr, flush=True)
        raise
    finally:
        stop_viewer(viewer_process)
        _stop_owned_process(preview_process)
        _stop_owned_process(sensor_process)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="面向现场用户的机器人标定流程")
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="按 Enter 依次采集四个约 10 秒 scene")
    capture.add_argument("--output", type=Path, default=default_raw_dir())
    capture.add_argument("--duration", type=float, default=10.0)
    solve = commands.add_parser("solve", help="从一个 capture 一键生成 candidate artifact")
    solve.add_argument("--input", type=Path, required=True)
    solve.add_argument("--output", type=Path)
    solve.add_argument("--workspace", type=Path)
    return parser


def _capture_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按 Enter 依次采集四个约 10 秒标定 scene")
    parser.add_argument("--output", type=Path, default=default_raw_dir())
    parser.add_argument("--duration", type=float, default=10.0)
    return parser


def _solve_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从一个 capture 一键生成 candidate 标定文件")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workspace", type=Path)
    return parser


def capture_main(argv: list[str] | None = None) -> int:
    """Public interactive capture entry point."""
    args = _capture_parser().parse_args(argv)
    try:
        result, archive = capture_user_workflow(args.output, duration_s=args.duration)
    except (CaptureError, OSError, RuntimeError, subprocess.SubprocessError, TypeError, ValueError) as exc:
        print(f"标定采集失败: {exc}", file=sys.stderr)
        return 2
    print(f"标定数据已采集完成: {result}")
    print(f"raw 数据包路径: {archive.absolute()}")
    print(f"若是在板侧进行采集，请传至PC端进行标定文件生成: scp <user>@<board_IP>:{archive.absolute()} .")
    return 0


def solve_main(argv: list[str] | None = None) -> int:
    """Public one-input solve entry point."""
    args = _solve_parser().parse_args(argv)
    capture_id = args.input.name.removesuffix(".raw.tar")
    output = args.output or default_process_dir(capture_id)
    try:
        artifact = solve_user_workflow(args.input, output, workspace=args.workspace)
    except (CaptureError, OSError, RuntimeError, subprocess.SubprocessError, TypeError, ValueError) as exc:
        print(f"标定求解失败: {exc}", file=sys.stderr)
        return 2
    print(f"process 详细结果目录: {output.absolute()}")
    print(f"candidate 标定包绝对路径: {artifact.absolute()}")
    print(f"请查看 scene-04 测试叠加图: {output / 'test-overlay.png'}")
    print(f"板端先创建目录: ssh <板端用户>@<板端IP> 'mkdir -p {default_candidate_dir()}'")
    print(f"复制回板端示例: scp {artifact.absolute()} <板端用户>@<板端IP>:{default_candidate_dir() / artifact.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "capture":
            result, archive = capture_user_workflow(args.output, duration_s=args.duration)
            print(f"capture 已完成: {result}")
            print(f"复制到PC的文件: {archive}")
        else:
            capture_id = args.input.name.removesuffix(".raw.tar")
            output = args.output or default_process_dir(capture_id)
            result = solve_user_workflow(args.input, output, workspace=args.workspace)
            print(f"candidate 标定包已生成: {result}")
            print(f"请查看 scene-04 测试叠加图: {output / 'test-overlay.png'}")
            print(f"回传板端文件: {result}")
    except (
        CaptureError,
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        TypeError,
        ValueError,
    ) as exc:
        print(f"标定流程失败: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
