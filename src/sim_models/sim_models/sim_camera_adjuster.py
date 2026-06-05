"""sim_camera_adjuster.py — Interactive proxy camera pose adjuster for Gazebo.

Launches a static proxy camera in Gazebo and provides an OpenCV Trackbar
window to adjust its pose in real time.  The proxy camera publishes a ROS2
Image topic that can be consumed by any tool (e.g. camera_alignment).

Usage
-----
    # Terminal 1 — start proxy camera + trackbar
    ros2 run sim_models sim_camera_adjuster \\
        --camera /camera_align/top/image_raw \\
        --robot-config so101_single_arm \\
        --platform gazebo

    # Terminal 2 — view / align using the proxy camera topic
    ros2 run dataset_tools camera_alignment \\
        --use_sim \\
        --camera-source /camera_align/top/image_raw

Keys
----
    m : Toggle COARSE / FINE mode
    s : Save current pose to ~/.ros/ibrobot/sim_camera_overrides/<camera>.yaml
    r : Reset to initial pose (COARSE mode)
    q : Quit (despawns proxy camera, terminates ros_gz_bridge)

Design notes
------------
- Two precision modes share a single ``_current_pose`` dict (absolute values).
  Switching modes rebuilds the trackbar window centred on the current value.
- ros_gz_bridge is spawned as a subprocess and terminated on exit.
- Initial pose priority:
    1. ~/.ros/ibrobot/sim_camera_overrides/<camera>.yaml  (previous session)
    2. camera_presets.get_preset(platform, camera_name)   (use_default_transform=true)
    3. robot YAML transform field                          (use_default_transform=false)
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml
from ament_index_python.packages import get_package_share_directory

from dataset_tools.opencv_utils import require_opencv_gui

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OVERRIDE_BASE = Path.home() / ".ros" / "ibrobot" / "sim_camera_overrides"

# COARSE mode: 1 cm / step for xyz, 1 deg / step for angles, full range
_COARSE = {
    "xyz_step_m": 0.01,  # 1 cm per trackbar unit
    "xyz_range_m": 1.0,  # ±1 m total range → 200 steps
    "ang_step_rad": math.radians(1),  # 1° per unit
    "ang_range_rad": math.pi,  # ±180°  → 360 steps
    "fov_step_deg": 1.0,
    "fov_min": 20,
    "fov_max": 120,
}

# FINE mode: 0.1 mm / step for xyz, 0.1° / step for angles, ±10 mm / ±10°
_FINE = {
    "xyz_step_m": 0.0001,  # 0.1 mm per unit
    "xyz_half_m": 0.010,  # ±10 mm around current value → 200 steps
    "ang_step_rad": math.radians(0.1),  # 0.1° per unit
    "ang_half_rad": math.radians(10),  # ±10° around current value → 200 steps
    "fov_step_deg": 0.1,
    "fov_half_deg": 5.0,  # ±5° around current value
}

_STEPS = 200  # trackbar always has 0..200, centre at 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _camera_name_from_topic(topic: str) -> str:
    parts = topic.strip("/").split("/")
    return parts[1] if len(parts) >= 2 else topic.replace("/", "_")


def _resolve_proxy_topic(topic: str) -> str:
    """Map a real camera topic to its proxy camera topic.

    Examples
    --------
    /camera/top/image_raw       ->  /camera_align/top/image_raw
    /camera_align/top/image_raw ->  unchanged (already a proxy topic)
    """
    parts = topic.strip("/").split("/")
    if parts[0] == "camera_align":
        return topic  # already a proxy topic
    cam_name = parts[1] if len(parts) >= 2 else topic.replace("/", "_")
    return f"/camera_align/{cam_name}/image_raw"


def _load_initial_pose(
    camera_name: str,
    platform: str,
    robot_config: str | None,
) -> dict[str, float]:
    """Resolve initial pose with the priority chain documented above."""

    # 1. Check for existing override from a previous session
    # Skip stub files written by camera_alignment (they have _is_stub: true and all-zero pose).
    override_path = _OVERRIDE_BASE / f"{camera_name}.yaml"
    if override_path.exists():
        try:
            data = yaml.safe_load(override_path.read_text(encoding="utf-8"))
            p = data.get("pose", {})
            fovy = float(data.get("fovy_deg", 60.0))
            pose_vals = [p.get(k, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")]
            # Skip old-format stubs: _is_stub flag OR all-zero pose (never valid for a real camera)
            is_old_stub = all(v == 0.0 for v in pose_vals) and fovy == 60.0
            if data.get("_is_stub") or is_old_stub:
                print(f"[adjuster] Skipping all-zero stub override — using presets instead: {override_path}")
                override_path.unlink(missing_ok=True)  # Remove so presets take over permanently
            elif all(k in p for k in ("x", "y", "z", "roll", "pitch", "yaw")):
                print(f"[adjuster] Loaded initial pose from override: {override_path}")
                return {
                    "x": float(p["x"]),
                    "y": float(p["y"]),
                    "z": float(p["z"]),
                    "roll": float(p["roll"]),
                    "pitch": float(p["pitch"]),
                    "yaw": float(p["yaw"]),
                    "fovy_deg": fovy,
                }
        except Exception as exc:
            print(f"[adjuster] WARNING: failed to parse override YAML: {exc}")

    # 2. Try camera_presets (use_default_transform path)
    try:
        from robot_config.launch_builders.sim_backend.camera_presets import get_preset

        preset = get_preset(platform, camera_name)
        if preset:
            print(f"[adjuster] Loaded initial pose from camera_presets ({platform}/{camera_name})")
            return {
                "x": float(preset.get("x", 0.0)),
                "y": float(preset.get("y", 0.0)),
                "z": float(preset.get("z", 0.5)),
                "roll": float(preset.get("roll", 0.0)),
                "pitch": float(preset.get("pitch", 0.0)),
                "yaw": float(preset.get("yaw", 0.0)),
                "fovy_deg": float(preset.get("fovy", 60.0)),
            }
    except Exception as exc:
        print(f"[adjuster] camera_presets unavailable: {exc}")

    # 3. Try reading from robot YAML transform field
    if robot_config:
        try:
            pose = _load_pose_from_yaml(robot_config, camera_name)
            if pose:
                print(f"[adjuster] Loaded initial pose from robot YAML ({robot_config})")
                return pose
        except Exception as exc:
            print(f"[adjuster] WARNING: failed to read robot YAML: {exc}")

    # 4. Hard fallback (should rarely be reached)
    print("[adjuster] WARNING: using hard fallback pose (0, 0.5, 0.8)")
    return {"x": 0.0, "y": 0.0, "z": 0.5, "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "fovy_deg": 60.0}


def _load_pose_from_yaml(robot_config: str, camera_name: str) -> dict[str, float] | None:
    """Search ROS2 ament index for the robot YAML and extract the camera transform."""
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("robot_config")
        yaml_path = Path(share) / "config" / "robots" / f"{robot_config}.yaml"
        if not yaml_path.exists():
            return None
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        peripherals = data.get("robot", {}).get("peripherals", [])
        for cam in peripherals:
            if cam.get("name") == camera_name and cam.get("type") == "camera":
                t = cam.get("transform", {})
                return {
                    "x": float(t.get("x", 0.0)),
                    "y": float(t.get("y", 0.0)),
                    "z": float(t.get("z", 0.0)),
                    "roll": float(t.get("roll", 0.0)),
                    "pitch": float(t.get("pitch", 0.0)),
                    "yaw": float(t.get("yaw", 0.0)),
                    "fovy_deg": float(cam.get("fovy", 60.0)),
                }
    except Exception:
        pass
    return None


def _load_camera_geometry_from_yaml(robot_config: str, camera_name: str) -> dict[str, float]:
    defaults = {"width": 640, "height": 480, "fps": 30.0}
    try:
        share = Path(get_package_share_directory("robot_config"))
        yaml_path = share / "config" / "robots" / f"{robot_config}.yaml"
        if not yaml_path.exists():
            return defaults
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        peripherals = (data.get("robot", {}) or {}).get("peripherals", []) or []
        for peripheral in peripherals:
            if peripheral.get("type") != "camera" or peripheral.get("name") != camera_name:
                continue
            return {
                "width": int(peripheral.get("width", defaults["width"]) or defaults["width"]),
                "height": int(peripheral.get("height", defaults["height"]) or defaults["height"]),
                "fps": float(peripheral.get("fps", defaults["fps"]) or defaults["fps"]),
            }
    except Exception:
        return defaults
    return defaults


def _load_joint_names_from_yaml(robot_config: str) -> tuple[list[str], list[str]]:
    arm_defaults = ["1", "2", "3", "4", "5"]
    gripper_defaults = ["6"]
    try:
        share = Path(get_package_share_directory("robot_config"))
        yaml_path = share / "config" / "robots" / f"{robot_config}.yaml"
        if not yaml_path.exists():
            return arm_defaults, gripper_defaults
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        joints = (data.get("robot", {}) or {}).get("joints", {}) or {}
        arm = [str(name) for name in joints.get("arm", arm_defaults)]
        gripper = [str(name) for name in joints.get("gripper", gripper_defaults)]
        return arm or arm_defaults, gripper or gripper_defaults
    except Exception:
        return arm_defaults, gripper_defaults


def _env_with_domain(base_env: dict[str, str], domain: int, *, headless: bool = False) -> dict[str, str]:
    env = base_env.copy()
    env["ROS_DOMAIN_ID"] = str(domain)
    if headless:
        env["IBROBOT_GAZEBO_HEADLESS"] = "1"
        env["QT_QPA_PLATFORM"] = "offscreen"
    return env


def _wait_for_active_controllers(env: dict[str, str], timeout_s: float = 120.0) -> None:
    result = subprocess.run(
        [
            "ros2",
            "run",
            "robot_config",
            "wait_for_controllers",
            "joint_state_broadcaster",
            "arm_position_controller",
            "gripper_position_controller",
            "--controller-manager",
            "controller_manager",
            "--timeout",
            str(timeout_s),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s + 10.0,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        raise RuntimeError(f"Timed out waiting for calibration controllers: {detail}")
    print("[adjuster] Calibration controllers are active.")


def _load_first_yaml_document(text: str) -> dict | None:
    for document in yaml.safe_load_all(text):
        if isinstance(document, dict):
            return document
    return None


def _extract_joint_state_yaml(text: str) -> str:
    """Strip ros2 topic echo noise and keep only the JointState YAML payload."""
    lines = text.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line.startswith("header:")),
        None,
    )
    if start is None:
        raise RuntimeError("No JointState YAML payload found in ros2 topic echo output.")

    cleaned: list[str] = []
    for line in lines[start:]:
        if line.startswith("A message was lost!!!"):
            break
        if line.startswith("---"):
            continue
        cleaned.append(line)

    payload = "\n".join(cleaned).strip()
    if not payload:
        raise RuntimeError("JointState YAML payload was empty after sanitizing ros2 topic echo output.")
    return payload


def _read_joint_state_snapshot(
    env: dict[str, str],
    arm_joint_names: list[str],
    gripper_joint_names: list[str],
    timeout_s: float = 10.0,
) -> tuple[list[float], list[float]] | None:
    result = subprocess.run(
        ["ros2", "topic", "echo", "--once", "/joint_states"],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        raise RuntimeError(f"Failed to read /joint_states: {detail}")

    payload = _extract_joint_state_yaml(result.stdout)
    document = _load_first_yaml_document(payload)
    if not document:
        return None

    names = [str(name) for name in document.get("name", [])]
    positions = [float(position) for position in document.get("position", [])]
    if not names or len(names) != len(positions):
        return None

    by_name = dict(zip(names, positions, strict=True))
    arm = [by_name[name] for name in arm_joint_names if name in by_name]
    gripper = [by_name[name] for name in gripper_joint_names if name in by_name]
    if len(arm) != len(arm_joint_names):
        missing = [name for name in arm_joint_names if name not in by_name]
        raise RuntimeError(f"Primary /joint_states is missing arm joints: {missing}")
    if gripper_joint_names and len(gripper) != len(gripper_joint_names):
        missing = [name for name in gripper_joint_names if name not in by_name]
        raise RuntimeError(f"Primary /joint_states is missing gripper joints: {missing}")
    return arm, gripper


def _publish_joint_command(env: dict[str, str], topic: str, values: list[float], timeout_s: float = 8.0) -> None:
    payload = "{data: [" + ", ".join(f"{value:.12g}" for value in values) + "]}"
    result = subprocess.run(
        ["ros2", "topic", "pub", "--once", topic, "std_msgs/msg/Float64MultiArray", payload],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        raise RuntimeError(f"Failed to publish {topic}: {detail}")


def _sync_calibration_robot_pose_once(
    primary_env: dict[str, str],
    calibration_env: dict[str, str],
    robot_config: str,
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    arm_joint_names, gripper_joint_names = _load_joint_names_from_yaml(robot_config)
    snapshot = _read_joint_state_snapshot(primary_env, arm_joint_names, gripper_joint_names)
    if snapshot is None:
        return None
    arm_positions, gripper_positions = snapshot
    _publish_joint_command(calibration_env, "/arm_position_controller/commands", arm_positions)
    if gripper_positions:
        _publish_joint_command(calibration_env, "/gripper_position_controller/commands", gripper_positions)
    return tuple(arm_positions), tuple(gripper_positions)


def _save_override(
    camera_name: str,
    topic: str,
    pose: dict[str, float],
    parent_frame: str = "base",
) -> None:
    """Write pose override YAML.  *pose* is always link-relative for TF cameras
    (parent_frame != 'base') and world-absolute for static cameras."""
    _OVERRIDE_BASE.mkdir(parents=True, exist_ok=True)
    path = _OVERRIDE_BASE / f"{camera_name}.yaml"
    content = {
        "camera": camera_name,
        "topic": topic,
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        # parent_frame tells readers whether pose is world-absolute or link-relative.
        # 'base' (or absent) = world-absolute (top, front cameras).
        # 'gripper' etc.     = offset relative to that link (wrist camera).
        "parent_frame": parent_frame,
        "pose": {k: round(float(pose[k]), 6) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
        "fovy_deg": round(float(pose["fovy_deg"]), 2),
    }
    path.write_text(yaml.dump(content, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[adjuster] Override saved: {path}")


def _detect_platform_from_yaml(robot_config: str) -> str | None:
    """Read ``simulation.platform`` from the robot YAML (returns None on failure)."""
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("robot_config")
        yaml_path = Path(share) / "config" / "robots" / f"{robot_config}.yaml"
        if not yaml_path.exists():
            return None
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        return (data.get("robot", {}).get("simulation", {}) or {}).get("platform")
    except Exception:
        return None


def _write_calibration_yaml_patch(robot_config: str) -> Path:
    """Produce a patched YAML for headless Gazebo calibration.

    The patch:
      - forces ``simulation.platform: gazebo``
      - disables teleop / voice_asr / recording / all inference to keep the
        auxiliary stack as light as possible
    Returns the path to the patched file under ``/tmp``.
    The original YAML is **not** modified.
    """
    import os as _os

    from ament_index_python.packages import get_package_share_directory

    share = get_package_share_directory("robot_config")
    src = Path(share) / "config" / "robots" / f"{robot_config}.yaml"
    if not src.exists():
        raise FileNotFoundError(f"Robot YAML not found: {src}")

    data = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    robot = data.setdefault("robot", {})

    sim = robot.setdefault("simulation", {})
    sim["platform"] = "gazebo"

    # Best-effort disable heavy subsystems; keys may or may not exist.
    for key in ("teleoperation", "voice_asr", "recording"):
        section = robot.get(key)
        if isinstance(section, dict):
            section["enabled"] = False
    modes = robot.get("control_modes", {}) or {}
    for _mode_name, mode_cfg in modes.items():
        if isinstance(mode_cfg, dict):
            inf = mode_cfg.get("inference")
            if isinstance(inf, dict):
                inf["enabled"] = False

    # Tag so it's obvious in logs.
    robot["_calibration_patch"] = True

    out_dir = Path(_os.environ.get("TMPDIR", "/tmp"))
    out_path = out_dir / f"ibrobot_calib_{robot_config}.yaml"
    out_path.write_text(
        yaml.dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return out_path


class _CalibrationGazebo:
    """Lifecycle manager for a headless Gazebo stack spawned for camera
    calibration while MuJoCo owns the primary ROS_DOMAIN_ID.

    On ``__init__`` it:
      1. picks an isolated ROS_DOMAIN_ID (base + 1, overridable via
         IBROBOT_CALIB_DOMAIN);
      2. writes a patched YAML that forces platform=gazebo and strips
         teleop/inference/recording/voice_asr;
      3. launches ``ros2 launch robot_config robot.launch.py`` in that
         isolated domain with IBROBOT_GAZEBO_HEADLESS=1;
      4. waits until /clock is observed on that domain (or times out).

    On ``stop`` it terminates the launch process tree.
    """

    def __init__(self, robot_config: str, world_name: str) -> None:
        import signal as _signal

        self._robot_config = robot_config
        self._world_name = world_name
        self._proc: subprocess.Popen | None = None
        self._patched_yaml: Path | None = None
        self._signal = _signal
        self._mirror_stop = threading.Event()
        self._mirror_thread: threading.Thread | None = None
        self._last_mirrored_pose: tuple[tuple[float, ...], tuple[float, ...]] | None = None

        base_domain = int(os.environ.get("ROS_DOMAIN_ID", "0") or "0")
        override = os.environ.get("IBROBOT_CALIB_DOMAIN")
        self._base_domain = base_domain
        self._domain = int(override) if override else (base_domain + 1) % 128
        if self._domain == base_domain:
            self._domain = (base_domain + 2) % 128

        print(
            f"[adjuster] platform=mujoco detected -> starting headless Gazebo "
            f"for calibration on ROS_DOMAIN_ID={self._domain} "
            f"(primary domain was {base_domain})"
        )

        self._patched_yaml = _write_calibration_yaml_patch(robot_config)
        print(f"[adjuster] Patched YAML: {self._patched_yaml}")

        self._primary_env = _env_with_domain(dict(os.environ), self._base_domain)
        self._calibration_env = _env_with_domain(dict(os.environ), self._domain, headless=True)
        # Sanity: some users inherit LIBGL_ALWAYS_SOFTWARE from a prior session;
        # leaving it alone is fine.

        cmd = [
            "ros2",
            "launch",
            "robot_config",
            "robot.launch.py",
            f"config_path:={self._patched_yaml}",
            f"robot_config:={robot_config}",
            "use_sim:=true",
            "control_mode:=teleop",
            "auto_start_controllers:=true",
            "record:=false",
            "with_inference:=false",
            "with_moveit:=false",
        ]
        print(f"[adjuster] Launch: {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd,
            env=self._calibration_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,  # new process group for clean teardown
        )
        print(f"[adjuster] Calibration Gazebo launched (pid={self._proc.pid})")

        self._wait_for_clock(self._calibration_env, timeout_s=90.0)
        self._wait_for_calibration_ready()
        try:
            self._mirror_primary_pose_once()
        except Exception as exc:
            print(f"[adjuster] WARNING: failed to mirror initial calibration robot pose: {exc}")
        self._start_pose_mirror()

        # Export domain to parent so rclpy + ign bridge inherit it.
        os.environ["ROS_DOMAIN_ID"] = str(self._domain)
        print(
            f"[adjuster] NOTE: calibration runs on ROS_DOMAIN_ID={self._domain}. "
            f"Start camera_alignment in that same domain, e.g.:\n"
            f"           ROS_DOMAIN_ID={self._domain} ros2 run dataset_tools "
            f"camera_alignment --use_sim --camera-source /camera_align/<cam>/image_raw"
        )

    @property
    def domain(self) -> int:
        return self._domain

    def _wait_for_clock(self, env: dict, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        last_err = ""
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(f"Calibration Gazebo exited prematurely with code {self._proc.returncode}")
            try:
                out = subprocess.run(
                    ["ros2", "topic", "list"],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
                if "/clock" in out.stdout.splitlines():
                    print("[adjuster] Calibration Gazebo /clock ready.")
                    # Give controllers a few seconds to load too.
                    time.sleep(3.0)
                    return
                last_err = out.stderr.strip().splitlines()[-1] if out.stderr else ""
            except Exception as exc:
                last_err = str(exc)
            time.sleep(1.5)
        raise TimeoutError(
            f"Timed out waiting for /clock on calibration Gazebo (ROS_DOMAIN_ID={self._domain}). Last error: {last_err}"
        )

    def _wait_for_calibration_ready(self) -> None:
        _wait_for_active_controllers(self._calibration_env, timeout_s=120.0)

    def _mirror_primary_pose_once(self) -> None:
        mirrored = _sync_calibration_robot_pose_once(
            self._primary_env,
            self._calibration_env,
            self._robot_config,
        )
        if mirrored is None:
            print("[adjuster] WARNING: primary joint snapshot was empty; skipping initial pose mirror.")
            return
        self._last_mirrored_pose = mirrored
        print(
            f"[adjuster] Calibration robot pose mirrored from primary ROS_DOMAIN_ID="
            f"{self._base_domain} to calibration ROS_DOMAIN_ID={self._domain}"
        )

    def _pose_mirror_loop(self) -> None:
        while not self._mirror_stop.wait(1.0):
            if self._proc is not None and self._proc.poll() is not None:
                return
            try:
                mirrored = _sync_calibration_robot_pose_once(
                    self._primary_env,
                    self._calibration_env,
                    self._robot_config,
                )
                if mirrored is None or mirrored == self._last_mirrored_pose:
                    continue
                self._last_mirrored_pose = mirrored
            except Exception as exc:
                print(f"[adjuster] WARNING: calibration pose mirror update failed: {exc}")

    def _start_pose_mirror(self) -> None:
        self._mirror_thread = threading.Thread(
            target=self._pose_mirror_loop,
            name="calibration_pose_mirror",
            daemon=True,
        )
        self._mirror_thread.start()
        print(
            f"[adjuster] Calibration pose mirror started: primary ROS_DOMAIN_ID="
            f"{self._base_domain} -> calibration ROS_DOMAIN_ID={self._domain}"
        )

    def stop(self) -> None:
        self._mirror_stop.set()
        if self._mirror_thread is not None:
            self._mirror_thread.join(timeout=2.0)
        if self._proc is None:
            return
        if self._proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self._proc.pid, self._signal.SIGINT)
            try:
                self._proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                print("[adjuster] Calibration Gazebo did not exit on SIGINT; killing.")
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self._proc.pid, self._signal.SIGKILL)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self._proc.wait(timeout=5.0)
        print("[adjuster] Calibration Gazebo terminated.")
        self._proc = None


def _start_ros_gz_bridge(topic: str) -> subprocess.Popen:
    """Start ros_gz_bridge for the proxy camera topic."""
    # Bridge the full topic path (including /image_raw) so the ROS2 topic
    # matches exactly what callers subscribe to.
    bridge_topic = f"{topic}@sensor_msgs/msg/Image[gz.msgs.Image"
    proc = subprocess.Popen(
        ["ros2", "run", "ros_gz_bridge", "parameter_bridge", bridge_topic],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"[adjuster] ros_gz_bridge started (pid={proc.pid}): {bridge_topic}")
    return proc


def _wait_for_ros_topic(
    topic: str,
    timeout_s: float = 10.0,
    env: dict[str, str] | None = None,
) -> bool:
    """Wait until a ROS topic becomes visible in the current domain."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            out = subprocess.run(
                ["ros2", "topic", "list"],
                env=env,
                capture_output=True,
                text=True,
                timeout=3.0,
            )
            if topic in out.stdout.splitlines():
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


# ---------------------------------------------------------------------------
# TF snapshot helpers (wrist / moving-frame cameras)
# ---------------------------------------------------------------------------

# Parent frames that are fixed in the world — no TF lookup needed.
# "base" is intentionally NOT in this set: the so101 robot spawns at
# (x=0, y=0.30, z=-0.03) in world, so base ≠ world origin.  We must do a
# TF lookup for "base" to convert world coords → base-relative before saving,
# exactly the same as we do for "gripper".  Only truly static frames
# (map/odom/world) skip the TF lookup.
_STATIC_PARENT_FRAMES = frozenset({"world", "map", "odom"})


def _quat_to_rpy(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
    """Convert quaternion (x,y,z,w) to roll-pitch-yaw in radians (ZYX)."""
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return roll, pitch, yaw


def _compose_world_pose(
    parent_xyz: tuple,
    parent_rpy: tuple,
    offset_xyz: tuple,
    offset_rpy: tuple,
) -> tuple[tuple, tuple]:
    """Compose T_world_cam = T_world_parent @ T_parent_cam.

    All angles in radians (ZYX convention).  Returns (xyz, rpy).
    """
    import numpy as np

    def _mat4(xyz, rpy) -> np.ndarray:
        rx, ry, rz = rpy
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        T = np.eye(4)
        T[:3, :3] = Rz @ Ry @ Rx
        T[:3, 3] = xyz
        return T

    T_wc = _mat4(parent_xyz, parent_rpy) @ _mat4(offset_xyz, offset_rpy)
    xyz_out: tuple = (T_wc[0, 3], T_wc[1, 3], T_wc[2, 3])

    # Extract ZYX RPY from rotation matrix
    R = T_wc[:3, :3]
    p = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(math.cos(p)) > 1e-6:
        r = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        r = math.atan2(-R[1, 2], R[1, 1])
        y = 0.0
    rpy_out: tuple = (r, p, y)
    return xyz_out, rpy_out


def _lookup_tf_snapshot(
    parent_frame: str,
    world_frame: str = "world",
    timeout_s: float = 5.0,
) -> tuple[tuple, tuple] | None:
    """One-shot TF lookup for a fixed calibration posture.

    Starts a temporary rclpy node, waits up to *timeout_s* for the transform,
    destroys the node, then returns (xyz, rpy).  Returns None on any failure —
    caller must fall back to Static mode gracefully.

    This function is only called when parent_frame is not in _STATIC_PARENT_FRAMES
    (e.g. "gripper" for a wrist camera).  It is never called on the real-machine
    code path because SimCameraAdjuster is a sim-only class.
    """
    try:
        import rclpy
        import tf2_ros
    except ImportError:
        print("[adjuster] WARNING: tf2_ros not available — using Static mode")
        return None

    import threading

    if not rclpy.ok():
        rclpy.init()

    node = rclpy.create_node("_adjuster_tf_snapshot")
    buf = tf2_ros.Buffer()
    _lst = tf2_ros.TransformListener(buf, node)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    deadline = time.monotonic() + timeout_s
    result: tuple[tuple, tuple] | None = None
    while time.monotonic() < deadline:
        try:
            t = buf.lookup_transform(world_frame, parent_frame, rclpy.time.Time())
            tr = t.transform.translation
            ro = t.transform.rotation
            result = (
                (tr.x, tr.y, tr.z),
                _quat_to_rpy(ro.x, ro.y, ro.z, ro.w),
            )
            break
        except Exception:
            time.sleep(0.05)

    node.destroy_node()
    if result is None:
        print(
            f"[adjuster] WARNING: TF '{world_frame}<-{parent_frame}' not available "
            f"after {timeout_s:.0f}s — Static fallback (trackbar controls world pose directly)"
        )
    return result


def _lookup_gazebo_link_pose(
    model_name: str,
    link_name: str,
    world_name: str = "demo",
    timeout_s: float = 5.0,
) -> tuple[tuple, tuple] | None:
    """One-shot query of a model link's world pose via Ignition Gazebo CLI.

    Unlike TF (which needs robot_state_publisher), this reads directly from
    Gazebo's ``/world/{world}/dynamic_pose/info`` topic which is always
    publishing whenever Gazebo is running.  This is the reliable fallback when
    ROS TF is missing.

    Returns (xyz, rpy) of the link in world frame, or None on failure.
    """
    import subprocess

    # Collect several messages because dynamic_pose/info is chunked (root frame
    # only reports whole-model pose; link-level poses come in later samples).
    try:
        proc = subprocess.Popen(
            ["ign", "topic", "-e", "-t", f"/world/{world_name}/dynamic_pose/info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError:
        print("[adjuster] WARNING: 'ign' CLI not found — cannot query Gazebo link pose")
        return None

    target = f"{model_name}::{link_name}"
    lines: list[str] = []
    deadline = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline:
            import select as _select

            r, _, _ = _select.select([proc.stdout], [], [], 0.2)
            if not r:
                continue
            line = proc.stdout.readline()
            if not line:
                break
            lines.append(line)
            # Once we've seen the target, wait for its orientation line, then parse full buffer.
            if target in line and len(lines) > 200:
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except Exception:
            proc.kill()

    text = "".join(lines)

    # Protobuf text-format blocks look like:
    #   pose {
    #     name: "so101_single_arm::gripper"
    #     id: 23
    #     position { x: 0.1 y: 0.3 z: 0.2 }
    #     orientation { x: 0 y: 0 z: 0 w: 1 }
    #   }
    # Parse with a forward-scan state machine because nested braces rule out regex.
    import re as _re

    n = len(text)
    i = 0
    name_pat = _re.compile(r'name:\s*"([^"]+)"')
    num_pat = _re.compile(r"([xyzw]):\s*(-?[\deE\.\+\-]+)")
    best: tuple[tuple, tuple] | None = None
    while i < n:
        # find next top-level "pose {"
        j = text.find("pose {", i)
        if j < 0:
            break
        # find matching closing brace
        depth = 0
        k = j
        while k < n:
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        if k >= n:
            break
        block = text[j : k + 1]
        i = k + 1
        m = name_pat.search(block)
        if not m or m.group(1) != target:
            continue
        # extract position { ... }
        pos_match = _re.search(r"position\s*\{([^}]*)\}", block)
        ori_match = _re.search(r"orientation\s*\{([^}]*)\}", block)
        if not pos_match or not ori_match:
            continue
        pos = {k: float(v) for k, v in num_pat.findall(pos_match.group(1))}
        ori = {k: float(v) for k, v in num_pat.findall(ori_match.group(1))}
        xyz = (pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0))
        rpy = _quat_to_rpy(
            ori.get("x", 0.0),
            ori.get("y", 0.0),
            ori.get("z", 0.0),
            ori.get("w", 1.0),
        )
        best = (xyz, rpy)
        # keep scanning — later samples are more accurate after Gazebo settles

    if best is None:
        print(
            f"[adjuster] WARNING: Gazebo link '{target}' not found in "
            f"/world/{world_name}/dynamic_pose/info after {timeout_s:.0f}s"
        )
    return best


def _get_parent_frame(camera_name: str, platform: str) -> str:
    """Return parent_frame from camera_presets (e.g. 'gripper' or 'base')."""
    try:
        from robot_config.launch_builders.sim_backend.camera_presets import get_preset

        preset = get_preset(platform, camera_name)
        if preset:
            return str(preset.get("parent_frame", "base"))
    except Exception:
        pass
    return "base"


# ---------------------------------------------------------------------------
# Trackbar window
# ---------------------------------------------------------------------------


class _TrackbarWindow:
    """Manages OpenCV trackbar window for one precision mode."""

    COARSE = "COARSE"
    FINE = "FINE"

    def __init__(self, camera_name: str, pose: dict[str, float], tf_label: str = "") -> None:
        # Use require_opencv_gui() to get the GTK/Qt-capable system OpenCV,
        # not the headless venv build which lacks HighGUI window support.
        self._cv2 = require_opencv_gui()

        self._cam = camera_name
        self._pose = pose  # shared mutable dict — always absolute values
        self._mode = self.COARSE
        self._win: str = ""
        self._fine_centre: dict[str, float] = {}
        self._last_touched_key: str = "x"  # for Up/Down arrow step
        self._input_buffer: str = ""  # for numeric keyboard input
        self._saved_flash_ticks: int = 0
        self._saved_pose_snapshot: dict = {}
        # tf_label: non-empty => TF mode (trackbar = link-relative offset)
        # empty    => Static mode (trackbar = world-absolute pose)
        self._tf_label: str = tf_label
        self._rebuild()

    # ------------------------------------------------------------------
    def _win_name(self) -> str:
        return f"Camera_Adjuster [{self._cam} | {self._mode}]"

    def _rebuild(self) -> None:
        """Destroy old window and recreate trackbars for current mode."""
        cv2 = self._cv2
        if self._win:
            cv2.destroyWindow(self._win)
            time.sleep(0.05)

        self._win = self._win_name()
        # Create a dummy black image so the window has content
        import numpy as np

        img = np.zeros((1, 400, 3), dtype=np.uint8)
        cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
        cv2.imshow(self._win, img)

        if self._mode == self.COARSE:
            self._build_coarse(cv2)
        else:
            self._fine_centre = {k: self._pose[k] for k in ("x", "y", "z", "roll", "pitch", "yaw", "fovy_deg")}
            self._build_fine(cv2)

    def _build_coarse(self, cv2: Any) -> None:
        cfg = _COARSE
        win = self._win

        def _make_xyz(key: str) -> None:
            val_steps = round((self._pose[key] + cfg["xyz_range_m"]) / cfg["xyz_step_m"])
            val_steps = max(0, min(_STEPS * 2, val_steps))
            total = round(cfg["xyz_range_m"] * 2 / cfg["xyz_step_m"])
            cv2.createTrackbar(key + " (cm)", win, val_steps, total, lambda v, k=key: self._on_xyz_coarse(k, v))

        def _make_ang(key: str, label: str) -> None:
            val_steps = round((self._pose[key] + cfg["ang_range_rad"]) / cfg["ang_step_rad"])
            val_steps = max(0, min(round(cfg["ang_range_rad"] * 2 / cfg["ang_step_rad"]), val_steps))
            total = round(cfg["ang_range_rad"] * 2 / cfg["ang_step_rad"])
            cv2.createTrackbar(label, win, val_steps, total, lambda v, k=key: self._on_ang_coarse(k, v))

        for ax in ("x", "y", "z"):
            _make_xyz(ax)
        for ax, lbl in [("roll", "roll (deg)"), ("pitch", "pitch (deg)"), ("yaw", "yaw (deg)")]:
            _make_ang(ax, lbl)

        fov_val = round(self._pose["fovy_deg"]) - cfg["fov_min"]
        fov_val = max(0, min(cfg["fov_max"] - cfg["fov_min"], fov_val))
        cv2.createTrackbar(
            "fovy (deg)", win, fov_val, cfg["fov_max"] - cfg["fov_min"], lambda v: self._on_fov_coarse(v)
        )

    def _build_fine(self, cv2: Any) -> None:
        win = self._win

        def _make_xyz_fine(key: str) -> None:
            cv2.createTrackbar(key + " (0.1mm)", win, _STEPS // 2, _STEPS, lambda v, k=key: self._on_xyz_fine(k, v))

        def _make_ang_fine(key: str, label: str) -> None:
            cv2.createTrackbar(label, win, _STEPS // 2, _STEPS, lambda v, k=key: self._on_ang_fine(k, v))

        for ax in ("x", "y", "z"):
            _make_xyz_fine(ax)
        for ax, lbl in [("roll", "roll (0.1deg)"), ("pitch", "pitch (0.1deg)"), ("yaw", "yaw (0.1deg)")]:
            _make_ang_fine(ax, lbl)

        cv2.createTrackbar("fovy (0.1deg)", win, _STEPS // 2, _STEPS, lambda v: self._on_fov_fine(v))

    # ------------------------------------------------------------------
    # COARSE callbacks — convert trackbar position to absolute value

    def _on_xyz_coarse(self, key: str, val: int) -> None:
        self._pose[key] = round(val * _COARSE["xyz_step_m"] - _COARSE["xyz_range_m"], 4)
        self._last_touched_key = key

    def _on_ang_coarse(self, key: str, val: int) -> None:
        self._pose[key] = round(val * _COARSE["ang_step_rad"] - _COARSE["ang_range_rad"], 6)
        self._last_touched_key = key

    def _on_fov_coarse(self, val: int) -> None:
        self._pose["fovy_deg"] = val + _COARSE["fov_min"]
        self._last_touched_key = "fovy_deg"

    # FINE callbacks — delta from centre

    def _on_xyz_fine(self, key: str, val: int) -> None:
        delta = (val - _STEPS // 2) * _FINE["xyz_step_m"]
        self._pose[key] = round(self._fine_centre[key] + delta, 6)
        self._last_touched_key = key

    def _on_ang_fine(self, key: str, val: int) -> None:
        delta = (val - _STEPS // 2) * _FINE["ang_step_rad"]
        self._pose[key] = round(self._fine_centre[key] + delta, 6)
        self._last_touched_key = key

    def _on_fov_fine(self, val: int) -> None:
        delta = (val - _STEPS // 2) * _FINE["fov_step_deg"]
        self._pose["fovy_deg"] = round(self._fine_centre["fovy_deg"] + delta, 2)
        self._last_touched_key = "fovy_deg"

    # ------------------------------------------------------------------

    def toggle_mode(self) -> None:
        self._mode = self.FINE if self._mode == self.COARSE else self.COARSE
        self._rebuild()
        print(f"[adjuster] Switched to {self._mode} mode")

    def reset(self, initial_pose: dict[str, float]) -> None:
        self._pose.update(initial_pose)
        self._mode = self.COARSE
        self._rebuild()
        print("[adjuster] Reset to initial pose")

    # ------------------------------------------------------------------
    # Helpers for arrow-key step and numeric input

    def _trackbar_label(self, key: str) -> str:
        if self._mode == self.COARSE:
            m = {
                "x": "x (cm)",
                "y": "y (cm)",
                "z": "z (cm)",
                "roll": "roll (deg)",
                "pitch": "pitch (deg)",
                "yaw": "yaw (deg)",
                "fovy_deg": "fovy (deg)",
            }
        else:
            m = {
                "x": "x (0.1mm)",
                "y": "y (0.1mm)",
                "z": "z (0.1mm)",
                "roll": "roll (0.1deg)",
                "pitch": "pitch (0.1deg)",
                "yaw": "yaw (0.1deg)",
                "fovy_deg": "fovy (0.1deg)",
            }
        return m.get(key, key)

    def _pose_to_trackbar_pos(self, key: str) -> int:
        val = self._pose[key]
        if self._mode == self.COARSE:
            if key in ("x", "y", "z"):
                total = round(_COARSE["xyz_range_m"] * 2 / _COARSE["xyz_step_m"])
                return max(0, min(total, round((val + _COARSE["xyz_range_m"]) / _COARSE["xyz_step_m"])))
            elif key == "fovy_deg":
                return max(0, min(_COARSE["fov_max"] - _COARSE["fov_min"], round(val) - _COARSE["fov_min"]))
            else:
                total = round(_COARSE["ang_range_rad"] * 2 / _COARSE["ang_step_rad"])
                return max(0, min(total, round((val + _COARSE["ang_range_rad"]) / _COARSE["ang_step_rad"])))
        else:
            centre = self._fine_centre
            if key in ("x", "y", "z"):
                return max(0, min(_STEPS, round((val - centre[key]) / _FINE["xyz_step_m"]) + _STEPS // 2))
            elif key == "fovy_deg":
                return max(0, min(_STEPS, round((val - centre[key]) / _FINE["fov_step_deg"]) + _STEPS // 2))
            else:
                return max(0, min(_STEPS, round((val - centre[key]) / _FINE["ang_step_rad"]) + _STEPS // 2))

    def _sync_trackbar(self, key: str) -> None:
        with contextlib.suppress(Exception):
            self._cv2.setTrackbarPos(self._trackbar_label(key), self._win, self._pose_to_trackbar_pos(key))

    def _step_last_touched(self, direction: int) -> None:
        key = self._last_touched_key
        if key not in self._pose:
            return
        if self._mode == self.COARSE:
            steps = {
                "x": _COARSE["xyz_step_m"],
                "y": _COARSE["xyz_step_m"],
                "z": _COARSE["xyz_step_m"],
                "fovy_deg": _COARSE["fov_step_deg"],
            }
            step = steps.get(key, _COARSE["ang_step_rad"])
        else:
            steps = {
                "x": _FINE["xyz_step_m"],
                "y": _FINE["xyz_step_m"],
                "z": _FINE["xyz_step_m"],
                "fovy_deg": _FINE["fov_step_deg"],
            }
            step = steps.get(key, _FINE["ang_step_rad"])
        self._pose[key] = round(self._pose[key] + step * direction, 6)
        self._sync_trackbar(key)

    def _apply_input(self) -> None:
        buf = self._input_buffer
        self._input_buffer = ""
        if not buf or buf in ("-", "."):
            return
        try:
            val = float(buf)
            key = self._last_touched_key
            if key in ("x", "y", "z"):
                self._pose[key] = round(val / 100.0, 6)  # user types cm
            elif key == "fovy_deg":
                self._pose[key] = max(20.0, min(120.0, round(val, 2)))
            else:
                self._pose[key] = round(math.radians(val), 6)  # user types degrees
            self._sync_trackbar(key)
        except ValueError:
            pass

    # ------------------------------------------------------------------

    def flash_saved(self, pose: dict) -> None:
        self._saved_flash_ticks = 40
        self._saved_pose_snapshot = dict(pose)

    def wait_key(self, delay_ms: int = 50) -> int:
        cv2 = self._cv2
        import numpy as np

        p = self._pose
        x_cm = p["x"] * 100
        y_cm = p["y"] * 100
        z_cm = p["z"] * 100
        roll_d = math.degrees(p["roll"])
        pitch_d = math.degrees(p["pitch"])
        yaw_d = math.degrees(p["yaw"])
        fov_d = p["fovy_deg"]

        h = 90
        img = np.zeros((h, 980, 3), dtype=np.uint8)

        # Line 1: actual current values (truth — trackbar position != actual value)
        # Prefix: [TF:gripper] = offset relative to link; [STATIC] = world-absolute
        mode_prefix = f"[TF:{self._tf_label}] offset " if self._tf_label else "[STATIC] "
        vals = (
            mode_prefix + f"x={x_cm:.1f}cm  y={y_cm:.1f}cm  z={z_cm:.1f}cm  "
            f"roll={roll_d:.1f}deg  pitch={pitch_d:.1f}deg  "
            f"yaw={yaw_d:.1f}deg  fovy={fov_d:.1f}deg"
        )
        cv2.putText(img, vals, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (200, 210, 255), 1)

        # Line 2: SAVED flash OR key hint with last-touched param
        if self._saved_flash_ticks > 0:
            self._saved_flash_ticks -= 1
            sn = self._saved_pose_snapshot
            saved_line = (
                f"SAVED  x={sn['x'] * 100:.1f}cm y={sn['y'] * 100:.1f}cm "
                f"z={sn['z'] * 100:.1f}cm  fovy={sn['fovy_deg']:.1f}deg"
                "  -- check ghost window (v in camera_alignment)"
            )
            cv2.putText(img, saved_line, (8, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 0), 1)
        else:
            lk = self._last_touched_key
            hint = f"[{lk}]  S=save  R=reset  M=mode  Up/Down=step  type digits+Enter=set value  Q=quit"
            cv2.putText(img, hint, (8, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (140, 220, 140), 1)

        # Line 3: numeric input buffer
        if self._input_buffer:
            unit = "cm" if self._last_touched_key in ("x", "y", "z") else "deg"
            inp_line = f"Set [{self._last_touched_key}] ({unit}): {self._input_buffer}_   Enter=apply  Esc=cancel"
            cv2.putText(img, inp_line, (8, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 255), 1)

        cv2.imshow(self._win, img)

        raw = cv2.waitKey(delay_ms)
        if raw == -1:
            return 0xFF  # no key pressed

        # Arrow keys on Linux X11
        UP_KEY = 65362
        DOWN_KEY = 65364
        if raw == UP_KEY:
            self._step_last_touched(+1)
            return 0
        elif raw == DOWN_KEY:
            self._step_last_touched(-1)
            return 0

        c = raw & 0xFF

        # Numeric input: digits, minus sign (first char), decimal point
        if 0x30 <= c <= 0x39:  # 0-9
            self._input_buffer += chr(c)
            return 0
        if c == ord("-") and not self._input_buffer:
            self._input_buffer = "-"
            return 0
        if c == ord(".") and "." not in self._input_buffer:
            self._input_buffer += "."
            return 0
        if c == 13:  # Enter — apply buffered number
            self._apply_input()
            return 0
        if c == 27:  # Escape — cancel input
            self._input_buffer = ""
            return 0

        # For any regular key, discard buffered input to avoid accidental apply
        if self._input_buffer and c not in (0xFF,):
            self._input_buffer = ""

        return c

    def destroy(self) -> None:
        with contextlib.suppress(Exception):
            self._cv2.destroyWindow(self._win)


# ---------------------------------------------------------------------------
# Main adjuster logic
# ---------------------------------------------------------------------------


class SimCameraAdjuster:
    """Spawn a proxy camera in Gazebo and expose trackbar controls."""

    def __init__(
        self,
        topic: str,
        platform: str = "gazebo",
        robot_config: str = "so101_single_arm",
        world_name: str = "demo",
    ) -> None:
        from sim_models.aruco_spawner import (
            despawn_alignment_camera_gazebo,
            set_alignment_camera_pose_gazebo,
            spawn_alignment_camera_gazebo,
        )

        self._topic = topic
        self._cam_name = _camera_name_from_topic(topic)
        self._platform = platform
        self._robot_config = robot_config
        self._world = world_name
        self._spawn_fn = spawn_alignment_camera_gazebo
        self._set_pose_fn = set_alignment_camera_pose_gazebo
        self._despawn_fn = despawn_alignment_camera_gazebo
        self._geometry = _load_camera_geometry_from_yaml(robot_config, self._cam_name)
        print(
            f"[adjuster] Proxy camera geometry: "
            f"{int(self._geometry['width'])}x{int(self._geometry['height'])}"
            f"@{float(self._geometry['fps']):g}"
        )

        # Resolve initial pose (always stored as link-relative offset for TF cameras,
        # or world-absolute for static cameras — camera_presets stores the right values)
        self._initial_pose = _load_initial_pose(self._cam_name, platform, robot_config)
        self._pose = dict(self._initial_pose)

        # --- Parent-link world-pose snapshot for moving-frame cameras ---
        # Detect whether the camera is mounted on a moving link by reading parent_frame
        # from camera_presets.  For "world" / "map" / "odom" cameras, skip lookup.
        self._parent_frame: str = _get_parent_frame(self._cam_name, platform)
        self._tf_snap: tuple[tuple, tuple] | None = None  # (xyz, rpy) of parent in world

        if self._parent_frame not in _STATIC_PARENT_FRAMES:
            # PRIMARY: query Gazebo directly (works without robot_state_publisher / TF).
            # The entity path in Gazebo follows "<model_name>::<link_name>".
            gz_model = robot_config  # e.g. "so101_single_arm"
            print(f"[adjuster] Querying Gazebo for world pose of '{gz_model}::{self._parent_frame}' (max 5s)...")
            self._tf_snap = _lookup_gazebo_link_pose(
                model_name=gz_model,
                link_name=self._parent_frame,
                world_name=world_name,
            )
            if self._tf_snap is not None:
                print(
                    f"[adjuster] Gazebo pose OK: {gz_model}::{self._parent_frame} "
                    f"world pos = {tuple(round(v, 4) for v in self._tf_snap[0])}"
                )
            else:
                # FALLBACK: try ROS TF (requires robot_state_publisher).
                print(
                    f"[adjuster] Gazebo query failed — falling back to ROS TF 'world<-{self._parent_frame}' (max 5s)..."
                )
                self._tf_snap = _lookup_tf_snapshot(self._parent_frame, world_frame="world")
                if self._tf_snap is not None:
                    print(
                        f"[adjuster] TF snapshot OK: parent world pos = {tuple(round(v, 4) for v in self._tf_snap[0])}"
                    )
                else:
                    print(
                        "[adjuster] BOTH Gazebo and TF lookups failed — "
                        "Static fallback.  Trackbars will control world pose directly, "
                        "but the 's' key will REFUSE to save to avoid corrupting the "
                        f"'{self._parent_frame}'-relative calibration."
                    )

        # Compute spawn world pose (TF mode: compose; Static: passthrough)
        spawn_xyz, spawn_rpy = self._offset_to_world(
            (self._pose["x"], self._pose["y"], self._pose["z"]),
            (self._pose["roll"], self._pose["pitch"], self._pose["yaw"]),
        )

        # Spawn proxy camera
        self._model_name = spawn_alignment_camera_gazebo(
            topic=topic,
            pose_xyz=spawn_xyz,
            pose_rpy=spawn_rpy,
            fovy_deg=self._pose["fovy_deg"],
            width=int(self._geometry["width"]),
            height=int(self._geometry["height"]),
            fps=float(self._geometry["fps"]),
            world_name=world_name,
        )

        # Start ros_gz_bridge
        self._bridge_proc = _start_ros_gz_bridge(topic)
        if _wait_for_ros_topic(topic, timeout_s=10.0):
            print(f"[adjuster] proxy ROS topic ready: {topic}")
        else:
            print(f"[adjuster] WARNING: timed out waiting for proxy ROS topic: {topic}")

        # Build trackbar window.
        # tf_label non-empty → status bar shows "[TF:gripper] offset x=..."
        # tf_label empty     → status bar shows "[STATIC] x=..."
        tf_label = self._parent_frame if self._tf_snap is not None else ""
        self._tb = _TrackbarWindow(self._cam_name, self._pose, tf_label=tf_label)

    def _respawn_with_fovy(self, fovy_deg: float) -> None:
        """Despawn and respawn proxy camera with a new FOV.

        Also restarts ros_gz_bridge because the old bridge process drops the
        topic when Gazebo destroys the sensor and will not automatically
        reconnect to the new sensor instance.

        NOTE: Gazebo entity removal is async. We wait 1.0s after despawn so
        the old model is fully gone before we try to spawn the same name again.
        """
        print(f"[adjuster] Applying FOV={fovy_deg:.1f}° — respawning proxy camera...")

        # Step 1: terminate bridge first so it doesn't hold a ref to the old sensor
        if self._bridge_proc.poll() is None:
            self._bridge_proc.terminate()
            try:
                self._bridge_proc.wait(timeout=2)
            except Exception:
                self._bridge_proc.kill()
        print("[adjuster] bridge stopped")

        # Step 2: despawn old model
        try:
            self._despawn_fn(topic=self._topic, world_name=self._world)
        except Exception as exc:
            print(f"[adjuster] despawn warning (non-fatal): {exc}")

        # Wait long enough for Gazebo to finish the async entity removal
        print("[adjuster] waiting for Gazebo to remove old camera...")
        time.sleep(1.0)

        # Step 3: spawn new model with updated FOV (check=False so name collision doesn't crash)
        try:
            import math as _math

            from sim_models.aruco_spawner import _build_proxy_camera_sdf, _camera_name_from_topic

            cam_name = _camera_name_from_topic(self._topic)
            model_name = f"alignment_cam_{cam_name}"
            fovy_rad = _math.radians(fovy_deg)
            sdf_content = _build_proxy_camera_sdf(
                model_name,
                self._topic,
                fovy_rad,
                width=int(self._geometry["width"]),
                height=int(self._geometry["height"]),
                fps=float(self._geometry["fps"]),
            )
            sdf_escaped = sdf_content.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "")
            x, y, z = self._pose["x"], self._pose["y"], self._pose["z"]
            roll, pitch, yaw = self._pose["roll"], self._pose["pitch"], self._pose["yaw"]
            # Use world pose for spawn (TF mode: compose offset with snapshot)
            (x, y, z), (roll, pitch, yaw) = self._offset_to_world((x, y, z), (roll, pitch, yaw))
            cr, cp, cy = _math.cos(roll / 2), _math.cos(pitch / 2), _math.cos(yaw / 2)
            sr, sp, sy = _math.sin(roll / 2), _math.sin(pitch / 2), _math.sin(yaw / 2)
            qw = cr * cp * cy + sr * sp * sy
            qx = sr * cp * cy - cr * sp * sy
            qy = cr * sp * cy + sr * cp * sy
            qz = cr * cp * sy - sr * sp * cy
            req = (
                f'sdf: "{sdf_escaped}" '
                f'name: "{model_name}" '
                f"allow_renaming: false "
                f"pose {{ position {{ x: {x} y: {y} z: {z} }} "
                f"orientation {{ x: {qx} y: {qy} z: {qz} w: {qw} }} }}"
            )
            import subprocess as _sp

            result = _sp.run(
                [
                    "ign",
                    "service",
                    "-s",
                    f"/world/{self._world}/create",
                    "--reqtype",
                    "ignition.msgs.EntityFactory",
                    "--reptype",
                    "ignition.msgs.Boolean",
                    "--timeout",
                    "5000",
                    "-r",
                    req,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"[adjuster] spawn warning (rc={result.returncode}): {result.stderr.strip()}")
            else:
                print(f"[adjuster] new camera spawned with FOV={fovy_deg:.1f}°")
        except Exception as exc:
            print(f"[adjuster] spawn error: {exc}")
            return

        # Step 4: restart bridge
        self._bridge_proc = _start_ros_gz_bridge(self._topic)
        if _wait_for_ros_topic(self._topic, timeout_s=10.0):
            print(f"[adjuster] FOV updated to {fovy_deg:.1f}° ✓")
        else:
            print(f"[adjuster] WARNING: proxy ROS topic did not come back after FOV respawn: {self._topic}")

    def run(self) -> None:
        print(f"[adjuster] Camera: {self._cam_name}  Topic: {self._topic}")
        print("[adjuster] m: COARSE/FINE  s: save  r: reset  q: quit")

        _last_fovy = self._pose["fovy_deg"]
        _fovy_settle = 0  # countdown ticks; when it hits 0 → respawn
        _SETTLE_TICKS = 16  # 16 × 50 ms = 0.8 s debounce (user stops dragging)
        _pose_update_failures = 0
        _last_pose_error: str | None = None

        try:
            while True:
                key = self._tb.wait_key(50)

                if key == ord("q"):
                    break
                elif key == ord("m"):
                    self._tb.toggle_mode()
                elif key == ord("s"):
                    # Safety rule: for a camera whose preset parent is a moving link
                    # (e.g. wrist -> gripper), refuse to save if neither Gazebo nor TF
                    # could resolve the parent world pose.  Saving world-absolute
                    # coords with parent_frame="world" would make the production
                    # camera world-fixed and stop following the arm.
                    is_moving_parent = self._parent_frame not in _STATIC_PARENT_FRAMES
                    if is_moving_parent and self._tf_snap is None:
                        print(
                            f"[adjuster] REFUSED to save: camera '{self._cam_name}' is meant "
                            f"to follow '{self._parent_frame}', but no world pose of that link "
                            f"is available (Gazebo and ROS TF both failed).  Saving now would "
                            f"corrupt the calibration.  Fix the sim stack (ensure Gazebo is "
                            f"running and has the '{self._robot_config}' model loaded, or start "
                            f"robot_state_publisher) then retry."
                        )
                    else:
                        # In TF/Gazebo mode self._pose is already parent-relative.
                        # In Static mode the camera's preset parent is "world"/"map"/... so
                        # self._pose is world-absolute — save with the same parent_frame.
                        _save_override(self._cam_name, self._topic, self._pose, self._parent_frame)
                        self._tb.flash_saved(self._pose)
                        print(
                            "[adjuster] SAVED! Check ghost window (press v in camera_alignment) to confirm alignment."
                        )
                elif key == ord("r"):
                    self._tb.reset(self._initial_pose)
                    _last_fovy = self._pose["fovy_deg"]
                    _fovy_settle = 0

                # Detect fovy change and debounce before respawning
                current_fovy = self._pose["fovy_deg"]
                if current_fovy != _last_fovy:
                    _last_fovy = current_fovy
                    _fovy_settle = _SETTLE_TICKS
                    print(f"[adjuster] FOV trackbar moved to {current_fovy:.1f}° — will respawn in 1.5s if stable")
                elif _fovy_settle > 0:
                    _fovy_settle -= 1
                    if _fovy_settle == 0:
                        self._respawn_with_fovy(_last_fovy)

                # Push pose on every tick (non-blocking)
                try:
                    world_xyz, world_rpy = self._offset_to_world(
                        (self._pose["x"], self._pose["y"], self._pose["z"]),
                        (self._pose["roll"], self._pose["pitch"], self._pose["yaw"]),
                    )
                    ok, err = self._set_pose_fn(
                        topic=self._topic,
                        pose_xyz=world_xyz,
                        pose_rpy=world_rpy,
                        world_name=self._world,
                    )
                    if ok:
                        if _pose_update_failures > 0:
                            print("[adjuster] pose updates recovered.")
                        _pose_update_failures = 0
                        _last_pose_error = None
                    else:
                        _pose_update_failures += 1
                        _last_pose_error = err
                        if _pose_update_failures in (1, 10, 50) or _pose_update_failures % 200 == 0:
                            print(
                                f"[adjuster] WARNING: proxy camera pose update failed "
                                f"(count={_pose_update_failures}): {_last_pose_error}"
                            )
                except Exception:
                    _pose_update_failures += 1
                    if _pose_update_failures in (1, 10, 50) or _pose_update_failures % 200 == 0:
                        print("[adjuster] WARNING: proxy camera pose update raised unexpectedly; will retry.")

        finally:
            self._cleanup()

    def _offset_to_world(
        self,
        offset_xyz: tuple,
        offset_rpy: tuple,
    ) -> tuple[tuple, tuple]:
        """Convert trackbar offset to world pose.

        TF mode  (self._tf_snap is not None):
            world_pose = T_world_parent ⊕ trackbar_offset
        Static mode (self._tf_snap is None):
            world_pose = trackbar_offset  (passthrough, existing behaviour)
        """
        if self._tf_snap is None:
            return offset_xyz, offset_rpy
        parent_xyz, parent_rpy = self._tf_snap
        return _compose_world_pose(parent_xyz, parent_rpy, offset_xyz, offset_rpy)

    def _cleanup(self) -> None:
        self._tb.destroy()
        try:
            self._despawn_fn(topic=self._topic, world_name=self._world)
        except Exception as exc:
            print(f"[adjuster] Despawn failed (non-fatal): {exc}")
        if self._bridge_proc.poll() is None:
            self._bridge_proc.terminate()
            print(f"[adjuster] ros_gz_bridge terminated (pid={self._bridge_proc.pid})")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gazebo proxy camera adjuster with OpenCV Trackbar")
    p.add_argument(
        "--camera",
        required=True,
        metavar="TOPIC",
        help=(
            "Camera topic to adjust, e.g. /camera/top/image_raw. "
            "The proxy topic (/camera_align/...) is derived automatically."
        ),
    )
    p.add_argument(
        "--platform",
        default="gazebo",
        choices=["gazebo", "mujoco"],
        help="Simulation platform (default: gazebo)",
    )
    p.add_argument(
        "--robot-config",
        default="so101_single_arm",
        metavar="NAME",
        help="Robot config name for loading initial pose from YAML (default: so101_single_arm)",
    )
    p.add_argument(
        "--world",
        default="demo",
        metavar="NAME",
        help="Gazebo world name (default: demo)",
    )
    return p


def main(args: list[str] | None = None) -> int:
    parsed = build_parser().parse_args(args=args)

    # Accept real camera topics (/camera/top/image_raw) and map to proxy namespace.
    proxy_topic = _resolve_proxy_topic(parsed.camera)
    if proxy_topic != parsed.camera:
        print(f"[adjuster] {parsed.camera} -> proxy topic: {proxy_topic}")

    # Auto-promote platform from YAML if the user did not override on CLI.
    # argparse cannot distinguish "default" from "explicit 'gazebo'", so we only
    # promote when the explicit CLI token is absent from argv.
    cli_argv = args if args is not None else sys.argv[1:]
    platform_explicit = any(a == "--platform" or a.startswith("--platform=") for a in cli_argv)
    if not platform_explicit:
        yaml_platform = _detect_platform_from_yaml(parsed.robot_config)
        if yaml_platform in ("gazebo", "mujoco") and yaml_platform != parsed.platform:
            print(f"[adjuster] robot YAML declares simulation.platform={yaml_platform}; overriding --platform default.")
            parsed.platform = yaml_platform

    # When the primary simulator is MuJoCo we need a parallel headless Gazebo
    # for camera calibration (proxy camera spawning / link-pose queries require
    # Gazebo).  Spawn it in an isolated ROS_DOMAIN_ID so it does not disturb
    # the running MuJoCo stack.
    calib_gazebo: _CalibrationGazebo | None = None
    try:
        if parsed.platform == "mujoco":
            calib_gazebo = _CalibrationGazebo(
                robot_config=parsed.robot_config,
                world_name=parsed.world,
            )
            # SimCameraAdjuster itself is always a Gazebo-backed tool. From here
            # on the adjuster must behave as if it were running against the
            # calibration Gazebo; downgrade the platform label so downstream
            # code (aruco_spawner, ign queries, preset lookup) uses the Gazebo
            # code path.
            adjuster_platform = "gazebo"
        else:
            adjuster_platform = parsed.platform

        adjuster = SimCameraAdjuster(
            topic=proxy_topic,
            platform=adjuster_platform,
            robot_config=parsed.robot_config,
            world_name=parsed.world,
        )
        adjuster.run()
        return 0
    finally:
        if calib_gazebo is not None:
            calib_gazebo.stop()


if __name__ == "__main__":
    raise SystemExit(main())
