import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_SERVER = Path("/opt/ros/humble/lib/nav2_controller/controller_server")


@pytest.mark.parametrize("profile", ["nav2_params.yaml", "nav2_sim_params.yaml"])
def test_mppi_controller_configures_with_humble_pluginlib(profile, tmp_path):
    if not CONTROLLER_SERVER.is_file():
        pytest.skip("Nav2 controller_server is not installed")

    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = str(160 + os.getpid() % 40)
    log_path = tmp_path / f"{profile}.log"
    helper_processes = []
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [
                str(CONTROLLER_SERVER),
                "--ros-args",
                "--params-file",
                str(PACKAGE_ROOT / "config" / profile),
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )
        helper_processes.extend(
            [
                subprocess.Popen(
                    [
                        "/opt/ros/humble/lib/tf2_ros/static_transform_publisher",
                        "0",
                        "0",
                        "0",
                        "0",
                        "0",
                        "0",
                        "map",
                        "odom",
                    ],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid,
                ),
                subprocess.Popen(
                    [
                        "/opt/ros/humble/lib/tf2_ros/static_transform_publisher",
                        "0",
                        "0",
                        "0",
                        "0",
                        "0",
                        "0",
                        "odom",
                        "base_link",
                    ],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid,
                ),
                subprocess.Popen(
                    [
                        "/opt/ros/humble/lib/nav2_map_server/map_server",
                        "--ros-args",
                        "-p",
                        f"yaml_filename:={PACKAGE_ROOT / 'config' / 'maps' / 'sim_map.yaml'}",
                    ],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid,
                ),
            ]
        )
        try:
            deadline = time.monotonic() + 15.0
            result = None
            while time.monotonic() < deadline:
                result = subprocess.run(
                    ["ros2", "lifecycle", "set", "/controller_server", "configure"],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=5,
                )
                if result.returncode == 0:
                    break
                time.sleep(0.5)

            assert result is not None and result.returncode == 0, log_path.read_text(encoding="utf-8")
            map_result = subprocess.run(
                ["ros2", "lifecycle", "set", "/map_server", "configure"],
                capture_output=True,
                text=True,
                env=env,
                timeout=5,
            )
            if map_result.returncode == 0:
                subprocess.run(
                    ["ros2", "lifecycle", "set", "/map_server", "activate"],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=5,
                    check=True,
                )
            state = subprocess.run(
                ["ros2", "lifecycle", "get", "/controller_server"],
                capture_output=True,
                text=True,
                env=env,
                timeout=5,
                check=True,
            )
            assert "inactive [2]" in state.stdout
            activate = subprocess.run(
                ["ros2", "lifecycle", "set", "/controller_server", "activate"],
                capture_output=True,
                text=True,
                env=env,
                timeout=5,
            )
            assert activate.returncode == 0, log_path.read_text(encoding="utf-8")
            state = subprocess.run(
                ["ros2", "lifecycle", "get", "/controller_server"],
                capture_output=True,
                text=True,
                env=env,
                timeout=5,
                check=True,
            )
            assert "active [3]" in state.stdout
            log = log_path.read_text(encoding="utf-8")
            assert "Configured MPPI Controller: FollowPath" in log
            assert "Critic loaded : mppi::critics::CostCritic" in log
        finally:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=10)
            for helper in helper_processes:
                os.killpg(os.getpgid(helper.pid), signal.SIGTERM)
                helper.wait(timeout=10)
