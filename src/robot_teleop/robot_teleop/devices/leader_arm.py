"""
SO-101 Leader Arm teleoperation device

Implements teleoperation via SO-101 leader arm hardware,
reading joint positions from serial port and applying calibration.
"""

import logging
import math
from pathlib import Path
from typing import Any

from ..base_teleop import BaseTeleopDevice


class LeaderArmDevice(BaseTeleopDevice):
    """
    SO-101 leader arm teleoperation device.

    Reads joint positions from SO-101 leader arm via serial port,
    applies calibration offsets, and maps to follower joint names.
    """

    def __init__(self, config: dict, node=None):
        super().__init__(config, node=node)

        # Configuration
        self.port = config.get("port", "/dev/ttyACM1")
        calib_str = config.get("calib_file", "")
        self.calib_file = Path(calib_str).expanduser() if calib_str else None
        self.joint_mapping = config.get("joint_mapping", {})

        # Physical constants (4096 steps per 360 degrees)
        self.rad_per_step = (2 * math.pi) / 4096.0

        # Hardware interface
        self.motors_bus = None
        self.calibration = None
        # Joint definitions (SO-101 leader arm)
        self.joints = {
            "1": {"id": 1, "model": "sts3215"},
            "2": {"id": 2, "model": "sts3215"},
            "3": {"id": 3, "model": "sts3215"},
            "4": {"id": 4, "model": "sts3215"},
            "5": {"id": 5, "model": "sts3215"},
            "6": {"id": 6, "model": "sts3215"},
        }
        self.joint_names = list(self.joints.keys())
        self.logger = logging.getLogger(__name__)

    def connect(self) -> bool:
        try:
            from lerobot.motors import Motor, MotorNormMode
            from lerobot.motors.feetech import FeetechMotorsBus

            motors = {}
            for joint_name, joint_info in self.joints.items():
                norm_mode = MotorNormMode.RANGE_0_100 if joint_name == "6" else MotorNormMode.RANGE_M100_100
                motors[joint_name] = Motor(
                    id=joint_info["id"],
                    model=joint_info["model"],
                    norm_mode=norm_mode,
                )

            self.motors_bus = FeetechMotorsBus(port=self.port, motors=motors)
            # Connect to motors
            self.motors_bus.connect()
            self.logger.info(f"Motors bus connected on {self.port}")

            # Load and write calibration to firmware (CRITICAL for Feetech coordinate system)
            if self.calib_file and self.calib_file.exists():
                self.calibration = self._load_calibration()
                self.logger.info(f"Loaded calibration from {self.calib_file}")

                self.logger.info("Writing calibration to motor firmware...")
                self.motors_bus.write_calibration(self.calibration)
                self.logger.info("Calibration written to firmware. Motors will now output ~2048 at physical zero.")
            else:
                self.logger.warning("No calibration file found, using raw encoder positions")
                self.calibration = None

            self._is_connected = True
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect leader arm: {e}")
            self._is_connected = False
            raise ConnectionError(f"Cannot connect to leader arm on {self.port}: {e}") from e

    def get_joint_targets(self) -> dict[str, float]:
        if not self._is_connected or self.motors_bus is None:
            return {}

        try:
            # Read absolute positions. Because we wrote calibration to firmware,
            # the motors will output 2048 when they are at their home physical position.
            raw_positions = self.motors_bus.sync_read("Present_Position", normalize=False)

            joint_targets = {}
            for name in self.joint_names:
                if name not in raw_positions:
                    continue

                raw = raw_positions[name]

                # EXACTLY matching so101_hardware.cpp logic:
                # rad = (raw - 2048.0) * rad_per_step
                position_rad = (raw - 2048.0) * self.rad_per_step

                # Map to follower joint name
                follower_joint = self._map_joint(name)
                joint_targets[follower_joint] = position_rad

            return joint_targets

        except Exception as e:
            self.logger.error(f"Failed to read leader arm positions: {e}")
            return {}

    def disconnect(self):
        if self.motors_bus is not None:
            try:
                self.motors_bus.disconnect()
                self.logger.info("Leader arm disconnected")
            except Exception:
                pass
            finally:
                self.motors_bus = None
                self._is_connected = False

    def _load_calibration(self) -> dict[str, Any]:
        from so101_hardware.calibration.interactive import load_calibration as load_calib_so101

        return load_calib_so101(self.calib_file, self.joint_names, self.logger)

    def _map_joint(self, leader_joint: str) -> str:
        return self.joint_mapping.get(leader_joint, leader_joint)
