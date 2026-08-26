#!/usr/bin/env python3
"""Validation script for robot configuration files.

This script validates robot_config YAML files and reports any errors.
Usage:
    python validate_config.py /path/to/robot_config.yaml
"""

import logging
import sys
from pathlib import Path

# Configure basic logging for CLI tool
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Add robot_config to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from robot_config.loader import load_robot_config, validate_config  # noqa: E402


def main():
    if len(sys.argv) < 2:
        logger.info("Usage: python validate_config.py <robot_config.yaml>")
        logger.info("\nExample:")
        logger.info("  python validate_config.py config/robots/so101_single_arm.yaml")
        sys.exit(1)

    config_path = sys.argv[1]

    try:
        logger.info(f"Validating {config_path}...")
        config = load_robot_config(config_path)

        logger.info(f"  Robot: {config.name} ({config.robot_type})")
        logger.info(f"  Hardware plugin: {config.ros2_control.hardware_plugin}")
        cameras = [peripheral for peripheral in config.peripherals if hasattr(peripheral, "width")]
        microphones = [peripheral for peripheral in config.peripherals if peripheral.type == "microphone"]
        logger.info(f"  Cameras: {len(cameras)}")
        logger.info(f"  Microphones: {len(microphones)}")

        for cam in cameras:
            logger.info(f"    - {cam.name}: {cam.driver} @ {cam.width}x{cam.height} {cam.fps}fps")
        for microphone in microphones:
            logger.info(f"    - {microphone.name}: {microphone.driver} ({microphone.params.get('device', 'n/a')})")

        if config.speech_direction.enabled:
            logger.info(
                "  Speech direction: enabled (%s, microphone=%s)",
                config.speech_direction.profile,
                config.speech_direction.microphone,
            )

        errors = validate_config(config)

        if errors:
            logger.error(f"\n❌ Validation failed with {len(errors)} error(s):")
            for error in errors:
                logger.info(f"  - {error}")
            sys.exit(1)
        else:
            logger.info("\n✅ Configuration is valid!")
            sys.exit(0)

    except FileNotFoundError as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
