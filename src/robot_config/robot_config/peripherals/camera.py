"""Camera peripheral launch helpers for ROS2.

This module provides helper functions to launch existing ROS2 camera drivers
(usb_cam, realsense2_camera) based on camera configuration.
"""

from typing import Any

from robot_config.config import CameraConfig
from robot_config.utils import parse_bool


def camera_dict_requests_depth(config: dict[str, Any]) -> bool:
    """Return True when a YAML camera peripheral requests a depth-capable stream."""
    streams = config.get("streams") or []
    if isinstance(streams, str):
        streams = [streams]
    stream_names = {str(stream).strip().lower() for stream in streams}
    return (
        "depth" in stream_names
        or "pointcloud" in stream_names
        or parse_bool(config.get("align_depth", False), default=False)
        or parse_bool(config.get("enable_depth", False), default=False)
        or parse_bool(config.get("enable_pointcloud", False), default=False)
    )


def get_usb_cam_params(config: CameraConfig) -> dict[str, Any]:
    """Get parameters for usb_cam node.

    Args:
        config: Camera configuration

    Returns:
        Dictionary of parameters for usb_cam
    """
    params = {
        "camera_name": config.name,
        "framerate": config.fps,
        "image_width": config.width,
        "image_height": config.height,
        "pixel_format": config.pixel_format,
        "camera_frame_id": config.frame_id,
    }

    # Add optional parameters
    if config.brightness is not None:
        params["brightness"] = config.brightness
    if config.contrast is not None:
        params["contrast"] = config.contrast
    if config.saturation is not None:
        params["saturation"] = config.saturation
    if config.sharpness is not None:
        params["sharpness"] = config.sharpness

    return params


def get_realsense_params(config: CameraConfig) -> dict[str, Any]:
    """Get parameters for realsense2_camera node.

    Args:
        config: Camera configuration

    Returns:
        Dictionary of parameters for realsense2_camera
    """
    params = {
        "camera_name": config.name,
        "camera_fps": config.fps,
        "color_width": config.width,
        "color_height": config.height,
        "color_format": config.pixel_format.upper(),
        "camera_frame_id": config.frame_id,
        "enable_pointcloud": config.enable_pointcloud,
        "enable_sync": config.enable_sync,
        "align_depth": config.align_depth,
    }

    # Add depth parameters if specified
    if config.depth_width:
        params["depth_width"] = config.depth_width
        params["depth_height"] = config.depth_height
    if config.depth_fps:
        params["depth_fps"] = config.depth_fps

    # Add serial number if specified
    if isinstance(config.index_or_port, str) and len(config.index_or_port) > 0:
        params["serial_no"] = str(config.index_or_port)

    return params


def get_static_transforms(config: CameraConfig) -> list[tuple[str, str, dict[str, float]]]:
    """Generate static transform publishers for camera frames.

    Args:
        config: Camera configuration

    Returns:
        List of (parent_frame, child_frame, transform_dict) tuples
    """
    transforms = []

    # Camera frame (if transform is specified)
    if config.transform:
        transforms.append(
            (
                "base_link",  # TODO: Make parent configurable
                config.frame_id,
                config.transform,
            )
        )

    # Optical frame transform (standard ROS2 convention)
    if config.optical_frame_id:
        # Standard optical frame rotation: -90° around X, -90° around Y
        optical_transform = {"x": 0.0, "y": 0.0, "z": 0.0, "roll": -1.5708, "pitch": -1.5708, "yaw": 0.0}
        transforms.append(
            (
                config.frame_id,
                config.optical_frame_id,
                optical_transform,
            )
        )

    return transforms
