"""Perception system launch builders.

This module handles:
- Camera driver nodes (usb_cam, camera_ros, realsense2_camera)
- LiDAR driver nodes
- Static TF publishers for peripheral frames
- Virtual camera relays
"""

from __future__ import annotations

import os

from launch_ros.actions import Node

from robot_config.launch_builders.camera_isp_overrides import load_isp_override
from robot_config.logger_utils import get_colored_logger
from robot_config.utils import parse_bool

logger = get_colored_logger("robot_config.perception")


def generate_camera_nodes(robot_config, use_sim=False):
    """Generate physical camera driver nodes from configuration.

    Args:
        robot_config: Robot configuration dict
        use_sim: Simulation mode (if True, skip physical cameras)

    Returns:
        List of Node actions for cameras
    """
    is_sim = parse_bool(use_sim, default=False)
    if is_sim:
        logger.info("Skipping physical camera drivers in sim mode")
        return []

    nodes = []

    peripherals = robot_config.get("peripherals", [])
    logger.info(f"Generating nodes for {len(peripherals)} peripherals (use_sim={is_sim})")
    for periph in peripherals:
        periph_type = periph.get("type")

        # Skip virtual cameras in first pass
        if periph_type == "virtual_camera":
            continue

        if periph_type != "camera":
            continue

        name = periph["name"]
        driver = periph.get("driver", "opencv")
        logger.info(f"Creating camera node: {name} (driver={driver})")

        if driver == "opencv":
            # Use usb_cam package
            index = periph.get("index", 0)
            video_device = f"/dev/video{index}" if isinstance(index, int) else index

            params = {
                "use_sim_time": is_sim,
                "camera_name": name,
                "framerate": float(periph.get("fps", 30)),
                "image_width": periph.get("width", 640),
                "image_height": periph.get("height", 480),
                "pixel_format": periph.get("pixel_format", "mjpeg"),
                "brightness": periph.get("brightness", 0),
                "camera_frame_id": periph.get("frame_id", f"camera_{name}_frame"),
                "video_device": video_device,
            }

            if "camera_info_url" in periph:
                params["camera_info_url"] = periph["camera_info_url"]

            # Pass-through for all V4L2 / usb_cam ISP knobs declared in YAML.
            # Keys must match usb_cam_node.cpp:65-85 declarations exactly.
            # Any key absent from the YAML is left at the usb_cam default
            # (or the camera_isp_override below, which takes precedence).
            for key in (
                "contrast",
                "saturation",
                "sharpness",
                "gain",
                "auto_white_balance",
                "white_balance",
                "autoexposure",
                "exposure",
                "autofocus",
                "focus",
                "io_method",
            ):
                if key in periph:
                    params[key] = periph[key]

            # Apply per-camera ISP override (from camera_isp_calibrator).
            # Override values take precedence over YAML so calibration
            # results persist across launches without modifying SSOT.
            override = load_isp_override(name)
            if override:
                params.update(override)

            logger.info(f"  Camera params: {params}")

            nodes.append(
                Node(
                    package="usb_cam",
                    executable="usb_cam_node_exe",
                    name=f"{name}_camera",
                    parameters=[params],
                    remappings=[
                        ("image_raw", f"/camera/{name}/image_raw"),
                        ("camera_info", f"/camera/{name}/camera_info"),
                    ],
                    output="screen",
                )
            )

        elif driver == "camera_ros":
            params = {
                "camera": periph.get("index", periph.get("camera", 0)),
                "format": periph.get("format", "MJPEG"),
                "width": periph.get("width", 640),
                "height": periph.get("height", 480),
                "framerate": float(periph.get("fps", 30)),
            }
            if "camera_info_url" in periph:
                params["camera_info_url"] = periph["camera_info_url"]

            print(f"[robot_config]   camera_ros params: {params}")

            nodes.append(
                Node(
                    package="camera_ros",
                    executable="camera_node",
                    namespace=f"/camera/{name}",
                    name=f"{name}_camera",
                    parameters=[params],
                    output="screen",
                    respawn=bool(periph.get("respawn", True)),
                )
            )

        elif driver == "realsense":
            # Use rs_launch.py to match the board-validated RealSense startup path.
            w = periph.get("width", 640)
            h = periph.get("height", 480)
            fps = periph.get("fps", 30)
            driver_camera_name = periph.get("driver_camera_name", f"{name}_camera")
            driver_topic_prefix = periph.get("driver_topic_prefix", f"/camera/{driver_camera_name}")
            frame_id = periph.get("frame_id", f"{driver_camera_name}_link")
            direct_topic_remap = bool(periph.get("direct_topic_remap", False))
            align_depth = periph.get("align_depth", False)
            streams = periph.get("streams") or (
                ["color"]
                + (["depth"] if periph.get("enable_depth", True) else [])
                + (["pointcloud"] if periph.get("enable_pointcloud", False) else [])
            )
            align_depth = align_depth and "depth" in streams
            driver_params = {
                "camera_namespace": "camera",
                "camera_name": driver_camera_name,
                "base_frame_id": frame_id,
                "tf_prefix": "",
                "publish_tf": True,
                "enable_color": "color" in streams,
                "enable_depth": "depth" in streams,
                "enable_infra": False,
                "enable_infra1": False,
                "enable_infra2": False,
                "enable_motion": False,
                "enable_rgbd": False,
                "rgb_camera.color_profile": f"{w}x{h}x{fps}",
                "depth_module.depth_profile": f"{w}x{h}x{fps}",
                "align_depth.enable": align_depth,
                "pointcloud.enable": "pointcloud" in streams,
                "pointcloud.stream_filter": 2 if "pointcloud" in streams else 0,
                "pointcloud.ordered_pc": False,
                "enable_sync": bool(periph.get("enable_sync", True)),
                "initial_reset": bool(periph.get("initial_reset", True)),
                "enable_gyro": False,
                "enable_accel": False,
                "unite_imu_method": 0,
            }

            if "depth_width" in periph:
                depth_w = periph["depth_width"]
                depth_h = periph["depth_height"]
                depth_fps = periph.get("depth_fps", fps)
                driver_params["depth_module.depth_profile"] = f"{depth_w}x{depth_h}x{depth_fps}"
            if "serial_number" in periph:
                driver_params["serial_no"] = str(periph["serial_number"])

            logger.info(f"  RealSense driver params: {driver_params}")

            depth_source_topic = (
                f"{driver_topic_prefix}/aligned_depth_to_color/image_raw"
                if align_depth
                else f"{driver_topic_prefix}/depth/image_rect_raw"
            )
            depth_target_topic = (
                f"/camera/{name}/aligned_depth_to_color/image_raw"
                if align_depth
                else f"/camera/{name}/depth/image_rect_raw"
            )
            depth_camera_info_source_topic = (
                f"{driver_topic_prefix}/aligned_depth_to_color/camera_info"
                if align_depth
                else f"{driver_topic_prefix}/depth/camera_info"
            )
            driver_remappings = []
            if direct_topic_remap:
                # Large RGB-D payloads should not cross a second DDS
                # publisher/subscriber pair merely to normalize topic names.
                # CameraInfo still uses the small relay below so its frame_id is
                # normalized to the robot-config optical frame.
                driver_remappings.extend(
                    [
                        (f"{driver_topic_prefix}/color/image_raw", f"/camera/{name}/image_raw"),
                        (depth_source_topic, depth_target_topic),
                    ]
                )
                if "pointcloud" in streams:
                    driver_remappings.append(
                        (
                            f"{driver_topic_prefix}/depth/color/points",
                            f"/camera/{name}/depth/color/points",
                        )
                    )
                logger.info(f"  RealSense direct large-payload remappings: {driver_remappings}")

            nodes.append(
                Node(
                    package="realsense2_camera",
                    executable="realsense2_camera_node",
                    namespace="camera",
                    name=driver_camera_name,
                    parameters=[driver_params],
                    remappings=driver_remappings,
                    arguments=["--ros-args", "--log-level", "info"],
                    output="screen",
                    emulate_tty=True,
                )
            )

            # Keep a stable camera contract for downstream consumers while
            # isolating RealSense driver topic naming differences here.
            relay_topics = [
                (
                    f"{driver_topic_prefix}/color/camera_info",
                    f"/camera/{name}/camera_info",
                    f"{name}_camera_info_relay",
                    "sensor_msgs/msg/CameraInfo",
                    periph.get("optical_frame_id"),
                ),
                (
                    depth_camera_info_source_topic,
                    (
                        f"/camera/{name}/aligned_depth_to_color/camera_info"
                        if align_depth
                        else f"/camera/{name}/depth/camera_info"
                    ),
                    f"{name}_{'aligned_depth_' if align_depth else 'depth_'}camera_info_relay",
                    "sensor_msgs/msg/CameraInfo",
                    periph.get("optical_frame_id"),
                ),
            ]
            if not direct_topic_remap:
                relay_topics.extend(
                    [
                        (
                            f"{driver_topic_prefix}/color/image_raw",
                            f"/camera/{name}/image_raw",
                            f"{name}_color_image_relay",
                            "sensor_msgs/msg/Image",
                            periph.get("optical_frame_id"),
                        ),
                        (
                            depth_source_topic,
                            f"/camera/{name}/depth/image_rect_raw",
                            f"{name}_depth_image_relay",
                            "sensor_msgs/msg/Image",
                            periph.get("optical_frame_id"),
                        ),
                    ]
                )
                if align_depth:
                    relay_topics.append(
                        (
                            depth_source_topic,
                            depth_target_topic,
                            f"{name}_aligned_depth_image_relay",
                            "sensor_msgs/msg/Image",
                            periph.get("optical_frame_id"),
                        )
                    )
            if "pointcloud" in streams and not direct_topic_remap:
                relay_topics.append(
                    (
                        f"{driver_topic_prefix}/depth/color/points",
                        f"/camera/{name}/depth/color/points",
                        f"{name}_pointcloud_relay",
                        "sensor_msgs/msg/PointCloud2",
                        periph.get("optical_frame_id"),
                    )
                )
            for source_topic, target_topic, relay_name, message_type, target_frame_id in relay_topics:
                relay_args = [source_topic, target_topic, message_type]
                if target_frame_id:
                    relay_args.append(target_frame_id)
                nodes.append(
                    Node(
                        package="robot_config",
                        executable="topic_relay",
                        name=relay_name,
                        arguments=relay_args,
                        output="screen",
                    )
                )

    return nodes


def _is_openharmony_runtime() -> bool:
    """Detect the official OpenHarmony ROS runtime environment."""
    if os.environ.get("OHOS_ROS2_ROOT") or os.environ.get("OHOS_ROS2_SYSDEPS"):
        return True
    if os.environ.get("SETUP_PLATFORM_ID") == "openharmony-5.1.0-musl":
        return True

    try:
        with open("/etc/os-release") as fh:
            os_release = fh.read().lower()
    except OSError:
        return False
    return "openharmony" in os_release or "id=ohos" in os_release


def generate_lidar_nodes(robot_config, use_sim=False):
    """Generate physical LiDAR driver nodes from configuration."""
    is_sim = parse_bool(use_sim, default=False)
    if is_sim:
        print("[robot_config] Skipping physical lidar drivers in sim mode")
        return []

    nodes = []
    peripherals = robot_config.get("peripherals", [])
    print(f"[robot_config] Generating lidar nodes from {len(peripherals)} peripherals (use_sim={is_sim})")

    for periph in peripherals:
        if periph.get("type") != "lidar":
            continue

        name = periph["name"]
        driver = periph.get("driver", "")
        if driver != "ldlidar":
            continue

        params = dict(periph.get("params", {}))
        if periph.get("frame_id") and "frame_id" not in params:
            params["frame_id"] = periph["frame_id"]
        if "port" in periph and "port_name" not in params:
            params["port_name"] = periph["port"]
        params.setdefault("use_sim_time", is_sim)

        print(f"[robot_config] Creating lidar node: {name} (driver={driver})")
        print(f"[robot_config]   LiDAR params: {params}")

        nodes.append(
            Node(
                package="ldlidar_ros2",
                executable="ldlidar_ros2_node",
                name=f"{name}_lidar",
                parameters=[params],
                output="screen",
                respawn=bool(periph.get("respawn", True)),
            )
        )

    return nodes


def generate_virtual_camera_relays(robot_config):
    """Generate virtual camera relay nodes.

    Creates topic_tools relay nodes to duplicate existing camera topics
    for virtual cameras (e.g., wrist camera relayed from top camera).

    Args:
        robot_config: Robot configuration dict

    Returns:
        List of Node actions for virtual camera relays
    """
    nodes = []

    peripherals = robot_config.get("peripherals", [])
    for periph in peripherals:
        if periph.get("type") != "camera":
            continue

        driver = periph.get("driver", "")

        # Check if this is a virtual camera (driver == "virtual")
        if driver != "virtual":
            continue

        name = periph["name"]
        source_topic = periph.get("source_topic")

        # Construct target topic
        target_topic = f"/camera/{name}/image_raw"

        if not source_topic:
            logger.warning(f"Virtual camera {name} missing source_topic")
            continue

        logger.info(f"Creating virtual camera relay: {name}")
        logger.info(f"  {source_topic} -> {target_topic}")

        nodes.append(
            Node(
                package="topic_tools",
                executable="relay",
                name=f"{name}_relay",
                arguments=[source_topic, target_topic],
                output="screen",
            )
        )

    return nodes


def generate_tf_nodes(robot_config, use_sim=False):
    """Generate static TF publisher nodes for peripheral frames.

    Args:
        robot_config: Robot configuration dict
        use_sim: Simulation mode (if True, TF is published by robot_state_publisher from URDF)

    Returns:
        List of Node actions for TF publishers
    """
    is_sim = parse_bool(use_sim, default=False)
    if is_sim:
        # sim 模式下相机 TF 由 robot_state_publisher 从 URDF 发布，无需静态发布节点
        return []

    nodes = []

    peripherals = robot_config.get("peripherals", [])
    for periph in peripherals:
        name = periph.get("name")
        frame_id = periph.get("frame_id")
        optical_frame_id = periph.get("optical_frame_id")
        transform = periph.get("transform", {})

        if not all([frame_id, transform]):
            continue

        parent_frame = transform.get("parent_frame", "base_link")
        x = transform.get("x", 0.0)
        y = transform.get("y", 0.0)
        z = transform.get("z", 0.0)
        roll = transform.get("roll", 0.0)
        pitch = transform.get("pitch", 0.0)
        yaw = transform.get("yaw", 0.0)

        # Main frame TF
        nodes.append(
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name=f"static_tf_{name}",
                arguments=[
                    "--x",
                    str(x),
                    "--y",
                    str(y),
                    "--z",
                    str(z),
                    "--roll",
                    str(roll),
                    "--pitch",
                    str(pitch),
                    "--yaw",
                    str(yaw),
                    "--frame-id",
                    parent_frame,
                    "--child-frame-id",
                    frame_id,
                ],
                output="screen",
            )
        )

        # Optical frame TF (standard rotation for camera sensors)
        # For RealSense, keep the driver link bridge and also publish the contract
        # optical frame when one is declared in YAML so downstream consumers using
        # the normalized frame_id stay connected even if the driver keeps its own
        # native optical frame names.
        if periph.get("driver") == "realsense":
            driver_camera_name = periph.get("driver_camera_name", f"{name}_camera")
            # realsense2_camera prefixes the configured base_frame_id with the
            # camera name. Bridge the normalized robot-config frame to that
            # actual driver root so the native color/depth TF tree is connected
            # to the robot instead of becoming a second, disconnected tree.
            driver_frame_id = periph.get("driver_frame_id", f"{driver_camera_name}_{frame_id}")
            if frame_id != driver_frame_id:
                nodes.append(
                    Node(
                        package="tf2_ros",
                        executable="static_transform_publisher",
                        name=f"static_tf_{name}_driver_bridge",
                        arguments=[
                            "--x",
                            "0",
                            "--y",
                            "0",
                            "--z",
                            "0",
                            "--roll",
                            "0",
                            "--pitch",
                            "0",
                            "--yaw",
                            "0",
                            "--frame-id",
                            frame_id,
                            "--child-frame-id",
                            driver_frame_id,
                        ],
                        output="screen",
                    )
                )
            if optical_frame_id:
                nodes.append(
                    Node(
                        package="tf2_ros",
                        executable="static_transform_publisher",
                        name=f"static_tf_{name}_optical",
                        arguments=[
                            "--x",
                            "0",
                            "--y",
                            "0",
                            "--z",
                            "0",
                            "--qx",
                            "-0.5",
                            "--qy",
                            "0.5",
                            "--qz",
                            "-0.5",
                            "--qw",
                            "0.5",
                            "--frame-id",
                            frame_id,
                            "--child-frame-id",
                            optical_frame_id,
                        ],
                        output="screen",
                    )
                )
                logger.info(f"  Added RealSense optical frame compatibility TF: {frame_id} -> {optical_frame_id}")
        elif periph.get("type") == "camera" and optical_frame_id:
            nodes.append(
                Node(
                    package="tf2_ros",
                    executable="static_transform_publisher",
                    name=f"static_tf_{name}_optical",
                    arguments=[
                        "--x",
                        "0",
                        "--y",
                        "0",
                        "--z",
                        "0",
                        "--qx",
                        "-0.5",
                        "--qy",
                        "0.5",
                        "--qz",
                        "-0.5",
                        "--qw",
                        "0.5",  # ROS optical frame convention
                        "--frame-id",
                        frame_id,
                        "--child-frame-id",
                        optical_frame_id,
                    ],
                    output="screen",
                )
            )

    return nodes
