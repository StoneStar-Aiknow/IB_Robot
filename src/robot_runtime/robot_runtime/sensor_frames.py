"""Advertised camera frame conventions shared by every platform (D4/5b.3).

Moved out of the platform module (now ``robot_runtime_sim.gazebo``) (openspec sim-platform-registry D4):
the advertised-frame convention is a pure contract helper, not
simulator-specific, and the interface description (core) needs it without
importing a platform module. The Gazebo/MuJoCo sensor generation keeps
reading it through :func:`advertised_camera_frame` / this module.
"""

from __future__ import annotations


def advertised_camera_frames(periph: dict) -> tuple[str | None, str | None]:
    """(image frame, info frame) the description advertises — the single source.

    Convention: opencv/default → ``frame_id``; realsense → ``optical_frame_id``
    (the image frame is absent for ``direct_topic_remap`` — the native driver
    topic carries its own frames); camera_ros → no frames (the driver
    publishes its own). Used by the interface description and, via
    :func:`advertised_camera_frame`, by the gz ``frame_id`` and the MuJoCo
    sensor ``frame_name``.
    """
    name = periph.get("name", "camera")
    driver = str(periph.get("driver", "opencv"))
    if driver == "camera_ros":
        return None, None
    if driver == "realsense":
        optical = str(periph.get("optical_frame_id") or f"camera_{name}_optical_frame")
        return (None if periph.get("direct_topic_remap") else optical), optical
    frame = str(periph.get("frame_id") or f"camera_{name}_frame")
    return frame, frame


def advertised_camera_frame(periph: dict) -> str:
    """The frame id the runtime description advertises for this camera's images."""
    image_frame, _info_frame = advertised_camera_frames(periph)
    if image_frame is not None:
        return image_frame
    name = periph.get("name", "camera")
    if str(periph.get("driver", "opencv")) == "realsense":
        return str(periph.get("optical_frame_id") or f"camera_{name}_optical_frame")
    return str(periph.get("frame_id") or f"camera_{name}_frame")
